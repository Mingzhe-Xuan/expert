import copy

import pytest
import torch

from src.experts.optimized import VectorizedMaterialWeights
from src.experts.modules import ContinuousEdgeGate, material_point_group_stick_breaking_weights
from src.experts import PointGroupTensorModel
from src.configs import ArchitectureConfig
from src.irreps import O3FeatureBatch
from src.symmetry import ParentDAGSpec, PointGroupAncestorDAG, build_point_group_parent_dag
from src.experts.chain_routing import HierarchicalChainRouter, VectorizedChainRouter, migrate_legacy_edge_scales
from test_parent_dag_training import _edge
from test_dispatcher_readout import SMALL_LAYOUT, _graph, _record


@pytest.mark.parametrize("mode", ["zero", "mixed", "positive"])
def test_legacy_vectorized_weights_and_residual_gradients(mode):
    dag = ParentDAGSpec("test", 1, (_edge(3, 2), _edge(2, 1), _edge(4, 1)))
    gate = ContinuousEdgeGate(tuple(e.edge_id for e in dag.embeddings), initial_sigma=0.5).double()
    reference_gate = copy.deepcopy(gate)
    values = {"zero": [0., 0., 0.], "mixed": [0., .2, .3], "positive": [.1, .2, .3]}[mode]
    residuals = {e.checksum: torch.tensor(v, dtype=torch.float64, requires_grad=True)
                 for e, v in zip(dag.embeddings, values)}
    reference_residuals = {k: v.detach().clone().requires_grad_() for k, v in residuals.items()}
    router = VectorizedMaterialWeights()
    actual = router(dag, residuals, gate)
    expected = material_point_group_stick_breaking_weights(dag, reference_residuals, reference_gate)
    for k in actual:
        torch.testing.assert_close(actual[k], expected[k], atol=1e-14, rtol=1e-12)
    sum(k * v for k, v in actual.items()).backward()
    sum(k * v for k, v in expected.items()).backward()
    for p, q in zip(gate.parameters(), reference_gate.parameters()):
        torch.testing.assert_close(p.grad, q.grad, atol=1e-14, rtol=1e-12)
    for k in residuals:
        torch.testing.assert_close(residuals[k].grad, reference_residuals[k].grad, atol=1e-14, rtol=1e-12)
    # A fresh graph is built even when topology is reused, including after a parameter update.
    with torch.no_grad():
        next(gate.parameters()).add_(.1)
    sum(k * v for k, v in router(dag, residuals, gate).items()).backward()
    assert len(router.plans) == 1


@pytest.mark.parametrize("value", [-1., float("nan"), float("inf")])
def test_invalid_residual_rejected(value):
    dag = ParentDAGSpec("test", 1, (_edge(2, 1),))
    with pytest.raises(ValueError, match="finite and non-negative"):
        VectorizedMaterialWeights()(dag, {dag.embeddings[0].checksum: value},
                                    ContinuousEdgeGate((dag.embeddings[0].edge_id,)))


def test_singleton_and_bounded_cache():
    router = VectorizedMaterialWeights(capacity=1)
    assert router(ParentDAGSpec("single", 32, ()), {}, ContinuousEdgeGate(()))[32] == 1
    for name in ("a", "b"):
        dag = ParentDAGSpec(name, 1, (_edge(2, 1),))
        router(dag, {dag.embeddings[0].checksum: .1}, ContinuousEdgeGate((dag.embeddings[0].edge_id,)))
    assert len(router.plans) == 1


@pytest.mark.parametrize("current", [1, 5, 7, 8, 15, 20, 31, 32])
def test_real_dag_weight_and_sigma_gradient_parity(current):
    dag = build_point_group_parent_dag("real", current, PointGroupAncestorDAG.from_path())
    gate = ContinuousEdgeGate(tuple(sorted({e.edge_id for e in dag.embeddings}))).double()
    residuals = {e.checksum: (i % 5) * .025 for i, e in enumerate(dag.embeddings)}
    outputs = []
    gradients = []
    for function in (material_point_group_stick_breaking_weights, VectorizedMaterialWeights()):
        gate.zero_grad(set_to_none=True)
        weights = function(dag, residuals, gate)
        outputs.append(torch.stack(list(weights.values())))
        if dag.embeddings:
            sum(k * v for k, v in weights.items()).backward()
            gradients.append({k: p.grad.clone() for k, p in gate.named_parameters() if p.grad is not None})
    torch.testing.assert_close(outputs[0], outputs[1], atol=1e-12, rtol=1e-10)
    if gradients:
        assert gradients[0].keys() == gradients[1].keys()
        for k in gradients[0]:
            torch.testing.assert_close(gradients[0][k], gradients[1][k], atol=1e-12, rtol=1e-10)


@pytest.mark.parametrize("gates,routing", [(True, False), (True, True)])
@pytest.mark.parametrize("weighting", ["legacy", "within_cross_chain"])
def test_complete_model_checkpoint_output_and_gradient_parity(gates, routing, weighting):
    torch.manual_seed(12)
    config = ArchitectureConfig("B+A+PGE+R", "full_o3", "none", "full_o3", "full_pg")
    class_dag = PointGroupAncestorDAG.from_path()
    dags = tuple(build_point_group_parent_dag(str(i), 5, class_dag) for i in range(2))
    edge_ids = tuple(sorted({e.edge_id for dag in dags for e in dag.embeddings}))
    numbers = class_dag.ancestors(5)
    symbols = class_dag.symbols(numbers)
    kwargs = dict(hidden_layout=SMALL_LAYOUT, expert_point_groups=symbols,
                  material_edge_ids=edge_ids, cutoff=1.4, pg_weighting=weighting)
    reference = PointGroupTensorModel(config, "dielectric", **kwargs,
                                     grouped_pg_gates=False, vectorized_pg_routing=False)
    actual = PointGroupTensorModel(config, "dielectric", **kwargs,
                                  grouped_pg_gates=gates, vectorized_pg_routing=routing)
    assert reference.state_dict().keys() == actual.state_dict().keys()
    actual.load_state_dict(reference.state_dict(), strict=True)
    graph = _graph()
    metadata = dict(parent_dags=dags,
                    parent_residuals=tuple({e.checksum: .04 for e in d.embeddings} for d in dags))
    x = torch.randn(graph.num_nodes, SMALL_LAYOUT.dimension)
    outputs, gradients, inputs = [], [], []
    for model in (reference, actual):
        values = x.clone().requires_grad_()
        output = model(O3FeatureBatch(values, SMALL_LAYOUT, graph.node_batch), graph,
                       (_record("2/m"), _record("2/m")), **metadata).raw_cartesian
        output.square().sum().backward()
        outputs.append(output)
        gradients.append({n: p.grad for n, p in model.named_parameters() if p.grad is not None})
        inputs.append(values.grad)
    torch.testing.assert_close(outputs[0], outputs[1], atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(inputs[0], inputs[1], atol=2e-6, rtol=2e-5)
    assert gradients[0].keys() == gradients[1].keys()
    for key in gradients[0]:
        torch.testing.assert_close(gradients[0][key], gradients[1][key], atol=2e-6, rtol=2e-5)
    assert reference.last_dispatch_stats == actual.last_dispatch_stats


def test_standalone_profiler_complete_pass(tmp_path):
    from src.experts.dispatcher import _extract_graph
    from src.profiling.standalone_pg import compare_standalone_pg
    from src.training.benchmark import FrozenFeatureExample

    graph = _graph()
    rows = []
    for i in range(2):
        indices, local = _extract_graph(graph, i)
        rows.append(FrozenFeatureExample(str(i), torch.randn(len(indices), SMALL_LAYOUT.dimension),
                    local, _record("2/m"), torch.zeros(1, 6), torch.zeros(1, 3, 3)))
    config = ArchitectureConfig("B+A+PGE+R", "full_o3", "none", "full_o3", "full_pg")
    report = compare_standalone_pg(SMALL_LAYOUT, config, rows, expert_point_groups=("2/m",),
        edge_ids=(), output_dir=tmp_path / "profile", provenance={"test": True},
        device="cpu", warmup=1, repeats=1)
    assert report["status"] == "passed"
    assert set(report["variants"]) == {"legacy", "reference", "grouped_gates", "optimized"}
    for name, variant in report["variants"].items():
        assert variant["batch"]["samples"] == 2
        assert variant["median_ms"]["total_ms"] > 0
        assert (tmp_path / "profile" / name / "trace.json").is_file()
    with pytest.raises(FileExistsError):
        compare_standalone_pg(SMALL_LAYOUT, config, rows, expert_point_groups=("2/m",),
            edge_ids=(), output_dir=tmp_path / "profile", provenance={}, device="cpu")


def test_wrapper_router_is_shared_and_chain_formula_is_exact():
    from src.models.global_experts.routing import HierarchicalChainRouter as WrapperRouter
    from src.models.global_experts.vectorized_routing import VectorizedChainRouter as WrapperVectorized
    assert WrapperRouter is HierarchicalChainRouter
    assert WrapperVectorized is VectorizedChainRouter
    dag = build_point_group_parent_dag("chain", 5, PointGroupAncestorDAG.from_path())
    router = HierarchicalChainRouter(sorted({e.edge_id for e in dag.embeddings}), temperature=.7).double()
    residuals = {e.checksum: torch.tensor(.03 * (1+i % 4), dtype=torch.float64, requires_grad=True)
                 for i, e in enumerate(dag.embeddings)}
    result = router(dag, residuals)
    first = []
    for path, omega in zip(dag.current_to_root_embedding_paths(), result.omega):
        energies = [(residuals[e.checksum].detach() / router.scales[router.edge_indices[e.edge_id]])**2 for e in path]
        gates = torch.stack([-torch.expm1(-energy) for energy in energies])
        expected = [gates.prod()]
        for i, energy in enumerate(energies):
            expected.append(gates[:i].prod() * torch.exp(-energy))
        torch.testing.assert_close(omega, torch.stack(expected))
        first.append(energies[0])
    torch.testing.assert_close(result.pi, torch.softmax(-torch.stack(first)/.7, 0))
    sum(k*v for k, v in result.alpha.items()).backward()
    assert router.log_scales.grad is not None
    assert all(v.grad is None for v in residuals.values())


def test_legacy_checkpoint_migration_preserves_sigmas_and_other_weights():
    config = ArchitectureConfig("B+PGE+R", "none", "none", "full_o3", "full_pg")
    dag = build_point_group_parent_dag("migrate", 5, PointGroupAncestorDAG.from_path())
    ids = tuple(sorted({e.edge_id for e in dag.embeddings}))
    kwargs = dict(hidden_layout=SMALL_LAYOUT, expert_point_groups=("2/m",), material_edge_ids=ids)
    legacy = PointGroupTensorModel(config, "dielectric", **kwargs, pg_weighting="legacy")
    current = PointGroupTensorModel(config, "dielectric", **kwargs, pg_weighting="within_cross_chain")
    state = legacy.state_dict()
    migrated = migrate_legacy_edge_scales(state, current.edge_gate)
    current.load_state_dict(migrated, strict=True)
    expected = torch.stack([torch.nn.functional.softplus(legacy.edge_gate.log_sigmas["edge_"+key]).clamp_min(1e-8) for key in ids])
    torch.testing.assert_close(current.edge_gate.scales, expected)
    assert "edge_gate.log_scales" not in state
    for key in state:
        if not key.startswith("edge_gate."):
            assert torch.equal(state[key], migrated[key])
    with pytest.raises(ValueError, match="already contains"):
        migrate_legacy_edge_scales(migrated, current.edge_gate)


def test_new_routing_checkpoint_metadata_blocks_temperature_mismatch(tmp_path):
    from src.data import TrainingUnit
    from src.heads import TARGET_LAYOUTS
    from src.training import CoefficientNormalizer, save_checkpoint, load_checkpoint
    from test_dispatcher_readout import _convention

    config = ArchitectureConfig("B+PGE+R", "none", "none", "full_o3", "full_pg")
    model = PointGroupTensorModel(config, "dielectric", hidden_layout=SMALL_LAYOUT,
        expert_point_groups=("2/m",), material_edge_ids=(), pg_weighting="within_cross_chain")
    unit = TrainingUnit("jarvis_tensor", "dielectric")
    normalizer = CoefficientNormalizer.fit(torch.randn(3, 6), TARGET_LAYOUTS["dielectric"], unit, split="train")
    path = tmp_path / "new.pt"
    optimizer = torch.optim.Adam(model.parameters())
    save_checkpoint(path, model=model, optimizer=optimizer, architecture=config, unit=unit,
                    convention=_convention(), normalizer=normalizer, step=0)
    options = dict(model=model, optimizer=None, expected_architecture=config, expected_unit=unit,
                   expected_convention=_convention(), expected_layout=TARGET_LAYOUTS["dielectric"])
    load_checkpoint(path, **options)
    model.edge_gate.temperature = .5
    before = {k: v.clone() for k, v in model.state_dict().items()}
    with pytest.raises(ValueError, match="PG routing metadata"):
        load_checkpoint(path, **options)
    assert all(torch.equal(v, model.state_dict()[k]) for k, v in before.items())
