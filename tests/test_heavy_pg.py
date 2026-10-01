import io

import pytest
import torch

from src.experts import FullPointGroupExpert, hidden_layout_from_multiplicities
from src.experts.optimized import GroupedFullPointGroupExpert
from src.symmetry import PointGroupRegistry
LAYOUT = hidden_layout_from_multiplicities((16, 2, 2, 2, 2))


@pytest.mark.parametrize("depth", [0, -1, True, 2.5])
def test_invalid_depth(depth):
    with pytest.raises(ValueError, match="positive integer"):
        FullPointGroupExpert(PointGroupRegistry()["mmm"], LAYOUT, depth=depth)


@pytest.mark.parametrize("symbol", ["mmm", "4/mmm", "m-3m"])
def test_deep_grouped_parity_gradients_and_independent_blocks(symbol):
    group = PointGroupRegistry()[symbol]
    torch.manual_seed(42)
    reference = FullPointGroupExpert(group, LAYOUT, depth=12).double()
    grouped = GroupedFullPointGroupExpert(group, LAYOUT, depth=12).double()
    grouped.load_state_dict(reference.state_dict())
    assert len({b.weight.data_ptr() for b in reference.blocks}) == 12
    shallow = FullPointGroupExpert(group, LAYOUT)
    assert 2 * sum(p.numel() for p in reference.parameters()) == 12 * sum(p.numel() for p in shallow.parameters())
    x = torch.randn(4, LAYOUT.dimension, dtype=torch.float64, requires_grad=True)
    y = x.detach().clone().requires_grad_()
    a, b = reference(x), grouped(y)
    torch.testing.assert_close(a, b, atol=1e-9, rtol=1e-9)
    a.square().mean().backward()
    b.square().mean().backward()
    torch.testing.assert_close(x.grad, y.grad, atol=1e-8, rtol=1e-8)
    for p, q in zip(reference.parameters(), grouped.parameters()):
        assert p.grad is not None and torch.isfinite(p.grad).all()
        torch.testing.assert_close(p.grad, q.grad, atol=1e-8, rtol=1e-8)
    for block in reference.blocks:
        assert block.weight.grad.abs().sum() > 0
    stream = io.BytesIO()
    torch.save(grouped.state_dict(), stream)
    stream.seek(0)
    restored = GroupedFullPointGroupExpert(group, LAYOUT, depth=12).double()
    restored.load_state_dict(torch.load(stream, weights_only=True))
    torch.testing.assert_close(restored(x.detach()), b.detach())
    with pytest.raises(RuntimeError):
        shallow.load_state_dict(grouped.state_dict())


def test_default_depth_state_and_output_compatibility():
    group = PointGroupRegistry()["mmm"]
    torch.manual_seed(7)
    default = FullPointGroupExpert(group, LAYOUT)
    torch.manual_seed(7)
    explicit = FullPointGroupExpert(group, LAYOUT, depth=2)
    for key, value in default.state_dict().items():
        assert torch.equal(value, explicit.state_dict()[key])


def test_production_heavy_parameter_budget():
    from src.irreps import IrrepLayout, IrrepTerm
    from src.training.benchmark import CachedBackboneTensorModel
    from src.cli.reduced_dpa4_relative_pg_train import ARCHITECTURE
    from src.cli.reduced_protocol import REDUCED_POINT_GROUPS
    from src.symmetry import PointGroupAncestorDAG, build_point_group_parent_dag
    dag = PointGroupAncestorDAG.from_path()
    paths = [build_point_group_parent_dag(g, dag.number(g), dag) for g in REDUCED_POINT_GROUPS]
    edges = tuple(sorted({e.edge_id for path in paths for e in path.embeddings}))
    groups = dag.symbols(sorted({n for g in REDUCED_POINT_GROUPS for n in dag.ancestors(dag.number(g))}))
    source = IrrepLayout(tuple(IrrepTerm(64, l, p, f"dpa4_l{l}_{p}")
                              for l in range(5) for p in ("e", "o")))
    model = CachedBackboneTensorModel(source, ARCHITECTURE, "dielectric", groups, edges,
                                     hidden_layout=LAYOUT, pg_expert_depth=12, pg_weighting="within_cross_chain")
    assert len(groups) == 15 and len(edges) == 24
    count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert count == 769156
    assert abs(count - 738661) < 0.05 * 738661
    increment = sum(sum(p.numel() for p in expert.blocks[0].parameters())
                    for expert in model.downstream.pg_experts.values())
    assert abs(count - 738661) < min(abs(count + increment - 738661), abs(count - increment - 738661))


def test_deep_full_model_forward_backward_and_checkpoint(tmp_path):
    from src.cli.reduced_dpa4_relative_pg_train import ARCHITECTURE
    from src.training.benchmark import CachedBackboneTensorModel
    from src.training.checkpoint import save_checkpoint, load_checkpoint
    from src.training.normalization import CoefficientNormalizer
    from src.training.smoke import default_convention_metadata
    from src.heads import TARGET_LAYOUTS
    from src.data import TrainingUnit
    from src.irreps import O3FeatureBatch
    from test_dispatcher_readout import SMALL_LAYOUT, _graph, _record

    def build(depth):
        return CachedBackboneTensorModel(SMALL_LAYOUT, ARCHITECTURE, "dielectric",
                                        ("2/m",), hidden_layout=LAYOUT, pg_expert_depth=depth)
    model = build(12)
    graph = _graph()
    features = O3FeatureBatch(torch.randn(graph.num_nodes, SMALL_LAYOUT.dimension),
                             SMALL_LAYOUT, graph.node_batch)
    symmetries = (_record("2/m"), _record("2/m"))
    output = model(features, graph, symmetries).raw_cartesian
    output.square().mean().backward()
    for block in next(iter(model.downstream.pg_experts.values())).blocks:
        assert block.weight.grad is not None and torch.isfinite(block.weight.grad).all()
    optimizer = torch.optim.AdamW(model.parameters())
    optimizer.step()
    expected = model(features, graph, symmetries).raw_cartesian.detach()
    unit = TrainingUnit("curated_reduced_total", "dielectric")
    layout = TARGET_LAYOUTS["dielectric"]
    normalizer = CoefficientNormalizer.fit(torch.randn(3, layout.dimension), layout, unit, split="train")
    path = tmp_path / "deep.pt"
    save_checkpoint(path, model=model, optimizer=optimizer, architecture=ARCHITECTURE,
                    unit=unit, convention=default_convention_metadata(), normalizer=normalizer, step=120)
    kwargs = dict(optimizer=None, expected_architecture=ARCHITECTURE, expected_unit=unit,
                  expected_convention=default_convention_metadata(), expected_layout=layout)
    restored = build(12)
    load_checkpoint(path, model=restored, **kwargs)
    torch.testing.assert_close(restored(features, graph, symmetries).raw_cartesian, expected)
    with pytest.raises(ValueError, match="depth mismatch"):
        load_checkpoint(path, model=build(2), **kwargs)
