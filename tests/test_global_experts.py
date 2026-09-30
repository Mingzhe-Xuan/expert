from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from src.models.global_experts.routing import HierarchicalChainRouter
from src.models.global_experts.config import GlobalExpertsConfig, layout_from_irreps
from src.models.global_experts.adapter import IdentityMessageAdapter
from src.models.global_experts.model import GlobalExpertsModel, CrystalRouting
from src.models.global_experts.frames import standardize_o3
from src.symmetry import PointGroupAncestorDAG, build_point_group_parent_dag
from src.symmetry.registry import _layout_irreps
from e3nn import o3
from test_gmtnet_attention import official


def routing_case(number=8):
    classes = PointGroupAncestorDAG.from_path()
    dag = build_point_group_parent_dag("sample", number, classes)
    ids = tuple(edge.edge_id for edge in dag.embeddings)
    residuals = {e.checksum: 0.03 + 0.01 * i for i, e in enumerate(dag.embeddings)}
    return classes, dag, ids, residuals


def test_chain_formula_gradients_and_exact_collapse():
    classes, dag, ids, residuals = routing_case()
    router = HierarchicalChainRouter(ids, class_dag=classes).double()
    result = router(dag, residuals)
    assert len(result.pi) > 1
    explicit = []
    features = {pg: torch.randn(5, dtype=torch.float64) for pg in result.alpha}
    for k, path in enumerate(dag.current_to_root_embedding_paths()):
        remaining, ancestors = torch.tensor(1.0, dtype=torch.float64), []
        for edge in path:
            q = residuals[edge.checksum] / router.scales[ids.index(edge.edge_id)]
            a = 1 - torch.exp(-q.square())
            ancestors.append(remaining * (1 - a))
            remaining = remaining * a
        expected = torch.stack([remaining, *ancestors])
        torch.testing.assert_close(result.omega[k], expected)
        explicit.append(
            result.pi[k]
            * sum(w * features[pg] for pg, w in zip(result.groups[k], expected))
        )
    collapsed = sum(w * features[pg] for pg, w in result.alpha.items())
    torch.testing.assert_close(sum(explicit), collapsed)
    collapsed.square().sum().backward()
    assert torch.isfinite(router.log_scales.grad).all()
    assert router.log_scales.grad.abs().sum() > 0
    with pytest.raises(ValueError, match="exactly"):
        router(dag, {})
    invalid = dict(residuals)
    invalid[next(iter(invalid))] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        router(dag, invalid)


def test_chain_boundary_and_detached_residuals():
    classes, dag, ids, residuals = routing_case()
    router = HierarchicalChainRouter(ids, class_dag=classes).double()
    zero = {key: 0.0 for key in residuals}
    result = router(dag, zero)
    for weights in result.omega:
        assert weights[0] == 0 and weights[1] == 1 and weights[2:].sum() == 0
    torch.testing.assert_close(result.pi, torch.ones_like(result.pi) / len(result.pi))
    measured = {
        key: torch.tensor(value, dtype=torch.float64, requires_grad=True)
        for key, value in residuals.items()
    }
    result = router(dag, measured)
    sum(number * weight for number, weight in result.alpha.items()).backward()
    assert all(value.grad is None for value in measured.values())
    assert router.log_scales.grad is not None
    # Removing a cover edge cannot silently turn a maximal path into a shorter one.
    immediate_edge = next(
        edge
        for edge in dag.embeddings
        if edge.child_point_group_number == dag.current_point_group_number
    )
    incomplete = replace(dag, embeddings=(immediate_edge,))
    with pytest.raises(ValueError, match="exact offline"):
        router(
            incomplete,
            {edge.checksum: residuals[edge.checksum] for edge in incomplete.embeddings},
        )


def test_singleton_router():
    classes, dag, ids, residuals = routing_case(32)
    result = HierarchicalChainRouter(ids, class_dag=classes)(dag, residuals)
    assert result.groups == ((32,),)
    assert result.alpha[32] == 1


@pytest.mark.parametrize("backend", ["full_o3", "o2_tp"])
@pytest.mark.parametrize("sign", [1, -1])
def test_adapter_identity_and_o3(sign, backend):
    layout = layout_from_irreps("2x0e + 1x1o + 1x2e")
    adapter = IdentityMessageAdapter(layout, backend=backend).double()
    x = torch.randn(4, layout.dimension, dtype=torch.float64)
    edges = torch.tensor([[0, 0, 1, 2], [1, 2, 3, 0]])
    vectors = torch.randn(4, 3, dtype=torch.float64)
    assert torch.equal(adapter(x, edges, vectors), x)
    adapter.residual_logit.data.fill_(0.3)
    q = sign * o3.rand_matrix(dtype=torch.float64)
    d = _layout_irreps(layout).D_from_matrix(q)
    torch.testing.assert_close(
        adapter(x @ d.T, edges, vectors @ q.T),
        adapter(x, edges, vectors) @ d.T,
        atol=1e-6,
        rtol=1e-5,
    )
    assert torch.equal(adapter(x, edges[:, :0], vectors[:0]), x)


@pytest.mark.parametrize("sign", [1, -1])
def test_parity_aware_standardization(sign):
    cell = torch.tensor(
        [[3.0, 0.1, 0.2], [0.2, 4.0, 0.1], [0.1, 0.3, 5.0]], dtype=torch.float64
    )
    positions = (
        torch.tensor([[0.1, 0.2, 0.3], [0.3, 0.7, 0.4]], dtype=torch.float64) @ cell
    )
    species = torch.tensor([14, 8])
    _, frame = standardize_o3(positions, cell, species)
    q = sign * o3.rand_matrix(dtype=torch.float64)
    _, transformed = standardize_o3(positions @ q.T, cell @ q.T, species)
    torch.testing.assert_close(transformed @ q, frame, atol=1e-7, rtol=1e-7)


def test_full_model_forward_backward_baseline_and_o3(official):
    torch.manual_seed(2026)
    classes, dag, ids, residuals = routing_case(8)
    config = GlobalExpertsConfig(
        expert_irreps="2x0e + 1x1o + 1x2e",
        adapter_initial_logit=0.2,
        branch_initial_logit=-1.0,
    )
    args = SimpleNamespace(target="dielectric", use_mask=False, reduce_cell=False)
    baseline = official.gmtnet.GMTNet(args).double().eval()
    model = (
        GlobalExpertsModel(
            baseline,
            official.gmtnet.equality_adjustment,
            expert_numbers=classes.ancestors(8),
            edge_ids=ids,
            config=config,
            class_dag=classes,
        )
        .double()
        .eval()
    )
    data = SimpleNamespace(
        x=torch.randn(4, 92, dtype=torch.float64),
        edge_index=torch.tensor([[0, 0, 1, 2, 3], [1, 2, 2, 3, 0]]),
        edge_attr=torch.randn(5, 3, dtype=torch.float64),
        batch=torch.zeros(4, dtype=torch.long),
    )
    frame = o3.rand_matrix(dtype=torch.float64)
    routing = (CrystalRouting("sample", dag, residuals, frame),)
    equality = torch.zeros(1, 9, 9, dtype=torch.bool)
    mask = torch.eye(32, dtype=torch.float64).unsqueeze(0)
    prediction, diagnostics = model(
        data, mask, equality, routing, return_diagnostics=True
    )
    assert prediction.shape == (1, 3, 3)
    for sign in [1, -1]:
        q = sign * o3.rand_matrix(dtype=torch.float64)
        changed = SimpleNamespace(**vars(data))
        changed.edge_attr = data.edge_attr @ q.T
        transformed_routing = (replace(routing[0], input_to_standard=frame @ q.T),)
        actual, rotated = model(
            changed, mask, equality, transformed_routing, return_diagnostics=True
        )
        torch.testing.assert_close(
            rotated["fused_features"],
            diagnostics["fused_features"] @ model.global_irreps.D_from_matrix(q).T,
            atol=2e-5,
            rtol=2e-5,
        )
        # Official dielectric probes use the default float32 dtype even for double carriers.
        tensor_q = q.to(prediction)
        torch.testing.assert_close(
            actual, tensor_q @ prediction @ tensor_q.T, atol=2e-5, rtol=2e-5
        )
    prediction.square().sum().backward()
    for parameter in (
        model.branch_logit,
        model.router.log_scales,
        model.adapter.residual_logit,
        model.input_map.weight,
        model.output_map.weight,
    ):
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0
    model.config = replace(config, auxiliary_enabled=False)
    torch.testing.assert_close(
        model(data, mask, equality), baseline(data, mask, equality), rtol=0, atol=0
    )


def test_nontrivial_transported_mask_equivariance(official):
    model, splits = _tiny_model_and_splits(official)
    from src.training.global_experts.runner import collate
    from src.symmetry import PointGroupRegistry
    from src.models.global_experts.config import layout_from_irreps

    model.eval()
    data, mask, equality, routing, _, _ = collate(splits["train"][:1], "cpu")
    projector = (
        PointGroupRegistry()[8]
        .invariant_projector(layout_from_irreps(str(model.global_irreps)))
        .float()
    )
    mask[0] = projector
    original = model(data, mask, equality, routing)
    q = -o3.rand_matrix()
    d = model.global_irreps.D_from_matrix(q)
    data.edge_attr = data.edge_attr @ q.T
    rotated_routing = (
        replace(routing[0], input_to_standard=routing[0].input_to_standard @ q.T),
    )
    actual = model(data, (d @ mask[0] @ d.T).unsqueeze(0), equality, rotated_routing)
    torch.testing.assert_close(actual, q @ original @ q.T, atol=2e-5, rtol=2e-5)


def test_invariant_input_archive_and_order_guard(tmp_path):
    from src.training.global_experts.data import (
        attach_invariant_inputs,
        file_sha256,
        row_digest,
    )

    row = {
        "sample_id": "a",
        "graph": {
            "x": torch.randn(2, 92),
            "edge_index": torch.tensor([[0, 1], [1, 0]]),
            "edge_attr": torch.randn(2, 3),
        },
    }
    archive = tmp_path / "inputs.pt"
    torch.save(
        {
            "schema_version": 1,
            "features": {"a": torch.randn(2, 6, requires_grad=True)},
            "graph_rows": {"a": row_digest(row)},
            "kind": "dpa4_invariant",
        },
        archive,
    )
    modified, metadata = attach_invariant_inputs(
        {"train": [row]}, archive, expected_sha256=file_sha256(archive)
    )
    assert modified["train"][0]["graph"]["x"].shape == (2, 6)
    assert not modified["train"][0]["graph"]["x"].requires_grad
    assert row["graph"]["x"].shape == (2, 92)
    assert metadata["input_dimension"] == 6
    changed = {**row, "graph": {**row["graph"], "x": row["graph"]["x"].flip(0)}}
    with pytest.raises(ValueError, match="graph/order"):
        attach_invariant_inputs(
            {"train": [changed]}, archive, expected_sha256=file_sha256(archive)
        )


def _tiny_model_and_splits(official, config=None):
    classes, dag, ids, residuals = routing_case(8)
    config = config or GlobalExpertsConfig(expert_irreps="2x0e + 1x1o + 1x2e")
    baseline = official.gmtnet.GMTNet(
        SimpleNamespace(target="dielectric", use_mask=True, reduce_cell=False)
    )
    model = GlobalExpertsModel(
        baseline,
        official.gmtnet.equality_adjustment,
        expert_numbers=classes.ancestors(8),
        edge_ids=ids,
        config=config,
        class_dag=classes,
    )
    splits = {}
    for name, size in [("train", 2), ("validation", 1), ("test", 1)]:
        rows = []
        for index in range(size):
            key = f"{name}-{index}"
            rows.append(
                {
                    "sample_id": key,
                    "graph": {
                        "x": torch.randn(4, 92),
                        "edge_index": torch.tensor([[0, 0, 1, 2, 3], [1, 2, 2, 3, 0]]),
                        "edge_attr": torch.randn(5, 3),
                    },
                    "feature_mask": torch.eye(32),
                    "equality": torch.zeros(9, 9, dtype=torch.bool),
                    "target": torch.eye(3),
                    "routing": CrystalRouting(
                        key, replace(dag, material_id=key), residuals, torch.eye(3)
                    ),
                }
            )
        splits[name] = rows
    return model, splits


def test_batched_ownership_unique_expert_calls_and_checkpoint(official, tmp_path):
    from src.training.global_experts.runner import collate, load_checkpoint

    model, splits = _tiny_model_and_splits(official)
    model.eval()
    rows = splits["train"]
    data, mask, equality, routing, _, _ = collate(rows, "cpu")
    calls = {key: 0 for key in model.experts}

    def increment(key):
        def hook(*_):
            calls[key] += 1

        return hook

    handles = [
        module.register_forward_hook(increment(key))
        for key, module in model.experts.items()
    ]
    joint = model(data, mask, equality, routing)
    assert all(value == 1 for value in calls.values())
    for handle in handles:
        handle.remove()
    independent = []
    for row in rows:
        graph, m, e, r, _, _ = collate([row], "cpu")
        independent.append(model(graph, m, e, r))
    torch.testing.assert_close(joint, torch.cat(independent), atol=2e-6, rtol=2e-5)
    checkpoint = tmp_path / "model.pt"
    torch.save(
        {
            "schema_version": 1,
            "model_metadata": model.metadata(),
            "provenance": {"x": "test"},
            "model_state": model.state_dict(),
        },
        checkpoint,
    )
    restored, _ = _tiny_model_and_splits(official)
    load_checkpoint(restored, checkpoint, provenance={"x": "test"})
    restored.eval()
    torch.testing.assert_close(restored(data, mask, equality, routing), joint)
    with pytest.raises(ValueError, match="identity"):
        load_checkpoint(restored, checkpoint, provenance={"x": "stale"})
    with pytest.raises(ValueError, match="sample order"):
        model(data, mask, equality, tuple(reversed(routing)))


def test_training_lifecycle_and_frozen_global(official, tmp_path):
    from src.training.global_experts import (
        GlobalExpertsTrainConfig,
        train_global_experts,
    )

    config = GlobalExpertsConfig(
        expert_irreps="2x0e + 1x1o + 1x2e", freeze_global=True, use_equiv_attn=True
    )
    model, splits = _tiny_model_and_splits(official, config)
    global_before = {k: v.clone() for k, v in model.global_model.state_dict().items()}
    branch_before = model.branch_logit.detach().clone()
    summary = train_global_experts(
        model,
        splits,
        output_dir=tmp_path / "run",
        provenance={"dataset": "synthetic"},
        config=GlobalExpertsTrainConfig(epochs=1, batch_size=2, checkpoint_interval=1,
                                       minimum_checkpoint_epoch_exclusive=0),
        device="cpu",
    )
    assert summary["status"] == "passed" and summary["best_epoch"] == 1
    assert (tmp_path / "run/epoch-0001.pt").is_file()
    assert len((tmp_path / "run/predictions.jsonl").read_text().splitlines()) == 1
    assert not torch.equal(model.branch_logit, branch_before)
    for key, value in model.global_model.state_dict().items():
        assert torch.equal(value, global_before[key])
    for expert in model.experts.values():
        assert any(
            p.grad is not None
            and torch.isfinite(p.grad).all()
            and p.grad.abs().sum() > 0
            for p in expert.parameters()
        )


def test_masked_huber_missing_entries():
    from src.training.global_experts.runner import masked_huber

    prediction = torch.tensor([1.0, 4.0], requires_grad=True)
    target = torch.tensor([0.0, float("nan")])
    valid = torch.tensor([True, False])
    loss = masked_huber(prediction, target, valid)
    loss.backward()
    torch.testing.assert_close(prediction.grad, torch.tensor([1.0, 0.0]))


def test_routing_cache_integrity_and_frame_covariance(tmp_path):
    from src.training.global_experts.data import prepare_routing
    from src.features import cgcnn_node_features

    cell = torch.diag(torch.tensor([3.0, 4.0, 5.0], dtype=torch.float64))
    sample = SimpleNamespace(
        sample_id="sample",
        atomic_numbers=torch.tensor([14]),
        cartesian_positions=torch.zeros(1, 3, dtype=torch.float64),
        lattice=cell,
    )
    vectors = torch.cat([cell, -cell]).float()
    row = {
        "sample_id": "sample",
        "graph": {
            "x": cgcnn_node_features(sample.atomic_numbers),
            "edge_index": torch.zeros(2, 6, dtype=torch.long),
            "edge_attr": vectors,
        },
    }
    splits = {"train": [row]}
    path = tmp_path / "routing.pt"
    enriched, identity = prepare_routing(
        splits, [sample], path, dataset_sha256="dataset", graph_cache_sha256="graph"
    )
    loaded, _ = prepare_routing(
        splits, [sample], path, dataset_sha256="dataset", graph_cache_sha256="graph"
    )
    assert (
        loaded["train"][0]["routing"].residuals
        == enriched["train"][0]["routing"].residuals
    )
    q = -o3.rand_matrix(dtype=torch.float64)
    rotated_sample = SimpleNamespace(**vars(sample))
    rotated_sample.lattice = cell @ q.T
    rotated_row = {
        **row,
        "graph": {**row["graph"], "edge_attr": vectors.double() @ q.T},
    }
    transformed, _ = prepare_routing(
        {"train": [rotated_row]},
        [rotated_sample],
        tmp_path / "rotated.pt",
        dataset_sha256="dataset",
        graph_cache_sha256="rotated",
    )
    torch.testing.assert_close(
        transformed["train"][0]["routing"].input_to_standard @ q,
        loaded["train"][0]["routing"].input_to_standard,
        atol=1e-7,
        rtol=1e-7,
    )
    for key, value in loaded["train"][0]["routing"].residuals.items():
        assert transformed["train"][0]["routing"].residuals[key] == pytest.approx(
            value, abs=1e-7
        )
    with pytest.raises(ValueError, match="identity"):
        prepare_routing(
            splits, [sample], path, dataset_sha256="wrong", graph_cache_sha256="graph"
        )
    broken = torch.load(path, weights_only=True)
    broken["records"]["sample"]["residuals"] = {}
    torch.save(broken, path)
    with pytest.raises(ValueError, match="keys"):
        prepare_routing(
            splits, [sample], path, dataset_sha256="dataset", graph_cache_sha256="graph"
        )


def test_cli_exposes_all_model_and_training_parameters():
    from dataclasses import fields
    from src.cli.global_experts_train import parser
    from src.training.global_experts import GlobalExpertsTrainConfig

    parsed = parser().parse_args(
        [
            "--official-root",
            "source",
            "--graph-cache",
            "graphs.pt",
            "--routing-cache",
            "route.pt",
            "--output-dir",
            "out",
            "--use-equiv-attn",
            "--no-auxiliary-enabled",
            "--chain-temperature",
            ".4",
        ]
    )
    for config in (GlobalExpertsConfig, GlobalExpertsTrainConfig):
        assert all(hasattr(parsed, field.name) for field in fields(config))
    assert parsed.use_equiv_attn and not parsed.auxiliary_enabled
    assert parsed.chain_temperature == 0.4
    assert parsed.input_features == "dpa4"
    assert parsed.feature_shards == 64


def test_cli_input_conflicts_fail_before_loading():
    from src.cli.global_experts_train import parser, run

    base = ["--official-root", "source", "--graph-cache", "graph.pt",
            "--routing-cache", "route.pt", "--output-dir", "out"]
    with pytest.raises(ValueError, match="required together"):
        run(parser().parse_args(base + ["--node-features", "features.pt"]))
    with pytest.raises(ValueError, match="conflicts"):
        run(parser().parse_args(base + ["--input-features", "cgcnn",
            "--node-features", "features.pt", "--node-feature-sha256", "abc"]))


def test_profiler_artifacts_and_cleanup(official, tmp_path):
    from src.profiling.global_experts import profile_global_experts
    from src.training.global_experts import GlobalExpertsTrainConfig
    model, splits = _tiny_model_and_splits(official)
    report = profile_global_experts(
        model, splits, output_dir=tmp_path / "profile", provenance={},
        config=GlobalExpertsTrainConfig(batch_size=2), warmup=1, repeats=1)
    assert report["status"] == "passed"
    assert report["parameters_with_grad"] > 0
    assert report["timings"][0]["backward_ms"] > 0
    names = {e["name"] for e in report["events"]}
    assert {"pass/forward", "pass/backward", "module/adapter", "module/router"} <= names
    assert any(name.startswith("module/experts.") for name in names)
    assert "encode_nodes" not in model.__dict__
    assert "forward" not in model.adapter.__dict__
    assert (tmp_path / "profile" / "trace.json").stat().st_size > 0


def test_frame_representation_uses_detached_cpu_metadata(official, monkeypatch):
    from src.training.global_experts.runner import collate
    model, splits = _tiny_model_and_splits(official, GlobalExpertsConfig(
        expert_irreps="2x0e + 1x1o + 1x2e", cache_frames=False))
    model.train()
    data, mask, equality, routing, target, _ = collate(splits["train"], "cpu")
    routing = tuple(replace(r, input_to_standard=r.input_to_standard.clone().requires_grad_())
                    for r in routing)
    original = o3.Irreps.D_from_matrix
    seen = []
    def checked(irreps, frame):
        assert frame.device.type == "cpu"
        assert not frame.requires_grad
        seen.append(frame.dtype)
        return original(irreps, frame)
    monkeypatch.setattr(o3.Irreps, "D_from_matrix", checked)
    output = model(data, mask, equality, routing)
    (output - target).square().mean().backward()
    assert seen == [data.x.dtype] * len(routing)
    assert model.input_map.weight.grad is not None
    assert all(r.input_to_standard.grad is None for r in routing)


@pytest.mark.parametrize("input_features", ["dpa4", "cgcnn"])
def test_default_cli_real_graph_cache_and_training(official, monkeypatch, tmp_path, input_features):
    from pathlib import Path
    from src.cli import global_experts_train as cli
    from src.data import TrainingUnit, IndependentTensorDataset, TensorSample
    from src.data.contracts import SplitManifest
    from src.heads import TARGET_LAYOUTS

    root = Path(official.gmtnet.__file__).parent
    unit = TrainingUnit("curated_reduced_total", "dielectric")
    samples = []
    for i, name in enumerate(("train", "validation", "test")):
        samples.append(
            TensorSample(
                name,
                unit,
                torch.diag(
                    torch.tensor([2.8 + 0.02 * i, 3.1, 3.4], dtype=torch.float64)
                ),
                torch.zeros(1, 3, dtype=torch.float64),
                torch.tensor([14]),
                torch.eye(3),
                torch.zeros(TARGET_LAYOUTS["dielectric"].dimension),
                "relative",
                {"manifest_sha256": "a" * 64},
            )
        )
    manifest = SplitManifest(
        seed=42,
        source="published",
        train=("train",),
        validation=("validation",),
        test=("test",),
        sample_to_group={name: name for name in ("train", "validation", "test")},
    )
    dataset = IndependentTensorDataset(unit, samples, manifest)
    monkeypatch.setattr(cli, "load_training_dataset", lambda *a, **kw: dataset)
    from types import SimpleNamespace
    from src.models.global_experts.config import layout_from_irreps

    layout = layout_from_irreps("2x0e + 2x1o")
    calls = []

    def frozen_features(**kwargs):
        calls.append(kwargs)
        return layout, {
            name: [SimpleNamespace(
                sample_id=name,
                graph=SimpleNamespace(atomic_numbers=torch.tensor([14])),
                features=torch.ones(1, layout.dimension),
            )] for name in ("train", "validation", "test")
        }

    monkeypatch.setattr(cli, "_load_dpa4_feature_splits", frozen_features)
    args = cli.parser().parse_args(
        [
            "--official-root",
            str(root),
            "--graph-cache",
            str(tmp_path / "graph.pt"),
            "--routing-cache",
            str(tmp_path / "route.pt"),
            "--output-dir",
            str(tmp_path / "run"),
            "--device",
            "cpu",
            "--epochs",
            "1",
            "--minimum-checkpoint-epoch-exclusive",
            "0",
            "--batch-size",
            "1",
            "--checkpoint-interval",
            "1",
            "--prepare-only",
        ]
    )
    if input_features == "cgcnn":
        args.input_features = "cgcnn"
    prepared = cli.run(args)
    assert prepared["status"] == "prepared"
    args.prepare_only = False
    summary = cli.run(args)
    assert summary["status"] == "passed"
    assert (
        summary["model_metadata"]["config"]["expert_irreps"]
        == GlobalExpertsConfig().expert_irreps
    )
    assert summary["provenance"]["graph_cache_sha256"]
    assert summary["provenance"]["routing_cache_sha256"]
    embedding = summary["provenance"]["input_embedding"]
    if input_features == "dpa4":
        assert len(calls) == 2
        assert calls[0]["selected_ids"] == calls[0]["full_ids"]
        assert embedding["input_dimension"] == 4
        assert embedding["kind"] == "frozen_dpa4_o3_invariant_per_copy"
        def missing_features(**kwargs):
            raise FileNotFoundError("missing frozen DPA4 feature shard")
        monkeypatch.setattr(cli, "_load_dpa4_feature_splits", missing_features)
        with pytest.raises(FileNotFoundError, match="DPA4"):
            cli.run(args)
    else:
        assert not calls
        assert embedding["kind"] == "jarvis_cgcnn"
        assert embedding["input_dimension"] == 92
