from dataclasses import replace
import torch
import pytest

from src.experts.modules import FullPointGroupExpert
from src.models.global_experts.optimized import GroupedFullPointGroupExpert, FrameRepresentationCache
from src.models.global_experts.routing import HierarchicalChainRouter
from src.models.global_experts.vectorized_routing import VectorizedChainRouter
from src.models.global_experts.config import layout_from_irreps, GlobalExpertsConfig
from src.symmetry import PointGroupRegistry
from test_global_experts import routing_case, _tiny_model_and_splits
from test_gmtnet_attention import official
from e3nn import o3


def assert_gradients_equal(reference, actual, atol=1e-8, rtol=1e-7):
    expected = dict(reference.named_parameters())
    for name, parameter in actual.named_parameters():
        other = expected[name]
        assert (parameter.grad is None) == (other.grad is None), name
        if parameter.grad is not None:
            torch.testing.assert_close(parameter.grad, other.grad, atol=atol, rtol=rtol)


@pytest.mark.parametrize("pg", range(1, 33))
def test_grouped_pg_equivalence_gradients_and_group_action(pg):
    group = PointGroupRegistry()[pg]
    layout = layout_from_irreps("2x0e + 1x1o + 1x2e")
    reference = FullPointGroupExpert(group, layout).double()
    actual = GroupedFullPointGroupExpert(group, layout).double()
    actual.load_state_dict(reference.state_dict(), strict=True)
    x = torch.randn(4, layout.dimension, dtype=torch.float64, requires_grad=True)
    y = x.detach().clone().requires_grad_()
    left, right = reference(x), actual(y)
    torch.testing.assert_close(right, left, atol=1e-10, rtol=1e-9)
    probe = torch.randn_like(left)
    (left * probe).sum().backward()
    (right * probe).sum().backward()
    torch.testing.assert_close(y.grad, x.grad, atol=1e-10, rtol=1e-9)
    assert_gradients_equal(reference, actual)
    for d in group.representation(layout)[::max(1, group.order // 3)]:
        torch.testing.assert_close(actual(x.detach() @ d.T), right.detach() @ d.T, atol=1e-7, rtol=1e-6)


@pytest.mark.parametrize("pg", [1, 5, 7, 8, 15, 20, 31, 32])
@pytest.mark.parametrize("zero", [False, True, "mixed"])
def test_vector_router_matches_values_and_gradients(pg, zero):
    classes, dag, ids, residuals = routing_case(pg)
    if zero is True:
        residuals = {k: 0.0 for k in residuals}
    elif zero == "mixed":
        residuals = {k: (v if i % 2 else 0.0) for i, (k, v) in enumerate(residuals.items())}
    reference = HierarchicalChainRouter(ids, class_dag=classes).double()
    actual = VectorizedChainRouter(ids, class_dag=classes).double()
    actual.load_state_dict(reference.state_dict())
    left, right = reference(dag, residuals), actual(dag, residuals)
    assert left.groups == right.groups
    for l, r in zip(left.omega, right.omega):
        torch.testing.assert_close(l, r, atol=1e-12, rtol=1e-10)
    torch.testing.assert_close(left.pi, right.pi, atol=1e-12, rtol=1e-10)
    for key in left.alpha:
        torch.testing.assert_close(left.alpha[key], right.alpha[key], atol=1e-12, rtol=1e-10)
    if ids:
        sum(k * v for k, v in left.alpha.items()).backward()
        sum(k * v for k, v in right.alpha.items()).backward()
        assert_gradients_equal(reference, actual)
    actual(dag, residuals)  # reuse immutable plan
    with pytest.raises(ValueError, match="exactly"):
        actual(dag, {**residuals, "unknown": 0.0})
    if residuals:
        with pytest.raises(ValueError, match="finite"):
            actual(dag, {**residuals, next(iter(residuals)): float("nan")})


def test_frame_cache_keys_eviction_and_invalid_frames(monkeypatch):
    cache = FrameRepresentationCache(2)
    irreps = o3.Irreps("1x0e + 1x1o")
    original = o3.Irreps.D_from_matrix
    calls = []
    def counted(self, frame):
        calls.append(frame.clone())
        return original(self, frame)
    monkeypatch.setattr(o3.Irreps, "D_from_matrix", counted)
    frame = torch.eye(3)
    ref = torch.zeros(1)
    first = cache.get(irreps, frame, ref)
    assert cache.get(irreps, frame.clone(), ref) is first
    frame.mul_(-1)
    cache.get(irreps, frame, ref)
    cache.get(irreps, frame, ref.double())
    assert len(calls) == 3 and len(cache.values) == 2
    cache.get(irreps, torch.eye(3), ref)
    assert len(calls) == 4
    with pytest.raises(ValueError, match="orthogonal"):
        cache.get(irreps, frame * 2, ref)


def test_vector_router_detachment_and_invalid_dag():
    classes, dag, ids, residuals = routing_case(8)
    router = VectorizedChainRouter(ids, class_dag=classes).double()
    measured = {k: torch.tensor(v, dtype=torch.float64, requires_grad=True) for k, v in residuals.items()}
    weights = router(dag, measured)
    sum(k * v for k, v in weights.alpha.items()).backward()
    assert all(v.grad is None for v in measured.values())
    edge = next(e for e in dag.embeddings if e.child_point_group_number == 8)
    incomplete = replace(dag, embeddings=(edge,))
    with pytest.raises(ValueError, match="exact offline"):
        router(incomplete, {edge.checksum: 0.0})
    with pytest.raises(ValueError, match="scalar"):
        router(dag, {**residuals, next(iter(residuals)): torch.ones(2)})
    router.float()
    assert not router._plans


def test_full_model_optimized_reference_parity(official):
    from src.training.global_experts.runner import collate
    torch.manual_seed(42)
    config = GlobalExpertsConfig(adapter_initial_logit=.3)
    actual, splits = _tiny_model_and_splits(official, config)
    reference, _ = _tiny_model_and_splits(official, replace(config,
        grouped_pg_gates=False, cache_frames=False, vectorized_routing=False))
    reference.load_state_dict(actual.state_dict(), strict=True)
    reference.eval()
    actual.eval()
    data, mask, equality, routing, target, _ = collate(splits["train"], "cpu")
    left = reference(data, mask, equality, routing)
    right = actual(data, mask, equality, routing)
    torch.testing.assert_close(left, right, atol=2e-5, rtol=2e-5)
    (left - target).square().mean().backward()
    (right - target).square().mean().backward()
    assert_gradients_equal(reference, actual, atol=2e-4, rtol=2e-3)
    assert actual.frame_cache.values
    actual.double()
    assert not actual.frame_cache.values


def test_paired_benchmark_checks_all_variants(official, monkeypatch, tmp_path):
    from src.profiling import compare_optimized as module
    from src.training.global_experts import GlobalExpertsTrainConfig
    model, splits = _tiny_model_and_splits(official)
    def measured(candidate, splits, **kwargs):
        assert kwargs["warmup"] == 6 and kwargs["repeats"] == 7
        return {"timings": [{"forward_loss_ms": 1., "backward_ms": 2., "total_ms": 3.}] * 7,
                "peak_allocated_bytes": None, "cuda_kernel_calls": 0,
                "batch": {}, "loss": 1., "runtime": {}, "device_name": "CPU"}
    monkeypatch.setattr(module, "profile_global_experts", measured)
    result = module.compare_optimized(model, splits, output_dir=tmp_path / "paired",
        provenance={}, config=GlobalExpertsTrainConfig(batch_size=2), device="cpu")
    assert result["status"] == "passed"
    assert len(result["variants"]) == 4
    assert (tmp_path / "paired" / "comparison.json").is_file()


def test_profile_sample_selection_preserves_representatives_and_splits():
    from types import SimpleNamespace
    from src.cli.global_experts_compare import select_profile_splits
    dataset = SimpleNamespace(split_manifest=SimpleNamespace(train=("a", "b", "c", "d")))
    selected = {"train": ("c", "a"), "validation": ("v",), "test": ("t",)}
    actual = select_profile_splits(dataset, selected, 3)
    assert actual == {**selected, "train": ("c", "a", "b")}
    with pytest.raises(ValueError):
        select_profile_splits(dataset, selected, 1)
