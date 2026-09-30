import pytest
import torch

from src.models.global_experts.dense import DenseO3Branch
from e3nn import o3
from test_gmtnet_attention import official


def case(hidden=None, initial_logit=0.2):
    torch.manual_seed(71)
    irreps = o3.Irreps("2x0e + 1x0o + 2x1o + 1x2e")
    model = DenseO3Branch(irreps, hidden_irreps=hidden, depth=2,
                          initial_logit=initial_logit).double()
    x = torch.randn(4, irreps.dim, dtype=torch.float64)
    edge = torch.tensor([[0, 1, 2, 3, 0], [1, 2, 3, 0, 2]])
    vec = torch.randn(5, 3, dtype=torch.float64)
    return irreps, model, x, edge, vec


@pytest.mark.parametrize("sign", [1, -1])
@pytest.mark.parametrize("hidden", [None, "4x0e + 2x0o + 3x1o + 2x2e"])
def test_dense_o3(sign, hidden):
    irreps, model, x, edge, vec = case(hidden)
    rotation = sign * o3.rand_matrix(dtype=torch.float64)
    representation = irreps.D_from_matrix(rotation)
    torch.testing.assert_close(model(x @ representation.T, edge, vec @ rotation.T),
                               model(x, edge, vec) @ representation.T,
                               atol=2e-7, rtol=2e-7)


@pytest.mark.parametrize("hidden", [None, "4x0e + 2x0o + 3x1o + 2x2e"])
def test_identity_and_empty_edges(hidden):
    _, model, x, edge, vec = case(hidden, 0.0)
    assert torch.equal(model(x, edge, vec), x)
    assert torch.equal(model(x, edge[:, :0], vec[:0]), x)


def test_permutation_and_cutoff():
    _, model, x, edge, vec = case()
    permutation = torch.tensor([2, 0, 3, 1])
    inverse = torch.argsort(permutation)
    torch.testing.assert_close(model(x[permutation], inverse[edge], vec),
                               model(x, edge, vec)[permutation])
    torch.testing.assert_close(model(x, edge, vec / vec.norm(dim=-1, keepdim=True) * 7), x)


def test_gradients_after_identity_warmup():
    _, model, x, edge, vec = case(initial_logit=0.0)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    target = torch.randn_like(x)
    for step in range(2):
        optimizer.zero_grad()
        (model(x, edge, vec) - target).square().mean().backward()
        for name, parameter in model.named_parameters():
            assert parameter.grad is not None, name
            assert torch.isfinite(parameter.grad).all(), name
            if step == 1:
                assert parameter.grad.abs().sum() > 0, name
        optimizer.step()


def test_state_roundtrip():
    _, model, x, edge, vec = case()
    _, clone, _, _, _ = case()
    clone.load_state_dict(model.state_dict(), strict=True)
    torch.testing.assert_close(clone(x, edge, vec), model(x, edge, vec))


def test_wrapper_dense_equivariance_backward_and_paired_shared(official):
    from dataclasses import replace
    from test_global_experts import _tiny_model_and_splits
    from src.models.global_experts import GlobalExpertsConfig
    from src.training.global_experts.runner import collate

    config = GlobalExpertsConfig(expert_irreps="2x0e + 1x1o + 1x2e")
    torch.manual_seed(42)
    pg, _ = _tiny_model_and_splits(official, config)
    torch.manual_seed(42)
    dense, splits = _tiny_model_and_splits(official, replace(
        config, auxiliary_type="dense", dense_depth=2, dense_initial_logit=0.2))
    assert not dense.experts and dense.router is None
    for name, value in pg.state_dict().items():
        if name.startswith(("global_model.", "input_map.", "output_map.", "adapter.")):
            assert torch.equal(value, dense.state_dict()[name]), name
    data, mask, equality, _, _, _ = collate(splits["train"], "cpu")
    dense.eval()
    prediction, info = dense(data, mask, equality, return_diagnostics=True)
    prediction.square().sum().backward()
    assert dense.branch_logit.grad.abs() > 0
    assert dense.dense.blocks[0].tp.weight.grad.abs().sum() > 0
    q = -o3.rand_matrix()
    data.edge_attr = data.edge_attr @ q.T
    actual, rotated = dense(data, mask, equality, return_diagnostics=True)
    torch.testing.assert_close(rotated["fused_features"],
                               info["fused_features"] @ dense.global_irreps.D_from_matrix(q).T,
                               atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(actual, q @ prediction @ q.T, atol=2e-5, rtol=2e-5)
    dense.config = replace(dense.config, auxiliary_enabled=False)
    torch.testing.assert_close(dense(data, mask, equality),
                               dense.global_model(data, mask, equality), atol=0, rtol=0)


def test_pg_metadata_legacy_fields():
    from src.models.global_experts import GlobalExpertsConfig
    old = GlobalExpertsConfig().metadata()
    assert "auxiliary_type" not in old
    assert not any(key.startswith("dense_") for key in old)
    assert GlobalExpertsConfig(**old).metadata() == old


@pytest.mark.parametrize("hidden", ["2x0e + 1x1o + 1x2e", "4x0e + 2x1o + 2x2e"])
def test_parameter_formula_and_capacity_search(hidden):
    from src.models.global_experts.dense.budget import dense_parameter_count, trainable_count, match_dense_config
    from src.models.global_experts import GlobalExpertsConfig
    external = "2x0e + 1x1o + 1x2e"
    branch = DenseO3Branch(external, hidden_irreps=hidden, depth=3, gate_width=27)
    assert trainable_count([branch]) == dense_parameter_count(external, hidden, 3, 8, 27, 2)
    budget = {"mean_active_experts_router": 20000.5, "shared_parameters": 100,
              "mean_active_auxiliary": 20100.5}
    config, report = match_dense_config(GlobalExpertsConfig(expert_irreps=external), budget)
    assert report["auxiliary_relative_error"] < 0.01
    chosen = DenseO3Branch(external, hidden_irreps=config.dense_hidden_irreps,
                           depth=config.dense_depth, gate_width=config.dense_gate_width)
    assert trainable_count([chosen]) == report["dense_experts_replacement_parameters"]


def test_training_budget_and_dense_checkpoint_lifecycle(official, tmp_path):
    from dataclasses import replace
    from test_global_experts import _tiny_model_and_splits
    from src.models.global_experts import GlobalExpertsConfig
    from src.models.global_experts.dense.budget import pg_active_budget, trainable_count
    from src.training.global_experts.runner import train_global_experts, GlobalExpertsTrainConfig, load_checkpoint
    config = GlobalExpertsConfig(expert_irreps="2x0e + 1x1o + 1x2e")
    pg, splits = _tiny_model_and_splits(official, config)
    report = pg_active_budget(pg, splits["train"])
    expected = trainable_count([pg.experts]) + len(pg.router.edge_ids)
    assert report["mean_active_experts_router"] == expected
    assert report["training_crystals"] == 2
    assert pg_active_budget(pg, splits["train"] * 2)["mean_active_auxiliary"] == report["mean_active_auxiliary"]
    dense, splits = _tiny_model_and_splits(official, replace(config, auxiliary_type="dense", dense_depth=2))
    result = train_global_experts(dense, splits, output_dir=tmp_path / "dense",
        provenance={"test": True}, config=GlobalExpertsTrainConfig(
            epochs=2, batch_size=2, checkpoint_interval=1, minimum_checkpoint_epoch_exclusive=0))
    assert result["status"] == "passed"
    payload = load_checkpoint(dense, tmp_path / "dense" / "best.pt", provenance={"test": True})
    assert payload["model_metadata"]["config"]["auxiliary_type"] == "dense"
