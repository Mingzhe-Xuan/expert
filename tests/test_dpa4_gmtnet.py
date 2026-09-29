from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from src.baselines.gmtnet.runner import (
    GMTNetConfig,
    _attach_dpa4_node_embeddings,
    _replace_atom_embedding,
    _validation_history_metrics,
    dpa4_invariant_node_embedding,
)
from src.evaluation import tensor_benchmark_metrics
from src.features import cgcnn_node_features
from src.irreps import IrrepLayout, IrrepTerm


ROOT = Path(__file__).resolve().parents[1]
LAYOUT = IrrepLayout(
    (
        IrrepTerm(2, 0, "e", "scalar_even"),
        IrrepTerm(1, 0, "o", "scalar_odd"),
        IrrepTerm(2, 1, "e", "vector_even"),
        IrrepTerm(1, 2, "o", "tensor_odd"),
    )
)


def test_validation_history_metrics_match_common_dielectric_definition() -> None:
    prediction = torch.tensor(
        [[[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]],
         [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]]
    )
    target = torch.zeros_like(prediction)
    actual = _validation_history_metrics(prediction, target)
    expected = tensor_benchmark_metrics(prediction, target, task="dielectric")
    assert actual["validation_mae"] == pytest.approx(float(prediction.abs().mean()))
    assert actual["validation_fnorm"] == pytest.approx(expected["fnorm"])


@pytest.mark.parametrize(
    ("prediction", "target"),
    (
        (torch.zeros((2, 3, 3)), torch.zeros((1, 3, 3))),
        (torch.full((2, 3, 3), torch.nan), torch.zeros((2, 3, 3))),
    ),
)
def test_validation_history_metrics_reject_invalid_tensors(
    prediction: torch.Tensor, target: torch.Tensor
) -> None:
    with pytest.raises(ValueError):
        _validation_history_metrics(prediction, target)


def test_dpa4_scalarization_is_o3_invariant_and_preserves_signed_even_scalars() -> None:
    torch.manual_seed(91)
    features = torch.randn(4, LAYOUT.dimension, dtype=torch.float64)
    baseline = dpa4_invariant_node_embedding(features, LAYOUT)
    assert baseline.shape == (4, 6)
    assert torch.equal(baseline[:, :2], features[:, :2])
    transformed = features.clone()
    offset = 0
    for term in LAYOUT.terms:
        width = 2 * term.degree + 1
        stop = offset + term.multiplicity * width
        block = transformed[:, offset:stop].reshape(4, term.multiplicity, width)
        if term.degree == 0 and term.parity == "o":
            block.mul_(-1.0)
        elif term.degree > 0:
            orthogonal, _ = torch.linalg.qr(torch.randn(width, width, dtype=torch.float64))
            block.copy_(block @ orthogonal)
        offset = stop
    actual = dpa4_invariant_node_embedding(transformed, LAYOUT)
    assert torch.allclose(actual, baseline, atol=1e-10, rtol=1e-10)


def test_dpa4_scalarization_rejects_width_and_nonfinite_drift() -> None:
    with pytest.raises(ValueError, match=r"declared O\(3\) layout"):
        dpa4_invariant_node_embedding(torch.zeros((2, LAYOUT.dimension - 1)), LAYOUT)
    malformed = torch.zeros((2, LAYOUT.dimension))
    malformed[0, 0] = torch.nan
    with pytest.raises(ValueError, match="finite"):
        dpa4_invariant_node_embedding(malformed, LAYOUT)


def _feature_row(sample_id: str):
    atomic_numbers = torch.tensor([14, 8], dtype=torch.long)
    return SimpleNamespace(
        sample_id=sample_id,
        features=torch.arange(2 * LAYOUT.dimension, dtype=torch.float32).reshape(2, -1),
        graph=SimpleNamespace(atomic_numbers=atomic_numbers),
    )


def _graph_row(sample_id: str):
    atomic_numbers = torch.tensor([14, 8], dtype=torch.long)
    return {
        "sample_id": sample_id,
        "graph": {
            "x": cgcnn_node_features(atomic_numbers),
            "edge_index": torch.empty((2, 0), dtype=torch.long),
            "edge_attr": torch.empty((0, 3)),
        },
    }


def test_dpa4_embedding_attachment_preserves_exact_sample_and_node_order() -> None:
    graphs = {name: [_graph_row(f"{name}-0")] for name in ("train", "validation", "test")}
    features = {name: [_feature_row(f"{name}-0")] for name in graphs}
    attached = _attach_dpa4_node_embeddings(graphs, features, LAYOUT)
    for name in graphs:
        assert attached[name][0]["graph"]["x"].shape == (2, 6)
        assert graphs[name][0]["graph"]["x"].shape == (2, 92)
    features["test"][0].sample_id = "wrong"
    with pytest.raises(ValueError, match="sample order"):
        _attach_dpa4_node_embeddings(graphs, features, LAYOUT)


def test_dpa4_embedding_attachment_rejects_gmtnet_node_order_drift() -> None:
    graphs = {name: [_graph_row(f"{name}-0")] for name in ("train", "validation", "test")}
    features = {name: [_feature_row(f"{name}-0")] for name in graphs}
    graphs["validation"][0]["graph"]["x"] = graphs["validation"][0]["graph"]["x"].flip(0)
    with pytest.raises(ValueError, match="node ordering"):
        _attach_dpa4_node_embeddings(graphs, features, LAYOUT)


def test_dpa4_override_replaces_only_atom_embedding_and_preserves_output_width() -> None:
    model = torch.nn.Module()
    model.atom_embedding = torch.nn.Linear(92, 128)
    model.message_passing = torch.nn.Linear(128, 128)
    message_passing = model.message_passing
    _replace_atom_embedding(model, 640)
    assert model.atom_embedding.in_features == 640
    assert model.atom_embedding.out_features == 128
    assert model.message_passing is message_passing


def test_dpa4_gmtnet_launcher_freezes_the_comparison_protocol() -> None:
    launcher = (ROOT / "slurm" / "train_reduced_dpa4_gmtnet.sbatch").read_text(encoding="utf-8")
    assert "src.cli.reduced_dpa4_gmtnet_train" in launcher
    assert 'EXPERT_GMTNET_VENV:?' in launcher
    assert 'EXPERT_GMTNET_ROOT:?' in launcher
    assert '--feature-shards "${EXPERT_DPA4_FEATURE_SHARDS:-64}"' in launcher
    assert '--batch-size "${EXPERT_DPA4_GMTNET_BATCH_SIZE:-64}"' in launcher
    assert '--seed "${EXPERT_DPA4_GMTNET_SEED:-42}"' in launcher
    assert '--checkpoint-interval "${checkpoint_interval}"' in launcher
    assert "#SBATCH --time=3-00:00:00" in launcher
    assert GMTNetConfig().checkpoint_interval == 0


def test_dpa4_gmtnet_300e_launcher_is_isolated_and_protocol_matched() -> None:
    launcher = (
        ROOT / "slurm" / "train_reduced_dpa4_gmtnet_300e.sbatch"
    ).read_text(encoding="utf-8")
    assert "src.cli.reduced_dpa4_gmtnet_train" in launcher
    assert 'run_root="results/reduced-benchmark/dpa4-gmtnet-300e"' in launcher
    assert "#SBATCH --output=logs/slurm/dpa4-gmtnet-300e-%j.out" in launcher
    assert "#SBATCH --error=logs/slurm/dpa4-gmtnet-300e-%j.err" in launcher
    assert 'EXPERT_GMTNET_VENV:?' in launcher
    assert 'EXPERT_GMTNET_ROOT:?' in launcher
    assert "--feature-shards 64" in launcher
    assert "--epochs 300" in launcher
    assert "--batch-size 64" in launcher
    assert "--learning-rate 0.001" in launcher
    assert "--end-learning-rate 0.00001" in launcher
    assert "--weight-decay 0.00001" in launcher
    assert "--seed 42" in launcher
    assert "--checkpoint-interval 20" in launcher
    assert "#SBATCH --time=3-00:00:00" in launcher
    assert 'run_root="results/reduced-benchmark/dpa4-gmtnet"' not in launcher


def test_dpa4_gmtnet_equiv_attn_300e_launcher_is_isolated_and_matched() -> None:
    launcher = (
        ROOT / "slurm" / "train_reduced_dpa4_gmtnet_equiv_attn_300e.sbatch"
    ).read_text(encoding="utf-8")
    assert "src.cli.reduced_dpa4_gmtnet_train" in launcher
    assert 'run_root="results/reduced-benchmark/dpa4-gmtnet-equiv-attn-300e"' in launcher
    assert "#SBATCH --output=logs/slurm/dpa4-gmtnet-equiv-attn-300e-%j.out" in launcher
    assert "#SBATCH --error=logs/slurm/dpa4-gmtnet-equiv-attn-300e-%j.err" in launcher
    assert 'EXPERT_GMTNET_VENV:?' in launcher
    assert 'EXPERT_GMTNET_ROOT:?' in launcher
    assert "--feature-shards 64" in launcher
    assert "--epochs 300" in launcher
    assert "--batch-size 64" in launcher
    assert "--learning-rate 0.001" in launcher
    assert "--end-learning-rate 0.00001" in launcher
    assert "--weight-decay 0.00001" in launcher
    assert "--seed 42" in launcher
    assert "--checkpoint-interval 20" in launcher
    assert "--use-equiv-attn" in launcher
    assert "#SBATCH --time=3-00:00:00" in launcher
    assert 'run_root="results/reduced-benchmark/dpa4-gmtnet-300e"' not in launcher
    assert 'run_root="results/reduced-benchmark/dpa4-gmtnet"' not in launcher
