from __future__ import annotations

import math

import pytest
import torch

from src.configs import ArchitectureConfig
from src.data import TensorSample, TrainingUnit
from src.evaluation import tensor_benchmark_metrics
from src.experts import default_hidden_layout
from src.heads import TensorReadout, cartesian_to_irreps
from src.irreps import IrrepLayout, IrrepTerm
from src.training import (
    BenchmarkConfig,
    FrozenFeatureExample,
    collate_frozen_examples,
    load_frozen_feature_cache,
    prepare_tensor_batch,
    save_frozen_feature_cache,
    train_cached_backbone_readout,
)


def test_tensor_benchmark_metrics_use_sample_frobenius_distances() -> None:
    target = torch.stack((torch.eye(3), 2.0 * torch.eye(3)))
    prediction = target.clone()
    prediction[0, 0, 0] += 0.2
    prediction[1, 0, 0] += 1.0

    report = tensor_benchmark_metrics(prediction, target)

    assert report["sample_count"] == 2
    assert report["fnorm"] == pytest.approx(0.6)
    assert report["ewt_25"] == pytest.approx(50.0)
    assert report["ewt_10"] == pytest.approx(0.0)
    assert report["ewt_5"] == pytest.approx(0.0)


def test_tensor_benchmark_metrics_define_zero_target_behavior() -> None:
    target = torch.zeros((2, 3, 3))
    prediction = target.clone()
    prediction[1, 0, 0] = 1.0
    report = tensor_benchmark_metrics(prediction, target)
    assert report["fnorm"] == pytest.approx(0.5)
    assert report["ewt_25"] == pytest.approx(50.0)
    assert math.isfinite(float(report["fnorm"]))


def test_readout_edge_layout_reaches_target_maximum_degree() -> None:
    config = ArchitectureConfig("B+R", "none", "none", "full_o3", "none")
    hidden = default_hidden_layout(config)
    dielectric = TensorReadout(hidden, "dielectric", "full_o3")
    elastic = TensorReadout(hidden, "elastic", "full_o3")
    assert max(term.degree for term in dielectric.edge_layout.terms) == 2
    assert max(term.degree for term in elastic.edge_layout.terms) == 4


def _cached_examples(count: int = 7):
    unit = TrainingUnit("jarvis_tensor", "dielectric")
    layout = IrrepLayout((IrrepTerm(2, 0, "e", "cached_scalar"),))
    lattice = 5.0 * torch.eye(3, dtype=torch.float64)
    output = []
    for index in range(count):
        target = torch.diag(torch.tensor([2.0, 2.5, 3.0]) + 0.1 * index)
        sample = TensorSample(
            sample_id=f"cached-{index}",
            unit=unit,
            lattice=lattice,
            fractional_positions=torch.tensor(
                [[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]], dtype=torch.float64
            ),
            atomic_numbers=torch.tensor([14, 8]),
            target_cartesian=target,
            target_coefficients=cartesian_to_irreps(target, "dielectric"),
            target_unit="dimensionless",
            source={"fixture": "cached-benchmark"},
        )
        prepared = prepare_tensor_batch((sample,), cutoff=3.0)
        features = torch.tensor(
            [[1.0, float(index)], [0.5, float(index)]], dtype=torch.float32
        )
        output.append(
            FrozenFeatureExample(
                sample.sample_id,
                features,
                prepared.graph,
                prepared.symmetries[0],
                prepared.target_coefficients,
                target.unsqueeze(0),
            )
        )
    return unit, layout, tuple(output)


def test_cached_feature_collation_and_training_emit_public_metrics(tmp_path) -> None:
    unit, layout, examples = _cached_examples()
    batch = collate_frozen_examples(examples[:2], layout, device="cpu")
    assert batch.graph.num_graphs == 2
    assert batch.features.node_features.shape == (4, 2)
    assert batch.target_coefficients.shape == (2, 6)

    checkpoint = tmp_path / "benchmark.pt"
    report = train_cached_backbone_readout(
        backbone_family="analytic",
        unit=unit,
        source_layout=layout,
        train_examples=examples[:3],
        validation_examples=examples[3:5],
        test_examples=examples[5:],
        checkpoint_path=checkpoint,
        config=BenchmarkConfig(max_epochs=2, batch_size=2, patience=2),
        device="cpu",
    )
    assert checkpoint.is_file()
    assert report["status"] == "passed"
    assert report["split_counts"] == {"train": 3, "validation": 2, "test": 2}
    assert set(report["test_metrics"]) == {
        "sample_count", "fnorm", "ewt_25", "ewt_10", "ewt_5"
    }
    assert set(report["exceeds_public_target"]) == {
        "fnorm", "ewt_25", "ewt_10", "ewt_5"
    }


def test_frozen_feature_cache_round_trip_is_metadata_gated(tmp_path) -> None:
    unit, layout, examples = _cached_examples(2)
    path = tmp_path / "features.pt"
    digest = "a" * 64
    save_frozen_feature_cache(
        path,
        backbone_family="mace",
        checkpoint_sha256=digest,
        unit=unit,
        split="test",
        layout=layout,
        examples=examples,
    )
    restored_layout, restored = load_frozen_feature_cache(
        path,
        expected_backbone="mace",
        expected_checkpoint_sha256=digest,
        expected_unit=unit,
        expected_split="test",
        expected_sample_ids=[example.sample_id for example in examples],
    )
    assert restored_layout == layout
    assert [example.sample_id for example in restored] == ["cached-0", "cached-1"]
    assert torch.equal(restored[1].features, examples[1].features)
    with pytest.raises(ValueError, match="checkpoint_sha256 mismatch"):
        load_frozen_feature_cache(
            path,
            expected_backbone="mace",
            expected_checkpoint_sha256="b" * 64,
            expected_unit=unit,
            expected_split="test",
            expected_sample_ids=["cached-0", "cached-1"],
        )
