from __future__ import annotations

from collections import Counter

import pytest
import torch

from src.data import LEGAL_TRAINING_UNITS, TrainingUnit, make_seeded_split
from src.heads import TARGET_LAYOUTS
from src.training import CoefficientNormalizer


def test_independent_training_units_are_legal() -> None:
    units = [TrainingUnit(dataset, target) for dataset, target in LEGAL_TRAINING_UNITS]
    assert [unit.namespace for unit in units] == [
        "jarvis_tensor__dielectric",
        "jarvis_tensor__elastic",
        "dtnet__dielectric",
        "matten__elastic",
        "jarvis_dfpt__bec",
        "curated_reduced_total__dielectric",
    ]
    with pytest.raises(ValueError):
        TrainingUnit("jarvis_tensor", "bec")
    with pytest.raises(ValueError):
        TrainingUnit("matten", "dielectric")
    with pytest.raises(ValueError):
        TrainingUnit("dtnet", "elastic")


def test_seeded_split_is_deterministic_group_preserving_8_1_1() -> None:
    samples = [f"sample-{index}" for index in range(12)]
    groups = ["duplicate-a", "duplicate-a", *[f"material-{index}" for index in range(10)]]
    first = make_seeded_split(samples, groups)
    second = make_seeded_split(samples, groups)
    assert first == second
    group_split = {}
    for split_name in ("train", "validation", "test"):
        for sample in getattr(first, split_name):
            group = first.sample_to_group[sample]
            assert group_split.setdefault(group, split_name) == split_name
    assert Counter(
        group_split.values()
    ) == {"train": 9, "validation": 1, "test": 1}
    assert set(first.train) | set(first.validation) | set(first.test) == set(samples)


def test_five_group_smoke_split_is_three_one_one() -> None:
    manifest = make_seeded_split(
        [f"sample-{index}" for index in range(5)],
        [f"material-{index}" for index in range(5)],
    )
    assert tuple(map(len, (manifest.train, manifest.validation, manifest.test))) == (3, 1, 1)


@pytest.mark.parametrize("mode", ["rms", "variance"])
def test_normalizer_is_copy_aware_and_invertible(mode: str) -> None:
    torch.manual_seed(9)
    layout = TARGET_LAYOUTS["elastic"]
    coefficients = torch.randn(7, layout.dimension, dtype=torch.float64)
    # Give the two scalar copies deliberately different distributions.
    coefficients[:, 0] = torch.linspace(10.0, 16.0, 7)
    coefficients[:, 1] = torch.linspace(-3.0, -1.0, 7)
    unit = TrainingUnit("matten", "elastic")
    normalizer = CoefficientNormalizer.fit(
        coefficients, layout, unit, split="train", mode=mode
    )
    assert len(normalizer.blocks) == 5
    assert [block.copy_label for block in normalizer.blocks] == [
        "scalar_bulk",
        "scalar_shear",
        "quadrupole_a",
        "quadrupole_b",
        "hexadecapole",
    ]
    assert normalizer.blocks[0].scale != normalizer.blocks[1].scale
    if mode == "variance":
        assert normalizer.center[0] != 0 and normalizer.center[1] != 0
        assert torch.equal(normalizer.center[2:], torch.zeros_like(normalizer.center[2:]))
    else:
        assert torch.equal(normalizer.center, torch.zeros_like(normalizer.center))
    normalized = normalizer.transform(coefficients)
    recovered = normalizer.inverse_transform(normalized)
    assert torch.allclose(recovered, coefficients, atol=1e-12, rtol=1e-12)


def test_normalizer_fit_and_load_fail_closed_against_leakage() -> None:
    layout = TARGET_LAYOUTS["dielectric"]
    unit = TrainingUnit("jarvis_tensor", "dielectric")
    values = torch.ones(3, layout.dimension)
    with pytest.raises(ValueError, match="train split"):
        CoefficientNormalizer.fit(values, layout, unit, split="validation")
    normalizer = CoefficientNormalizer.fit(values, layout, unit, split="train")
    state = normalizer.state_dict()
    loaded = CoefficientNormalizer.from_state_dict(
        state, expected_unit=unit, expected_layout=layout
    )
    assert torch.equal(loaded.scale, normalizer.scale)
    with pytest.raises(ValueError, match="different training unit"):
        CoefficientNormalizer.from_state_dict(
            state,
            expected_unit=TrainingUnit("jarvis_tensor", "elastic"),
            expected_layout=TARGET_LAYOUTS["elastic"],
        )
    tampered = {**state, "layout": list(TARGET_LAYOUTS["bec"].to_spec())}
    with pytest.raises(ValueError, match="layout"):
        CoefficientNormalizer.from_state_dict(
            tampered, expected_unit=unit, expected_layout=layout
        )
