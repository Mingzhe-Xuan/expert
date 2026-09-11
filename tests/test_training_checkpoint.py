from __future__ import annotations

import json

import pytest
import torch
from torch import nn

from src.configs import ArchitectureConfig
from src.data import TrainingUnit
from src.heads import TARGET_LAYOUTS
from src.irreps import ConventionMetadata
from src.training import (
    CoefficientNormalizer,
    coefficient_mse,
    load_checkpoint,
    physical_coefficient_metrics,
    save_checkpoint,
)


def _architecture(readout: str = "full_o3") -> ArchitectureConfig:
    return ArchitectureConfig(
        branch="B+R",
        adaptation_backend="none",
        o3e_backend="none",
        readout_backend=readout,
        pg_hidden_mode="none",
    )


def _convention(copy_ordering: str = "target-contract-v1") -> ConventionMetadata:
    return ConventionMetadata(
        schema_version=1,
        real_harmonic_convention="e3nn-real-y-lm",
        cg_convention="e3nn-wigner-3j",
        path_ordering="input1-input2-output-lexicographic",
        copy_ordering=copy_ordering,
        subduction_checksum="a" * 64,
    )


@pytest.mark.parametrize("task", ["dielectric", "elastic", "bec"])
def test_copy_aware_coefficient_loss_and_metrics(task: str) -> None:
    torch.manual_seed(3)
    layout = TARGET_LAYOUTS[task]
    item_count = 5 if task == "bec" else 2
    prediction = torch.randn(item_count, layout.dimension, requires_grad=True)
    target = torch.randn_like(prediction)
    unit = TrainingUnit("jarvis_dfpt", "bec") if task == "bec" else TrainingUnit(
        "jarvis_tensor", task
    )
    normalizer = CoefficientNormalizer.fit(target, layout, unit, split="train")
    result = coefficient_mse(prediction, target, layout, normalizer)
    assert list(result.by_copy) == [term.copy_label for term in layout.terms]
    assert result.total.ndim == 0 and torch.isfinite(result.total)
    result.total.backward()
    assert prediction.grad is not None
    assert prediction.grad.shape[0] == item_count
    assert torch.isfinite(prediction.grad).all()
    metrics = physical_coefficient_metrics(prediction, target, layout)
    assert json.loads(json.dumps(metrics)).keys() == metrics.keys()


def test_normalizer_scale_weights_loss_per_copy() -> None:
    layout = TARGET_LAYOUTS["dielectric"]
    unit = TrainingUnit("jarvis_tensor", "dielectric")
    target = torch.tensor(
        [[1.0, 2.0, 0.0, 0.0, 0.0, 0.0], [3.0, 4.0, 0.0, 0.0, 0.0, 0.0]]
    )
    normalizer = CoefficientNormalizer.fit(target, layout, unit, split="train")
    prediction = target + 1.0
    result = coefficient_mse(prediction, target, layout, normalizer)
    expected_scalar = normalizer.scale[0].reciprocal().square()
    expected_quadrupole = normalizer.scale[1].reciprocal().square()
    assert torch.allclose(result.by_copy["trace"], expected_scalar)
    assert torch.allclose(result.by_copy["traceless"], expected_quadrupole)


def test_optimizer_checkpoint_round_trip_and_metadata(tmp_path) -> None:
    torch.manual_seed(12)
    layout = TARGET_LAYOUTS["dielectric"]
    unit = TrainingUnit("jarvis_tensor", "dielectric")
    architecture = _architecture()
    convention = _convention()
    model = nn.Linear(4, layout.dimension)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    inputs = torch.randn(3, 4)
    targets = torch.randn(3, layout.dimension)
    normalizer = CoefficientNormalizer.fit(targets, layout, unit, split="train")
    loss = coefficient_mse(model(inputs), targets, layout, normalizer).total
    loss.backward()
    optimizer.step()
    expected = model(inputs).detach().clone()
    checkpoint = tmp_path / "unit.pt"
    save_checkpoint(
        checkpoint,
        model=model,
        optimizer=optimizer,
        architecture=architecture,
        unit=unit,
        convention=convention,
        normalizer=normalizer,
        step=1,
    )
    assert checkpoint.is_file()
    assert not (tmp_path / ".unit.pt.tmp").exists()

    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(10.0)
    loaded = load_checkpoint(
        checkpoint,
        model=model,
        optimizer=optimizer,
        expected_architecture=architecture,
        expected_unit=unit,
        expected_convention=convention,
        expected_layout=layout,
    )
    assert loaded.step == 1
    assert loaded.normalizer.unit == unit
    assert torch.allclose(model(inputs), expected)
    assert optimizer.state_dict()["state"]


@pytest.mark.parametrize(
    "wrong_architecture,wrong_unit,wrong_convention,error",
    [(_architecture("o2_tp"), TrainingUnit("jarvis_tensor", "dielectric"), _convention(), "architecture"),
     (_architecture(), TrainingUnit("matten", "elastic"), _convention(), "training unit"),
     (_architecture(), TrainingUnit("jarvis_tensor", "dielectric"), _convention("different"), "convention")],
)
def test_checkpoint_mismatch_fails_before_model_mutation(
    tmp_path, wrong_architecture, wrong_unit, wrong_convention, error
) -> None:
    layout = TARGET_LAYOUTS["dielectric"]
    unit = TrainingUnit("jarvis_tensor", "dielectric")
    model = nn.Linear(2, layout.dimension)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    normalizer = CoefficientNormalizer.fit(
        torch.ones(2, layout.dimension), layout, unit, split="train"
    )
    checkpoint = tmp_path / "strict.pt"
    save_checkpoint(
        checkpoint,
        model=model,
        optimizer=optimizer,
        architecture=_architecture(),
        unit=unit,
        convention=_convention(),
        normalizer=normalizer,
        step=0,
    )
    before = [parameter.detach().clone() for parameter in model.parameters()]
    with pytest.raises(ValueError, match=error):
        load_checkpoint(
            checkpoint,
            model=model,
            optimizer=optimizer,
            expected_architecture=wrong_architecture,
            expected_unit=wrong_unit,
            expected_convention=wrong_convention,
            expected_layout=TARGET_LAYOUTS[wrong_unit.target],
        )
    assert all(torch.equal(parameter, saved) for parameter, saved in zip(model.parameters(), before))
