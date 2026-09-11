from __future__ import annotations

from dataclasses import asdict, dataclass
import os
from pathlib import Path

import torch
from torch import nn

from ..configs import ArchitectureConfig
from ..data import TrainingUnit
from ..irreps import ConventionMetadata, IrrepLayout
from .normalization import CoefficientNormalizer


CHECKPOINT_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class LoadedCheckpoint:
    step: int
    normalizer: CoefficientNormalizer


def save_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    architecture: ArchitectureConfig,
    unit: TrainingUnit,
    convention: ConventionMetadata,
    normalizer: CoefficientNormalizer,
    step: int,
) -> None:
    """Atomically save a plain-tensor checkpoint with all frozen compatibility keys."""

    if step < 0:
        raise ValueError("checkpoint step must be non-negative")
    if normalizer.unit != unit:
        raise ValueError("normalizer belongs to a different training unit")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp")
    payload = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "step": step,
        "architecture": architecture.to_dict(),
        "training_unit": unit.namespace,
        "conventions": asdict(convention),
        "convention_checksum": convention.checksum,
        "normalizer": normalizer.state_dict(),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
    }
    try:
        torch.save(payload, temporary)
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    expected_architecture: ArchitectureConfig,
    expected_unit: TrainingUnit,
    expected_convention: ConventionMetadata,
    expected_layout: IrrepLayout,
    map_location: torch.device | str = "cpu",
) -> LoadedCheckpoint:
    """Validate all metadata before mutating model or optimizer state."""

    payload = torch.load(path, map_location=map_location, weights_only=True)
    if payload.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("unsupported checkpoint schema")
    if payload.get("architecture") != expected_architecture.to_dict():
        raise ValueError("checkpoint architecture mismatch")
    if payload.get("training_unit") != expected_unit.namespace:
        raise ValueError("checkpoint training unit mismatch")
    expected_convention.require_compatible(payload)
    if payload.get("conventions") != asdict(expected_convention):
        raise ValueError("checkpoint convention metadata mismatch")
    normalizer = CoefficientNormalizer.from_state_dict(
        payload["normalizer"],
        expected_unit=expected_unit,
        expected_layout=expected_layout,
        device=map_location,
    )
    # Metadata gates are complete. State mutation is now permitted and remains strict.
    model.load_state_dict(payload["model_state"], strict=True)
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer_state"])
    return LoadedCheckpoint(step=int(payload["step"]), normalizer=normalizer)
