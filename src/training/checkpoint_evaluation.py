from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Sequence

import torch

from ..configs import ArchitectureConfig
from ..data import TrainingUnit
from ..heads import TARGET_LAYOUTS
from ..irreps import IrrepLayout
from .benchmark import (
    CachedBackboneTensorModel,
    FrozenFeatureExample,
    _evaluate,
)
from .checkpoint import load_checkpoint
from .smoke import default_convention_metadata


def evaluate_cached_backbone_checkpoint(
    *,
    unit: TrainingUnit,
    source_layout: IrrepLayout,
    test_examples: Sequence[FrozenFeatureExample],
    checkpoint_path: str | Path,
    predictions_path: str | Path,
    architecture: ArchitectureConfig,
    expert_point_groups: tuple[str, ...],
    material_edge_ids: tuple[str, ...],
    hidden_layout: IrrepLayout,
    batch_size: int,
    minimum_checkpoint_epoch_exclusive: int = 100,
    device: torch.device | str = "cuda",
) -> dict[str, object]:
    """Evaluate a retained legacy relative-PG checkpoint without optimizer steps."""

    if not test_examples:
        raise ValueError("checkpoint evaluation requires non-empty test examples")
    if batch_size < 1 or minimum_checkpoint_epoch_exclusive < 0:
        raise ValueError("batch size and checkpoint epoch threshold are invalid")
    model = CachedBackboneTensorModel(
        source_layout,
        architecture,
        unit.target,
        expert_point_groups,
        material_edge_ids,
        hidden_layout,
        pg_weighting="legacy",
        grouped_pg_gates=False,
        vectorized_pg_routing=False,
    ).to(device)
    checkpoint = Path(checkpoint_path)
    loaded = load_checkpoint(
        checkpoint,
        model=model,
        optimizer=None,
        expected_architecture=architecture,
        expected_unit=unit,
        expected_convention=default_convention_metadata(),
        expected_layout=TARGET_LAYOUTS[unit.target],
        map_location=device,
    )
    if loaded.step <= minimum_checkpoint_epoch_exclusive:
        raise ValueError(
            "checkpoint epoch must be strictly greater than the evaluation threshold"
        )
    prediction_rows: list[dict[str, object]] = []
    test_loss, test_mae, test_metrics, irrep_metrics = _evaluate(
        model,
        test_examples,
        source_layout,
        unit,
        loaded.normalizer,
        batch_size=batch_size,
        device=device,
        training_protocol="gmtnet",
        prediction_rows=prediction_rows,
    )
    target = Path(predictions_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            for row in prediction_rows:
                stream.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    return {
        "checkpoint": str(checkpoint),
        "checkpoint_bytes": checkpoint.stat().st_size,
        "checkpoint_epoch": loaded.step,
        "checkpoint_sha256": digest,
        "minimum_checkpoint_epoch_exclusive": minimum_checkpoint_epoch_exclusive,
        "selection_rule": "best_validation_mae_among_retained_eligible_checkpoints",
        "training_performed": False,
        "optimizer_steps": 0,
        "predictions": str(target),
        "test_loss": test_loss,
        "test_mae": test_mae,
        "test_metrics": test_metrics,
        "test_irrep_metrics": irrep_metrics,
        "sample_count": len(prediction_rows),
    }
