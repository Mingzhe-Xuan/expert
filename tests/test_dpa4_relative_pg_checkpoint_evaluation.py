from __future__ import annotations

from pathlib import Path

import pytest

from src.cli.reduced_dpa4_relative_pg_evaluate import select_retained_checkpoint
from src.training.benchmark import BenchmarkConfig, _checkpoint_epoch_is_eligible


ROOT = Path(__file__).resolve().parents[1]


def _source_summary() -> dict[str, object]:
    return {
        "history": [
            {
                "epoch": epoch,
                "validation_mae": 1.0 if epoch == 100 else 2.0,
                "validation_fnorm": 3.0,
            }
            for epoch in range(1, 201)
        ],
        "periodic_checkpoints": [
            {"epoch": epoch, "path": f"checkpoint-epoch-{epoch:03d}.pt"}
            for epoch in range(20, 201, 20)
        ],
    }


def test_retained_checkpoint_selection_excludes_epoch_100() -> None:
    summary = _source_summary()
    summary["history"][119]["validation_mae"] = 1.5
    summary["history"][139]["validation_mae"] = 1.6
    selected = select_retained_checkpoint(
        summary,
        minimum_checkpoint_epoch_exclusive=100,
    )
    assert selected["epoch"] == 120


def test_training_checkpoint_threshold_is_strict_and_short_runs_can_override() -> None:
    production = BenchmarkConfig(
        max_epochs=200,
        minimum_checkpoint_epoch_exclusive=100,
    )
    assert not _checkpoint_epoch_is_eligible(production, 100)
    assert _checkpoint_epoch_is_eligible(production, 101)
    assert _checkpoint_epoch_is_eligible(BenchmarkConfig(max_epochs=1), 1)
    with pytest.raises(ValueError, match="minimum checkpoint epoch"):
        BenchmarkConfig(max_epochs=100, minimum_checkpoint_epoch_exclusive=100)


def test_retained_checkpoint_selection_rejects_missing_eligible_archive() -> None:
    summary = _source_summary()
    summary["periodic_checkpoints"] = summary["periodic_checkpoints"][:5]
    with pytest.raises(ValueError, match="no retained checkpoint"):
        select_retained_checkpoint(
            summary,
            minimum_checkpoint_epoch_exclusive=100,
        )


def test_post100_evaluation_launcher_reuses_exact_archives() -> None:
    launcher = (
        ROOT / "slurm" / "evaluate_reduced_dpa4_relative_pg_post100.sbatch"
    ).read_text(encoding="utf-8")
    assert "src.cli.reduced_dpa4_relative_pg_evaluate" in launcher
    assert "--minimum-checkpoint-epoch-exclusive 100" in launcher
    assert "checkpoint-${source_job}-epoch-120.pt" in launcher
    assert "dpa4-relative-pg-post100-eval" in launcher
    assert "evaluate_one 56d" in launcher
    assert "evaluate_one 64d" in launcher
    assert "evaluate_one 80d" in launcher
    assert "train_cached_backbone_readout" not in launcher
