from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from src.cli.reduced_dpa4_train import (
    _merge_sharded_examples,
    _shard_sample_ids,
    _validate_shard_arguments,
)
from src.cli.reduced_protocol import REDUCED_POINT_GROUPS, point_group_stratified_smoke_ids
from src.data import TrainingUnit, load_training_dataset


ROOT = Path(__file__).resolve().parents[1]


def test_dpa4_feature_shards_are_disjoint_exhaustive_and_restore_order() -> None:
    expected = tuple(f"sample-{index}" for index in range(17))
    shard_ids = [_shard_sample_ids(expected, 4, index) for index in range(4)]
    assert set().union(*(set(ids) for ids in shard_ids)) == set(expected)
    assert sum(len(ids) for ids in shard_ids) == len(expected)
    shards = [tuple(SimpleNamespace(sample_id=sample_id) for sample_id in ids)
              for ids in reversed(shard_ids)]
    assert tuple(row.sample_id for row in _merge_sharded_examples(expected, shards)) == expected


def test_dpa4_feature_shard_merge_rejects_coverage_drift() -> None:
    rows = tuple(SimpleNamespace(sample_id=value) for value in ("a", "b"))
    with pytest.raises(ValueError, match="duplicate"):
        _merge_sharded_examples(("a", "b"), (rows, rows[:1]))
    with pytest.raises(ValueError, match="coverage mismatch"):
        _merge_sharded_examples(("a", "b", "c"), (rows,))
    with pytest.raises(ValueError, match="index/count"):
        _shard_sample_ids(("a",), 0, 0)


def test_dpa4_launcher_separates_cache_partitions_from_concurrent_workers() -> None:
    launcher = (ROOT / "slurm" / "train_reduced_dpa4_full_pg.sbatch").read_text(
        encoding="utf-8"
    )
    assert "default_feature_shards=64" in launcher
    assert "EXPERT_DPA4_FEATURE_WORKERS" in launcher
    assert 'claim_root="${run_root}/claims-${SLURM_JOB_ID}"' in launcher
    assert 'mkdir "${claim_root}/${candidate}" 2>/dev/null' in launcher
    assert "candidate<feature_shards" in launcher
    assert "worker_index<feature_workers" in launcher
    assert "run_feature_worker &" in launcher
    assert '--feature-shard-index "${shard_index}"' in launcher


@pytest.mark.parametrize(
    ("count", "index", "prepare_only", "message"),
    (
        (0, None, False, "positive"),
        (8, None, False, "smallest selected split"),
        (4, 0, False, "prepare-only"),
        (4, 4, True, "outside"),
    ),
)
def test_dpa4_feature_shard_cli_validation(count, index, prepare_only, message) -> None:
    with pytest.raises(ValueError, match=message):
        _validate_shard_arguments(count, index, prepare_only, minimum_split_size=7)
    _validate_shard_arguments(4, 3, True, minimum_split_size=7)


def test_real_smoke_selection_covers_every_retained_pg_in_each_split() -> None:
    dataset = load_training_dataset(TrainingUnit("curated_reduced_total", "dielectric"))
    selected = point_group_stratified_smoke_ids(dataset)
    assert set(selected) == {"train", "validation", "test"}
    for split, ids in selected.items():
        assert len(ids) == len(REDUCED_POINT_GROUPS) == 7
        assert tuple(dataset.by_id(sample_id).source["point_group"] for sample_id in ids) == (
            REDUCED_POINT_GROUPS
        )
        assert set(ids) <= set(getattr(dataset.split_manifest, split))


def test_comparison_cli_validates_common_ids_and_writes_table(tmp_path) -> None:
    dpa4 = tmp_path / "dpa4.jsonl"
    gmtnet = tmp_path / "gmtnet.jsonl"
    target = [[2.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 4.0]]
    rotated = [[3.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 4.0]]
    with dpa4.open("w", encoding="utf-8") as left, gmtnet.open("w", encoding="utf-8") as right:
        for index in range(677):
            left.write(json.dumps({"sample_id": f"sample-{index}", "prediction": target,
                                   "target": target}) + "\n")
            right.write(json.dumps({"sample_id": f"sample-{index}", "prediction": rotated,
                                    "target": rotated}) + "\n")
    summary = tmp_path / "comparison.json"
    table = tmp_path / "comparison.md"
    subprocess.run(
        [sys.executable, "-m", "src.cli.compare_reduced_benchmark", "--dpa4", str(dpa4),
         "--gmtnet", str(gmtnet), "--summary", str(summary), "--table", str(table)],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads(summary.read_text(encoding="utf-8"))
    assert report["status"] == "passed"
    assert report["test_count"] == 677
    assert report["metrics"]["GMTNet"]["rmse"] == 0.0
    assert "| Model | RMSE" in table.read_text(encoding="utf-8")
