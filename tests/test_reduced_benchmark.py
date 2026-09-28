from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from src.configs import ArchitectureConfig
from src.cli.reduced_dpa4_train import (
    _merge_sharded_examples,
    _shard_sample_ids,
    _validate_shard_arguments,
)
from src.cli.reduced_dpa4_relative_pg_train import (
    DEFAULT_HIDDEN_MULTIPLICITIES,
    _require_cuda_expert_dispatch,
)
from src.cli.reduced_protocol import REDUCED_POINT_GROUPS, point_group_stratified_smoke_ids
from src.data import TrainingUnit, load_training_dataset
from src.experts import hidden_layout_from_multiplicities
from src.irreps import IrrepLayout, IrrepTerm
from src.symmetry import PointGroupAncestorDAG, build_point_group_parent_dag
from src.training.benchmark import CachedBackboneTensorModel


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


def test_dpa4_relative_pg_launcher_matches_gmtnet_and_archives_every_20_epochs() -> None:
    launcher = (ROOT / "slurm" / "train_reduced_dpa4_relative_pg.sbatch").read_text(
        encoding="utf-8"
    )
    assert "src.cli.reduced_dpa4_relative_pg_train" in launcher
    assert '--batch-size "${EXPERT_REDUCED_DPA4_BATCH_SIZE:-64}"' in launcher
    assert '--seed "${EXPERT_REDUCED_DPA4_SEED:-42}"' in launcher
    assert '--checkpoint-interval "${EXPERT_DPA4_CHECKPOINT_INTERVAL:-20}"' in launcher
    assert "#SBATCH --time=3-00:00:00" in launcher


def test_dpa4_relative_pg_64d_launcher_is_isolated_and_protocol_matched() -> None:
    launcher = (ROOT / "slurm" / "train_reduced_dpa4_relative_pg_64d.sbatch").read_text(
        encoding="utf-8"
    )
    assert "src.cli.reduced_dpa4_relative_pg_train" in launcher
    assert 'run_root="results/reduced-benchmark/dpa4-relative-pg-64d"' in launcher
    assert "--hidden-multiplicities 16 2 2 2 2" in launcher
    assert '--batch-size "${EXPERT_REDUCED_DPA4_64D_BATCH_SIZE:-64}"' in launcher
    assert '--seed "${EXPERT_REDUCED_DPA4_64D_SEED:-42}"' in launcher
    assert '--checkpoint-interval "${EXPERT_DPA4_64D_CHECKPOINT_INTERVAL:-20}"' in launcher
    assert "#SBATCH --time=3-00:00:00" in launcher
    assert 'run_root="results/reduced-benchmark/dpa4-relative-pg"' not in launcher


def test_scalar_widened_relative_pg_has_dataset_layout_and_parameter_count() -> None:
    source_layout = IrrepLayout(
        tuple(
            IrrepTerm(64, degree, parity, f"dpa4_l{degree}_{parity}")
            for degree in range(5)
            for parity in ("e", "o")
        )
    )
    hidden_layout = hidden_layout_from_multiplicities((16, 2, 2, 2, 2))
    groups = (
        "2/m", "mm2", "mmm", "4/m", "4mm", "-42m", "4/mmm", "-3m",
        "6/m", "6mm", "-6m2", "6/mmm", "m-3", "-43m", "m-3m",
    )
    class_dag = PointGroupAncestorDAG.from_path()
    edge_ids = tuple(
        sorted(
            {
                embedding.edge_id
                for point_group in REDUCED_POINT_GROUPS
                for embedding in build_point_group_parent_dag(
                    f"test-{point_group}", class_dag.number(point_group), class_dag
                ).embeddings
            }
        )
    )
    architecture = ArchitectureConfig(
        "B+A+PGE+R", "full_o3", "none", "full_o3", "full_pg"
    )
    model = CachedBackboneTensorModel(
        source_layout,
        architecture,
        "dielectric",
        groups,
        edge_ids,
        hidden_layout,
    )
    assert DEFAULT_HIDDEN_MULTIPLICITIES == (8, 2, 2, 2, 2)
    assert len(edge_ids) == 24
    assert tuple(term.multiplicity for term in hidden_layout.terms) == (16, 2, 2, 2, 2)
    assert hidden_layout.dimension == 64
    assert sum(parameter.numel() for parameter in model.parameters()) == 130_196
    assert sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    ) == 130_196


@pytest.mark.parametrize(
    "values",
    ((8, 2, 2, 2), (8, 2, 0, 2, 2), (8, 2, -1, 2, 2), (8, 2, 2.5, 2, 2)),
)
def test_hidden_multiplicity_profiles_fail_closed(values) -> None:
    with pytest.raises(ValueError, match="five positive integers"):
        hidden_layout_from_multiplicities(values)


def test_dpa4_relative_pg_requires_observed_distinct_cuda_streams() -> None:
    _require_cuda_expert_dispatch(
        {
            "expert_buckets": 3,
            "max_structures_per_expert": 4,
            "asynchronous_cuda": True,
            "cuda_streams": 3,
        }
    )
    for stats in (
        None,
        {"expert_buckets": 1, "asynchronous_cuda": False, "cuda_streams": 1},
        {"expert_buckets": 3, "asynchronous_cuda": True, "cuda_streams": 2},
    ):
        with pytest.raises(RuntimeError, match="expert-stream dispatch"):
            _require_cuda_expert_dispatch(stats)


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
    cgcnn = tmp_path / "cgcnn.jsonl"
    parent = tmp_path / "parent.jsonl"
    target = [[2.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 4.0]]
    rotated = [[3.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 4.0]]
    with (
        dpa4.open("w", encoding="utf-8") as left,
        gmtnet.open("w", encoding="utf-8") as right,
        cgcnn.open("w", encoding="utf-8") as third,
        parent.open("w", encoding="utf-8") as fourth,
    ):
        for index in range(677):
            left.write(json.dumps({"sample_id": f"sample-{index}", "prediction": target,
                                   "target": target}) + "\n")
            right.write(json.dumps({"sample_id": f"sample-{index}", "prediction": rotated,
                                    "target": rotated}) + "\n")
            third.write(json.dumps({"sample_id": f"sample-{index}", "prediction": target,
                                    "target": target}) + "\n")
            fourth.write(json.dumps({"sample_id": f"sample-{index}", "prediction": target,
                                     "target": target}) + "\n")
    summary = tmp_path / "comparison.json"
    table = tmp_path / "comparison.md"
    subprocess.run(
        [sys.executable, "-m", "src.cli.compare_reduced_benchmark", "--dpa4", str(dpa4),
         "--gmtnet", str(gmtnet), "--cgcnn-full-pg", str(cgcnn),
         "--cgcnn-parent-dag", str(parent),
         "--cgcnn-relative-parent-dag", str(parent),
         "--summary", str(summary), "--table", str(table)],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads(summary.read_text(encoding="utf-8"))
    assert report["status"] == "passed"
    assert report["test_count"] == 677
    assert report["metrics"]["GMTNet"]["rmse"] == 0.0
    assert report["metrics"]["CGCNN B+A+PGE+R full_pg (current-group only)"]["rmse"] == 0.0
    assert report["metrics"][
        "CGCNN B+A+PGE+R full_pg (PG parent-DAG all ancestors)"
    ]["rmse"] == 0.0
    assert report["metrics"][
        "CGCNN B+A+PGE+R full_pg (relative-PG path-weighted, 56D)"
    ]["rmse"] == 0.0
    assert "| Model | RMSE" in table.read_text(encoding="utf-8")


def test_comparison_launcher_requires_explicit_artifacts_and_job_scopes_outputs() -> None:
    launcher = (ROOT / "slurm" / "compare_reduced_benchmark.sbatch").read_text(
        encoding="utf-8"
    )
    assert "EXPERT_CGCNN_VENV:?" in launcher
    assert "EXPERT_REDUCED_DPA4_PREDICTIONS:?" in launcher
    assert "EXPERT_REDUCED_GMTNET_PREDICTIONS:?" in launcher
    assert "EXPERT_REDUCED_CGCNN_PREDICTIONS:?" in launcher
    assert "EXPERT_REDUCED_CGCNN_PARENT_PREDICTIONS:?" in launcher
    assert "EXPERT_REDUCED_CGCNN_RELATIVE_PARENT_PREDICTIONS:?" in launcher
    assert '--cgcnn-full-pg "${EXPERT_REDUCED_CGCNN_PREDICTIONS}"' in launcher
    assert '--cgcnn-parent-dag "${EXPERT_REDUCED_CGCNN_PARENT_PREDICTIONS}"' in launcher
    assert (
        '--cgcnn-relative-parent-dag "${EXPERT_REDUCED_CGCNN_RELATIVE_PARENT_PREDICTIONS}"'
        in launcher
    )
    assert 'summary-${SLURM_JOB_ID}.json' in launcher
    assert 'table-${SLURM_JOB_ID}.md' in launcher


def test_parent_launcher_uses_offline_point_group_asset_without_hall_registry() -> None:
    launcher = (ROOT / "slurm" / "train_reduced_cgcnn_parent_dag.sbatch").read_text(
        encoding="utf-8"
    )
    assert "EXPERT_REDUCED_HALL_EMBEDDING_ROOT" not in launcher
    assert "--hall-embedding-root" not in launcher
    assert "#SBATCH --time=3-00:00:00" in launcher
