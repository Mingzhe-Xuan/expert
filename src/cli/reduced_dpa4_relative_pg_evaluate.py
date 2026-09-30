from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import time

from ..backbones import BackboneResourceRegistry
from ..data import TrainingUnit, load_training_dataset
from ..experts import hidden_layout_from_multiplicities
from ..symmetry import PointGroupAncestorDAG
from ..training import write_smoke_report
from ..training.checkpoint_evaluation import evaluate_cached_backbone_checkpoint
from .reduced_cgcnn_parent_dag_train import _parent_enriched_examples, _shared_edge_ids
from .reduced_dpa4_relative_pg_train import (
    ARCHITECTURE,
    DEFAULT_HIDDEN_MULTIPLICITIES,
    _load_feature_splits,
)
from .reduced_protocol import canonical_expert_point_groups
from .reporting import execution_metadata, write_single_case_junit


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def select_retained_checkpoint(
    source_summary: dict[str, object],
    *,
    minimum_checkpoint_epoch_exclusive: int,
) -> dict[str, object]:
    history = source_summary.get("history")
    archives = source_summary.get("periodic_checkpoints")
    if not isinstance(history, list) or not isinstance(archives, list):
        raise ValueError("source summary lacks history or retained checkpoints")
    by_epoch = {int(row["epoch"]): row for row in history}
    eligible = [
        archive
        for archive in archives
        if int(archive["epoch"]) > minimum_checkpoint_epoch_exclusive
    ]
    if not eligible:
        raise ValueError("source run has no retained checkpoint after the epoch threshold")
    return min(eligible, key=lambda row: float(by_epoch[int(row["epoch"])]["validation_mae"]))


def run(arguments: argparse.Namespace) -> dict[str, object]:
    source_summary_path = arguments.source_summary
    source_summary = json.loads(source_summary_path.read_text(encoding="utf-8"))
    if source_summary.get("status") != "passed":
        raise ValueError("source summary must be passed")
    if source_summary.get("routing") != "point_group_relative_edge_stick_breaking":
        raise ValueError("checkpoint-only evaluation requires legacy relative-PG routing")
    selected = select_retained_checkpoint(
        source_summary,
        minimum_checkpoint_epoch_exclusive=arguments.minimum_checkpoint_epoch_exclusive,
    )
    checkpoint = arguments.checkpoint
    if Path(str(selected["path"])).name != checkpoint.name:
        raise ValueError("checkpoint is not the selected retained eligible archive")
    if checkpoint.stat().st_size != int(selected["bytes"]) or _sha256(checkpoint) != str(
        selected["sha256"]
    ):
        raise ValueError("retained checkpoint bytes or SHA-256 mismatch")

    unit = TrainingUnit("curated_reduced_total", "dielectric")
    dataset = load_training_dataset(unit, manifest_path=arguments.manifest)
    full_ids = {
        name: tuple(getattr(dataset.split_manifest, name))
        for name in ("train", "validation", "test")
    }
    if {name: len(ids) for name, ids in full_ids.items()} != {
        "train": 5001,
        "validation": 637,
        "test": 677,
    }:
        raise ValueError("reduced dielectric_total split drift")
    dataset_sha256 = str(dataset[0].source["manifest_sha256"])
    if source_summary.get("dataset_sha256") != dataset_sha256:
        raise ValueError("source summary dataset SHA-256 mismatch")
    resource = BackboneResourceRegistry()["dpa4"]
    loader_arguments = SimpleNamespace(
        feature_shards=arguments.feature_shards,
        feature_shard_index=None,
        prepare_only=False,
        smoke=False,
        cache_root=arguments.cache_root,
        device=arguments.device,
    )
    source_layout, splits = _load_feature_splits(
        loader_arguments,
        unit,
        dataset,
        full_ids,
        dataset_sha256,
        resource,
    )
    class_dag = PointGroupAncestorDAG.from_path()
    for split in ("train", "validation", "test"):
        routing_cache = (
            arguments.cache_root
            / "dpa4-material-parent-dag-point-group-v3"
            / unit.namespace
            / "full"
            / f"{split}.pt"
        )
        splits[split], _ = _parent_enriched_examples(
            splits[split],
            routing_cache,
            sample_ids=full_ids[split],
            dataset_sha256=dataset_sha256,
            class_dag=class_dag,
        )
    expert_point_groups = canonical_expert_point_groups(
        splits["train"], splits["validation"], splits["test"]
    )
    if list(expert_point_groups) != source_summary.get("expert_point_groups"):
        raise ValueError("source expert point groups differ from frozen inputs")
    hidden_layout = hidden_layout_from_multiplicities(arguments.hidden_multiplicities)
    source_hidden = source_summary.get("hidden_multiplicities")
    if source_hidden is not None and list(arguments.hidden_multiplicities) != source_hidden:
        raise ValueError("requested hidden multiplicities differ from source run")
    result = evaluate_cached_backbone_checkpoint(
        unit=unit,
        source_layout=source_layout,
        test_examples=splits["test"],
        checkpoint_path=checkpoint,
        predictions_path=arguments.predictions,
        architecture=ARCHITECTURE,
        expert_point_groups=expert_point_groups,
        material_edge_ids=_shared_edge_ids(splits),
        hidden_layout=hidden_layout,
        batch_size=arguments.batch_size,
        minimum_checkpoint_epoch_exclusive=arguments.minimum_checkpoint_epoch_exclusive,
        device=arguments.device,
    )
    selected_history = source_summary["history"][int(result["checkpoint_epoch"]) - 1]
    return {
        "schema_version": 1,
        "status": "passed",
        "model": source_summary.get("model"),
        "mode": "checkpoint_only_evaluation",
        "source_summary": str(source_summary_path),
        "source_summary_sha256": _sha256(source_summary_path),
        "dataset_sha256": dataset_sha256,
        "hidden_multiplicities": list(arguments.hidden_multiplicities),
        "split_counts": {name: len(rows) for name, rows in splits.items()},
        "selected_validation_mae": selected_history["validation_mae"],
        "selected_validation_fnorm": selected_history["validation_fnorm"],
        "execution": execution_metadata(),
        **result,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a retained post-100 DPA relative-PG checkpoint without training"
    )
    parser.add_argument("--manifest", type=Path, default=Path("data/manifests/curated_tensors_reduced_gt_5pct.json"))
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--source-summary", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--junit", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--feature-shards", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--minimum-checkpoint-epoch-exclusive", type=int, default=100)
    parser.add_argument(
        "--hidden-multiplicities",
        type=int,
        nargs=5,
        default=DEFAULT_HIDDEN_MULTIPLICITIES,
        metavar=("L0", "L1", "L2", "L3", "L4"),
    )
    arguments = parser.parse_args()
    started = time.perf_counter()
    error = None
    try:
        report = run(arguments)
    except Exception as caught:
        error = caught
        report = {
            "schema_version": 1,
            "status": "failed",
            "error_type": type(caught).__name__,
            "error": str(caught),
            "execution": execution_metadata(),
        }
    write_smoke_report(arguments.summary, report)
    write_single_case_junit(
        arguments.junit,
        suite_name="reduced_dpa4_relative_pg_checkpoint_evaluation",
        seconds=time.perf_counter() - started,
        error=error,
    )
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    if error is not None:
        raise error


if __name__ == "__main__":
    main()
