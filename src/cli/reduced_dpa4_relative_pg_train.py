from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import torch

from ..backbones import BackboneResourceRegistry
from ..configs import ArchitectureConfig
from ..data import TrainingUnit, load_training_dataset
from ..experts import default_hidden_layout
from ..models import build_backbone_adapter
from ..symmetry import PointGroupAncestorDAG, parent_detection_config_sha256
from ..training import (
    BenchmarkConfig,
    extract_frozen_examples,
    load_frozen_feature_cache,
    save_frozen_feature_cache,
    train_cached_backbone_readout,
    write_smoke_report,
)
from .reduced_cgcnn_parent_dag_train import _parent_enriched_examples, _shared_edge_ids
from .reduced_dpa4_train import (
    _cache_path,
    _merge_sharded_examples,
    _ordered_samples,
    _shard_sample_ids,
    _validate_shard_arguments,
)
from .reduced_protocol import canonical_expert_point_groups, point_group_stratified_smoke_ids
from .reporting import execution_metadata, write_single_case_junit


ARCHITECTURE = ArchitectureConfig("B+A+PGE+R", "full_o3", "none", "full_o3", "full_pg")
MODEL_NAME = "DPA4 B+A+PGE+R full_pg relative-position PG parent-DAG path-weighted"


def _require_cuda_expert_dispatch(stats) -> None:
    if (
        not isinstance(stats, dict)
        or int(stats.get("expert_buckets", 0)) < 2
        or stats.get("asynchronous_cuda") is not True
        or int(stats.get("cuda_streams", 0)) != int(stats["expert_buckets"])
    ):
        raise RuntimeError("DPA-relative-PG CUDA expert-stream dispatch was not observed")


def _load_feature_splits(arguments, unit, dataset, selected_ids, dataset_sha256, resource):
    shard_count = arguments.feature_shards
    shard_index = arguments.feature_shard_index
    _validate_shard_arguments(
        shard_count,
        shard_index,
        arguments.prepare_only,
        min(len(ids) for ids in selected_ids.values()),
    )
    adapter = None
    source_layout = None
    splits = {}
    scope = "smoke" if arguments.smoke else "full"

    def progress(current: int, total: int, sample_id: str) -> None:
        if current == 1 or current == total or current % 25 == 0:
            print(
                json.dumps(
                    {
                        "event": "feature_extraction",
                        "current": current,
                        "total": total,
                        "sample_id": sample_id,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    for split in ("train", "validation", "test"):
        if shard_count > 1 and shard_index is None:
            shard_layout = None
            shards = []
            for current_shard in range(shard_count):
                sample_ids = _shard_sample_ids(selected_ids[split], shard_count, current_shard)
                cache = _cache_path(
                    arguments.cache_root, unit, scope, split, shard_count, current_shard
                )
                if not cache.is_file():
                    raise FileNotFoundError(f"missing frozen feature shard {cache}")
                layout, examples = load_frozen_feature_cache(
                    cache,
                    expected_backbone="dpa4",
                    expected_checkpoint_sha256=resource.sha256,
                    expected_unit=unit,
                    expected_split=f"{split}:shard:{current_shard}/{shard_count}",
                    expected_sample_ids=sample_ids,
                    expected_dataset_sha256=dataset_sha256,
                )
                if shard_layout is None:
                    shard_layout = layout
                elif shard_layout != layout:
                    raise ValueError("source layout differs across frozen feature shards")
                shards.append(examples)
            layout = shard_layout
            examples = _merge_sharded_examples(selected_ids[split], shards)
        else:
            current_shard = 0 if shard_index is None else shard_index
            sample_ids = (
                selected_ids[split]
                if shard_count == 1
                else _shard_sample_ids(selected_ids[split], shard_count, current_shard)
            )
            _, samples = _ordered_samples(dataset, sample_ids)
            cache = _cache_path(
                arguments.cache_root, unit, scope, split, shard_count, current_shard
            )
            cache_split = (
                split
                if shard_count == 1
                else f"{split}:shard:{current_shard}/{shard_count}"
            )
            if cache.is_file():
                layout, examples = load_frozen_feature_cache(
                    cache,
                    expected_backbone="dpa4",
                    expected_checkpoint_sha256=resource.sha256,
                    expected_unit=unit,
                    expected_split=cache_split,
                    expected_sample_ids=sample_ids,
                    expected_dataset_sha256=dataset_sha256,
                )
            else:
                if adapter is None:
                    adapter = build_backbone_adapter(
                        "dpa4", default_hidden_layout(ARCHITECTURE), device=arguments.device
                    )
                layout, examples = extract_frozen_examples(
                    adapter,
                    samples,
                    cutoff=resource.cutoff_angstrom,
                    device=arguments.device,
                    progress=progress,
                )
                save_frozen_feature_cache(
                    cache,
                    backbone_family="dpa4",
                    checkpoint_sha256=resource.sha256,
                    unit=unit,
                    split=cache_split,
                    layout=layout,
                    examples=examples,
                    dataset_sha256=dataset_sha256,
                )
        if source_layout is None:
            source_layout = layout
        elif source_layout != layout:
            raise ValueError("source layout differs across cached splits")
        splits[split] = examples
    return source_layout, splits


def run(arguments: argparse.Namespace) -> dict[str, object]:
    unit = TrainingUnit("curated_reduced_total", "dielectric")
    dataset = load_training_dataset(unit, manifest_path=arguments.manifest)
    expected = {"train": 5001, "validation": 637, "test": 677}
    full_ids = {
        name: tuple(getattr(dataset.split_manifest, name))
        for name in ("train", "validation", "test")
    }
    if {name: len(ids) for name, ids in full_ids.items()} != expected:
        raise ValueError("reduced dielectric_total split drift")
    selected_ids = point_group_stratified_smoke_ids(dataset) if arguments.smoke else full_ids
    dataset_sha256 = str(dataset[0].source["manifest_sha256"])
    resource = BackboneResourceRegistry()["dpa4"]
    if resource.sha256 is None:
        raise ValueError("DPA4 resource lacks a frozen checkpoint SHA-256")
    source_layout, splits = _load_feature_splits(
        arguments, unit, dataset, selected_ids, dataset_sha256, resource
    )

    class_dag = PointGroupAncestorDAG.from_path()
    scope = "smoke" if arguments.smoke else "full"
    coverage = {}
    if arguments.feature_shard_index is None:
        for split in ("train", "validation", "test"):
            routing_cache = (
                arguments.cache_root
                / "dpa4-material-parent-dag-point-group-v3"
                / unit.namespace
                / scope
                / f"{split}.pt"
            )
            splits[split], coverage[split] = _parent_enriched_examples(
                splits[split],
                routing_cache,
                sample_ids=selected_ids[split],
                dataset_sha256=dataset_sha256,
                class_dag=class_dag,
            )

    common = {
        "schema_version": 1,
        "status": "passed",
        "model": MODEL_NAME,
        "routing": "point_group_relative_edge_stick_breaking",
        "training_unit": unit.namespace,
        "dataset_sha256": dataset_sha256,
        "feature_embedding": {
            "backbone": "dpa4",
            "checkpoint_sha256": resource.sha256,
            "feature_tap": resource.feature_tap,
            "feature_layout": resource.feature_layout,
        },
        "parent_detection": {
            "config_sha256": parent_detection_config_sha256(class_dag),
            "coverage_by_split": coverage,
            "topology": "offline_complete_oriented_point_group_paths",
            "class_dag_sha256": class_dag.asset_sha256,
        },
        "path_fusion": {
            "path_definition": "maximal_current_point_group_to_root",
            "between_path_prior": "node_count_normalized",
            "within_path_weighting": "relative_vector_point_group_edge_stick_breaking",
            "duplicate_destination_reduction": "sum_then_normalize",
        },
        "split_counts": {name: len(rows) for name, rows in splits.items()},
        "execution": execution_metadata(),
    }
    if arguments.prepare_only:
        return {
            **common,
            "mode": "prepare_only",
            "feature_shards": arguments.feature_shards,
            "feature_shard_index": arguments.feature_shard_index,
        }

    expert_point_groups = canonical_expert_point_groups(
        splits["train"], splits["validation"], splits["test"]
    )
    material_edge_ids = _shared_edge_ids(splits)
    report = train_cached_backbone_readout(
        backbone_family="dpa4",
        unit=unit,
        source_layout=source_layout,
        train_examples=splits["train"],
        validation_examples=splits["validation"],
        test_examples=splits["test"],
        checkpoint_path=arguments.checkpoint,
        predictions_path=arguments.predictions,
        architecture=ARCHITECTURE,
        expert_point_groups=expert_point_groups,
        material_edge_ids=material_edge_ids,
        config=BenchmarkConfig(
            max_epochs=arguments.epochs,
            batch_size=arguments.batch_size,
            learning_rate=arguments.learning_rate,
            end_learning_rate=arguments.end_learning_rate,
            weight_decay=arguments.weight_decay,
            patience=arguments.epochs,
            seed=arguments.seed,
            training_protocol="gmtnet",
            checkpoint_interval=arguments.checkpoint_interval,
        ),
        device=arguments.device,
    )
    if torch.device(arguments.device).type == "cuda":
        _require_cuda_expert_dispatch(report.get("last_dispatch_stats"))
    report.update(common)
    report["expert_point_groups"] = list(expert_point_groups)
    report["source_retained_point_groups"] = list(expert_point_groups)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train DPA4 features with relative-position point-group parent routing"
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/manifests/curated_tensors_reduced_gt_5pct.json"),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--junit", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--end-learning-rate", type=float, default=1.0e-5)
    parser.add_argument("--weight-decay", type=float, default=1.0e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint-interval", type=int, default=20)
    parser.add_argument("--feature-shards", type=int, default=64)
    parser.add_argument("--feature-shard-index", type=int)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--smoke", action="store_true")
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
        suite_name="reduced_dpa4_relative_pg",
        seconds=time.perf_counter() - started,
        error=error,
    )
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    if error is not None:
        raise error


if __name__ == "__main__":
    main()
