from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from ..backbones import BackboneResourceRegistry
from ..baselines.gmtnet import GMTNetConfig, run_gmtnet_benchmark
from ..data import TrainingUnit, load_training_dataset
from ..training import load_frozen_feature_cache, write_smoke_report
from .reduced_dpa4_train import _cache_path, _merge_sharded_examples, _shard_sample_ids
from .reduced_protocol import point_group_stratified_smoke_ids
from .reporting import execution_metadata, write_single_case_junit


def _load_dpa4_feature_splits(
    *,
    cache_root: Path,
    unit: TrainingUnit,
    full_ids: dict[str, tuple[str, ...]],
    selected_ids: dict[str, tuple[str, ...]],
    shard_count: int,
    checkpoint_sha256: str,
    dataset_sha256: str,
):
    if shard_count < 1 or shard_count > min(len(ids) for ids in full_ids.values()):
        raise ValueError("DPA4 feature shard count is outside the full split range")
    source_layout = None
    selected_splits = {}
    for split in ("train", "validation", "test"):
        shards = []
        for shard_index in range(shard_count):
            sample_ids = _shard_sample_ids(full_ids[split], shard_count, shard_index)
            path = _cache_path(cache_root, unit, "full", split, shard_count, shard_index)
            if not path.is_file():
                raise FileNotFoundError(f"missing frozen DPA4 feature shard {path}")
            layout, examples = load_frozen_feature_cache(
                path,
                expected_backbone="dpa4",
                expected_checkpoint_sha256=checkpoint_sha256,
                expected_unit=unit,
                expected_split=f"{split}:shard:{shard_index}/{shard_count}",
                expected_sample_ids=sample_ids,
                expected_dataset_sha256=dataset_sha256,
            )
            if source_layout is None:
                source_layout = layout
            elif source_layout != layout:
                raise ValueError("DPA4 source layout differs across cached shards")
            shards.append(examples)
        merged = _merge_sharded_examples(full_ids[split], shards)
        by_id = {example.sample_id: example for example in merged}
        selected_splits[split] = tuple(by_id[sample_id] for sample_id in selected_ids[split])
    if source_layout is None:
        raise RuntimeError("DPA4 feature cache did not provide a source layout")
    return source_layout, selected_splits


def run(arguments: argparse.Namespace) -> dict[str, object]:
    unit = TrainingUnit("curated_reduced_total", "dielectric")
    dataset = load_training_dataset(unit, manifest_path=arguments.manifest)
    full_ids = {
        name: tuple(getattr(dataset.split_manifest, name))
        for name in ("train", "validation", "test")
    }
    expected = {"train": 5001, "validation": 637, "test": 677}
    if {name: len(ids) for name, ids in full_ids.items()} != expected:
        raise ValueError("reduced dielectric_total split drift")
    selected_ids = point_group_stratified_smoke_ids(dataset) if arguments.smoke else full_ids
    dataset_sha256 = str(dataset[0].source["manifest_sha256"])
    resource = BackboneResourceRegistry()["dpa4"]
    if resource.sha256 is None:
        raise ValueError("DPA4 resource lacks a frozen checkpoint SHA-256")
    source_layout, feature_splits = _load_dpa4_feature_splits(
        cache_root=arguments.dpa_cache_root,
        unit=unit,
        full_ids=full_ids,
        selected_ids=selected_ids,
        shard_count=arguments.feature_shards,
        checkpoint_sha256=resource.sha256,
        dataset_sha256=dataset_sha256,
    )
    report = run_gmtnet_benchmark(
        dataset,
        official_root=arguments.official_root,
        cache_path=arguments.cache,
        checkpoint_path=arguments.checkpoint,
        predictions_path=arguments.predictions,
        config=GMTNetConfig(
            epochs=arguments.epochs,
            batch_size=arguments.batch_size,
            learning_rate=arguments.learning_rate,
            end_learning_rate=arguments.end_learning_rate,
            weight_decay=arguments.weight_decay,
            seed=arguments.seed,
            checkpoint_interval=arguments.checkpoint_interval,
        ),
        device=arguments.device,
        split_ids=selected_ids,
        dpa4_feature_layout=source_layout,
        dpa4_feature_splits=feature_splits,
    )
    report["dpa4_checkpoint_sha256"] = resource.sha256
    report["feature_shards"] = arguments.feature_shards
    report["execution"] = execution_metadata()
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train official GMTNet with frozen invariant DPA4 node embeddings"
    )
    parser.add_argument("--official-root", type=Path, required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/manifests/curated_tensors_reduced_gt_5pct.json"),
    )
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--dpa-cache-root", type=Path, required=True)
    parser.add_argument("--feature-shards", type=int, default=64)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--junit", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--end-learning-rate", type=float, default=1.0e-5)
    parser.add_argument("--weight-decay", type=float, default=1.0e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint-interval", type=int, default=20)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Use one real sample per retained point group in each frozen split",
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
        suite_name="reduced_dpa4_gmtnet",
        seconds=time.perf_counter() - started,
        error=error,
    )
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    if error is not None:
        raise error


if __name__ == "__main__":
    main()
