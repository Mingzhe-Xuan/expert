from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from ..backbones import BackboneResourceRegistry
from ..configs import ArchitectureConfig
from ..data import TrainingUnit, load_training_dataset
from ..experts import default_hidden_layout
from ..models import build_backbone_adapter
from ..training import (
    BenchmarkConfig,
    extract_frozen_examples,
    load_frozen_feature_cache,
    save_frozen_feature_cache,
    train_cached_backbone_readout,
    write_smoke_report,
)
from .reporting import execution_metadata, write_single_case_junit
from .reduced_protocol import REDUCED_POINT_GROUPS, point_group_stratified_smoke_ids


ARCHITECTURE = ArchitectureConfig("B+A+PGE+R", "full_o3", "none", "full_o3", "full_pg")
def _ordered_samples(dataset, ids):
    return ids, tuple(dataset.by_id(sample_id) for sample_id in ids)


def _shard_sample_ids(sample_ids, shard_count: int, shard_index: int):
    if shard_count < 1 or not 0 <= shard_index < shard_count:
        raise ValueError("feature shard index/count are invalid")
    return tuple(sample_ids[shard_index::shard_count])


def _worker_shard_indices(shard_count: int, worker_count: int, worker_index: int):
    """Assign fine-grained cache partitions exactly once across persistent workers."""

    if shard_count < 1 or worker_count < 1 or worker_count > shard_count:
        raise ValueError("feature worker/shard counts are invalid")
    if not 0 <= worker_index < worker_count:
        raise ValueError("feature worker index is outside feature_workers")
    return tuple(range(worker_index, shard_count, worker_count))


def _merge_sharded_examples(expected_ids, shards):
    by_id = {}
    for shard in shards:
        for example in shard:
            if example.sample_id in by_id:
                raise ValueError(f"duplicate frozen shard sample ID {example.sample_id!r}")
            by_id[example.sample_id] = example
    expected = tuple(expected_ids)
    missing = [sample_id for sample_id in expected if sample_id not in by_id]
    extra = sorted(set(by_id) - set(expected))
    if missing or extra:
        raise ValueError(f"frozen shard coverage mismatch: missing={missing[:3]}, extra={extra[:3]}")
    return tuple(by_id[sample_id] for sample_id in expected)


def _cache_path(root, unit, scope, split, shard_count, shard_index):
    base = root / "dpa4" / unit.namespace / scope
    if shard_count == 1:
        return base / f"{split}.pt"
    return base / f"shards-{shard_count}" / f"shard-{shard_index}" / f"{split}.pt"


def _validate_shard_arguments(shard_count, shard_index, prepare_only, minimum_split_size):
    if shard_count < 1:
        raise ValueError("feature_shards must be positive")
    if shard_count > minimum_split_size:
        raise ValueError("feature_shards cannot exceed the smallest selected split")
    if shard_index is not None:
        if not prepare_only:
            raise ValueError("a feature shard worker must use --prepare-only")
        if not 0 <= shard_index < shard_count:
            raise ValueError("feature_shard_index is outside feature_shards")


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
    shard_count = arguments.feature_shards
    shard_index = arguments.feature_shard_index
    _validate_shard_arguments(
        shard_count,
        shard_index,
        arguments.prepare_only,
        min(len(ids) for ids in selected_ids.values()),
    )
    actual = {
        name: len(
            ids
            if shard_index is None
            else _shard_sample_ids(ids, shard_count, shard_index)
        )
        for name, ids in selected_ids.items()
    }
    dataset_sha256 = str(dataset[0].source["manifest_sha256"])
    resource = BackboneResourceRegistry()["dpa4"]
    if resource.sha256 is None:
        raise ValueError("DPA4 resource lacks a frozen checkpoint SHA-256")
    adapter = None
    source_layout = None
    splits = {}

    def progress(current: int, total: int, sample_id: str) -> None:
        if current == 1 or current == total or current % 25 == 0:
            print(json.dumps({"event": "feature_extraction", "current": current,
                              "total": total, "sample_id": sample_id}, sort_keys=True), flush=True)

    for split in ("train", "validation", "test"):
        cache_scope = "smoke" if arguments.smoke else "full"
        if shard_count > 1 and shard_index is None:
            shard_layout = None
            shard_examples = []
            for current_shard in range(shard_count):
                sample_ids = _shard_sample_ids(
                    selected_ids[split], shard_count, current_shard
                )
                cache = _cache_path(
                    arguments.cache_root,
                    unit,
                    cache_scope,
                    split,
                    shard_count,
                    current_shard,
                )
                if not cache.is_file():
                    raise FileNotFoundError(f"missing frozen feature shard {cache}")
                current_layout, current_examples = load_frozen_feature_cache(
                    cache,
                    expected_backbone="dpa4",
                    expected_checkpoint_sha256=resource.sha256,
                    expected_unit=unit,
                    expected_split=f"{split}:shard:{current_shard}/{shard_count}",
                    expected_sample_ids=sample_ids,
                    expected_dataset_sha256=dataset_sha256,
                )
                if shard_layout is None:
                    shard_layout = current_layout
                elif shard_layout != current_layout:
                    raise ValueError("source layout differs across frozen feature shards")
                shard_examples.append(current_examples)
            layout = shard_layout
            examples = _merge_sharded_examples(selected_ids[split], shard_examples)
        else:
            current_shard = 0 if shard_index is None else shard_index
            sample_ids = (
                selected_ids[split]
                if shard_count == 1
                else _shard_sample_ids(selected_ids[split], shard_count, current_shard)
            )
            _, samples = _ordered_samples(dataset, sample_ids)
            cache = _cache_path(
                arguments.cache_root,
                unit,
                cache_scope,
                split,
                shard_count,
                current_shard,
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
    if arguments.prepare_only:
        return {
            "schema_version": 1,
            "status": "passed",
            "mode": "prepare_only",
            "backbone": "dpa4",
            "training_unit": unit.namespace,
            "dataset_sha256": dataset_sha256,
            "architecture": ARCHITECTURE.to_dict(),
            "split_counts": actual,
            "feature_shards": shard_count,
            "feature_shard_index": shard_index,
            "execution": execution_metadata(),
        }
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
        expert_point_groups=REDUCED_POINT_GROUPS,
        config=BenchmarkConfig(
            max_epochs=arguments.max_epochs,
            batch_size=arguments.batch_size,
            learning_rate=arguments.learning_rate,
            weight_decay=arguments.weight_decay,
            patience=arguments.patience,
            seed=arguments.seed,
            normalization=arguments.normalization,
            gradient_clip_norm=arguments.gradient_clip_norm,
        ),
        device=arguments.device,
    )
    report["dataset_sha256"] = dataset_sha256
    report["execution"] = execution_metadata()
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Train exact DPA4 B+A+PGE+R/full_pg on reduced total dielectric")
    parser.add_argument("--manifest", type=Path, default=Path("data/manifests/curated_tensors_reduced_gt_5pct.json"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--junit", type=Path, required=True)
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-6)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260911)
    parser.add_argument("--normalization", choices=("rms", "variance"), default="rms")
    parser.add_argument("--gradient-clip-norm", type=float, default=10.0)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--feature-shards", type=int, default=1)
    parser.add_argument("--feature-shard-index", type=int)
    parser.add_argument("--smoke", action="store_true",
                        help="Use one real sample per retained point group in each split")
    arguments = parser.parse_args()
    started = time.perf_counter()
    error = None
    try:
        report = run(arguments)
    except Exception as caught:
        error = caught
        report = {"schema_version": 1, "status": "failed", "error_type": type(caught).__name__,
                  "error": str(caught), "execution": execution_metadata()}
    write_smoke_report(arguments.summary, report)
    write_single_case_junit(arguments.junit, suite_name="reduced_dpa4_full_pg",
                            seconds=time.perf_counter() - started, error=error)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    if error is not None:
        raise error


if __name__ == "__main__":
    main()
