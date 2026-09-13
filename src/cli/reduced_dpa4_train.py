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


ARCHITECTURE = ArchitectureConfig("B+A+PGE+R", "full_o3", "none", "full_o3", "full_pg")
POINT_GROUPS = ("2/m", "mm2", "mmm", "4/mmm", "-3m", "-43m", "m-3m")


def _ordered_samples(dataset, split: str):
    ids = getattr(dataset.split_manifest, split)
    return ids, tuple(dataset.by_id(sample_id) for sample_id in ids)


def run(arguments: argparse.Namespace) -> dict[str, object]:
    unit = TrainingUnit("curated_reduced_total", "dielectric")
    dataset = load_training_dataset(unit, manifest_path=arguments.manifest)
    expected = {"train": 5001, "validation": 637, "test": 677}
    actual = {
        name: len(getattr(dataset.split_manifest, name))
        for name in ("train", "validation", "test")
    }
    if actual != expected:
        raise ValueError(f"reduced dielectric_total split drift: {actual}")
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
        sample_ids, samples = _ordered_samples(dataset, split)
        cache = arguments.cache_root / "dpa4" / unit.namespace / f"{split}.pt"
        if cache.is_file():
            layout, examples = load_frozen_feature_cache(
                cache,
                expected_backbone="dpa4",
                expected_checkpoint_sha256=resource.sha256,
                expected_unit=unit,
                expected_split=split,
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
                split=split,
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
        expert_point_groups=POINT_GROUPS,
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
