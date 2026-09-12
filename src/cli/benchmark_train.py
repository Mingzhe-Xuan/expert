from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from ..backbones import BACKBONE_FAMILIES, BackboneResourceRegistry
from ..configs import ArchitectureConfig
from ..data import TrainingUnit, load_training_dataset
from ..experts import default_hidden_layout
from ..models import build_backbone_adapter
from ..training import (
    BenchmarkConfig,
    extract_frozen_examples,
    load_frozen_feature_cache,
    require_published_split_counts,
    save_frozen_feature_cache,
    train_cached_backbone_readout,
    write_smoke_report,
)
from .reporting import execution_metadata, write_single_case_junit


def _ordered_samples(dataset, split: str):
    ids = getattr(dataset.split_manifest, split)
    return ids, tuple(dataset.by_id(sample_id) for sample_id in ids)


def run_benchmark(arguments: argparse.Namespace) -> dict[str, object]:
    unit = TrainingUnit("jarvis_tensor", arguments.target)
    dataset = load_training_dataset(unit)
    require_published_split_counts(unit, dataset.split_manifest)
    resource = BackboneResourceRegistry()[arguments.backbone]
    if resource.sha256 is None:
        raise ValueError("backbone resource lacks a frozen checkpoint SHA-256")
    architecture = ArchitectureConfig("B+R", "none", "none", "full_o3", "none")
    adapter = None
    source_layout = None
    splits = {}

    def progress(current: int, total: int, sample_id: str) -> None:
        if current == 1 or current == total or current % 25 == 0:
            print(
                json.dumps(
                    {"event": "feature_extraction", "current": current, "total": total,
                     "sample_id": sample_id},
                    sort_keys=True,
                ),
                flush=True,
            )

    for split in ("train", "validation", "test"):
        sample_ids, samples = _ordered_samples(dataset, split)
        cache = arguments.cache_root / arguments.backbone / unit.namespace / f"{split}.pt"
        if cache.is_file():
            layout, examples = load_frozen_feature_cache(
                cache,
                expected_backbone=arguments.backbone,
                expected_checkpoint_sha256=resource.sha256,
                expected_unit=unit,
                expected_split=split,
                expected_sample_ids=sample_ids,
            )
        else:
            if adapter is None:
                adapter = build_backbone_adapter(
                    arguments.backbone,
                    default_hidden_layout(architecture),
                    device=arguments.device,
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
                backbone_family=arguments.backbone,
                checkpoint_sha256=resource.sha256,
                unit=unit,
                split=split,
                layout=layout,
                examples=examples,
            )
        if source_layout is None:
            source_layout = layout
        elif layout != source_layout:
            raise ValueError("source layout differs across cached dataset splits")
        splits[split] = examples
    if arguments.prepare_only:
        return {
            "schema_version": 1,
            "status": "passed",
            "mode": "prepare_only",
            "backbone": arguments.backbone,
            "training_unit": unit.namespace,
            "source_layout": list(source_layout.to_spec()),
            "split_counts": {name: len(rows) for name, rows in splits.items()},
            "execution": execution_metadata(),
        }
    report = train_cached_backbone_readout(
        backbone_family=arguments.backbone,
        unit=unit,
        source_layout=source_layout,
        train_examples=splits["train"],
        validation_examples=splits["validation"],
        test_examples=splits["test"],
        checkpoint_path=arguments.checkpoint,
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
    report["execution"] = execution_metadata()
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a frozen backbone plus direct readout on a full JARVIS tensor split"
    )
    parser.add_argument("--backbone", choices=BACKBONE_FAMILIES, required=True)
    parser.add_argument("--target", choices=("dielectric", "elastic"), required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
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
        report = run_benchmark(arguments)
    except Exception as caught:
        error = caught
        report = {
            "schema_version": 1,
            "status": "failed",
            "backbone": arguments.backbone,
            "training_unit": f"jarvis_tensor__{arguments.target}",
            "error_type": type(caught).__name__,
            "error": str(caught),
            "execution": execution_metadata(),
        }
    write_smoke_report(arguments.summary, report)
    write_single_case_junit(
        arguments.junit,
        suite_name=f"benchmark_{arguments.backbone}_{arguments.target}",
        seconds=time.perf_counter() - started,
        error=error,
    )
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    if error is not None:
        raise error


if __name__ == "__main__":
    main()
