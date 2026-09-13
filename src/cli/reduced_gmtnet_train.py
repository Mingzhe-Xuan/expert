from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from ..baselines.gmtnet import GMTNetConfig, run_gmtnet_benchmark
from ..data import TrainingUnit, load_training_dataset
from ..training import write_smoke_report
from .reporting import execution_metadata, write_single_case_junit
from .reduced_protocol import point_group_stratified_smoke_ids


def main() -> None:
    parser = argparse.ArgumentParser(description="Train official GMTNet on reduced total dielectric")
    parser.add_argument("--official-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=Path("data/manifests/curated_tensors_reduced_gt_5pct.json"))
    parser.add_argument("--cache", type=Path, required=True)
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
    parser.add_argument("--smoke", action="store_true",
                        help="Use one real sample per retained point group in each split")
    arguments = parser.parse_args()
    started = time.perf_counter()
    error = None
    try:
        dataset = load_training_dataset(
            TrainingUnit("curated_reduced_total", "dielectric"),
            manifest_path=arguments.manifest,
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
            ),
            device=arguments.device,
            split_ids=(point_group_stratified_smoke_ids(dataset) if arguments.smoke else None),
        )
        report["execution"] = execution_metadata()
    except Exception as caught:
        error = caught
        report = {"schema_version": 1, "status": "failed", "error_type": type(caught).__name__,
                  "error": str(caught), "execution": execution_metadata()}
    write_smoke_report(arguments.summary, report)
    write_single_case_junit(arguments.junit, suite_name="reduced_gmtnet",
                            seconds=time.perf_counter() - started, error=error)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    if error is not None:
        raise error


if __name__ == "__main__":
    main()
