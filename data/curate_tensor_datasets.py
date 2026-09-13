from __future__ import annotations

import argparse
from pathlib import Path

from curation.pipeline import run_pipeline


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Physically audit, deduplicate, merge, and summarize tensor datasets."
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "data" / "processed" / "curated_tensors"
    )
    parser.add_argument(
        "--manifest", type=Path, default=ROOT / "data" / "manifests" / "curated_tensors.json"
    )
    parser.add_argument(
        "--report-json",
        type=Path,
        default=ROOT / "docs" / "analysis" / "curated_tensor_datasets.json",
    )
    parser.add_argument(
        "--report-markdown",
        type=Path,
        default=ROOT / "docs" / "analysis" / "curated_tensor_datasets.md",
    )
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    report = run_pipeline(
        output_dir=args.output_dir,
        manifest_path=args.manifest,
        report_json_path=args.report_json,
        report_markdown_path=args.report_markdown,
        workers=args.workers,
    )
    for subtype, row in report["subtypes"].items():
        print(
            f"{subtype}: input={row['input_records']} physical={row['physical_valid_records']} "
            f"recommended={row['recommended_records']}"
        )


if __name__ == "__main__":
    main()
