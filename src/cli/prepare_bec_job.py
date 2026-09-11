from __future__ import annotations

import argparse
from pathlib import Path

from data.prepare_jarvis_bec import load_dfpt_index, prepare_dataset
from data.summarize_jarvis_bec import summarize_dataset

from .reporting import EvidenceFailure, run_recorded_case


def prepare_and_validate(
    *,
    index_zip: Path,
    dataset_output: Path,
    archive_dir: Path,
    errors: Path,
    dataset_summary: Path,
    workers: int,
) -> dict[str, object]:
    """Run resumable BEC extraction and reject any incomplete/invalid result."""

    if workers < 1:
        raise ValueError("workers must be at least 1")
    sources = load_dfpt_index(index_zip)
    successes, failures = prepare_dataset(
        sources=sources,
        output_path=dataset_output,
        archive_dir=archive_dir,
        error_path=errors,
        workers=workers,
        keep_archives=False,
    )
    dataset_audit = summarize_dataset(dataset_output, errors, dataset_summary)
    evidence: dict[str, object] = {
        "status": "passed",
        "indexed_archives": len(sources),
        "new_successes": successes,
        "new_failures": failures,
        "dataset_output": str(dataset_output),
        "dataset_summary": str(dataset_summary),
        "dataset_audit": dataset_audit,
    }
    if failures or not bool(dataset_audit["valid"]):
        raise EvidenceFailure("BEC extraction or validation reported failures", evidence)
    if int(dataset_audit["records"]) != len(sources):
        raise EvidenceFailure("BEC dataset does not cover every indexed archive", evidence)
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare and validate the complete JARVIS-DFPT BEC dataset")
    parser.add_argument("--index-zip", type=Path, required=True)
    parser.add_argument("--dataset-output", type=Path, required=True)
    parser.add_argument("--archive-dir", type=Path, required=True)
    parser.add_argument("--errors", type=Path, required=True)
    parser.add_argument("--dataset-summary", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--junit", type=Path, required=True)
    arguments = parser.parse_args()
    run_recorded_case(
        lambda: prepare_and_validate(
            index_zip=arguments.index_zip,
            dataset_output=arguments.dataset_output,
            archive_dir=arguments.archive_dir,
            errors=arguments.errors,
            dataset_summary=arguments.dataset_summary,
            workers=arguments.workers,
        ),
        output=arguments.summary,
        junit=arguments.junit,
        suite_name="prepare_jarvis_dfpt_bec",
    )


if __name__ == "__main__":
    main()
