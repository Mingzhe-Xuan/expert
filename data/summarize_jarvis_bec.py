"""Validate and summarize an extracted JARVIS-DFPT BEC JSONL dataset."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--errors", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    ids: set[str] = set()
    residuals: list[float] = []
    atom_counts: list[int] = []
    duplicate_ids: list[str] = []
    invalid_records: list[dict[str, object]] = []
    with args.input.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                record = json.loads(line)
                sample_id = str(record["sample_id"])
                elements = record["elements"]
                coords = record["fractional_coordinates"]
                bec = record["born_effective_charge_e"]
                if sample_id in ids:
                    duplicate_ids.append(sample_id)
                ids.add(sample_id)
                if not (len(elements) == len(coords) == len(bec)):
                    raise ValueError("atom/coordinate/BEC length mismatch")
                if any(
                    len(tensor) != 3
                    or any(len(row) != 3 for row in tensor)
                    for tensor in bec
                ):
                    raise ValueError("BEC is not N x 3 x 3")
                if not all(
                    math.isfinite(float(value))
                    for tensor in bec
                    for row in tensor
                    for value in row
                ):
                    raise ValueError("non-finite BEC")
                atom_counts.append(len(elements))
                residuals.append(
                    float(record["quality"]["acoustic_sum_rule_frobenius_e"])
                )
            except Exception as exc:
                invalid_records.append(
                    {"line": line_number, "error": f"{type(exc).__name__}: {exc}"}
                )

    upstream_errors = 0
    if args.errors and args.errors.exists():
        with args.errors.open("r", encoding="utf-8") as handle:
            upstream_errors = sum(1 for line in handle if line.strip())
    sorted_residuals = sorted(residuals)

    def percentile(fraction: float) -> float | None:
        if not sorted_residuals:
            return None
        index = round(fraction * (len(sorted_residuals) - 1))
        return sorted_residuals[index]

    summary = {
        "records": len(atom_counts),
        "unique_sample_ids": len(ids),
        "duplicate_sample_ids": sorted(set(duplicate_ids)),
        "invalid_records": invalid_records,
        "upstream_extraction_errors": upstream_errors,
        "natoms": {
            "min": min(atom_counts) if atom_counts else None,
            "max": max(atom_counts) if atom_counts else None,
            "total": sum(atom_counts),
        },
        "acoustic_sum_rule_frobenius_e": {
            "median": percentile(0.5),
            "p95": percentile(0.95),
            "max": max(residuals) if residuals else None,
        },
        "valid": not duplicate_ids and not invalid_records and upstream_errors == 0,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return int(not summary["valid"])


if __name__ == "__main__":
    raise SystemExit(main())
