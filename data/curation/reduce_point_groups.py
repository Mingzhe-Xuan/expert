"""Build per-property reduced datasets from frequent recommended point groups."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping, Sequence

from .plot_point_groups import POINT_GROUPS, SUBTYPE_STYLES


SCHEMA_VERSION = 1
DEFAULT_FREQUENCY_THRESHOLD = 0.05
SUBTYPES = tuple(style[0] for style in SUBTYPE_STYLES)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_text_writer(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    return descriptor, Path(temporary_name)


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    descriptor, temporary = _atomic_text_writer(path)
    try:
        with open(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, ensure_ascii=False)
            stream.write("\n")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _validated_point_group_counts(
    report: Mapping[str, Any], subtype: str
) -> tuple[dict[str, int], int]:
    try:
        section = report["subtypes"][subtype]
        counts = section["point_group_counts"]["recommended"]
        total = section["recommended_records"]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"missing recommended point-group data for {subtype}") from exc
    if not isinstance(counts, Mapping):
        raise ValueError(f"point-group counts for {subtype} must be an object")
    missing = sorted(set(POINT_GROUPS) - set(counts))
    extra = sorted(set(counts) - set(POINT_GROUPS))
    if missing or extra:
        raise ValueError(f"invalid point groups for {subtype}: missing={missing}, extra={extra}")

    normalized: dict[str, int] = {}
    for point_group in POINT_GROUPS:
        value = counts[point_group]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"count for {subtype}/{point_group} must be a non-negative integer")
        normalized[point_group] = value
    if isinstance(total, bool) or not isinstance(total, int) or total <= 0:
        raise ValueError(f"recommended_records for {subtype} must be a positive integer")
    if sum(normalized.values()) != total:
        raise ValueError(
            f"point-group counts for {subtype} sum to {sum(normalized.values())}, expected {total}"
        )
    return normalized, total


def select_available_point_groups(
    report: Mapping[str, Any],
    subtype: str,
    *,
    threshold: float = DEFAULT_FREQUENCY_THRESHOLD,
) -> tuple[str, ...]:
    """Select groups whose within-property recommended frequency is strictly above threshold."""

    if not math.isfinite(threshold) or not 0.0 <= threshold < 1.0:
        raise ValueError("frequency threshold must be finite and in [0, 1)")
    counts, total = _validated_point_group_counts(report, subtype)
    return tuple(
        point_group
        for point_group in POINT_GROUPS
        if counts[point_group] / total > threshold
    )


def _validated_artifact(
    curated_manifest: Mapping[str, Any], subtype: str, repository_root: Path
) -> tuple[Path, Mapping[str, Any]]:
    key = f"recommended/{subtype}"
    try:
        artifact = curated_manifest["artifacts"][key]
        relative_path = artifact["path"]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"curated manifest is missing {key}") from exc
    if not isinstance(relative_path, str):
        raise ValueError(f"artifact path for {key} must be a string")
    path = (repository_root / relative_path).resolve()
    try:
        path.relative_to(repository_root.resolve())
    except ValueError as exc:
        raise ValueError(f"artifact path for {key} escapes the repository") from exc
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.stat().st_size != artifact.get("size_bytes"):
        raise ValueError(f"size mismatch for {key}")
    if _sha256_file(path) != artifact.get("sha256"):
        raise ValueError(f"SHA-256 mismatch for {key}")
    return path, artifact


def _reduction_metadata(
    *,
    available: Sequence[str],
    point_group: str,
    count: int,
    total: int,
    threshold: float,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "source_scope": "recommended",
        "criterion": "within_property_point_group_frequency_strictly_greater_than",
        "frequency_threshold": threshold,
        "property_available_point_groups": list(available),
        "point_group_count": count,
        "point_group_frequency": count / total,
        "property_recommended_records": total,
    }


def _reduce_one(
    *,
    input_path: Path,
    output_path: Path,
    subtype: str,
    counts: Mapping[str, int],
    total: int,
    available: Sequence[str],
    threshold: float,
    expected_splits: Mapping[str, int],
) -> dict[str, Any]:
    descriptor, temporary = _atomic_text_writer(output_path)
    observed_counts: Counter[str] = Counter()
    observed_splits: Counter[str] = Counter()
    written_counts: Counter[str] = Counter()
    written_splits: Counter[str] = Counter()
    records = 0
    written = 0
    available_set = set(available)
    try:
        with input_path.open("r", encoding="utf-8") as source, open(
            descriptor, "w", encoding="utf-8", newline="\n"
        ) as destination:
            for line_number, line in enumerate(source, start=1):
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSON at {input_path}:{line_number}") from exc
                if not isinstance(row, dict):
                    raise ValueError(f"record at {input_path}:{line_number} must be an object")
                if row.get("property_subtype") != subtype:
                    raise ValueError(f"subtype mismatch at {input_path}:{line_number}")
                point_group = row.get("point_group")
                if point_group not in counts:
                    raise ValueError(f"unknown point group at {input_path}:{line_number}")
                split = row.get("split")
                if split not in {"train", "validation", "test"}:
                    raise ValueError(f"invalid split at {input_path}:{line_number}")
                if "reduction" in row:
                    raise ValueError(
                        f"record already has reduction metadata at {input_path}:{line_number}"
                    )
                records += 1
                observed_counts[point_group] += 1
                observed_splits[split] += 1
                if point_group not in available_set:
                    continue
                row["reduction"] = _reduction_metadata(
                    available=available,
                    point_group=point_group,
                    count=counts[point_group],
                    total=total,
                    threshold=threshold,
                )
                destination.write(
                    json.dumps(row, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
                )
                destination.write("\n")
                written += 1
                written_counts[point_group] += 1
                written_splits[split] += 1

        if records != total:
            raise ValueError(f"row count for {subtype} is {records}, expected {total}")
        if dict(observed_counts) != {group: count for group, count in counts.items() if count}:
            raise ValueError(f"point-group counts in {input_path} disagree with the report")
        if dict(observed_splits) != dict(expected_splits):
            raise ValueError(f"split counts in {input_path} disagree with the curated manifest")
        expected_written = sum(counts[group] for group in available)
        if written != expected_written:
            raise ValueError(
                f"reduced row count for {subtype} is {written}, expected {expected_written}"
            )
        temporary.replace(output_path)
    finally:
        temporary.unlink(missing_ok=True)

    return {
        "path": output_path.as_posix(),
        "records": written,
        "size_bytes": output_path.stat().st_size,
        "sha256": _sha256_file(output_path),
        "point_group_counts": dict(written_counts),
        "split_counts": dict(sorted(written_splits.items())),
        "available_point_groups": list(available),
        "retained_fraction": written / total,
    }


def reduce_recommended_datasets(
    *,
    report_path: Path,
    curated_manifest_path: Path,
    output_dir: Path,
    output_manifest_path: Path,
    repository_root: Path,
    threshold: float = DEFAULT_FREQUENCY_THRESHOLD,
) -> dict[str, Any]:
    """Validate and reduce every supported recommended tensor dataset."""

    report = json.loads(report_path.read_text(encoding="utf-8"))
    curated_manifest = json.loads(curated_manifest_path.read_text(encoding="utf-8"))
    report_hash = _sha256_file(report_path)
    try:
        expected_report_hash = curated_manifest["report"]["json_sha256"]
    except (KeyError, TypeError) as exc:
        raise ValueError("curated manifest is missing its report hash") from exc
    if report_hash != expected_report_hash:
        raise ValueError("curation report SHA-256 does not match the curated manifest")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary_dir = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent)
    )
    artifacts: dict[str, Any] = {}
    try:
        for subtype in SUBTYPES:
            counts, total = _validated_point_group_counts(report, subtype)
            available = select_available_point_groups(report, subtype, threshold=threshold)
            input_path, input_artifact = _validated_artifact(
                curated_manifest, subtype, repository_root
            )
            temporary_output = temporary_dir / f"{subtype}.jsonl"
            artifact = _reduce_one(
                input_path=input_path,
                output_path=temporary_output,
                subtype=subtype,
                counts=counts,
                total=total,
                available=available,
                threshold=threshold,
                expected_splits=input_artifact["split_counts"],
            )
            final_output = output_dir / temporary_output.name
            try:
                artifact["path"] = (
                    final_output.resolve().relative_to(repository_root.resolve()).as_posix()
                )
            except ValueError:
                artifact["path"] = final_output.resolve().as_posix()
            artifacts[subtype] = artifact

        output_dir.mkdir(parents=True, exist_ok=True)
        for subtype in SUBTYPES:
            (temporary_dir / f"{subtype}.jsonl").replace(
                output_dir / f"{subtype}.jsonl"
            )
    finally:
        shutil.rmtree(temporary_dir, ignore_errors=True)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "criterion": {
            "source_scope": "recommended",
            "frequency_denominator": "recommended_records_within_property_subtype",
            "operator": ">",
            "threshold": threshold,
        },
        "source": {
            "curated_manifest": curated_manifest_path.resolve()
            .relative_to(repository_root.resolve())
            .as_posix(),
            "curated_manifest_sha256": _sha256_file(curated_manifest_path),
            "report": report_path.resolve().relative_to(repository_root.resolve()).as_posix(),
            "report_sha256": report_hash,
        },
        "artifacts": artifacts,
    }
    _atomic_write_json(output_manifest_path, manifest)
    return manifest


def _parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report",
        type=Path,
        default=root / "docs" / "analysis" / "curated_tensor_datasets.json",
    )
    parser.add_argument(
        "--curated-manifest",
        type=Path,
        default=root / "data" / "manifests" / "curated_tensors.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "data" / "processed" / "curated_tensors" / "reduced_gt_5pct",
    )
    parser.add_argument(
        "--output-manifest",
        type=Path,
        default=root / "results" / "tensor-curation" / "reduced_gt_5pct.json",
    )
    parser.add_argument("--threshold", type=float, default=DEFAULT_FREQUENCY_THRESHOLD)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = Path(__file__).resolve().parents[2]
    reduce_recommended_datasets(
        report_path=args.report,
        curated_manifest_path=args.curated_manifest,
        output_dir=args.output_dir,
        output_manifest_path=args.output_manifest,
        repository_root=root,
        threshold=args.threshold,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
