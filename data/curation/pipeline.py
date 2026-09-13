from __future__ import annotations

from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from functools import partial
import hashlib
import json
import math
from pathlib import Path
import tempfile
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import spglib

from .physics import AuditThresholds, audit_record
from .records import AuditResult, PROPERTY_SUBTYPES
from .sources import iter_all_records, sha256_file


SCHEMA_VERSION = 1
LABEL_AGREEMENT_RELATIVE_TOLERANCE = 1.0e-3


def _finite_json(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Mapping):
        return {str(key): _finite_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite_json(item) for item in value]
    return value


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with open(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _modified_z(values: np.ndarray) -> np.ndarray:
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    if mad <= 1.0e-15:
        return np.where(np.abs(values - median) <= 1.0e-15, 0.0, np.inf)
    return 0.6744897501960817 * (values - median) / mad


def _outlier_features(result: AuditResult) -> dict[str, float]:
    metrics = result.metrics
    output = {
        "log1p_tensor_frobenius": math.log1p(float(metrics["tensor_frobenius"])),
        "log1p_tensor_max_abs": math.log1p(float(metrics["tensor_max_abs"])),
    }
    if result.record.subtype.startswith("dielectric_"):
        output["log1p_abs_trace_mean"] = math.log1p(abs(float(metrics["trace_mean"])))
        eigen_min = float(metrics["eigenvalue_min"])
        eigen_max = float(metrics["eigenvalue_max"])
        output["log_condition"] = math.log(max(eigen_max / max(eigen_min, 1.0e-12), 1.0))
    else:
        output["log1p_kelvin_max"] = math.log1p(
            abs(float(metrics["kelvin_eigenvalue_max_gpa"]))
        )
        output["log_condition"] = math.log(max(float(metrics["kelvin_condition"]), 1.0))
    return output


def assign_outliers(results: Sequence[AuditResult], threshold: float) -> None:
    by_subtype: dict[str, list[AuditResult]] = defaultdict(list)
    for result in results:
        if result.physical_valid:
            by_subtype[result.record.subtype].append(result)
    for subtype_results in by_subtype.values():
        features = [_outlier_features(result) for result in subtype_results]
        for name in sorted(features[0]) if features else ():
            values = np.asarray([row[name] for row in features], dtype=np.float64)
            scores = _modified_z(values)
            for result, score in zip(subtype_results, scores):
                result.outlier_scores[name] = float(score)
                if abs(float(score)) > threshold:
                    result.outlier = True
        for result in subtype_results:
            if result.outlier:
                result.warnings.append("robust_statistical_outlier")


def resolve_duplicates(results: Sequence[AuditResult]) -> None:
    groups: dict[tuple[str, str], list[AuditResult]] = defaultdict(list)
    for result in results:
        if result.physical_valid and result.structure_fingerprint is not None:
            groups[(result.record.subtype, result.structure_fingerprint)].append(result)
    for (subtype, fingerprint), members in groups.items():
        group_id = hashlib.sha256(f"{subtype}:{fingerprint}".encode()).hexdigest()[:20]
        members.sort(key=lambda item: item.record.record_id)
        provenance = [
            {
                "record_id": member.record.record_id,
                "source_dataset": member.record.source_dataset,
                "source_id": member.record.source_id,
            }
            for member in members
        ]
        for member in members:
            member.duplicate_group = group_id
            member.duplicate_members = provenance
        if len(members) == 1:
            members[0].recommended = not members[0].outlier
            continue
        reference = members[0].canonical_tensor
        assert reference is not None
        residuals = [
            float(np.linalg.norm(member.canonical_tensor - reference) / max(np.linalg.norm(reference), 1.0))
            for member in members[1:]
            if member.canonical_tensor is not None
        ]
        if residuals and max(residuals) <= LABEL_AGREEMENT_RELATIVE_TOLERANCE:
            members[0].duplicate_status = "representative"
            members[0].recommended = not members[0].outlier
            for member in members[1:]:
                member.duplicate_status = "collapsed_agreeing"
                member.recommended = False
        else:
            for member in members:
                member.duplicate_status = "conflicting_labels"
                member.warnings.append("duplicate_label_conflict")
                member.recommended = False


def deterministic_split(group_id: str) -> str:
    value = int(hashlib.sha256(group_id.encode()).hexdigest()[:16], 16) / float(16**16)
    if value < 0.8:
        return "train"
    if value < 0.9:
        return "validation"
    return "test"


def _record_json(result: AuditResult, *, tensor: np.ndarray) -> dict[str, Any]:
    record = result.record
    group = result.duplicate_group or result.structure_fingerprint or record.record_id
    return _finite_json(
        {
            "schema_version": SCHEMA_VERSION,
            "record_id": record.record_id,
            "property_subtype": record.subtype,
            "unit": record.unit,
            "lattice_angstrom": record.lattice,
            "fractional_coordinates": record.fractional_positions,
            "atomic_numbers": record.atomic_numbers,
            "tensor": tensor,
            "point_group": result.point_group,
            "space_group": result.space_group,
            "structure_fingerprint": result.structure_fingerprint,
            "duplicate_group": result.duplicate_group,
            "duplicate_status": result.duplicate_status,
            "duplicate_members": result.duplicate_members,
            "split": deterministic_split(group),
            "provenance": {
                "source_dataset": record.source_dataset,
                "source_id": record.source_id,
                **dict(record.provenance),
            },
            "quality": {
                "warnings": result.warnings,
                "metrics": result.metrics,
                "outlier": result.outlier,
                "outlier_scores": result.outlier_scores,
            },
        }
    )


def _audit_json(result: AuditResult) -> dict[str, Any]:
    return _finite_json(
        {
            "schema_version": SCHEMA_VERSION,
            "record_id": result.record.record_id,
            "source_dataset": result.record.source_dataset,
            "source_id": result.record.source_id,
            "property_subtype": result.record.subtype,
            "physical_valid": result.physical_valid,
            "recommended": result.recommended,
            "reasons": result.reasons,
            "warnings": result.warnings,
            "metrics": result.metrics,
            "outlier": result.outlier,
            "outlier_scores": result.outlier_scores,
            "structure_fingerprint": result.structure_fingerprint,
            "duplicate_group": result.duplicate_group,
            "duplicate_status": result.duplicate_status,
            "duplicate_members": result.duplicate_members,
        }
    )


def _summary(results: Sequence[AuditResult]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for subtype in sorted(PROPERTY_SUBTYPES):
        rows = [result for result in results if result.record.subtype == subtype]
        valid = [result for result in rows if result.physical_valid]
        recommended = [result for result in rows if result.recommended]
        reason_counts = Counter(reason for result in rows for reason in result.reasons)
        warning_counts = Counter(warning for result in rows for warning in result.warnings)
        pg_counts = {
            scope: dict(sorted(Counter(result.point_group for result in selected).items()))
            for scope, selected in (("physical_valid", valid), ("recommended", recommended))
        }
        descriptors: dict[str, Any] = {}
        for key in (
            "tensor_frobenius",
            "tensor_max_abs",
            "eigenvalue_min",
            "eigenvalue_max",
            "trace_mean",
            "kelvin_eigenvalue_min_gpa",
            "kelvin_eigenvalue_max_gpa",
            "kelvin_condition",
        ):
            values = np.asarray(
                [float(result.metrics[key]) for result in recommended if key in result.metrics],
                dtype=np.float64,
            )
            values = values[np.isfinite(values)]
            if values.size:
                descriptors[key] = {
                    "min": float(values.min()),
                    "q01": float(np.quantile(values, 0.01)),
                    "median": float(np.median(values)),
                    "mean": float(values.mean()),
                    "q99": float(np.quantile(values, 0.99)),
                    "max": float(values.max()),
                    "std": float(values.std()),
                }
        counts = np.asarray(list(pg_counts["recommended"].values()), dtype=np.float64)
        all_pg = 32
        padded = np.pad(counts, (0, max(0, all_pg - len(counts))))[:all_pg]
        probabilities = padded / padded.sum() if padded.sum() else padded
        positive = probabilities[probabilities > 0]
        entropy = -float(np.sum(positive * np.log(positive))) if positive.size else 0.0
        output[subtype] = _finite_json(
            {
                "input_records": len(rows),
                "physical_valid_records": len(valid),
                "outlier_records": sum(result.outlier for result in valid),
                "recommended_records": len(recommended),
                "source_counts": dict(sorted(Counter(r.record.source_dataset for r in rows).items())),
                "reason_counts": dict(sorted(reason_counts.items())),
                "warning_counts": dict(sorted(warning_counts.items())),
                "duplicate_status_counts": dict(
                    sorted(Counter(result.duplicate_status for result in valid).items())
                ),
                "point_group_counts": pg_counts,
                "point_group_metrics_recommended": {
                    "covered": len(pg_counts["recommended"]),
                    "zero": 32 - len(pg_counts["recommended"]),
                    "largest_share": float(padded.max() / padded.sum()) if padded.sum() else None,
                    "coefficient_of_variation_32": float(padded.std() / padded.mean())
                    if padded.mean()
                    else None,
                    "normalized_entropy_32": entropy / math.log(32) if positive.size else None,
                    "effective_point_groups": math.exp(entropy) if positive.size else 0.0,
                },
                "descriptive_statistics_recommended": descriptors,
            }
        )
    return output


def _render_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Curated dielectric and elastic tensor datasets",
        "",
        "Physical validity and robust statistical outliers are reported separately. Recommended",
        "sets contain physical-valid, non-outlier representatives of agreeing duplicate groups and",
        "exclude unresolved duplicate-label conflicts.",
        "",
        "| Subtype | Input | Physical valid | Outliers | Recommended | PG coverage | Largest PG share | CV (32 PGs) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for subtype, row in report["subtypes"].items():
        pg = row["point_group_metrics_recommended"]
        share = "—" if pg["largest_share"] is None else f"{100 * pg['largest_share']:.1f}%"
        cv = "—" if pg["coefficient_of_variation_32"] is None else f"{pg['coefficient_of_variation_32']:.3f}"
        lines.append(
            f"| `{subtype}` | {row['input_records']} | {row['physical_valid_records']} | "
            f"{row['outlier_records']} | {row['recommended_records']} | {pg['covered']}/32 | "
            f"{share} | {cv} |"
        )
    lines.extend(["", "## Exclusion reasons", ""])
    for subtype, row in report["subtypes"].items():
        reasons = row["reason_counts"] or {"none": 0}
        lines.append(f"- `{subtype}`: " + ", ".join(f"{key}={value}" for key, value in reasons.items()))
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "Point-group coverage, largest share, CV over all 32 crystallographic point groups,",
            "normalized entropy, effective point-group count, full frequencies, duplicate outcomes,",
            "and property quantiles are available in the machine-readable JSON report. A high CV or",
            "large dominant-group share means the merged data remains unsuitable for claiming balanced",
            "per-point-group evaluation without stratified reporting or a balanced-training ablation.",
            "",
        ]
    )
    return "\n".join(lines)


def run_pipeline(
    *,
    output_dir: Path,
    manifest_path: Path,
    report_json_path: Path,
    report_markdown_path: Path,
    workers: int = 1,
    thresholds: AuditThresholds = AuditThresholds(),
    records: Iterable[Any] | None = None,
) -> dict[str, Any]:
    normalized = list(iter_all_records() if records is None else records)
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            results = list(
                executor.map(partial(audit_record, thresholds=thresholds), normalized, chunksize=32)
            )
    else:
        results = [audit_record(record, thresholds) for record in normalized]
    assign_outliers(results, thresholds.outlier_modified_z)
    resolve_duplicates(results)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts: dict[str, Any] = {}
    audit_path = output_dir / "audit.jsonl"
    _atomic_write(audit_path, "".join(json.dumps(_audit_json(r), sort_keys=True) + "\n" for r in results))
    artifacts["audit"] = {
        "path": audit_path.as_posix(),
        "records": len(results),
        "size_bytes": audit_path.stat().st_size,
        "sha256": sha256_file(audit_path),
    }
    for scope in ("physical_valid", "recommended"):
        for subtype in sorted(PROPERTY_SUBTYPES):
            selected = [
                result
                for result in results
                if result.record.subtype == subtype
                and (result.physical_valid if scope == "physical_valid" else result.recommended)
            ]
            path = output_dir / scope / f"{subtype}.jsonl"
            _atomic_write(
                path,
                "".join(
                    json.dumps(_record_json(result, tensor=result.clean_tensor), sort_keys=True) + "\n"
                    for result in selected
                    if result.clean_tensor is not None
                ),
            )
            artifacts[f"{scope}/{subtype}"] = {
                "path": path.as_posix(),
                "records": len(selected),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "split_counts": dict(
                    sorted(
                        Counter(
                            deterministic_split(
                                result.duplicate_group
                                or result.structure_fingerprint
                                or result.record.record_id
                            )
                            for result in selected
                        ).items()
                    )
                ),
            }
    report = {
        "schema_version": SCHEMA_VERSION,
        "spglib_version": spglib.__version__,
        "thresholds": asdict(thresholds),
        "label_agreement_relative_tolerance": LABEL_AGREEMENT_RELATIVE_TOLERANCE,
        "subtypes": _summary(results),
    }
    _atomic_write(report_json_path, json.dumps(report, indent=2, sort_keys=True) + "\n")
    _atomic_write(report_markdown_path, _render_markdown(report))
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "sources": [
            "data/manifests/dtnet_dielectric.json",
            "data/manifests/jarvis_dielectric.json",
            "data/manifests/jarvis_elastic.json",
            "data/manifests/matten_elastic.json",
        ],
        "semantics": {
            "dielectric_electronic": "clamped-ion electronic relative permittivity",
            "dielectric_ionic": "ionic/lattice contribution to relative permittivity",
            "dielectric_total": "static relative permittivity; electronic + ionic",
            "elastic_stiffness": "fourth-rank stiffness tensor in GPa",
        },
        "thresholds": asdict(thresholds),
        "label_agreement_relative_tolerance": LABEL_AGREEMENT_RELATIVE_TOLERANCE,
        "artifacts": artifacts,
        "report": {
            "json": report_json_path.as_posix(),
            "markdown": report_markdown_path.as_posix(),
            "json_sha256": sha256_file(report_json_path),
        },
    }
    _atomic_write(manifest_path, json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return report
