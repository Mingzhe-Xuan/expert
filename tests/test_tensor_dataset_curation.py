from __future__ import annotations

import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

from data.curation.physics import AuditThresholds, audit_record
from data.curation.plot_point_groups import POINT_GROUPS, render_point_group_frequency_svg
from data.curation.reduce_point_groups import (
    reduce_recommended_datasets,
    select_available_point_groups,
)
from data.curation.pipeline import run_pipeline
from data.curation.records import NormalizedRecord
from data.curation.sources import jarvis_voigt_to_standard, voigt_to_elastic


def _record(
    record_id: str,
    subtype: str,
    tensor: np.ndarray,
    *,
    source: str = "fixture",
) -> NormalizedRecord:
    return NormalizedRecord(
        record_id=record_id,
        source_dataset=source,
        source_id=record_id,
        subtype=subtype,
        lattice=np.eye(3) * 4.0,
        fractional_positions=np.asarray([[0.0, 0.0, 0.0]]),
        atomic_numbers=np.asarray([14], dtype=np.int32),
        tensor=tensor,
        unit="dimensionless" if subtype.startswith("dielectric") else "GPa",
    )


def _cubic_stiffness(c11: float = 200.0, c12: float = 100.0, c44: float = 50.0) -> np.ndarray:
    value = np.zeros((6, 6))
    value[:3, :3] = c12
    np.fill_diagonal(value[:3, :3], c11)
    value[3, 3] = value[4, 4] = value[5, 5] = c44
    return voigt_to_elastic(value)


def test_dielectric_physical_constraints_and_point_group_projection() -> None:
    electronic = audit_record(_record("good", "dielectric_electronic", np.eye(3) * 2.0))
    ionic = audit_record(_record("ionic", "dielectric_ionic", np.zeros((3, 3))))
    below_vacuum = audit_record(
        _record("below", "dielectric_electronic", np.eye(3) * 0.5)
    )
    wrong_cubic_shape = audit_record(
        _record("anisotropic", "dielectric_total", np.diag([2.0, 3.0, 4.0]))
    )
    antisymmetric = np.eye(3) * 2.0
    antisymmetric[0, 1] = 1.0
    intrinsic = audit_record(_record("antisym", "dielectric_electronic", antisymmetric))

    assert electronic.physical_valid and ionic.physical_valid
    assert electronic.point_group == "m-3m"
    assert np.allclose(electronic.clean_tensor, np.eye(3) * 2.0)
    assert "dielectric_not_positive_semidefinite" in below_vacuum.reasons
    assert "point_group_residual" in wrong_cubic_shape.reasons
    assert "intrinsic_symmetry_residual" in intrinsic.reasons


def test_elastic_stability_intrinsic_symmetry_and_voigt_mapping() -> None:
    stable = audit_record(_record("stable", "elastic_stiffness", _cubic_stiffness()))
    unstable = audit_record(
        _record("unstable", "elastic_stiffness", _cubic_stiffness(c44=-1.0))
    )
    broken = _cubic_stiffness()
    broken[0, 1, 2, 2] += 20.0
    nonsymmetric = audit_record(_record("broken", "elastic_stiffness", broken))

    assert stable.physical_valid
    assert stable.metrics["kelvin_eigenvalue_min_gpa"] > 0
    assert "elastic_not_positive_definite" in unstable.reasons
    assert "intrinsic_symmetry_residual" in nonsymmetric.reasons
    assert stable.clean_tensor[1, 2, 1, 2] == 50.0
    assert stable.clean_tensor[2, 1, 2, 1] == 50.0


def test_jarvis_vasp_voigt_order_is_rearranged_before_cartesian_expansion() -> None:
    source = np.diag([1.0, 2.0, 3.0, 40.0, 50.0, 60.0])
    standard = jarvis_voigt_to_standard(source)
    tensor = voigt_to_elastic(standard)

    assert tensor[0, 1, 0, 1] == 40.0
    assert tensor[1, 2, 1, 2] == 50.0
    assert tensor[0, 2, 0, 2] == 60.0


def test_pipeline_collapses_agreeing_duplicates_and_excludes_conflicts(tmp_path: Path) -> None:
    records = [
        _record("a", "dielectric_electronic", np.eye(3) * 2.0, source="one"),
        _record("b", "dielectric_electronic", np.eye(3) * 2.0, source="two"),
        _record("c", "dielectric_total", np.eye(3) * 4.0, source="one"),
        _record("d", "dielectric_total", np.eye(3) * 5.0, source="two"),
        _record("e", "dielectric_ionic", np.zeros((3, 3))),
        _record("f", "elastic_stiffness", _cubic_stiffness()),
    ]
    output = tmp_path / "processed"
    manifest = tmp_path / "manifest.json"
    report_json = tmp_path / "report.json"
    report_md = tmp_path / "report.md"
    thresholds = AuditThresholds(outlier_modified_z=1.0e9)
    report = run_pipeline(
        output_dir=output,
        manifest_path=manifest,
        report_json_path=report_json,
        report_markdown_path=report_md,
        records=records,
        thresholds=thresholds,
    )

    electronic = report["subtypes"]["dielectric_electronic"]
    total = report["subtypes"]["dielectric_total"]
    assert electronic["physical_valid_records"] == 2
    assert electronic["recommended_records"] == 1
    assert electronic["duplicate_status_counts"] == {
        "collapsed_agreeing": 1,
        "representative": 1,
    }
    assert total["physical_valid_records"] == 2
    assert total["recommended_records"] == 0
    assert total["duplicate_status_counts"] == {"conflicting_labels": 2}

    rows = [
        json.loads(line)
        for line in (output / "recommended" / "dielectric_electronic.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert len(rows) == 1
    assert rows[0]["duplicate_status"] == "representative"
    assert {member["record_id"] for member in rows[0]["duplicate_members"]} == {"a", "b"}
    assert rows[0]["split"] in {"train", "validation", "test"}
    saved_manifest = json.loads(manifest.read_text(encoding="utf-8"))
    assert saved_manifest["artifacts"]["recommended/dielectric_electronic"]["records"] == 1
    assert "CV (32 PGs)" in report_md.read_text(encoding="utf-8")


def _point_group_plot_report() -> dict[str, object]:
    counts = {point_group: index + 1 for index, point_group in enumerate(POINT_GROUPS)}
    subtype = {
        "recommended_records": sum(counts.values()),
        "point_group_counts": {"recommended": counts},
    }
    return {
        "subtypes": {
            name: subtype
            for name in (
                "dielectric_electronic",
                "dielectric_ionic",
                "dielectric_total",
                "elastic_stiffness",
            )
        }
    }


def test_point_group_frequency_svg_is_deterministic_and_complete() -> None:
    report = _point_group_plot_report()
    first = render_point_group_frequency_svg(report)
    second = render_point_group_frequency_svg(report)

    assert first == second
    assert first.count("<rect ") == 1 + 4 * 32
    assert first.count("Frequency (records)") == 4
    assert "Cubic" in first and "Triclinic" in first
    assert "m-3m — 32 (6.1%)" in first
    ET.fromstring(first)


def test_point_group_frequency_svg_rejects_inconsistent_counts() -> None:
    report = _point_group_plot_report()
    electronic = report["subtypes"]["dielectric_electronic"]
    electronic["point_group_counts"]["recommended"].pop("1")

    with pytest.raises(ValueError, match="missing=\\['1'\\]"):
        render_point_group_frequency_svg(report)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _reduction_fixture(root: Path) -> tuple[Path, Path]:
    counts = {point_group: 3 for point_group in POINT_GROUPS}
    counts["1"] = 6
    counts["-1"] = 5
    counts[POINT_GROUPS[-1]] = 2
    assert sum(counts.values()) == 100
    report = {
        "subtypes": {
            subtype: {
                "recommended_records": 100,
                "point_group_counts": {"recommended": counts},
            }
            for subtype in (
                "dielectric_electronic",
                "dielectric_ionic",
                "dielectric_total",
                "elastic_stiffness",
            )
        }
    }
    report_path = root / "docs" / "analysis" / "report.json"
    report_path.parent.mkdir(parents=True)
    report_path.write_text(json.dumps(report, sort_keys=True), encoding="utf-8")

    artifacts = {}
    for subtype in report["subtypes"]:
        path = root / "data" / "recommended" / f"{subtype}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = []
        index = 0
        for point_group, count in counts.items():
            for _ in range(count):
                split = "train" if index < 80 else "validation" if index < 90 else "test"
                rows.append(
                    {
                        "record_id": f"{subtype}:{index}",
                        "property_subtype": subtype,
                        "point_group": point_group,
                        "split": split,
                        "tensor": [index],
                    }
                )
                index += 1
        path.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
            encoding="utf-8",
        )
        artifacts[f"recommended/{subtype}"] = {
            "path": path.relative_to(root).as_posix(),
            "records": 100,
            "size_bytes": path.stat().st_size,
            "sha256": _sha256(path),
            "split_counts": {"train": 80, "validation": 10, "test": 10},
        }
    manifest = {
        "report": {"json_sha256": _sha256(report_path)},
        "artifacts": artifacts,
    }
    manifest_path = root / "data" / "manifests" / "curated.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    return report_path, manifest_path


def test_reducer_uses_strict_frequency_threshold_and_annotates_records(tmp_path: Path) -> None:
    report_path, manifest_path = _reduction_fixture(tmp_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert select_available_point_groups(report, "dielectric_total") == ("1",)

    output_dir = tmp_path / "reduced"
    output_manifest = tmp_path / "reduced-manifest.json"
    manifest = reduce_recommended_datasets(
        report_path=report_path,
        curated_manifest_path=manifest_path,
        output_dir=output_dir,
        output_manifest_path=output_manifest,
        repository_root=tmp_path,
    )
    first_manifest = output_manifest.read_bytes()
    first_outputs = {path.name: path.read_bytes() for path in output_dir.iterdir()}

    repeated = reduce_recommended_datasets(
        report_path=report_path,
        curated_manifest_path=manifest_path,
        output_dir=output_dir,
        output_manifest_path=output_manifest,
        repository_root=tmp_path,
    )
    assert manifest == repeated
    assert output_manifest.read_bytes() == first_manifest
    assert {path.name: path.read_bytes() for path in output_dir.iterdir()} == first_outputs

    for subtype, artifact in manifest["artifacts"].items():
        assert artifact["available_point_groups"] == ["1"]
        assert artifact["records"] == 6
        rows = [
            json.loads(line)
            for line in (output_dir / f"{subtype}.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        assert len(rows) == 6
        assert all(row["point_group"] == "1" for row in rows)
        assert all(row["reduction"]["property_available_point_groups"] == ["1"] for row in rows)
        assert all(row["reduction"]["point_group_frequency"] == 0.06 for row in rows)


def test_reducer_fails_closed_on_tampered_recommended_input(tmp_path: Path) -> None:
    report_path, manifest_path = _reduction_fixture(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifact = manifest["artifacts"]["recommended/dielectric_electronic"]
    path = tmp_path / artifact["path"]
    path.write_text("{not-json}\n", encoding="utf-8")
    artifact["size_bytes"] = path.stat().st_size
    artifact["sha256"] = _sha256(path)
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    output_dir = tmp_path / "reduced"
    output_dir.mkdir()
    existing = output_dir / "dielectric_electronic.jsonl"
    existing.write_text("previous-valid-output\n", encoding="utf-8")

    with pytest.raises(ValueError, match="invalid JSON"):
        reduce_recommended_datasets(
            report_path=report_path,
            curated_manifest_path=manifest_path,
            output_dir=output_dir,
            output_manifest_path=tmp_path / "reduced-manifest.json",
            repository_root=tmp_path,
        )
    assert existing.read_text(encoding="utf-8") == "previous-valid-output\n"
