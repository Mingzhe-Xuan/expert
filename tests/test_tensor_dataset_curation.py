from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from data.curation.physics import AuditThresholds, audit_record
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
