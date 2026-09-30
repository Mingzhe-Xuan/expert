from __future__ import annotations

import json
from pathlib import Path
import struct
import xml.etree.ElementTree as ET

import pytest

from src.evaluation.training_history import (
    file_sha256,
    load_experiment_history,
    load_current_group_history,
    load_gmtnet_history,
    load_parent_dag_history,
    render_current_group_history,
    render_all_experiment_histories,
    render_routing_comparison_history,
)


def _summary(path: Path) -> Path:
    history = [
        {
            "epoch": epoch,
            "train_loss": 3.0 / epoch,
            "validation_loss": 4.0 / epoch,
            "validation_mae": 5.0 / epoch,
            "validation_fnorm": 20.0 / epoch,
            "learning_rate": 1.0e-3 - (epoch / 200) * 9.9e-4,
        }
        for epoch in range(1, 201)
    ]
    payload = {
        "status": "passed",
        "model": "CGCNN B+A+PGE+R full_pg",
        "routing": "current_point_group_only",
        "best_epoch": 200,
        "best_validation_mae": history[-1]["validation_mae"],
        "history": history,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _gmtnet_summary(path: Path, *, include_validation_fnorm: bool = False) -> Path:
    history = [
        {
            "epoch": epoch,
            "training_loss": 2.5 / epoch,
            "validation_mae": 4.0 / epoch,
            "learning_rate": 1.0e-3 - (epoch / 200) * 9.9e-4,
        }
        for epoch in range(1, 201)
    ]
    if include_validation_fnorm:
        for row in history:
            row["validation_fnorm"] = 12.0 / row["epoch"]
    payload = {
        "status": "passed",
        "model": "GMTNet",
        "best_epoch": 200,
        "best_validation_mae": history[-1]["validation_mae"],
        "history": history,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _parent_summary(path: Path) -> Path:
    history = [
        {
            "epoch": epoch,
            "train_loss": 2.8 / epoch,
            "validation_loss": 3.8 / epoch,
            "validation_mae": 4.8 / epoch,
            "validation_fnorm": 18.0 / epoch,
            "learning_rate": 1.0e-3 - (epoch / 200) * 9.9e-4,
        }
        for epoch in range(1, 201)
    ]
    payload = {
        "status": "passed",
        "model": "CGCNN B+A+PGE+R full_pg PG-parent-DAG all-ancestors",
        "routing": "point_group_parent_dag_all_ancestors",
        "point_group_dag": {"activation": "current_plus_all_transitive_parents"},
        "best_epoch": 200,
        "best_validation_mae": history[-1]["validation_mae"],
        "history": history,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _relative_parent_summary(path: Path) -> Path:
    path = _parent_summary(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.update(
        {
            "model": "CGCNN B+A+PGE+R full_pg relative-position PG parent-DAG path-weighted",
            "routing": "point_group_relative_edge_stick_breaking",
            "parent_detection": {
                "topology": "offline_complete_oriented_point_group_paths"
            },
            "path_fusion": {
                "path_definition": "maximal_current_point_group_to_root",
                "between_path_prior": "node_count_normalized",
                "within_path_weighting": "relative_vector_point_group_edge_stick_breaking",
                "duplicate_destination_reduction": "sum_then_normalize",
            },
        }
    )
    payload.pop("point_group_dag")
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_current_group_history_validates_identity_and_complete_epochs(tmp_path) -> None:
    path = _summary(tmp_path / "summary.json")
    payload = load_current_group_history(path, expected_sha256=file_sha256(path))
    assert len(payload["history"]) == 200
    assert payload["best_epoch"] == 200
    with pytest.raises(ValueError, match="SHA-256"):
        load_current_group_history(path, expected_sha256="0" * 64)
    changed = json.loads(path.read_text(encoding="utf-8"))
    changed["routing"] = "material_parent_dag"
    path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="current-group-only"):
        load_current_group_history(path)


def test_training_history_render_writes_parseable_labeled_svg_and_png(tmp_path) -> None:
    path = _summary(tmp_path / "summary.json")
    payload = load_current_group_history(path)
    gmtnet_path = _gmtnet_summary(
        tmp_path / "gmtnet.json", include_validation_fnorm=True
    )
    gmtnet = load_gmtnet_history(
        gmtnet_path, expected_sha256=file_sha256(gmtnet_path)
    )
    svg = tmp_path / "curve.svg"
    png = tmp_path / "curve.png"
    render_current_group_history(
        payload, gmtnet_summary=gmtnet, svg_path=svg, png_path=png
    )
    ET.parse(svg)
    svg_text = svg.read_text(encoding="utf-8")
    assert "current-pg" in svg_text
    assert "current-pg vs GMTNet" in svg_text
    assert "current-pg train Huber" in svg_text
    assert "current-pg best epoch 200" in svg_text
    assert "GMTNet best epoch 200" in svg_text
    assert "GMTNet validation Fnorm" in svg_text
    assert "Shared LR schedule (both models)" in svg_text
    assert "\ufffd" not in svg_text
    assert all(line == line.rstrip() for line in svg_text.splitlines())
    data = png.read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    width, height = struct.unpack(">II", data[16:24])
    assert width >= 1500 and height >= 1200


def test_gmtnet_history_rejects_wrong_model_and_epoch_gaps(tmp_path) -> None:
    path = _gmtnet_summary(tmp_path / "gmtnet.json")
    assert len(load_gmtnet_history(path)["history"]) == 200
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["model"] = "not-gmtnet"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="passed GMTNet"):
        load_gmtnet_history(path)


def test_gmtnet_history_accepts_legacy_missing_fnorm_and_rejects_partial_or_nonfinite(
    tmp_path,
) -> None:
    legacy_path = _gmtnet_summary(tmp_path / "legacy.json")
    assert "validation_fnorm" not in load_gmtnet_history(legacy_path)["history"][0]

    path = _gmtnet_summary(tmp_path / "current.json", include_validation_fnorm=True)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["history"][0].pop("validation_fnorm")
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="present for every epoch or absent"):
        load_gmtnet_history(path)

    path = _gmtnet_summary(tmp_path / "nonfinite.json", include_validation_fnorm=True)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["history"][0]["validation_fnorm"] = float("nan")
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="non-finite"):
        load_gmtnet_history(path)


def test_gmtnet_history_respects_exclusive_checkpoint_epoch_threshold(tmp_path) -> None:
    path = _gmtnet_summary(tmp_path / "threshold.json", include_validation_fnorm=True)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["config"] = {"minimum_checkpoint_epoch_exclusive": 100}
    payload["history"][39]["validation_mae"] = 0.01
    payload["history"][139]["validation_mae"] = 0.02
    payload["best_epoch"] = 140
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert load_gmtnet_history(path)["best_epoch"] == 140

    payload["config"]["minimum_checkpoint_epoch_exclusive"] = 200
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="no checkpoint-eligible epoch"):
        load_gmtnet_history(path)


def test_parent_dag_history_validates_hash_routing_and_best_epoch(tmp_path) -> None:
    path = _parent_summary(tmp_path / "parent.json")
    payload = load_parent_dag_history(path, expected_sha256=file_sha256(path))
    assert len(payload["history"]) == 200
    assert payload["best_epoch"] == 200
    changed = json.loads(path.read_text(encoding="utf-8"))
    changed["routing"] = "material_parent_dag"
    path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="supported parent-DAG"):
        load_parent_dag_history(path)


def test_relative_parent_history_requires_complete_path_fusion_contract(tmp_path) -> None:
    path = _relative_parent_summary(tmp_path / "relative.json")
    assert load_parent_dag_history(path)["routing"] == "point_group_relative_edge_stick_breaking"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["path_fusion"]["between_path_prior"] = "equal"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="path-fusion"):
        load_parent_dag_history(path)


def test_routing_comparison_render_is_labeled_and_parseable(tmp_path) -> None:
    current_path = _summary(tmp_path / "current.json")
    parent_path = _parent_summary(tmp_path / "parent.json")
    svg = tmp_path / "routing.svg"
    png = tmp_path / "routing.png"
    render_routing_comparison_history(
        load_current_group_history(current_path),
        load_parent_dag_history(parent_path),
        svg_path=svg,
        png_path=png,
    )
    ET.parse(svg)
    text = svg.read_text(encoding="utf-8")
    assert "current-pg vs all-ancestor PG-DAG" in text
    assert "current-pg best epoch 200" in text
    assert "parent-DAG best epoch 200" in text
    assert "Shared LR schedule" in text
    assert "\ufffd" not in text
    assert all(line == line.rstrip() for line in text.splitlines())
    data = png.read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    width, height = struct.unpack(">II", data[16:24])
    assert width >= 1500 and height >= 1200


def test_relative_routing_comparison_uses_distinct_labels(tmp_path) -> None:
    current_path = _summary(tmp_path / "current.json")
    parent_path = _relative_parent_summary(tmp_path / "relative.json")
    svg = tmp_path / "relative.svg"
    png = tmp_path / "relative.png"
    render_routing_comparison_history(
        load_current_group_history(current_path),
        load_parent_dag_history(parent_path),
        svg_path=svg,
        png_path=png,
    )
    ET.parse(svg)
    text = svg.read_text(encoding="utf-8")
    assert "current-pg vs relative-PG path-weighted" in text
    assert "relative-PG best epoch 200" in text
    assert "\ufffd" not in text


def test_all_experiment_history_plot_preserves_missing_metrics(tmp_path) -> None:
    current_path = _summary(tmp_path / "current.json")
    gmtnet_path = _gmtnet_summary(tmp_path / "gmtnet.json")
    current = json.loads(current_path.read_text(encoding="utf-8"))
    current["test_metrics"] = {"fnorm": 19.5}
    current_path.write_text(json.dumps(current), encoding="utf-8")
    gmtnet = json.loads(gmtnet_path.read_text(encoding="utf-8"))
    gmtnet["test_metrics"] = {"fnorm": 19.1}
    gmtnet_path.write_text(json.dumps(gmtnet), encoding="utf-8")
    svg = tmp_path / "all.svg"
    png = tmp_path / "all.png"
    render_all_experiment_histories(
        (
            ("current-PG", load_experiment_history(current_path)),
            ("GMTNet", load_experiment_history(gmtnet_path)),
        ),
        svg_path=svg,
        png_path=png,
    )
    ET.parse(svg)
    text = svg.read_text(encoding="utf-8")
    assert "all recorded experiment histories" in text
    assert "current-PG" in text and "GMTNet" in text
    assert "Every drawn series has contiguous epochs" in text
    assert "exact overlaps may occlude a line" in text
    assert "\ufffd" not in text
    assert png.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
