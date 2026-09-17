from __future__ import annotations

import json
from pathlib import Path
import struct
import xml.etree.ElementTree as ET

import pytest

from src.evaluation.training_history import (
    file_sha256,
    load_current_group_history,
    load_gmtnet_history,
    render_current_group_history,
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


def _gmtnet_summary(path: Path) -> Path:
    history = [
        {
            "epoch": epoch,
            "training_loss": 2.5 / epoch,
            "validation_mae": 4.0 / epoch,
            "learning_rate": 1.0e-3 - (epoch / 200) * 9.9e-4,
        }
        for epoch in range(1, 201)
    ]
    payload = {
        "status": "passed",
        "model": "GMTNet",
        "best_epoch": 200,
        "best_validation_mae": history[-1]["validation_mae"],
        "history": history,
    }
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
    gmtnet_path = _gmtnet_summary(tmp_path / "gmtnet.json")
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
    assert "current-group only" in svg_text
    assert "CGCNN best epoch 200" in svg_text
    assert "GMTNet best epoch 200" in svg_text
    assert "Shared LR schedule (both models)" in svg_text
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
