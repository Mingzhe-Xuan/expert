from __future__ import annotations

import json
from pathlib import Path
import struct
import xml.etree.ElementTree as ET

import pytest

from src.evaluation.training_history import (
    file_sha256,
    load_current_group_history,
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
    svg = tmp_path / "curve.svg"
    png = tmp_path / "curve.png"
    render_current_group_history(payload, svg_path=svg, png_path=png)
    ET.parse(svg)
    svg_text = svg.read_text(encoding="utf-8")
    assert "current-group only" in svg_text
    assert "Best epoch 200" in svg_text
    assert all(line == line.rstrip() for line in svg_text.splitlines())
    data = png.read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    width, height = struct.unpack(">II", data[16:24])
    assert width >= 1500 and height >= 1200
