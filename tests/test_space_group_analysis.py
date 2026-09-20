from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from src.evaluation.space_group_analysis import (
    analyze_space_groups,
    render_relationship_plot,
    write_csv,
    write_markdown,
)


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def _fixtures(tmp_path: Path) -> tuple[Path, Path, Path]:
    dataset: list[dict] = []
    for space_group, point_group, train_count in ((1, "1", 1), (2, "-1", 10), (3, "2", 100)):
        for index in range(train_count):
            dataset.append({"record_id": f"train-{space_group}-{index}", "split": "train", "space_group": space_group, "point_group": point_group})
        dataset.append({"record_id": f"val-{space_group}", "split": "validation", "space_group": space_group, "point_group": point_group})
        dataset.append({"record_id": f"test-{space_group}", "split": "test", "space_group": space_group, "point_group": point_group})
    dataset_path = _write_jsonl(tmp_path / "dataset.jsonl", dataset)
    current_rows = []
    gmt_rows = []
    for space_group, error in ((1, 3.0), (2, 2.0), (3, 1.0)):
        target = [[float(space_group), 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]]
        current = [row[:] for row in target]
        gmt = [row[:] for row in target]
        current[0][0] += error
        gmt[0][0] += error / 2
        base = {"sample_id": f"test-{space_group}", "target": target}
        current_rows.append({**base, "prediction": current})
        gmt_rows.append({**base, "prediction": gmt})
    return dataset_path, _write_jsonl(tmp_path / "current.jsonl", current_rows), _write_jsonl(tmp_path / "gmt.jsonl", gmt_rows)


def test_space_group_analysis_reports_counts_metrics_and_relationships(tmp_path: Path) -> None:
    dataset, current, gmt = _fixtures(tmp_path)
    report = analyze_space_groups(dataset, {"current-pg": current, "GMTNet": gmt}, cohort_thresholds=(1, 2))
    assert report["test_id_count"] == 3
    assert report["space_group_count"] == 3
    assert [row["train_count"] for row in report["groups"]] == [1, 10, 100]
    first = report["groups"][0]
    assert first["models"]["current-pg"]["fnorm"] == pytest.approx(3.0)
    assert first["models"]["current-pg"]["rmse"] == pytest.approx(1.0)
    assert first["delta_current_pg_minus_gmtnet"]["fnorm"] == pytest.approx(1.5)
    relation = report["relationships"]["test_count_gte_1"]["models"]["current-pg"]["fnorm"]
    assert relation["group_count"] == 3
    assert relation["spearman_rho"] == pytest.approx(-1.0)
    unavailable = report["relationships"]["test_count_gte_2"]["models"]["GMTNet"]["rmse"]
    assert unavailable["group_count"] == 0
    assert unavailable["spearman_rho"] is None


def test_space_group_analysis_fails_closed_on_prediction_id_mismatch(tmp_path: Path) -> None:
    dataset, current, gmt = _fixtures(tmp_path)
    rows = [json.loads(line) for line in gmt.read_text(encoding="utf-8").splitlines()]
    rows[0]["sample_id"] = "unknown"
    _write_jsonl(gmt, rows)
    with pytest.raises(ValueError, match="IDs or order differ"):
        analyze_space_groups(dataset, {"current-pg": current, "GMTNet": gmt}, cohort_thresholds=(1,))


def test_space_group_outputs_are_parseable_and_labeled(tmp_path: Path) -> None:
    dataset, current, gmt = _fixtures(tmp_path)
    report = analyze_space_groups(dataset, {"current-pg": current, "GMTNet": gmt}, cohort_thresholds=(1,))
    csv_path = tmp_path / "groups.csv"
    markdown_path = tmp_path / "groups.md"
    svg_path = tmp_path / "groups.svg"
    write_csv(report, csv_path)
    write_markdown(report, markdown_path)
    render_relationship_plot(report, svg_path, minimum_test_count=1)
    assert "current-pg_fnorm" in csv_path.read_text(encoding="utf-8").splitlines()[0]
    assert "Training-count relationship" in markdown_path.read_text(encoding="utf-8")
    ET.parse(svg_path)
    svg = svg_path.read_text(encoding="utf-8")
    assert "error vs training coverage" in svg
    assert "current-pg" in svg
