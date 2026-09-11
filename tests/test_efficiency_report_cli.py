from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.cli.efficiency_report import (
    FLOPS_SCOPE,
    aggregate_efficiency_reports,
    render_efficiency_markdown,
)


def _summary(index: int, *, suite: str = "point_group") -> dict[str, object]:
    task = "elastic"
    efficiency = {
        "schema_version": 1,
        "architecture": "4-b-pge-r__a-none__o3e-none__pg-full_pg__r-full_o3",
        "task": task,
        "point_groups": ["m-3m"],
        "pg_hidden_mode": "full_pg",
        "adaptation_backend": "none",
        "o3e_backend": "none",
        "readout_backend": "full_o3",
        "total_parameters": 100,
        "torch_registered_parameters": 90,
        "external_frozen_parameters": 10,
        "trainable_parameters": 40,
        "active_nonbackbone_parameters": 30,
        "active_expert_count_per_sample": [2],
        "active_expert_count_mean": 2.0,
        "active_expert_count_max": 2,
        "active_downstream_flops": 1234,
        "flops_scope": FLOPS_SCOPE,
        "end_to_end_forward_latency_ms": 2.5,
        "latency_repetitions": 1,
        "peak_cuda_allocated_bytes": 1_048_576,
        "device": "cuda:0",
    }
    summary = {
        "status": "passed",
        "index": index,
        "backbone": "mace",
        "architecture": {"variant_id": efficiency["architecture"]},
        "efficiency": efficiency,
    }
    if suite == "point_group":
        summary.update({"task": task, "purpose": "test", "point_group": "m-3m"})
    else:
        summary["training_unit"] = f"matten::{task}"
    return summary


def _write(path: Path, value: object) -> Path:
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def test_aggregate_complete_matrices_and_render_scoped_markdown(tmp_path: Path) -> None:
    point = [_write(tmp_path / f"p-{index}.json", _summary(index)) for index in range(2)]
    real = [
        _write(tmp_path / "r-0.json", _summary(0, suite="real_subset"))
    ]
    report = aggregate_efficiency_reports(
        {"point_group": reversed(point), "real_subset": real},
        expected_rows={"point_group": 2, "real_subset": 1},
    )
    assert report["status"] == "passed"
    assert report["row_count"] == 3
    assert [(row["suite"], row["index"]) for row in report["records"]] == [
        ("point_group", 0),
        ("point_group", 1),
        ("real_subset", 0),
    ]
    markdown = render_efficiency_markdown(report)
    assert "Active downstream FLOPs" in markdown
    assert "one_real_end_to_end_forward" in markdown
    assert "full_pg" in markdown
    assert "1.00" in markdown


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda value: value.update(status="failed"), "not passed"),
        (lambda value: value["efficiency"].pop("task"), "missing efficiency fields"),
        (lambda value: value["efficiency"].update(active_nonbackbone_parameters=5_000_000), "budget"),
        (lambda value: value["efficiency"].update(flops_scope="total_model"), "FLOPs scope"),
        (lambda value: value["efficiency"].update(active_expert_count_mean=1.0), "mean"),
        (lambda value: value["efficiency"].update(device="cpu", peak_cuda_allocated_bytes=None), "CUDA"),
    ],
)
def test_aggregation_rejects_invalid_evidence(
    tmp_path: Path, mutation, match: str
) -> None:
    summary = _summary(0)
    mutation(summary)
    path = _write(tmp_path / "invalid.json", summary)
    with pytest.raises(ValueError, match=match):
        aggregate_efficiency_reports(
            {"point_group": [path]}, expected_rows={"point_group": 1}
        )


def test_aggregation_rejects_missing_or_duplicate_row_indices(tmp_path: Path) -> None:
    first = _write(tmp_path / "first.json", _summary(0))
    duplicate = _write(tmp_path / "duplicate.json", _summary(0))
    with pytest.raises(ValueError, match="expected indices"):
        aggregate_efficiency_reports(
            {"point_group": [first, duplicate]}, expected_rows={"point_group": 2}
        )
