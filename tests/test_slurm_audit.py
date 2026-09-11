from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.cli import slurm_audit


def _manifest() -> dict[str, object]:
    return {
        "schema_version": 1,
        "jobs": [
            {"name": "test_all", "job_id": "100"},
            {
                "name": "point_group_smoke",
                "job_id": "101",
                "array": {"start": 0, "end": 2},
            },
        ],
    }


def _row(identifier: str, state: str = "COMPLETED", exit_code: str = "0:0") -> str:
    return f"{identifier}|job|{state}|{exit_code}|00:00:03|cpu=1|"


def test_manifest_expands_array_tasks_and_clean_sacct_passes() -> None:
    manifest = _manifest()
    assert slurm_audit.expected_job_ids(manifest) == {
        "100": "test_all",
        "101_0": "point_group_smoke",
        "101_1": "point_group_smoke",
        "101_2": "point_group_smoke",
    }
    text = "\n".join(
        [
            _row("100"),
            _row("101"),
            _row("101_0"),
            _row("101_1"),
            _row("101_2"),
            _row("100.batch"),
        ]
    )
    report = slurm_audit.audit_sacct(manifest, slurm_audit.parse_sacct(text))
    assert report["status"] == "passed"
    assert report["expected_count"] == report["observed_count"] == 4


@pytest.mark.parametrize(
    ("rows", "field"),
    [
        ([_row("100"), _row("101_0"), _row("101_1")], "missing_job_ids"),
        (
            [_row("100"), _row("101_0"), _row("101_0"), _row("101_1"), _row("101_2")],
            "duplicate_job_ids",
        ),
        (
            [_row("100", "FAILED", "1:0"), _row("101_0"), _row("101_1"), _row("101_2")],
            "failed_jobs",
        ),
    ],
)
def test_audit_rejects_incomplete_duplicate_or_failed_jobs(
    rows: list[str], field: str
) -> None:
    report = slurm_audit.audit_sacct(_manifest(), slurm_audit.parse_sacct("\n".join(rows)))
    assert report["status"] == "failed"
    assert report[field]


def test_manifest_and_sacct_schema_fail_closed() -> None:
    with pytest.raises(ValueError, match="schema_version"):
        slurm_audit.expected_job_ids({"schema_version": 2, "jobs": []})
    with pytest.raises(ValueError, match="field count"):
        slurm_audit.parse_sacct("100|too|short|")


def test_run_audit_uses_allocation_only_machine_fields(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = tmp_path / "jobs.json"
    manifest.write_text(json.dumps(_manifest()), encoding="utf-8")
    output = "\n".join([_row("100"), _row("101_0"), _row("101_1"), _row("101_2")])
    captured: dict[str, object] = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        return SimpleNamespace(returncode=0, stdout=output, stderr="")

    monkeypatch.setattr(slurm_audit.subprocess, "run", fake_run)
    report = slurm_audit.run_audit(manifest, tmp_path / "sacct.txt")
    command = captured["command"]
    assert command[:4] == ["sacct", "-X", "-n", "-P"]
    assert command[command.index("-j") + 1] == "100,101"
    assert command[-1] == "--format=" + ",".join(slurm_audit.SACCT_FIELDS)
    assert report["status"] == "passed"
