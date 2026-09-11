from __future__ import annotations

import json
from pathlib import Path
import xml.etree.ElementTree as ET

import pytest

from src.cli import dpa4_smoke
from src.cli import reporting
from src.cli.reporting import run_recorded_smoke, write_single_case_junit
from src.cli.test_all import junit_counts


def test_junit_counts_supports_testsuites_and_testsuite_roots(tmp_path: Path) -> None:
    aggregate = tmp_path / "aggregate.xml"
    aggregate.write_text(
        '<testsuites><testsuite tests="3" failures="1" errors="0" skipped="1"/>'
        '<testsuite tests="2" failures="0" errors="1" skipped="0"/></testsuites>',
        encoding="utf-8",
    )
    assert junit_counts(aggregate) == {
        "tests": 5,
        "failures": 1,
        "errors": 1,
        "skipped": 1,
    }

    single = tmp_path / "single.xml"
    single.write_text(
        '<testsuite tests="4" failures="0" errors="0" skipped="0"/>',
        encoding="utf-8",
    )
    assert junit_counts(single) == {
        "tests": 4,
        "failures": 0,
        "errors": 0,
        "skipped": 0,
    }


def test_single_case_junit_records_failure_without_a_skip(tmp_path: Path) -> None:
    path = tmp_path / "smoke.xml"
    write_single_case_junit(
        path,
        suite_name="smoke_7",
        seconds=1.25,
        error=ValueError("resource unavailable"),
    )
    root = ET.parse(path).getroot()
    assert root.attrib == {
        "name": "smoke_7",
        "tests": "1",
        "failures": "1",
        "errors": "0",
        "skipped": "0",
        "time": "1.250000",
    }
    failure = root.find("./testcase/failure")
    assert failure is not None
    assert failure.attrib["type"] == "ValueError"


def test_recorded_smoke_persists_success_summary_and_junit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(reporting, "execution_metadata", lambda: {"slurm_job_id": "42"})
    output = tmp_path / "success.json"
    junit = tmp_path / "success.xml"
    result = run_recorded_smoke(
        lambda device: {"status": "passed", "device": device},
        device="cuda",
        output=output,
        junit=junit,
        suite_name="adapter_smoke",
    )
    assert result["execution"] == {"slurm_job_id": "42"}
    assert json.loads(output.read_text(encoding="utf-8")) == result
    assert ET.parse(junit).getroot().attrib["failures"] == "0"


def test_recorded_smoke_persists_failure_before_reraising(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(reporting, "execution_metadata", lambda: {"slurm_job_id": "43"})
    output = tmp_path / "failure.json"
    junit = tmp_path / "failure.xml"

    def fail(_device: str) -> dict[str, object]:
        raise ValueError("checkpoint rejected")

    with pytest.raises(ValueError, match="checkpoint rejected"):
        run_recorded_smoke(
            fail,
            device="cuda",
            output=output,
            junit=junit,
            suite_name="adapter_smoke",
        )
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["status"] == "failed"
    assert result["error_type"] == "ValueError"
    assert ET.parse(junit).getroot().attrib["failures"] == "1"


def test_dpa4_smoke_imports_its_reported_source_layout() -> None:
    assert dpa4_smoke.DPA4_SO3_LAYOUT.dimension == 1600
