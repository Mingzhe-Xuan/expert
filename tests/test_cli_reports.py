from __future__ import annotations

from pathlib import Path
import xml.etree.ElementTree as ET

from src.cli.reporting import write_single_case_junit
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
