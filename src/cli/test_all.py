from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import xml.etree.ElementTree as ET


def junit_counts(path: Path) -> dict[str, int]:
    """Aggregate pytest JUnit counts across either supported root shape."""

    root = ET.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
    if not suites:
        raise ValueError(f"JUnit file has no testsuite records: {path}")
    keys = ("tests", "failures", "errors", "skipped")
    return {key: sum(int(suite.attrib.get(key, 0)) for suite in suites) for key in keys}


def _git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the complete project pytest suite")
    parser.add_argument("--junit", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    arguments = parser.parse_args()
    arguments.junit.parent.mkdir(parents=True, exist_ok=True)
    arguments.summary.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "pytest",
        "tests",
        f"--junitxml={arguments.junit}",
        "-ra",
    ]
    completed = subprocess.run(command, check=False)
    counts_error = None
    try:
        counts = junit_counts(arguments.junit)
    except (OSError, ET.ParseError, ValueError) as error:
        counts_error = error
        counts = {"tests": 0, "failures": 0, "errors": 1, "skipped": 0}
    acceptance_exit = completed.returncode or (
        1
        if counts_error is not None
        or counts["failures"]
        or counts["errors"]
        or counts["skipped"]
        else 0
    )
    report: dict[str, object] = {
        "status": "passed" if acceptance_exit == 0 else "failed",
        "exit_code": acceptance_exit,
        "pytest_exit_code": completed.returncode,
        "git_commit": _git_commit(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "python": platform.python_version(),
        "junit": str(arguments.junit),
    }
    report.update(counts)
    if counts_error is not None:
        report["junit_error"] = str(counts_error)
    arguments.summary.write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    raise SystemExit(acceptance_exit)


if __name__ == "__main__":
    main()
