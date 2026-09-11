from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
from typing import Iterable, Mapping


SACCT_FIELDS = ("JobIDRaw", "JobName", "State", "ExitCode", "Elapsed", "AllocTRES")


def expected_job_ids(manifest: Mapping[str, object]) -> dict[str, str]:
    """Expand a versioned acceptance manifest into JobIDRaw -> logical-name."""

    if manifest.get("schema_version") != 1:
        raise ValueError("Slurm job manifest schema_version must equal 1")
    jobs = manifest.get("jobs")
    if not isinstance(jobs, list) or not jobs:
        raise ValueError("Slurm job manifest must contain a non-empty jobs list")
    expected: dict[str, str] = {}
    for item in jobs:
        if not isinstance(item, dict) or set(item) not in (
            {"name", "job_id"},
            {"name", "job_id", "array"},
        ):
            raise ValueError("each job must contain only name, job_id, and optional array")
        name = item["name"]
        job_id = item["job_id"]
        if not isinstance(name, str) or not name or not isinstance(job_id, str) or not job_id.isdigit():
            raise ValueError("job name must be non-empty and job_id must contain only digits")
        array = item.get("array")
        if array is None:
            identifiers = (job_id,)
        else:
            if not isinstance(array, dict) or set(array) != {"start", "end"}:
                raise ValueError("array must contain exactly start and end")
            start, end = array["start"], array["end"]
            if not isinstance(start, int) or not isinstance(end, int) or start < 0 or end < start:
                raise ValueError("array range must satisfy 0 <= start <= end")
            identifiers = tuple(f"{job_id}_{index}" for index in range(start, end + 1))
        for identifier in identifiers:
            if identifier in expected:
                raise ValueError(f"duplicate expected JobIDRaw {identifier}")
            expected[identifier] = name
    return expected


def parse_sacct(text: str) -> list[dict[str, str]]:
    """Parse headerless pipe-delimited sacct output with the frozen field order."""

    rows = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        values = line.split("|")
        if values[-1] == "":
            values.pop()
        if len(values) != len(SACCT_FIELDS):
            raise ValueError(f"invalid sacct field count at line {line_number}")
        rows.append(dict(zip(SACCT_FIELDS, values)))
    return rows


def audit_sacct(
    manifest: Mapping[str, object], rows: Iterable[Mapping[str, str]]
) -> dict[str, object]:
    """Require every expected allocation/task to be uniquely COMPLETED with exit 0:0."""

    expected = expected_job_ids(manifest)
    observed: dict[str, Mapping[str, str]] = {}
    duplicates: list[str] = []
    for row in rows:
        identifier = row.get("JobIDRaw", "")
        if identifier not in expected:
            continue
        if identifier in observed:
            duplicates.append(identifier)
        else:
            observed[identifier] = row
    missing = sorted(set(expected) - set(observed))
    failed = [
        {
            "job_id_raw": identifier,
            "name": expected[identifier],
            "state": observed[identifier].get("State"),
            "exit_code": observed[identifier].get("ExitCode"),
        }
        for identifier in sorted(observed)
        if observed[identifier].get("State") != "COMPLETED"
        or observed[identifier].get("ExitCode") != "0:0"
    ]
    passed = not missing and not duplicates and not failed
    return {
        "schema_version": 1,
        "status": "passed" if passed else "failed",
        "expected_count": len(expected),
        "observed_count": len(observed),
        "missing_job_ids": missing,
        "duplicate_job_ids": sorted(set(duplicates)),
        "failed_jobs": failed,
        "jobs": [
            {
                "job_id_raw": identifier,
                "name": expected[identifier],
                **{field: observed[identifier].get(field) for field in SACCT_FIELDS[1:]},
            }
            for identifier in sorted(observed)
        ],
    }


def run_audit(manifest_path: Path, raw_path: Path) -> dict[str, object]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_job_ids(manifest)
    job_ids = sorted({str(item["job_id"]) for item in manifest["jobs"]})
    completed = subprocess.run(
        [
            "sacct",
            "-X",
            "-n",
            "-P",
            "-j",
            ",".join(job_ids),
            "--format=" + ",".join(SACCT_FIELDS),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_text(completed.stdout, encoding="utf-8")
    if completed.returncode:
        raise RuntimeError(f"sacct failed with exit {completed.returncode}: {completed.stderr.strip()}")
    return audit_sacct(manifest, parse_sacct(completed.stdout))


def main() -> None:
    parser = argparse.ArgumentParser(description="Strictly audit terminal Slurm acceptance states")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    try:
        report = run_audit(arguments.manifest, arguments.raw)
    except Exception as error:
        report = {
            "schema_version": 1,
            "status": "failed",
            "error_type": type(error).__name__,
            "error": str(error),
        }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
