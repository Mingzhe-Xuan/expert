from __future__ import annotations

import os
from pathlib import Path
import platform
import subprocess
import xml.etree.ElementTree as ET

import torch


def execution_metadata() -> dict[str, object]:
    return {
        "git_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_device": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
    }


def write_single_case_junit(
    path: str | Path,
    *,
    suite_name: str,
    seconds: float,
    error: BaseException | None = None,
) -> None:
    """Write one machine-readable smoke result without masking the original failure."""

    suite = ET.Element(
        "testsuite",
        name=suite_name,
        tests="1",
        failures="1" if error is not None else "0",
        errors="0",
        skipped="0",
        time=f"{seconds:.6f}",
    )
    case = ET.SubElement(
        suite,
        "testcase",
        classname=suite_name,
        name=suite_name,
        time=f"{seconds:.6f}",
    )
    if error is not None:
        failure = ET.SubElement(
            case,
            "failure",
            message=str(error),
            type=type(error).__name__,
        )
        failure.text = f"{type(error).__name__}: {error}"
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(suite).write(target, encoding="utf-8", xml_declaration=True)
