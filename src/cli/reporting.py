from __future__ import annotations

import json
import os
from pathlib import Path
import platform
import subprocess
import time
from typing import Callable
import xml.etree.ElementTree as ET

import torch


def representation_matrix_for_features(
    representation: object,
    rotation: torch.Tensor,
    features: torch.Tensor,
) -> torch.Tensor:
    """Build an e3nn representation on CPU, then match the feature tensor."""

    matrix = representation.D_from_matrix(rotation.detach().cpu())
    return matrix.to(device=features.device, dtype=features.dtype)


class EvidenceFailure(RuntimeError):
    """A failed case carrying structured evidence gathered before rejection."""

    def __init__(self, message: str, evidence: dict[str, object]) -> None:
        super().__init__(message)
        self.evidence = evidence


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


def run_recorded_smoke(
    run_case: Callable[[str], dict[str, object]],
    *,
    device: str,
    output: str | Path,
    junit: str | Path,
    suite_name: str,
) -> dict[str, object]:
    """Run one smoke case while persisting success or failure evidence."""

    return run_recorded_case(
        lambda: run_case(device),
        output=output,
        junit=junit,
        suite_name=suite_name,
    )


def run_recorded_case(
    run_case: Callable[[], dict[str, object]],
    *,
    output: str | Path,
    junit: str | Path,
    suite_name: str,
) -> dict[str, object]:
    """Run one callable while persisting success or failure evidence."""

    started = time.perf_counter()
    error: Exception | None = None
    try:
        result = run_case()
        result["execution"] = execution_metadata()
    except Exception as caught:
        error = caught
        result = dict(getattr(caught, "evidence", {}))
        result.update({
            "status": "failed",
            "error_type": type(caught).__name__,
            "error": str(caught),
            "execution": execution_metadata(),
        })
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    write_single_case_junit(
        junit,
        suite_name=suite_name,
        seconds=time.perf_counter() - started,
        error=error,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    if error is not None:
        raise error
    return result
