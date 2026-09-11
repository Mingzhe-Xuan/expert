from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import subprocess

import torch

from ..configs import enumerate_architecture_configs
from ..data import TrainingUnit
from ..training import run_five_structure_smoke, write_smoke_report


DEFAULT_SCHEDULE = Path(__file__).resolve().parents[1] / "configs" / "real_smoke_schedule.json"


def _git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def run_schedule(index: int, *, device: str, output_root: Path) -> dict[str, object]:
    schedule = json.loads(DEFAULT_SCHEDULE.read_text(encoding="utf-8"))
    runs = schedule["runs"]
    if not 0 <= index < len(runs):
        raise ValueError(f"schedule index must be in [0, {len(runs) - 1}]")
    row = runs[index]
    architecture = enumerate_architecture_configs()[int(row["variant_index"])]
    unit = TrainingUnit(str(row["dataset"]), str(row["target"]))
    name = f"{index:02d}-{row['backbone']}-{unit.namespace}-{architecture.variant_id}"
    checkpoint = output_root / "checkpoints" / f"{name}.pt"
    report = run_five_structure_smoke(
        backbone_family=str(row["backbone"]),
        architecture=architecture,
        unit=unit,
        checkpoint_path=checkpoint,
        device=device,
    )
    report["execution"] = {
        "git_commit": _git_commit(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_device": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
    }
    report_path = output_root / "reports" / f"{name}.json"
    write_smoke_report(report_path, report)
    return {"report": str(report_path), **report}


def main() -> None:
    parser = argparse.ArgumentParser(description="One frozen real-data 3/1/1 smoke run")
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-root", type=Path, required=True)
    arguments = parser.parse_args()
    report = run_schedule(
        arguments.index,
        device=arguments.device,
        output_root=arguments.output_root,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
