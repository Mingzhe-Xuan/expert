from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from .reporting import execution_metadata, write_single_case_junit
from ..configs import enumerate_architecture_configs
from ..data import TrainingUnit
from ..training import run_five_structure_smoke, write_smoke_report


DEFAULT_SCHEDULE = Path(__file__).resolve().parents[1] / "configs" / "real_smoke_schedule.json"


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
    report["index"] = index
    report["execution"] = execution_metadata()
    report_path = output_root / "reports" / f"{name}.json"
    write_smoke_report(report_path, report)
    return {"report": str(report_path), **report}


def main() -> None:
    parser = argparse.ArgumentParser(description="One frozen real-data 3/1/1 smoke run")
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--junit", type=Path, required=True)
    arguments = parser.parse_args()
    started = time.perf_counter()
    error = None
    try:
        report = run_schedule(
            arguments.index,
            device=arguments.device,
            output_root=arguments.output_root,
        )
    except Exception as caught:
        error = caught
        report = {
            "status": "failed",
            "index": arguments.index,
            "error_type": type(caught).__name__,
            "error": str(caught),
            "execution": execution_metadata(),
        }
    arguments.summary.parent.mkdir(parents=True, exist_ok=True)
    arguments.summary.write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    write_single_case_junit(
        arguments.junit,
        suite_name=f"real_subset_smoke_{arguments.index}",
        seconds=time.perf_counter() - started,
        error=error,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if error is not None:
        raise error


if __name__ == "__main__":
    main()
