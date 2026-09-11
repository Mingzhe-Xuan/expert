from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from .reporting import execution_metadata, write_single_case_junit
from ..evaluation import run_point_group_smoke


def main() -> None:
    parser = argparse.ArgumentParser(description="One real-checkpoint 32-PG smoke row")
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--junit", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    arguments = parser.parse_args()
    started = time.perf_counter()
    error = None
    try:
        report = run_point_group_smoke(
            arguments.index,
            fixture_path=arguments.fixtures,
            device=arguments.device,
        )
    except Exception as caught:
        error = caught
        report = {
            "status": "failed",
            "index": arguments.index,
            "error_type": type(caught).__name__,
            "error": str(caught),
        }
    report["execution"] = execution_metadata()
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    write_single_case_junit(
        arguments.junit,
        suite_name=f"point_group_smoke_{arguments.index}",
        seconds=time.perf_counter() - started,
        error=error,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if error is not None:
        raise error


if __name__ == "__main__":
    main()
