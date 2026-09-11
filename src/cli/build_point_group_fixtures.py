from __future__ import annotations

import argparse
from pathlib import Path

from ..data import TrainingUnit, load_structure_candidates
from ..evaluation import select_point_group_fixtures, write_point_group_fixture_manifest
from .reporting import run_recorded_case


SOURCE_UNITS = (
    TrainingUnit("jarvis_tensor", "dielectric"),
    TrainingUnit("jarvis_tensor", "elastic"),
    TrainingUnit("matten", "elastic"),
)


def build(output: Path) -> dict[str, object]:
    candidates = tuple(
        candidate
        for unit in SOURCE_UNITS
        for candidate in load_structure_candidates(unit)
    )
    manifest = select_point_group_fixtures(candidates)
    write_point_group_fixture_manifest(output, manifest)
    return {
        "status": "passed",
        "candidate_count": len(candidates),
        "fixture_count": len(manifest["fixtures"]),
        "output": str(output),
        "source_units": [unit.namespace for unit in SOURCE_UNITS],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build 32 real-equilibrium PG fixtures")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--junit", type=Path, required=True)
    arguments = parser.parse_args()
    run_recorded_case(
        lambda: build(arguments.output),
        output=arguments.summary,
        junit=arguments.junit,
        suite_name="build_point_group_fixtures",
    )


if __name__ == "__main__":
    main()
