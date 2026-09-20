from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from ..evaluation.space_group_analysis import (
    analyze_space_groups,
    render_relationship_plot,
    write_csv,
    write_markdown,
)


def _write_json_atomic(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze reduced dielectric errors by space group")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--current-pg", type=Path, required=True)
    parser.add_argument("--gmtnet", type=Path, required=True)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    parser.add_argument("--plot", type=Path, required=True)
    arguments = parser.parse_args()
    report = analyze_space_groups(
        arguments.dataset,
        {"current-pg": arguments.current_pg, "GMTNet": arguments.gmtnet},
    )
    _write_json_atomic(arguments.json, report)
    write_csv(report, arguments.csv)
    write_markdown(report, arguments.markdown)
    render_relationship_plot(report, arguments.plot)
    print(json.dumps({key: report[key] for key in ("test_id_count", "space_group_count", "relationships")}, indent=2))


if __name__ == "__main__":
    main()
