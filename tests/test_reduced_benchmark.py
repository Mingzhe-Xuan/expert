from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def test_comparison_cli_validates_common_ids_and_writes_table(tmp_path) -> None:
    dpa4 = tmp_path / "dpa4.jsonl"
    gmtnet = tmp_path / "gmtnet.jsonl"
    target = [[2.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 4.0]]
    rotated = [[3.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 4.0]]
    with dpa4.open("w", encoding="utf-8") as left, gmtnet.open("w", encoding="utf-8") as right:
        for index in range(677):
            left.write(json.dumps({"sample_id": f"sample-{index}", "prediction": target,
                                   "target": target}) + "\n")
            right.write(json.dumps({"sample_id": f"sample-{index}", "prediction": rotated,
                                    "target": rotated}) + "\n")
    summary = tmp_path / "comparison.json"
    table = tmp_path / "comparison.md"
    subprocess.run(
        [sys.executable, "-m", "src.cli.compare_reduced_benchmark", "--dpa4", str(dpa4),
         "--gmtnet", str(gmtnet), "--summary", str(summary), "--table", str(table)],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads(summary.read_text(encoding="utf-8"))
    assert report["status"] == "passed"
    assert report["test_count"] == 677
    assert report["metrics"]["GMTNet"]["rmse"] == 0.0
    assert "| Model | RMSE" in table.read_text(encoding="utf-8")
