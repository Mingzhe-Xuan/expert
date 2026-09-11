from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from validate_point_group_outputs import build_report


@pytest.mark.skipif(
    os.environ.get("RUN_FULL_PG_OUTPUT_TEST") != "1",
    reason="set RUN_FULL_PG_OUTPUT_TEST=1 to run the 32-PG/64-forward regression",
)
def test_final_outputs_obey_all_32_point_groups():
    report = build_report()
    assert len(report["point_groups"]) == 32
    for row in report["point_groups"]:
        assert row["detected_point_group"] == row["point_group"]
        assert row["dielectric"]["passed"], row["point_group"]
        assert row["elastic"]["passed"], row["point_group"]
