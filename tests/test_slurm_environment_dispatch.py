from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess

import pytest

from src.evaluation import point_group_smoke_schedule


ROOT = Path(__file__).resolve().parents[1]
SELECTOR = ROOT / "slurm" / "select_backbone_venv.sh"
FAMILIES = ("mace", "grace", "dpa4", "equiformerv2")
VARIABLES = {
    "mace": "EXPERT_MACE_VENV",
    "grace": "EXPERT_GRACE_VENV",
    "dpa4": "EXPERT_DPA4_VENV",
    "equiformerv2": "EXPERT_EQUIFORMERV2_VENV",
}


def _real_subset_schedule() -> list[dict[str, object]]:
    path = ROOT / "src" / "configs" / "real_smoke_schedule.json"
    return json.loads(path.read_text(encoding="utf-8"))["runs"]


def test_both_mixed_schedules_follow_the_frozen_modulo_four_mapping() -> None:
    for rows in (point_group_smoke_schedule(), _real_subset_schedule()):
        assert all(row["backbone"] == FAMILIES[index % 4] for index, row in enumerate(rows))


@pytest.mark.parametrize("index", range(8))
def test_shell_selector_chooses_exact_backbone_environment(index: int) -> None:
    environment = os.environ.copy()
    for family, variable in VARIABLES.items():
        environment[variable] = f"/recorded/{family}"
    completed = subprocess.run(
        [
            "bash",
            "-c",
            f'source "{SELECTOR.as_posix()}"; select_backbone_venv {index}; '
            'printf "%s" "$EXPERT_SELECTED_VENV"',
        ],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=environment,
    )
    assert completed.stdout == f"/recorded/{FAMILIES[index % 4]}"


def test_shell_selector_fails_closed_when_selected_environment_is_missing() -> None:
    environment = os.environ.copy()
    for variable in VARIABLES.values():
        environment.pop(variable, None)
    completed = subprocess.run(
        ["bash", "-c", f'source "{SELECTOR.as_posix()}"; select_backbone_venv 2'],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=environment,
    )
    assert completed.returncode != 0
    assert "EXPERT_DPA4_VENV" in completed.stderr


def test_slurm_launchers_use_only_explicit_environment_contracts() -> None:
    mixed = (
        ROOT / "slurm" / "smoke_32_point_groups.sbatch",
        ROOT / "slurm" / "smoke_real_subsets.sbatch",
    )
    for path in mixed:
        text = path.read_text(encoding="utf-8")
        assert "select_backbone_venv.sh" in text
        assert "EXPERT_SELECTED_VENV" in text

    expected = {
        ROOT / "slurm" / "test_all.sbatch": "EXPERT_MACE_VENV",
        ROOT / "slurm" / "build_point_group_fixtures.sbatch": "EXPERT_MACE_VENV",
        ROOT / "scripts" / "slurm" / "prepare_jarvis_bec.sbatch": "EXPERT_MACE_VENV",
        ROOT / "scripts" / "slurm" / "mace_adapter_smoke.sbatch": "EXPERT_MACE_VENV",
        ROOT / "scripts" / "slurm" / "grace_adapter_smoke.sbatch": "EXPERT_GRACE_VENV",
        ROOT / "scripts" / "slurm" / "dpa4_adapter_smoke.sbatch": "EXPERT_DPA4_VENV",
        ROOT / "scripts" / "slurm" / "equiformerv2_adapter_smoke.sbatch": (
            "EXPERT_EQUIFORMERV2_VENV"
        ),
    }
    all_launchers = (*mixed, *expected)
    for path, variable in expected.items():
        assert variable in path.read_text(encoding="utf-8")
    assert all("EXPERT_VENV" not in path.read_text(encoding="utf-8") for path in all_launchers)


def test_grace_slurm_paths_select_auditable_tensorflow_cpu_fallback() -> None:
    standalone = (ROOT / "scripts" / "slurm" / "grace_adapter_smoke.sbatch").read_text(
        encoding="utf-8"
    )
    selector = SELECTOR.read_text(encoding="utf-8")
    assert "EXPERT_GRACE_TF_DEVICE=cpu" in standalone
    assert 'EXPERT_GRACE_TF_DEVICE="${EXPERT_GRACE_TF_DEVICE:-cpu}"' in selector


def test_standalone_adapter_launchers_persist_complete_evidence() -> None:
    for family in FAMILIES:
        path = ROOT / "scripts" / "slurm" / f"{family}_adapter_smoke.sbatch"
        text = path.read_text(encoding="utf-8")
        assert "git rev-parse HEAD" in text
        assert "python -m pip freeze" in text
        assert "--output" in text
        assert "--junit" in text
        assert "${SLURM_JOB_ID}" in text


def test_required_data_job_launchers_persist_complete_evidence() -> None:
    for path in (
        ROOT / "slurm" / "build_point_group_fixtures.sbatch",
        ROOT / "scripts" / "slurm" / "prepare_jarvis_bec.sbatch",
    ):
        text = path.read_text(encoding="utf-8")
        assert "git rev-parse HEAD" in text
        assert "python -m pip freeze" in text
        assert "--summary" in text
        assert "--junit" in text
        assert "${SLURM_JOB_ID}" in text
