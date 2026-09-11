from __future__ import annotations

from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name


ROOT = Path(__file__).resolve().parents[1]
MACE_LOCK = ROOT / "requirements" / "guqq" / "mace-core.txt"
GRACE_LOCK = ROOT / "requirements" / "guqq" / "grace.txt"
DPA4_LOCK = ROOT / "requirements" / "guqq" / "dpa4.txt"


def _locked_requirements(path: Path) -> dict[str, Requirement]:
    requirements: dict[str, Requirement] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or line.startswith("--"):
            continue
        requirement = Requirement(line)
        name = canonicalize_name(requirement.name)
        assert name not in requirements, f"duplicate lock entry: {name}"
        assert len(requirement.specifier) == 1
        specifier = next(iter(requirement.specifier))
        assert specifier.operator == "==", f"{name} is not exactly pinned"
        assert requirement.marker is None
        requirements[name] = requirement
    return requirements


def test_mace_lock_uses_official_cuda_128_index() -> None:
    lines = MACE_LOCK.read_text(encoding="utf-8").splitlines()
    assert "--extra-index-url https://download.pytorch.org/whl/cu128" in lines


def test_mace_lock_has_unique_exact_requirements() -> None:
    requirements = _locked_requirements(MACE_LOCK)
    assert len(requirements) == 71


def test_mace_lock_freezes_acceptance_critical_versions() -> None:
    requirements = _locked_requirements(MACE_LOCK)
    expected = {
        "torch": "==2.11.0+cu128",
        "cuda-toolkit": "==12.8.1",
        "e3nn": "==0.4.4",
        "mace-torch": "==0.3.16",
        "spglib": "==2.6.0",
        "ase": "==3.26.0",
        "numpy": "==1.26.4",
        "scipy": "==1.15.3",
        "pytest": "==8.4.2",
    }
    actual = {name: str(requirements[name].specifier) for name in expected}
    assert actual == expected


def test_grace_lock_has_official_cuda_index_and_unique_exact_requirements() -> None:
    lines = GRACE_LOCK.read_text(encoding="utf-8").splitlines()
    assert "--extra-index-url https://download.pytorch.org/whl/cu128" in lines
    assert len(_locked_requirements(GRACE_LOCK)) == 95


def test_grace_lock_freezes_acceptance_critical_versions() -> None:
    requirements = _locked_requirements(GRACE_LOCK)
    expected = {
        "tensorpotential": "==0.6.0",
        "tensorflow": "==2.20.0",
        "torch": "==2.11.0+cu128",
        "cuda-toolkit": "==12.8.1",
        "e3nn": "==0.5.9",
        "numpy": "==1.26.4",
        "scipy": "==1.15.3",
        "spglib": "==2.6.0",
    }
    actual = {name: str(requirements[name].specifier) for name in expected}
    assert actual == expected


def test_dpa4_lock_has_official_cuda_index_and_unique_exact_requirements() -> None:
    lines = DPA4_LOCK.read_text(encoding="utf-8").splitlines()
    assert "--extra-index-url https://download.pytorch.org/whl/cu128" in lines
    assert len(_locked_requirements(DPA4_LOCK)) == 80


def test_dpa4_lock_freezes_acceptance_critical_versions() -> None:
    requirements = _locked_requirements(DPA4_LOCK)
    expected = {
        "deepmd-kit": "==3.2.0",
        "torch": "==2.11.0+cu128",
        "cuda-toolkit": "==12.8.1",
        "e3nn": "==0.5.9",
        "numpy": "==1.26.4",
        "scipy": "==1.15.3",
        "spglib": "==2.6.0",
    }
    actual = {name: str(requirements[name].specifier) for name in expected}
    assert actual == expected
