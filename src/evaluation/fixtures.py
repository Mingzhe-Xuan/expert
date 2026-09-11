from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Callable, Mapping, Sequence

import spglib
import torch

from ..data import StructureCandidate
from ..symmetry import PointGroupRegistry, canonicalize_structure
from ..symmetry.registry import canonical_point_group_symbol


Classifier = Callable[[StructureCandidate], tuple[str, int, int]]


def _record_checksum(record: Mapping[str, object]) -> str:
    payload = {key: value for key, value in record.items() if key != "checksum"}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def classify_candidate(candidate: StructureCandidate) -> tuple[str, int, int]:
    canonical = canonicalize_structure(
        candidate.cartesian_positions.double(),
        candidate.lattice.double(),
        candidate.atomic_numbers,
    )
    return (
        canonical.symmetry.current_point_group,
        canonical.symmetry.current_space_group,
        canonical.symmetry.hall_number,
    )


def select_point_group_fixtures(
    candidates: Sequence[StructureCandidate],
    *,
    classifier: Classifier = classify_candidate,
) -> dict[str, object]:
    """Select one deterministic equilibrium structure for every crystallographic PG."""

    registry = PointGroupRegistry()
    expected = tuple(group.symbol for group in registry)
    source_priority = {
        "jarvis_tensor__dielectric": 0,
        "jarvis_tensor__elastic": 1,
        "matten__elastic": 2,
    }
    selected: dict[str, tuple[tuple[object, ...], dict[str, object]]] = {}
    seen_sources = set()
    for candidate in candidates:
        point_group, space_group, hall_number = classifier(candidate)
        symbol = canonical_point_group_symbol(point_group)
        if symbol not in expected:
            raise ValueError(f"candidate classified to non-crystallographic PG {symbol!r}")
        seen_sources.add((candidate.source_dataset, candidate.source_manifest_sha256))
        record = {
            "point_group": symbol,
            "sample_id": candidate.sample_id,
            "source_dataset": candidate.source_dataset,
            "source_manifest_sha256": candidate.source_manifest_sha256,
            "expected_space_group": int(space_group),
            "expected_hall_number": int(hall_number),
            "atomic_numbers": candidate.atomic_numbers.tolist(),
            "lattice_angstrom": candidate.lattice.double().tolist(),
            "fractional_coordinates": candidate.fractional_positions.double().tolist(),
            "construction": "selected_unmodified_equilibrium_structure",
        }
        record["checksum"] = _record_checksum(record)
        score = (
            len(record["atomic_numbers"]),
            source_priority.get(candidate.source_dataset, 99),
            candidate.sample_id,
        )
        if symbol not in selected or score < selected[symbol][0]:
            selected[symbol] = (score, record)
    missing = [symbol for symbol in expected if symbol not in selected]
    if missing:
        raise ValueError(f"equilibrium sources do not cover point groups {missing}")
    fixtures = [selected[symbol][1] for symbol in expected]
    manifest = {
        "schema_version": 1,
        "generator": "src.evaluation.fixtures.select_point_group_fixtures",
        "spglib_version": str(spglib.__version__),
        "selection_order": "minimum_natoms_then_source_priority_then_sample_id",
        "source_manifests": [
            {"dataset": dataset, "sha256": digest}
            for dataset, digest in sorted(seen_sources)
        ],
        "fixtures": fixtures,
    }
    validate_point_group_fixture_manifest(manifest, redetect=False)
    return manifest


def validate_point_group_fixture_manifest(
    manifest: Mapping[str, object],
    *,
    redetect: bool = True,
) -> None:
    if manifest.get("schema_version") != 1:
        raise ValueError("unsupported point-group fixture schema")
    fixtures = manifest.get("fixtures")
    if not isinstance(fixtures, list):
        raise ValueError("fixture manifest requires a fixture list")
    expected = tuple(group.symbol for group in PointGroupRegistry())
    actual = tuple(str(record.get("point_group")) for record in fixtures)
    if actual != expected:
        raise ValueError("fixture manifest must contain exactly the canonical 32 point groups")
    ids = [str(record.get("sample_id")) for record in fixtures]
    if len(ids) != len(set(ids)):
        raise ValueError("fixture sample IDs must be unique")
    for record in fixtures:
        if record.get("checksum") != _record_checksum(record):
            raise ValueError(f"fixture checksum mismatch for {record.get('point_group')}")
        lattice = torch.tensor(record["lattice_angstrom"], dtype=torch.float64)
        fractional = torch.tensor(record["fractional_coordinates"], dtype=torch.float64)
        numbers = torch.tensor(record["atomic_numbers"], dtype=torch.long)
        if lattice.shape != (3, 3) or abs(float(torch.linalg.det(lattice))) < 1e-12:
            raise ValueError("fixture lattice must be finite and invertible")
        if fractional.shape != (len(numbers), 3) or len(numbers) < 1:
            raise ValueError("fixture sites/species must be non-empty and aligned")
        if not torch.isfinite(lattice).all() or not torch.isfinite(fractional).all():
            raise ValueError("fixture geometry must be finite")
        if redetect:
            detected = canonicalize_structure(fractional @ lattice, lattice, numbers)
            if detected.symmetry.current_point_group != record["point_group"]:
                raise ValueError("fixture point group changed during re-detection")
            if detected.symmetry.hall_number != int(record["expected_hall_number"]):
                raise ValueError("fixture Hall setting changed during re-detection")


def write_point_group_fixture_manifest(path: str | Path, manifest: Mapping[str, object]) -> None:
    validate_point_group_fixture_manifest(manifest)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
