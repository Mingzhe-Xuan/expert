from __future__ import annotations

import hashlib
import json
from pathlib import Path
import pickle
from typing import Any, Iterable, Mapping

import numpy as np

from .records import NormalizedRecord


ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = ROOT / "data"
_VOIGT_PAIRS = ((0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1))
_JARVIS_VASP_TO_STANDARD = (0, 1, 2, 4, 5, 3)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _manifest(name: str) -> tuple[Path, dict[str, Any]]:
    path = DATA_ROOT / "manifests" / name
    return path, json.loads(path.read_text(encoding="utf-8"))


def _resource(manifest_path: Path, manifest: Mapping[str, Any], *, processed: bool = False) -> Path:
    key = "processed_output" if processed else "local_file"
    entry = manifest[key]
    relative = entry["path"] if isinstance(entry, Mapping) else entry
    path = (manifest_path.parent.parent / str(relative)).resolve()
    root = manifest_path.parent.parent.resolve()
    if path != root and root not in path.parents:
        raise ValueError("dataset resource escapes data root")
    size = entry["size_bytes"] if isinstance(entry, Mapping) else manifest["bytes"]
    sha = entry["sha256"] if isinstance(entry, Mapping) else manifest["sha256"]
    if not path.is_file() or path.stat().st_size != int(size):
        raise ValueError(f"resource size mismatch: {path}")
    if sha256_file(path) != str(sha).lower():
        raise ValueError(f"resource hash mismatch: {path}")
    return path


def _atomic_numbers(elements: Iterable[str]) -> np.ndarray:
    from ase.data import atomic_numbers

    try:
        return np.asarray([atomic_numbers[str(value)] for value in elements], dtype=np.int32)
    except KeyError as exc:
        raise ValueError(f"unknown element symbol {exc.args[0]!r}") from exc


def _jarvis_structure(raw: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lattice = np.asarray(raw["lattice_mat"], dtype=np.float64)
    coordinates = np.asarray(raw["coords"], dtype=np.float64)
    if bool(raw.get("cartesian", False)):
        coordinates = np.linalg.solve(lattice.T, coordinates.T).T
    return lattice, coordinates % 1.0, _atomic_numbers(raw["elements"])


def _matten_structure(raw: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    elements: list[str] = []
    fractional: list[list[float]] = []
    for site in raw["sites"]:
        species = site["species"]
        if len(species) != 1 or float(species[0].get("occu", 0.0)) != 1.0:
            raise ValueError("disordered MatTen sites are unsupported")
        elements.append(str(species[0]["element"]))
        fractional.append(site["abc"])
    return (
        np.asarray(raw["lattice"]["matrix"], dtype=np.float64),
        np.asarray(fractional, dtype=np.float64) % 1.0,
        _atomic_numbers(elements),
    )


def voigt_to_elastic(voigt: Any) -> np.ndarray:
    matrix = np.asarray(voigt, dtype=np.float64)
    if matrix.shape != (6, 6):
        raise ValueError("elastic Voigt tensor must have shape 6x6")
    output = np.empty((3, 3, 3, 3), dtype=np.float64)
    for a, (i, j) in enumerate(_VOIGT_PAIRS):
        for b, (k, ell) in enumerate(_VOIGT_PAIRS):
            for indices in (
                (i, j, k, ell),
                (j, i, k, ell),
                (i, j, ell, k),
                (j, i, ell, k),
                (k, ell, i, j),
                (ell, k, i, j),
                (k, ell, j, i),
                (ell, k, j, i),
            ):
                output[indices] = matrix[a, b]
    return output


def jarvis_voigt_to_standard(voigt: Any) -> np.ndarray:
    """Reorder VASP/JARVIS XX,YY,ZZ,XY,YZ,ZX axes to standard Voigt order."""

    matrix = np.asarray(voigt, dtype=np.float64)
    if matrix.shape != (6, 6):
        raise ValueError("JARVIS elastic tensor must have shape 6x6")
    return matrix[np.ix_(_JARVIS_VASP_TO_STANDARD, _JARVIS_VASP_TO_STANDARD)]


def iter_dtnet() -> Iterable[NormalizedRecord]:
    manifest_path, manifest = _manifest("dtnet_dielectric.json")
    path = _resource(manifest_path, manifest, processed=True)
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            raw = json.loads(line)
            source_id = str(raw["sample_id"])
            structure = (
                np.asarray(raw["lattice_angstrom"], dtype=np.float64),
                np.asarray(raw["fractional_coordinates"], dtype=np.float64),
                _atomic_numbers(raw["elements"]),
            )
            component_residual = float(raw["quality"]["component_sum_max_abs"])
            for component in ("electronic", "ionic", "total"):
                yield NormalizedRecord(
                    record_id=f"dtnet:{source_id}:{component}",
                    source_dataset="dtnet",
                    source_id=source_id,
                    subtype=f"dielectric_{component}",
                    lattice=structure[0],
                    fractional_positions=structure[1],
                    atomic_numbers=structure[2],
                    tensor=np.asarray(raw["dielectric_raw"][component], dtype=np.float64),
                    unit="dimensionless",
                    provenance={
                        "resource": path.name,
                        "source_commit": manifest["source_commit"],
                        "band_gap_ev": raw["band_gap_ev"],
                        "component_sum_max_abs": component_residual,
                        "label_origin": "published",
                    },
                )


def iter_gmtnet_dielectric() -> Iterable[NormalizedRecord]:
    manifest_path, manifest = _manifest("jarvis_dielectric.json")
    path = _resource(manifest_path, manifest)
    with path.open("rb") as stream:
        records = pickle.load(stream)  # noqa: S301 -- exact trusted hash verified first
    for raw in records:
        source_id = str(raw["JARVIS_ID"])
        structure = _jarvis_structure(raw["atoms"])
        electronic = raw.get("dielectric")
        ionic = raw.get("dielectric_ionic")
        components = {
            "electronic": electronic,
            "ionic": ionic,
            "total": None
            if electronic is None or ionic is None
            else np.asarray(electronic, dtype=np.float64) + np.asarray(ionic, dtype=np.float64),
        }
        for component, tensor in components.items():
            if tensor is None or np.size(tensor) == 0:
                continue
            yield NormalizedRecord(
                record_id=f"gmtnet:{source_id}:{component}",
                source_dataset="gmtnet",
                source_id=source_id,
                subtype=f"dielectric_{component}",
                lattice=structure[0],
                fractional_positions=structure[1],
                atomic_numbers=structure[2],
                tensor=np.asarray(tensor, dtype=np.float64),
                unit="dimensionless",
                provenance={
                    "resource": path.name,
                    "source_commit": manifest["source_commit"],
                    "label_origin": "derived_sum" if component == "total" else "published",
                },
            )


def iter_gmtnet_elastic() -> Iterable[NormalizedRecord]:
    manifest_path, manifest = _manifest("jarvis_elastic.json")
    path = _resource(manifest_path, manifest)
    with path.open("rb") as stream:
        records = pickle.load(stream)  # noqa: S301 -- exact trusted hash verified first
    for raw in records:
        value = raw.get("elastic_total_kbar")
        if value is None or np.size(value) == 0:
            continue
        source_id = str(raw["JARVIS_ID"])
        structure = _jarvis_structure(raw["atoms"])
        source_total_gpa = np.asarray(value, dtype=np.float64) / 10.0
        source_symmetric_gpa = (
            np.asarray(raw.get("elastic_sym_kbar", value), dtype=np.float64) / 10.0
        )
        total_gpa = jarvis_voigt_to_standard(source_total_gpa)
        source_symmetric = jarvis_voigt_to_standard(source_symmetric_gpa)
        yield NormalizedRecord(
            record_id=f"gmtnet:{source_id}:elastic_stiffness",
            source_dataset="gmtnet",
            source_id=source_id,
            subtype="elastic_stiffness",
            lattice=structure[0],
            fractional_positions=structure[1],
            atomic_numbers=structure[2],
            tensor=voigt_to_elastic(total_gpa),
            unit="GPa",
            provenance={
                "resource": path.name,
                "source_commit": manifest["source_commit"],
                "label_origin": "published_elastic_total_kbar",
                "source_symmetrization_relative_residual": float(
                    np.linalg.norm(total_gpa - source_symmetric)
                    / max(np.linalg.norm(source_symmetric), 1.0)
                ),
            },
        )


def iter_matten_elastic() -> Iterable[NormalizedRecord]:
    manifest_path, manifest = _manifest("matten_elastic.json")
    path = _resource(manifest_path, manifest)
    raw = json.loads(path.read_text(encoding="utf-8"))
    for index in sorted(raw["structure"], key=int):
        structure = _matten_structure(raw["structure"][index])
        yield NormalizedRecord(
            record_id=f"matten:{index}:elastic_stiffness",
            source_dataset="matten",
            source_id=f"matten-{index}",
            subtype="elastic_stiffness",
            lattice=structure[0],
            fractional_positions=structure[1],
            atomic_numbers=structure[2],
            tensor=np.asarray(raw["elastic_tensor"][index], dtype=np.float64),
            unit="GPa",
            provenance={
                "resource": path.name,
                "source_commit": manifest["source_commit"],
                "row_index": int(index),
                "formula": raw["formula_pretty"][index],
                "published_split": raw["split"][index],
                "label_origin": "published",
            },
        )


def iter_all_records() -> Iterable[NormalizedRecord]:
    yield from iter_dtnet()
    yield from iter_gmtnet_dielectric()
    yield from iter_gmtnet_elastic()
    yield from iter_matten_elastic()
