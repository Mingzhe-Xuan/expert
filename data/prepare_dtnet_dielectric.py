from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Mapping, Sequence
from urllib.request import urlopen

import numpy as np


OFFICIAL_SOURCE_COMMIT = "c09e5299890a5473dfa2accf39bedbe5bdf385ab"
OFFICIAL_SOURCE_URL = (
    "https://raw.githubusercontent.com/pfnet-research/dielectric-pred/"
    f"{OFFICIAL_SOURCE_COMMIT}/data/mp_dielectric.json"
)
OFFICIAL_RAW_SHA256 = "7dae31b2f95b60060bf2bab1ce91751f7a48cb18f4e3b6459e44a899337759e0"
DEFAULT_INPUT = Path("data/raw/dtnet/mp_dielectric.json")
DEFAULT_OUTPUT = Path("data/processed/dtnet/dielectric.jsonl")
DEFAULT_MANIFEST = Path("data/manifests/dtnet_dielectric.json")
_MP_ID = re.compile(r"mp-[0-9]+\Z")
_TENSOR_NAMES = ("electronic", "ionic", "total")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise ValueError(f"duplicate JSON key {key!r}")
        output[key] = value
    return output


def _finite_float(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be numeric") from exc
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _matrix3(value: Any, label: str) -> list[list[float]]:
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError(f"{label} must have shape 3x3")
    rows = []
    for row_index, row in enumerate(value):
        if not isinstance(row, list) or len(row) != 3:
            raise ValueError(f"{label} must have shape 3x3")
        rows.append(
            [_finite_float(item, f"{label}[{row_index},{column}]") for column, item in enumerate(row)]
        )
    return rows


def _determinant3(matrix: Sequence[Sequence[float]]) -> float:
    a, b, c = matrix
    return (
        a[0] * (b[1] * c[2] - b[2] * c[1])
        - a[1] * (b[0] * c[2] - b[2] * c[0])
        + a[2] * (b[0] * c[1] - b[1] * c[0])
    )


def _normalize_structure(raw: Any, sample_id: str) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise ValueError(f"{sample_id}: structure must be an object")
    lattice_raw = raw.get("lattice")
    if not isinstance(lattice_raw, Mapping):
        raise ValueError(f"{sample_id}: structure.lattice must be an object")
    lattice = _matrix3(lattice_raw.get("matrix"), f"{sample_id}: lattice matrix")
    if abs(_determinant3(lattice)) < 1.0e-12:
        raise ValueError(f"{sample_id}: lattice must be invertible")
    pbc = lattice_raw.get("pbc")
    if pbc is not None and pbc != [True, True, True]:
        raise ValueError(f"{sample_id}: DTNet structures must be periodic in all directions")
    sites = raw.get("sites")
    if not isinstance(sites, list) or not sites:
        raise ValueError(f"{sample_id}: structure sites must be a non-empty list")
    elements: list[str] = []
    fractional: list[list[float]] = []
    for site_index, site in enumerate(sites):
        if not isinstance(site, Mapping):
            raise ValueError(f"{sample_id}: site {site_index} must be an object")
        species = site.get("species")
        if not isinstance(species, list) or len(species) != 1:
            raise ValueError(f"{sample_id}: disordered site {site_index} is unsupported")
        species_row = species[0]
        if not isinstance(species_row, Mapping):
            raise ValueError(f"{sample_id}: site {site_index} species must be an object")
        element = species_row.get("element")
        if not isinstance(element, str) or not re.fullmatch(r"[A-Z][a-z]?", element):
            raise ValueError(f"{sample_id}: site {site_index} has an invalid element")
        occupancy = _finite_float(species_row.get("occu"), f"{sample_id}: site occupancy")
        if occupancy != 1.0:
            raise ValueError(f"{sample_id}: partial occupancy is outside the data contract")
        abc = site.get("abc")
        if not isinstance(abc, list) or len(abc) != 3:
            raise ValueError(f"{sample_id}: fractional coordinate must have width 3")
        coordinates = [
            _finite_float(item, f"{sample_id}: fractional coordinate") % 1.0 for item in abc
        ]
        elements.append(element)
        fractional.append(coordinates)
    return {
        "lattice_angstrom": lattice,
        "fractional_coordinates": fractional,
        "elements": elements,
    }


def _normalize_tensor(value: Any, label: str) -> tuple[list[list[float]], list[list[float]], float]:
    raw = _matrix3(value, label)
    symmetric = [
        [0.5 * (raw[row][column] + raw[column][row]) for column in range(3)]
        for row in range(3)
    ]
    residual = max(
        abs(raw[row][column] - raw[column][row]) for row in range(3) for column in range(3)
    )
    return raw, symmetric, residual


def normalize_record(sample_id: str, raw: Any) -> dict[str, Any]:
    if not isinstance(sample_id, str) or _MP_ID.fullmatch(sample_id) is None:
        raise ValueError(f"invalid Materials Project ID {sample_id!r}")
    if not isinstance(raw, Mapping):
        raise ValueError(f"{sample_id}: record must be an object")
    missing = {"structure", "band_gap", *_TENSOR_NAMES} - set(raw)
    if missing:
        raise ValueError(f"{sample_id}: missing fields {sorted(missing)}")
    structure = _normalize_structure(raw["structure"], sample_id)
    raw_tensors: dict[str, list[list[float]]] = {}
    symmetric_tensors: dict[str, list[list[float]]] = {}
    antisymmetric: dict[str, float] = {}
    for name in _TENSOR_NAMES:
        source, symmetric, residual = _normalize_tensor(raw[name], f"{sample_id}: {name}")
        raw_tensors[name] = source
        symmetric_tensors[name] = symmetric
        antisymmetric[name] = residual
    component_residual = max(
        abs(
            raw_tensors["total"][row][column]
            - raw_tensors["electronic"][row][column]
            - raw_tensors["ionic"][row][column]
        )
        for row in range(3)
        for column in range(3)
    )
    if component_residual > 1.0e-8:
        raise ValueError(
            f"{sample_id}: total is inconsistent with electronic + ionic "
            f"(max residual {component_residual:.6g})"
        )
    total_values = [value for row in raw_tensors["total"] for value in row]
    if min(total_values) < -10.0 or max(total_values) > 100.0:
        raise ValueError(f"{sample_id}: total tensor violates the published [-10, 100] filter")
    band_gap = _finite_float(raw["band_gap"], f"{sample_id}: band_gap")
    if band_gap < 0.0:
        raise ValueError(f"{sample_id}: band gap cannot be negative")
    return {
        "schema_version": 1,
        "sample_id": sample_id,
        **structure,
        "dielectric_raw": raw_tensors,
        "dielectric_symmetric": symmetric_tensors,
        "band_gap_ev": band_gap,
        "quality": {
            "antisymmetric_max_abs": antisymmetric,
            "component_sum_max_abs": component_residual,
        },
        "source": {"upstream_key": sample_id},
    }


def official_dtnet_split(sample_ids: Sequence[str], seed: int = 3) -> dict[str, list[str]]:
    """Reproduce the two seed-3 sklearn train_test_split calls in DTNet train.py."""

    keys = sorted(str(value) for value in sample_ids)
    if len(keys) != len(set(keys)):
        raise ValueError("DTNet split IDs must be unique")
    if len(keys) < 10:
        raise ValueError("DTNet 8:1:1 split requires at least ten records")
    test_count = math.ceil(0.1 * len(keys))
    permutation = np.random.RandomState(seed).permutation(len(keys))
    test = [keys[int(index)] for index in permutation[:test_count]]
    train_pool = [keys[int(index)] for index in permutation[test_count:]]
    validation_count = math.ceil((0.1 / 0.9) * len(train_pool))
    second_permutation = np.random.RandomState(seed).permutation(len(train_pool))
    validation = [train_pool[int(index)] for index in second_permutation[:validation_count]]
    train = [train_pool[int(index)] for index in second_permutation[validation_count:]]
    return {"train": train, "validation": validation, "test": test}


def _relative_to_data(path: Path, manifest_path: Path) -> str:
    data_root = manifest_path.resolve().parent.parent
    resolved = path.resolve()
    if resolved != data_root and data_root not in resolved.parents:
        raise ValueError("DTNet resources must stay inside the manifest data root")
    return resolved.relative_to(data_root).as_posix()


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def download_official(path: Path, *, url: str, expected_sha256: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream, urlopen(url, timeout=60) as response:
            while block := response.read(1024 * 1024):
                stream.write(block)
            stream.flush()
            os.fsync(stream.fileno())
        actual = sha256_file(temporary)
        if actual != expected_sha256.lower():
            raise ValueError(f"downloaded DTNet SHA-256 mismatch: {actual}")
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def prepare_dtnet_dataset(
    input_path: Path,
    output_path: Path,
    manifest_path: Path,
    *,
    expected_input_sha256: str | None = OFFICIAL_RAW_SHA256,
    source_url: str = OFFICIAL_SOURCE_URL,
    source_commit: str = OFFICIAL_SOURCE_COMMIT,
    seed: int = 3,
) -> dict[str, Any]:
    input_path = input_path.resolve()
    output_path = output_path.resolve()
    manifest_path = manifest_path.resolve()
    if len({input_path, output_path, manifest_path}) != 3:
        raise ValueError("DTNet input, output, and manifest paths must be distinct")
    raw_relative = _relative_to_data(input_path, manifest_path)
    output_relative = _relative_to_data(output_path, manifest_path)
    actual_input_sha256 = sha256_file(input_path)
    if expected_input_sha256 is not None and actual_input_sha256 != expected_input_sha256.lower():
        raise ValueError(f"DTNet input SHA-256 mismatch: {actual_input_sha256}")
    with input_path.open("r", encoding="utf-8") as stream:
        raw = json.load(stream, object_pairs_hook=_unique_object)
    if not isinstance(raw, Mapping) or not raw:
        raise ValueError("DTNet input must be a non-empty ID-to-record object")
    records = [normalize_record(str(sample_id), raw[sample_id]) for sample_id in sorted(raw)]
    split = official_dtnet_split([record["sample_id"] for record in records], seed=seed)
    lines = "".join(
        json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
        for record in records
    )
    _atomic_write_text(output_path, lines)
    antisymmetric = {
        name: {
            "records_above_1e-8": sum(
                record["quality"]["antisymmetric_max_abs"][name] > 1.0e-8
                for record in records
            ),
            "maximum_abs": max(
                record["quality"]["antisymmetric_max_abs"][name] for record in records
            ),
        }
        for name in _TENSOR_NAMES
    }
    element_symbols = sorted({element for record in records for element in record["elements"]})
    tensor_ranges = {
        name: {
            "minimum": min(
                value
                for record in records
                for row in record["dielectric_raw"][name]
                for value in row
            ),
            "maximum": max(
                value
                for record in records
                for row in record["dielectric_raw"][name]
                for value in row
            ),
        }
        for name in _TENSOR_NAMES
    }
    manifest = {
        "schema_version": 1,
        "dataset": "DTNet refined Materials Project dielectric tensors",
        "source": "https://github.com/pfnet-research/dielectric-pred",
        "source_commit": source_commit,
        "source_url": source_url,
        "source_dataset_version": "Materials Project v2023.11.1",
        "license_note": "Upstream repository is MIT licensed; Materials Project data terms also apply.",
        "local_file": raw_relative,
        "bytes": input_path.stat().st_size,
        "sha256": actual_input_sha256,
        "raw_records": len(records),
        "filtered_records": len(records),
        "filter": "no additional row filtering; strict schema/component validation",
        "upstream_filter": (
            "DTNet retained 6,648 of 7,277 MP rows by requiring every total-dielectric element "
            "in [-10, 100] and elements supported by its 72-element PFP runtime"
        ),
        "tensor_components": list(_TENSOR_NAMES),
        "training_target": "dielectric_symmetric.total = (dielectric_raw.total + transpose) / 2",
        "tensor_unit": "dimensionless",
        "structure_summary": {
            "element_count": len(element_symbols),
            "element_symbols": element_symbols,
            "minimum_atoms": min(len(record["elements"]) for record in records),
            "maximum_atoms": max(len(record["elements"]) for record in records),
        },
        "value_summary": {
            "band_gap_ev": {
                "minimum": min(record["band_gap_ev"] for record in records),
                "maximum": max(record["band_gap_ev"] for record in records),
            },
            "dielectric_raw": tensor_ranges,
        },
        "quality_summary": {
            "antisymmetric_max_abs": antisymmetric,
            "component_sum_max_abs": max(
                record["quality"]["component_sum_max_abs"] for record in records
            ),
        },
        "split_protocol": (
            "DTNet scripts/train.py: sorted IDs; sklearn train_test_split(test_size=0.1, "
            "random_state=3), then train_test_split(test_size=0.1/0.9, random_state=3)"
        ),
        "split_seed": seed,
        "split_scope": (
            "the public seed-3 training configuration; the paper reports five distinct random "
            "splits but does not publish all five seeds"
        ),
        "split_counts": {name: len(values) for name, values in split.items()},
        "splits": split,
        "processed_output": {
            "path": output_relative,
            "size_bytes": output_path.stat().st_size,
            "sha256": sha256_file(output_path),
            "records": len(records),
            "format": "JSON Lines; schema_version=1",
        },
    }
    _atomic_write_text(manifest_path, json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download and normalize the official DTNet Materials Project dielectric dataset"
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--source-url", default=OFFICIAL_SOURCE_URL)
    parser.add_argument("--source-commit", default=OFFICIAL_SOURCE_COMMIT)
    parser.add_argument("--expected-input-sha256", default=OFFICIAL_RAW_SHA256)
    parser.add_argument("--seed", type=int, default=3)
    arguments = parser.parse_args()
    expected = arguments.expected_input_sha256.lower()
    if not re.fullmatch(r"[0-9a-f]{64}", expected):
        parser.error("--expected-input-sha256 must be 64 lowercase hexadecimal characters")
    if arguments.download:
        download_official(arguments.input, url=arguments.source_url, expected_sha256=expected)
    manifest = prepare_dtnet_dataset(
        arguments.input,
        arguments.output,
        arguments.manifest,
        expected_input_sha256=expected,
        source_url=arguments.source_url,
        source_commit=arguments.source_commit,
        seed=arguments.seed,
    )
    print(json.dumps({
        "status": "passed",
        "records": manifest["filtered_records"],
        "split_counts": manifest["split_counts"],
        "processed_output": manifest["processed_output"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
