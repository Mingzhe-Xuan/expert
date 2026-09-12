"""Validate benchmark downloads and freeze their split metadata."""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import pickle
import random
from collections import Counter
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "manifests"


def digest(path: Path, algorithm: str) -> str:
    result = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def split_gmtnet(records: list[tuple]) -> dict:
    size = len(records)
    order = list(range(size))
    random.Random(32).shuffle(order)
    n_train, n_val, n_test = int(0.8 * size), int(0.1 * size), int(0.1 * size)
    positions = {
        "train": order[:n_train],
        "validation": order[-(n_val + n_test) : -n_test],
        "test": order[-n_test:],
    }
    used = set().union(*map(set, positions.values()))
    positions["unused_due_to_integer_rounding"] = sorted(set(range(size)) - used)
    output = {}
    for name, indices in positions.items():
        rows = []
        for index in indices:
            row = {
                "filtered_index": index,
                "raw_index": records[index][0],
                "jarvis_id": records[index][1]["JARVIS_ID"],
            }
            if len(records[index]) == 3:
                row["elastic_support_mask_bits"] = int(records[index][2])
            rows.append(row)
        output[name] = rows
    return output


@functools.lru_cache(maxsize=1)
def _elastic_protocol_objects():
    import torch
    from e3nn import o3
    from e3nn.io import CartesianTensor

    irreps = o3.Irreps(
        "2x0e + 2x0o + 2x1e + 2x1o + 2x2e + 2x2o + 2x3e + 2x3o + 1x4e"
    )
    converter = CartesianTensor("ijkl=ijlk=jikl=klij")
    probe = torch.arange(73, dtype=torch.float32) + 10.0
    probe[16:] *= 100.0
    return irreps, converter, probe


def _fractional_structure(atoms: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from ase.data import atomic_numbers

    lattice = np.asarray(atoms["lattice_mat"], dtype=np.float64)
    coordinates = np.asarray(atoms["coords"], dtype=np.float64)
    if bool(atoms.get("cartesian", False)):
        coordinates = np.linalg.solve(lattice.T, coordinates.T).T
    numbers = np.asarray([atomic_numbers[symbol] for symbol in atoms["elements"]])
    return lattice, coordinates % 1.0, numbers


def gmtnet_elastic_protocol_filter(record: dict) -> tuple[bool, int, np.ndarray]:
    """Reproduce GMTNet's second-stage symmetry-zero screen and projected label."""

    import spglib
    import torch

    total = np.asarray(record["elastic_total_kbar"], dtype=np.float64) / 10.0
    if total.shape != (6, 6) or not np.isfinite(total).all():
        raise ValueError("GMTNet elastic label must be a finite 6x6 matrix")
    lattice, fractional, numbers = _fractional_structure(record["atoms"])
    symmetry = spglib.get_symmetry_dataset(
        (lattice, fractional, numbers), symprec=1.0e-5
    )
    if symmetry is None:
        raise ValueError(f"spglib failed for {record['JARVIS_ID']}")
    fractional_rotations = np.asarray(symmetry.rotations, dtype=np.float64)
    lattice_columns = lattice.T
    cartesian = lattice_columns @ fractional_rotations @ np.linalg.inv(lattice_columns)
    unique = np.asarray(
        list(dict.fromkeys(tuple(matrix.reshape(-1)) for matrix in cartesian))
    ).reshape(-1, 3, 3)
    irreps, converter, probe = _elastic_protocol_objects()
    representations = irreps.D_from_matrix(torch.tensor(unique, dtype=torch.float32))
    feature_probe = representations.sum(dim=0) @ probe
    target_probe = feature_probe[
        [0, 1, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 64, 65, 66, 67, 68, 69, 70, 71, 72]
    ]
    fourth_order = converter.to_cartesian(target_probe)
    pairs = ((0, 0), (1, 1), (2, 2), (0, 1), (1, 2), (0, 2))
    ideal = np.asarray(
        [[float(fourth_order[i, j, k, ell]) for k, ell in pairs] for i, j in pairs]
    )
    support = np.abs(ideal) > 1.0
    forbidden_max = float(np.max(np.abs(total * ~support), initial=0.0))
    accepted = forbidden_max < 1.0e-4
    projected = total * support
    bits = sum(1 << index for index, allowed in enumerate(support.reshape(-1)) if allowed)
    return accepted, bits, projected


def jarvis_manifest(filename: str, task: str) -> dict:
    path = ROOT / "raw" / "jarvis_gmtnet" / filename
    with path.open("rb") as stream:
        raw = pickle.load(stream)
    if task == "dielectric":
        records = [
            (i, x)
            for i, x in enumerate(raw)
            if x.get("dielectric")
            and np.max(np.abs(np.asarray(x["dielectric"]))) < 100
        ]
        criterion = "dielectric present and max(abs(dielectric)) < 100"
    else:
        screened = [
            (i, x) for i, x in enumerate(raw)
            if x.get("elastic_total_kbar")
            and np.max(np.abs(np.asarray(x["elastic_total_kbar"]) / 10.0)) < 1500
        ]
        records = []
        for raw_index, record in screened:
            accepted, support_bits, _ = gmtnet_elastic_protocol_filter(record)
            if accepted:
                records.append((raw_index, record, support_bits))
        criterion = (
            "GMTNet official two-stage filter: elastic_total_kbar present; GPa max(abs)<1500; "
            "structure-derived forbidden entries <1e-4 GPa; forbidden entries zeroed"
        )
    splits = split_gmtnet(records)
    output = {
        "dataset": "GMTNet calculation-matched JARVIS " + task,
        "source": "https://github.com/YKQ98/GMTNet",
        "source_commit": "7a606a459ee48a320ed38450e391811fb43d5e19",
        "local_file": path.relative_to(ROOT).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": digest(path, "sha256"),
        "raw_records": len(raw),
        "filter": criterion,
        "filtered_records": len(records),
        "split_protocol": "GMTNet seed=32, ratios=0.8/0.1/0.1",
        "split_counts": {name: len(items) for name, items in splits.items()},
        "splits": splits,
    }
    if task == "elastic":
        output["elastic_protocol_version"] = 1
        output["pre_symmetry_screen_records"] = len(screened)
    return output


def matten_manifest() -> dict:
    path = ROOT / "raw" / "matten" / "crystal_elasticity_tensor.json"
    columns = json.loads(path.read_text(encoding="utf-8"))
    counts = Counter(columns["split"].values())
    return {
        "dataset": "MatTen elasticity tensors of 10276 crystals",
        "version": "v1.0.0",
        "doi": "10.5281/zenodo.8190849",
        "license": "CC-BY-4.0",
        "source_repository": "https://github.com/wengroup/matten",
        "source_commit": "0a04f1cb27a7c22451f375683a85cfdb28e9e7ab",
        "local_file": path.relative_to(ROOT).as_posix(),
        "bytes": path.stat().st_size,
        "md5": digest(path, "md5"),
        "sha256": digest(path, "sha256"),
        "records": len(columns["structure"]),
        "columns": list(columns),
        "split_counts": dict(counts),
        "split_indices": {
            name: [int(i) for i, value in columns["split"].items() if value == name]
            for name in ("train", "val", "test")
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    parser.add_argument(
        "--only", choices=("all", "dielectric", "elastic", "matten"), default="all"
    )
    arguments = parser.parse_args()
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {}
    if arguments.only in {"all", "dielectric"}:
        outputs["jarvis_dielectric.json"] = jarvis_manifest(
            "jarvis_diele_piezo.pkl", "dielectric"
        )
    if arguments.only in {"all", "elastic"}:
        outputs["jarvis_elastic.json"] = jarvis_manifest("jarvis_elastic.pkl", "elastic")
    if arguments.only in {"all", "matten"}:
        outputs["matten_elastic.json"] = matten_manifest()
    for filename, content in outputs.items():
        (arguments.output_dir / filename).write_text(
            json.dumps(content, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    print(
        json.dumps(
            {
                name: {
                    "records": value.get("filtered_records", value.get("records")),
                    "split_counts": value["split_counts"],
                    "sha256": value["sha256"],
                }
                for name, value in outputs.items()
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
