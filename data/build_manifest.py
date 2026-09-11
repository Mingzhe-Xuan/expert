"""Validate benchmark downloads and freeze their split metadata."""

from __future__ import annotations

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


def split_gmtnet(records: list[tuple[int, dict]]) -> dict:
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
    return {
        name: [
            {
                "filtered_index": index,
                "raw_index": records[index][0],
                "jarvis_id": records[index][1]["JARVIS_ID"],
            }
            for index in indices
        ]
        for name, indices in positions.items()
    }


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
        records = [
            (i, x)
            for i, x in enumerate(raw)
            if x.get("elastic_total_kbar")
            and np.max(np.abs(np.asarray(x["elastic_total_kbar"]) / 10.0)) < 1500
        ]
        criterion = "elastic_total_kbar present; divide by 10 for GPa; max(abs) < 1500"
    splits = split_gmtnet(records)
    return {
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
    OUT.mkdir(parents=True, exist_ok=True)
    outputs = {
        "jarvis_dielectric.json": jarvis_manifest(
            "jarvis_diele_piezo.pkl", "dielectric"
        ),
        "jarvis_elastic.json": jarvis_manifest("jarvis_elastic.pkl", "elastic"),
        "matten_elastic.json": matten_manifest(),
    }
    for filename, content in outputs.items():
        (OUT / filename).write_text(
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
