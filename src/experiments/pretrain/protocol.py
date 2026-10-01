"""Pure-Python provenance and nested-sampling protocol."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import random

FRACTIONS = (25, 50, 75, 100)
SEEDS = (42, 43, 44)
TASKS = {"dielectric": "dielectric_total", "elastic": "elastic_stiffness"}
SPLITS = ("train", "validation", "test")
SAMPLING_SEED = 20261001


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


def load_records(task, *, root=Path("."), group_scope="all"):
    if task not in TASKS or group_scope not in ("all", "seven"):
        raise ValueError("unsupported task or point-group scope")
    manifest_path = root / "data/manifests/curated_tensors.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entry = manifest["artifacts"]["recommended/" + TASKS[task]]
    path = root / entry["path"]
    if path.stat().st_size != entry["size_bytes"] or sha256(path) != entry["sha256"]:
        raise ValueError("curated source checksum mismatch")
    groups = {"2/m", "mm2", "mmm", "4/mmm", "-3m", "-43m", "m-3m"}
    with path.open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    if len(rows) != entry["records"]:
        raise ValueError("curated source count mismatch")
    rows = [row for row in rows if row["provenance"]["source_dataset"] == "gmtnet"
            and (group_scope == "all" or row["point_group"] in groups)]
    ids = [row["record_id"] for row in rows]
    if not rows or len(ids) != len(set(ids)):
        raise ValueError("empty or duplicate source records")
    memberships = {}
    for row in rows:
        if row["split"] not in SPLITS or row["property_subtype"] != TASKS[task]:
            raise ValueError("invalid source task/split")
        group = row["duplicate_group"]
        if group in memberships and memberships[group] != row["split"]:
            raise ValueError("duplicate structure group leaks across splits")
        memberships[group] = row["split"]
    provenance = {"source_manifest": "data/manifests/curated_tensors.json", "source_sha256": entry["sha256"],
                  "source_dataset": "jarvis-dft", "group_scope": group_scope,
                  "task": task, "curation": "recommended; JARVIS source only; symmetry-projected labels",
                  "split_protocol": "preserved curated duplicate-group 8:1:1 split"}
    provenance["dataset_sha256"] = hashlib.sha256(json.dumps(
        {"source": entry["sha256"], "ids": ids, "scope": group_scope},
        sort_keys=True).encode()).hexdigest()
    return rows, provenance


def nested_ids(rows, fraction, *, sampling_seed=SAMPLING_SEED, smoke=False):
    if fraction not in FRACTIONS:
        raise ValueError("fraction must be 25, 50, 75 or 100")
    splits = {split: [row["record_id"] for row in rows if row["split"] == split]
              for split in SPLITS}
    flat = sum(splits.values(), [])
    if len(flat) != len(set(flat)) or any(not ids for ids in splits.values()):
        raise ValueError("splits must be nonempty and disjoint")
    # One permutation is shared by all fractions, models and training seeds.
    ordered = sorted(splits["train"])
    random.Random(sampling_seed).shuffle(ordered)
    size = max(1, math.floor(len(ordered) * fraction / 100))
    selected = set(ordered[:size])
    splits["train"] = [sid for sid in splits["train"] if sid in selected]
    if smoke:
        splits = {name: ids[:min(8, len(ids))] for name, ids in splits.items()}
    return splits


def crystal_system(space_group):
    if not 1 <= space_group <= 230:
        raise ValueError("space group outside 1..230")
    for upper, name in ((2, "Triclinic"), (15, "Monoclinic"), (74, "Orthorhombic"),
                        (142, "Tetragonal"), (167, "Trigonal"), (194, "Hexagonal"),
                        (230, "Cubic")):
        if space_group <= upper:
            return name


def grid():
    return [{"task": task, "fraction": fraction, "seed": seed, "model": model}
            for task in TASKS for fraction in FRACTIONS for seed in SEEDS
            for model in ("pretrain", "O(3)")]


def run_name(spec):
    model = "pretrain" if spec["model"] == "pretrain" else "o3"
    return f"{spec['task']}/{model}/p{spec['fraction']}/seed{spec['seed']}"
