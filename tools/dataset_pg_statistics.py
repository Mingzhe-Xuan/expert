from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import pickle
from typing import Any, Iterable, Mapping

import numpy as np
import spglib


ROOT = Path(__file__).resolve().parents[1]
SYMPREC = 1.0e-5
ANGLE_TOLERANCE = -1.0


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _verify(path: Path, manifest: Mapping[str, Any]) -> None:
    if path.stat().st_size != int(manifest["bytes"]):
        raise ValueError(f"size mismatch for {path}")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != manifest["sha256"]:
        raise ValueError(f"SHA-256 mismatch for {path}")


def _species_ids(elements: Iterable[str]) -> np.ndarray:
    identities: dict[str, int] = {}
    return np.asarray(
        [identities.setdefault(str(element), len(identities) + 1) for element in elements],
        dtype=np.int32,
    )


def _point_group(lattice: Any, fractional: Any, elements: Iterable[str]) -> str:
    dataset = spglib.get_symmetry_dataset(
        (
            np.asarray(lattice, dtype=np.float64),
            np.asarray(fractional, dtype=np.float64) % 1.0,
            _species_ids(elements),
        ),
        symprec=SYMPREC,
        angle_tolerance=ANGLE_TOLERANCE,
    )
    if dataset is None:
        raise ValueError("spglib could not classify a structure")
    return str(dataset.pointgroup).replace(" ", "")


def _jarvis_structure(raw: Mapping[str, Any]) -> tuple[Any, Any, Iterable[str]]:
    atoms = raw["atoms"]
    lattice = np.asarray(atoms["lattice_mat"], dtype=np.float64)
    coordinates = np.asarray(atoms["coords"], dtype=np.float64)
    if atoms.get("cartesian", False):
        coordinates = np.linalg.solve(lattice.T, coordinates.T).T
    return lattice, coordinates, atoms["elements"]


def _matten_structure(raw: Mapping[str, Any]) -> tuple[Any, Any, Iterable[str]]:
    elements: list[str] = []
    fractional: list[Any] = []
    for site in raw["sites"]:
        species = site["species"]
        if len(species) != 1 or float(species[0].get("occu", 0.0)) != 1.0:
            raise ValueError("disordered MatTen site is outside the training contract")
        elements.append(str(species[0]["element"]))
        fractional.append(site["abc"])
    return raw["lattice"]["matrix"], fractional, elements


def _distribution_metrics(counts: Mapping[str, int], groups: list[str]) -> dict[str, Any]:
    values = np.asarray([counts.get(group, 0) for group in groups], dtype=np.float64)
    total = int(values.sum())
    nonzero = values[values > 0]
    if total == 0:
        return {
            "samples": 0,
            "covered_point_groups": 0,
            "zero_point_groups": len(groups),
        }
    probabilities = values / total
    positive = probabilities[probabilities > 0]
    entropy = -float(np.sum(positive * np.log(positive)))
    mean = float(values.mean())
    return {
        "samples": total,
        "covered_point_groups": int(nonzero.size),
        "zero_point_groups": int((values == 0).sum()),
        "minimum_nonzero": int(nonzero.min()),
        "maximum": int(values.max()),
        "maximum_to_minimum_nonzero": float(values.max() / nonzero.min()),
        "maximum_to_uniform_mean": float(values.max() / mean),
        "coefficient_of_variation_32": float(values.std() / mean),
        "normalized_entropy_32": float(entropy / math.log(len(groups))),
        "effective_point_groups": float(math.exp(entropy)),
        "largest_group_share": float(values.max() / total),
    }


def _jarvis_dataset(manifest_name: str, key: str) -> dict[str, Any]:
    manifest_path = ROOT / "data" / "manifests" / manifest_name
    manifest = _read_json(manifest_path)
    resource = ROOT / "data" / manifest["local_file"]
    _verify(resource, manifest)
    with resource.open("rb") as stream:
        records = pickle.load(stream)  # noqa: S301 -- verified frozen upstream resource
    split_by_id = {
        str(row["jarvis_id"]): split
        for split in ("train", "validation", "test")
        for row in manifest["splits"][split]
    }
    counts = {split: Counter() for split in ("train", "validation", "test")}
    seen: set[str] = set()
    for raw in records:
        sample_id = str(raw["JARVIS_ID"])
        split = split_by_id.get(sample_id)
        if split is None:
            continue
        if sample_id in seen:
            raise ValueError(f"duplicate JARVIS ID {sample_id}")
        seen.add(sample_id)
        counts[split][_point_group(*_jarvis_structure(raw))] += 1
    if seen != set(split_by_id):
        raise ValueError(f"{key} split/resource membership mismatch")
    return {
        "key": key,
        "label": manifest["dataset"],
        "manifest": str(manifest_path.relative_to(ROOT)).replace("\\", "/"),
        "sample_scope": "frozen train/validation/test IDs; pre-filter and unused records excluded",
        "counts": counts,
    }


def _matten_dataset() -> dict[str, Any]:
    manifest_path = ROOT / "data" / "manifests" / "matten_elastic.json"
    manifest = _read_json(manifest_path)
    resource = ROOT / "data" / manifest["local_file"]
    _verify(resource, manifest)
    raw = _read_json(resource)
    counts = {split: Counter() for split in ("train", "validation", "test")}
    source_names = {"train": "train", "validation": "val", "test": "test"}
    seen: set[int] = set()
    for split, source_name in source_names.items():
        for raw_index in manifest["split_indices"][source_name]:
            index = int(raw_index)
            if index in seen:
                raise ValueError(f"duplicate MatTen split index {index}")
            seen.add(index)
            counts[split][_point_group(*_matten_structure(raw["structure"][str(index)]))] += 1
    if len(seen) != int(manifest["records"]):
        raise ValueError("MatTen split indices do not cover the frozen resource")
    return {
        "key": "matten_elastic",
        "label": manifest["dataset"],
        "manifest": str(manifest_path.relative_to(ROOT)).replace("\\", "/"),
        "sample_scope": "published train/val/test split",
        "counts": counts,
    }


def _bec_snapshot() -> dict[str, Any]:
    path = ROOT / "data" / "processed" / "jarvis_dfpt_bec_10sample.jsonl"
    counts = {"local_available": Counter()}
    ids: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        raw = json.loads(line)
        sample_id = str(raw["sample_id"])
        if sample_id in ids:
            raise ValueError(f"duplicate BEC sample ID {sample_id}")
        ids.add(sample_id)
        counts["local_available"][_point_group(
            raw["lattice_angstrom"], raw["fractional_coordinates"], raw["elements"]
        )] += 1
    return {
        "key": "jarvis_dfpt_bec_local_snapshot",
        "label": "JARVIS-DFPT BEC local successful-extraction snapshot",
        "manifest": "data/manifests/jarvis_dfpt_bec.json",
        "sample_scope": "10 local successful records; full extraction and frozen split pending",
        "counts": counts,
    }


def _format_metric(value: Any, digits: int = 3) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _render_markdown(report: Mapping[str, Any]) -> str:
    groups = report["point_groups"]
    lines = [
        "# Local dataset point-group frequency and expert-balance audit",
        "",
        f"Generated: {report['generated_date']} (Asia/Shanghai).",
        "",
        "## Scope and interpretation",
        "",
        "Counts use each frozen dataset split and spglib with `symprec=1e-5`, "
        "`angle_tolerance=-1`, matching model symmetry discovery. “Expert balance” below means "
        "the load of the **current-PG expert** on the training split. Exact routed load cannot yet "
        "be computed: parent experts require a validated, material-specific Hall-level "
        "`ParentDAGSpec`; the class-level subgroup lattice must not be substituted for it.",
        "",
        "JARVIS-DFPT BEC is explicitly excluded from training-balance conclusions because its "
        "full processed output and split are pending; only the 10 successful local records are shown.",
        "",
        "## Training-split balance summary",
        "",
        "| Dataset | Train N | PG coverage | Zero PGs | Min nonzero | Max | Max/min | Max/uniform | CV (32 PGs) | Entropy / log 32 | Effective PGs | Largest share | Assessment |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for dataset in report["datasets"]:
        metrics = dataset.get("train_metrics")
        if metrics is None:
            lines.append(
                f"| {dataset['key']} | — | — | — | — | — | — | — | — | — | — | — | "
                "不可判定（正式 split 缺失） |"
            )
            continue
        assessment = (
            "严重不平衡"
            if metrics["zero_point_groups"] > 0
            or metrics["maximum_to_minimum_nonzero"] >= 10
            else "不平衡"
            if metrics["maximum_to_minimum_nonzero"] >= 3
            else "近似平衡"
        )
        lines.append(
            f"| {dataset['key']} | {metrics['samples']} | "
            f"{metrics['covered_point_groups']}/32 | {metrics['zero_point_groups']} | "
            f"{metrics['minimum_nonzero']} | {metrics['maximum']} | "
            f"{_format_metric(metrics['maximum_to_minimum_nonzero'], 1)} | "
            f"{_format_metric(metrics['maximum_to_uniform_mean'], 2)} | "
            f"{_format_metric(metrics['coefficient_of_variation_32'])} | "
            f"{_format_metric(metrics['normalized_entropy_32'])} | "
            f"{_format_metric(metrics['effective_point_groups'], 1)} | "
            f"{100 * metrics['largest_group_share']:.1f}% | {assessment} |"
        )
    lines.extend(
        [
            "",
            "`Max/uniform` is the busiest expert count divided by `Train N / 32`; CV includes "
            "all 32 experts, including zero-count groups. Effective PGs is `exp(Shannon entropy)`.",
            "",
            "## Frequency × PG",
            "",
        ]
    )
    for dataset in report["datasets"]:
        splits = [split for split in dataset["counts"] if split != "total"]
        lines.extend(
            [
                f"### {dataset['key']}",
                "",
                dataset["label"] + ". " + dataset["sample_scope"] + ".",
                "",
                "| PG | " + " | ".join(splits) + " | total |",
                "|---|" + "---:|" * (len(splits) + 1),
            ]
        )
        for group in groups:
            values = [int(dataset["counts"][split][group]) for split in splits]
            lines.append(
                f"| {group} | " + " | ".join(str(value) for value in values) +
                f" | {int(dataset['counts']['total'][group])} |"
            )
        split_totals = [sum(dataset["counts"][split].values()) for split in splits]
        total_count = sum(dataset["counts"]["total"].values())
        lines.extend(
            [
                "| **Total** | " + " | ".join(f"**{value}**" for value in split_totals) +
                f" | **{total_count}** |",
                "",
            ]
        )
    lines.extend(
        [
            "## Consequences for expert training",
            "",
            "1. Uniform random sampling trains current-PG experts in direct proportion to the "
            "tables above; it does not produce balanced expert updates.",
            "2. Report both natural-distribution metrics and a balanced-training ablation. For the "
            "latter, use inverse-frequency or temperature-smoothed PG sampling **within the train "
            "split only**; never rebalance validation/test.",
            "3. A PG with very few structures cannot be repaired by oversampling alone. Prefer "
            "shared adaptation, hierarchical parameter sharing/regularization, and report per-PG "
            "metrics with confidence intervals; merge neither labels nor datasets silently.",
            "4. Recompute actual expert activation counts after material-specific parent DAGs exist. "
            "A parent expert may receive gradients from several child PGs, so current-PG frequency "
            "is a lower-bound view of routed load, not the final load profile.",
            "5. Finish BEC extraction and freeze its split before choosing BEC sampling weights.",
            "",
            "## Reproduction",
            "",
            "```text",
            "python tools/dataset_pg_statistics.py",
            "```",
            "",
            "Machine-readable counts and metrics: "
            "[dataset_pg_balance.json](dataset_pg_balance.json).",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    registry = _read_json(ROOT / "assets" / "docs" / "subgroup_chain.json")
    groups = [registry["point_groups"][str(index)]["hm_symbol"] for index in range(1, 33)]
    datasets = [
        _jarvis_dataset("jarvis_dielectric.json", "jarvis_dielectric"),
        _jarvis_dataset("jarvis_elastic.json", "jarvis_elastic"),
        _matten_dataset(),
        _bec_snapshot(),
    ]
    allowed = set(groups)
    for dataset in datasets:
        unexpected = set().union(*[set(value) for value in dataset["counts"].values()]) - allowed
        if unexpected:
            raise ValueError(f"unexpected point groups in {dataset['key']}: {sorted(unexpected)}")
        total = Counter()
        for counts in dataset["counts"].values():
            total.update(counts)
        dataset["counts"]["total"] = total
        dataset["metrics_by_split"] = {
            split: _distribution_metrics(counts, groups)
            for split, counts in dataset["counts"].items()
        }
        dataset["train_metrics"] = dataset["metrics_by_split"].get("train")
        dataset["counts"] = {
            split: {group: int(counts.get(group, 0)) for group in groups}
            for split, counts in dataset["counts"].items()
        }
    report = {
        "schema_version": 1,
        "generated_date": "2026-09-12",
        "spglib_version": spglib.__version__,
        "symprec": SYMPREC,
        "angle_tolerance": ANGLE_TOLERANCE,
        "point_groups": groups,
        "datasets": datasets,
    }
    output_dir = ROOT / "docs" / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "dataset_pg_balance.json"
    markdown_path = output_dir / "dataset_pg_balance.md"
    json_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    markdown_path.write_text(_render_markdown(report), encoding="utf-8")
    print(f"wrote {json_path.relative_to(ROOT)}")
    print(f"wrote {markdown_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
