from __future__ import annotations

import hashlib
import json
import pickle

import pytest
import torch

from src.data import (
    TensorSample,
    TrainingUnit,
    load_five_structure_smoke,
    load_training_dataset,
    voigt_stiffness_to_cartesian,
)
from src.heads import irreps_to_cartesian


def _structure(elements=("Si", "O")):
    return {
        "lattice_mat": [[4.0, 0.0, 0.0], [0.0, 4.2, 0.0], [0.0, 0.0, 4.4]],
        "coords": [[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]],
        "elements": list(elements),
        "cartesian": False,
    }


def _write_manifest_resource(tmp_path, name, payload, suffix, extra):
    manifests = tmp_path / "manifests"
    raw = tmp_path / "raw"
    manifests.mkdir(exist_ok=True)
    raw.mkdir(exist_ok=True)
    resource = raw / f"{name}.{suffix}"
    if suffix == "pkl":
        resource.write_bytes(pickle.dumps(payload))
    else:
        resource.write_text(json.dumps(payload), encoding="utf-8")
    manifest = {
        "local_file": f"raw/{resource.name}",
        "bytes": resource.stat().st_size,
        "sha256": hashlib.sha256(resource.read_bytes()).hexdigest(),
        **extra,
    }
    path = manifests / f"{name}.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def _jarvis_splits(ids):
    return {
        "train": [{"jarvis_id": value} for value in ids[:3]],
        "validation": [{"jarvis_id": ids[3]}],
        "test": [{"jarvis_id": ids[4]}],
    }


def test_jarvis_dielectric_and_elastic_loaders_preserve_published_3_1_1(tmp_path) -> None:
    ids = [f"JVASP-{index}" for index in range(5)]
    dielectric_records = [
        {
            "JARVIS_ID": sample_id,
            "atoms": _structure(),
            "dielectric": (torch.eye(3) * (index + 1)).tolist(),
        }
        for index, sample_id in enumerate(ids)
    ]
    dielectric_manifest = _write_manifest_resource(
        tmp_path,
        "dielectric",
        dielectric_records,
        "pkl",
        {"splits": _jarvis_splits(ids)},
    )
    dielectric = load_five_structure_smoke(
        TrainingUnit("jarvis_tensor", "dielectric"),
        manifest_path=dielectric_manifest,
    )
    assert [sample.sample_id for sample in dielectric] == ids
    assert all(sample.target_unit == "dimensionless" for sample in dielectric)
    assert torch.allclose(
        irreps_to_cartesian(dielectric[2].target_coefficients, "dielectric"),
        dielectric[2].target_cartesian,
        atol=1e-10,
    )

    voigt_kbar = torch.arange(36, dtype=torch.float64).reshape(6, 6)
    voigt_kbar = 0.5 * (voigt_kbar + voigt_kbar.T)
    elastic_records = [
        {
            "JARVIS_ID": sample_id,
            "atoms": _structure(),
            "elastic_total_kbar": voigt_kbar.tolist(),
        }
        for sample_id in ids
    ]
    elastic_manifest = _write_manifest_resource(
        tmp_path,
        "elastic",
        elastic_records,
        "pkl",
        {"splits": _jarvis_splits(ids)},
    )
    elastic = load_training_dataset(
        TrainingUnit("jarvis_tensor", "elastic"),
        manifest_path=elastic_manifest,
        sample_ids=ids,
    )
    assert elastic[0].target_cartesian.shape == (3, 3, 3, 3)
    assert elastic[0].target_unit == "GPa"
    assert elastic[0].target_cartesian[1, 2, 1, 2] == voigt_kbar[3, 3] / 10.0
    assert torch.allclose(
        irreps_to_cartesian(elastic[0].target_coefficients, "elastic"),
        elastic[0].target_cartesian,
        atol=1e-10,
    )


def test_matten_loader_reads_column_oriented_structure_and_native_tensor(tmp_path) -> None:
    base = _structure()
    structure = {
        "lattice": {"matrix": base["lattice_mat"]},
        "sites": [
            {"species": [{"element": symbol, "occu": 1}], "abc": coordinates}
            for symbol, coordinates in zip(base["elements"], base["coords"])
        ],
    }
    tensor = voigt_stiffness_to_cartesian(torch.eye(6, dtype=torch.float64)).tolist()
    payload = {
        "structure": {str(index): structure for index in range(5)},
        "formula_pretty": {str(index): "SiO" for index in range(5)},
        "elastic_tensor": {str(index): tensor for index in range(5)},
    }
    manifest = _write_manifest_resource(
        tmp_path,
        "matten",
        payload,
        "json",
        {"split_indices": {"train": [0, 1, 2], "val": [3], "test": [4]}},
    )
    samples = load_five_structure_smoke(
        TrainingUnit("matten", "elastic"), manifest_path=manifest
    )
    assert [sample.sample_id for sample in samples] == [f"matten-{i}" for i in range(5)]
    assert samples[0].source["formula"] == "SiO"
    assert torch.equal(samples[0].atomic_numbers, torch.tensor([14, 8]))


def test_bec_loader_preserves_node_scope_site_order_source_and_saved_split(tmp_path) -> None:
    manifests = tmp_path / "manifests"
    processed = tmp_path / "processed"
    manifests.mkdir()
    processed.mkdir()
    ids = [f"JVASP-{100 + index}" for index in range(5)]
    records = []
    for index, sample_id in enumerate(ids):
        records.append(
            {
                "sample_id": sample_id,
                "elements": ["O", "Si"],
                "lattice_angstrom": torch.eye(3).mul(4 + index).tolist(),
                "fractional_coordinates": [[0.25, 0.0, 0.0], [0.0, 0.5, 0.0]],
                "born_effective_charge_e": torch.arange(18).reshape(2, 3, 3).tolist(),
                "source": {"archive_md5": f"md5-{index}", "vasprun_sha256": f"sha-{index}"},
                "quality": {"natoms": 2, "acoustic_sum_rule_frobenius_e": 1.5},
            }
        )
    resource = processed / "bec.jsonl"
    resource.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8"
    )
    output = {
        "path": "data/processed/bec.jsonl",
        "size_bytes": resource.stat().st_size,
        "sha256": hashlib.sha256(resource.read_bytes()).hexdigest(),
        "splits": {
            "seed": 20260911,
            "source": "seeded_8_1_1",
            "train": ids[:3],
            "validation": [ids[3]],
            "test": [ids[4]],
        },
    }
    manifest = manifests / "bec.json"
    manifest.write_text(json.dumps({"processed_output": output}), encoding="utf-8")
    samples = load_five_structure_smoke(
        TrainingUnit("jarvis_dfpt", "bec"), manifest_path=manifest
    )
    assert [sample.sample_id for sample in samples] == ids
    assert samples[0].target_cartesian.shape == (2, 3, 3)
    assert samples[0].target_coefficients.shape == (2, 9)
    assert torch.equal(samples[0].atomic_numbers, torch.tensor([8, 14]))
    assert samples[0].source["archive_md5"] == "md5-0"
    assert samples[0].source["quality"]["natoms"] == 2
    graph = samples[0].build_graph(4.5)
    assert graph.num_nodes == 2
    assert torch.equal(graph.atomic_numbers, samples[0].atomic_numbers)


def test_resource_gate_precedes_pickle_and_pending_bec_fails_closed(tmp_path) -> None:
    manifest = _write_manifest_resource(
        tmp_path,
        "invalid",
        "not a pickle",
        "json",
        {
            "splits": _jarvis_splits([f"JVASP-{index}" for index in range(5)]),
        },
    )
    raw = json.loads(manifest.read_text(encoding="utf-8"))
    raw["sha256"] = "0" * 64
    manifest.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256"):
        load_training_dataset(
            TrainingUnit("jarvis_tensor", "dielectric"), manifest_path=manifest
        )

    pending = tmp_path / "manifests" / "pending.json"
    pending.write_text(
        json.dumps({"processed_output": {"path": "data/processed/missing.jsonl"}}),
        encoding="utf-8",
    )
    with pytest.raises(FileNotFoundError, match="not finalized"):
        load_training_dataset(TrainingUnit("jarvis_dfpt", "bec"), manifest_path=pending)


def test_tensor_sample_rejects_global_target_for_bec() -> None:
    with pytest.raises(ValueError, match="scope or shape"):
        TensorSample(
            sample_id="bad",
            unit=TrainingUnit("jarvis_dfpt", "bec"),
            lattice=torch.eye(3),
            fractional_positions=torch.zeros(2, 3),
            atomic_numbers=torch.tensor([1, 1]),
            target_cartesian=torch.zeros(3, 3),
            target_coefficients=torch.zeros(2, 9),
            target_unit="elementary_charge",
            source={},
        )
