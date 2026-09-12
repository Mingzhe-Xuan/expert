from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from data.build_manifest import gmtnet_elastic_protocol_filter, screen_elastic_records
from src.data import (
    TensorSample,
    TrainingUnit,
    load_five_structure_smoke,
    load_structure_candidates,
    load_training_dataset,
    voigt_stiffness_to_cartesian,
)
from src.heads import irreps_to_cartesian


ROOT = Path(__file__).resolve().parents[1]


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
    candidates = load_structure_candidates(
        TrainingUnit("jarvis_tensor", "dielectric"),
        manifest_path=dielectric_manifest,
    )
    assert [candidate.sample_id for candidate in candidates] == ids
    assert candidates[0].source_dataset == "jarvis_tensor__dielectric"

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


def test_gmtnet_elastic_protocol_filters_structurally_forbidden_entries() -> None:
    allowed = torch.zeros((6, 6), dtype=torch.float64)
    allowed[:3, :3] = torch.tensor(
        [[200.0, 100.0, 100.0], [100.0, 200.0, 100.0], [100.0, 100.0, 200.0]]
    )
    allowed[3, 3] = allowed[4, 4] = allowed[5, 5] = 50.0
    record = {
        "JARVIS_ID": "JVASP-cubic",
        "atoms": {
            "lattice_mat": (4.0 * torch.eye(3)).tolist(),
            "coords": [[0.0, 0.0, 0.0]],
            "elements": ["Si"],
            "cartesian": False,
        },
        "elastic_total_kbar": (10.0 * allowed).tolist(),
    }
    accepted, bits, projected = gmtnet_elastic_protocol_filter(record)
    support = torch.tensor(
        [bool(bits & (1 << index)) for index in range(36)]
    ).reshape(6, 6)
    assert accepted
    assert int(support.sum()) == 12
    assert support[:3, :3].all()
    assert support[3:, 3:].diag().all()
    assert torch.allclose(torch.from_numpy(projected), allowed)

    record["elastic_total_kbar"][0][3] = 0.1  # 0.01 GPa > official 1e-4 cutoff
    accepted, _, _ = gmtnet_elastic_protocol_filter(record)
    assert not accepted


def test_elastic_protocol_objects_import_in_a_fresh_process() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "from data.build_manifest import _elastic_protocol_objects; "
            "_elastic_protocol_objects()",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    assert completed.returncode == 0, completed.stderr


def test_elastic_protocol_parallel_orchestration_is_ordered_and_observable(
    monkeypatch, capsys
) -> None:
    class InlinePool:
        def __init__(self, *, max_workers):
            assert max_workers == 3

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def map(self, function, values, *, chunksize):
            assert chunksize == 16
            return map(function, values)

    def fake_filter(record):
        index = int(record["index"])
        if index == 4:
            raise RuntimeError("worker failure")
        return index % 2 == 0, 1 << index, torch.zeros((6, 6)).numpy()

    monkeypatch.setattr("data.build_manifest.ProcessPoolExecutor", InlinePool)
    monkeypatch.setattr("data.build_manifest.gmtnet_elastic_protocol_filter", fake_filter)
    screened = [(index + 10, {"index": index}) for index in range(4)]
    sequential = screen_elastic_records(screened, workers=1)
    parallel = screen_elastic_records(screened, workers=3, progress_every=2)
    assert parallel == sequential
    assert [(raw_index, bits) for raw_index, _, bits in parallel] == [(10, 1), (12, 4)]
    progress = [json.loads(line) for line in capsys.readouterr().err.splitlines()]
    assert progress == [
        {"accepted": 1, "completed": 2, "event": "elastic_protocol_screen", "total": 4},
        {"accepted": 2, "completed": 4, "event": "elastic_protocol_screen", "total": 4},
    ]
    with pytest.raises(RuntimeError, match="worker failure"):
        screen_elastic_records([(14, {"index": 4})], workers=3)
    with pytest.raises(ValueError, match="workers"):
        screen_elastic_records([], workers=0)
    with pytest.raises(ValueError, match="progress"):
        screen_elastic_records([], progress_every=-1)


@pytest.mark.parametrize(
    "argument, message",
    [("--workers", "positive"), ("--progress-every", "cannot be negative")],
)
def test_elastic_manifest_cli_rejects_invalid_parallel_options(argument, message) -> None:
    completed = subprocess.run(
        [sys.executable, "data/build_manifest.py", "--only", "dielectric", argument, "-1"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    assert completed.returncode == 2
    assert message in completed.stderr


def test_elastic_loader_applies_manifest_support_mask(tmp_path) -> None:
    ids = [f"JVASP-mask-{index}" for index in range(5)]
    voigt_kbar = torch.full((6, 6), 10.0, dtype=torch.float64)
    records = [
        {
            "JARVIS_ID": sample_id,
            "atoms": _structure(),
            "elastic_total_kbar": voigt_kbar.tolist(),
        }
        for sample_id in ids
    ]
    splits = _jarvis_splits(ids)
    for rows in splits.values():
        for row in rows:
            row["elastic_support_mask_bits"] = 1  # retain C11 only
    manifest = _write_manifest_resource(
        tmp_path,
        "masked-elastic",
        records,
        "pkl",
        {"splits": splits, "elastic_protocol_version": 1},
    )
    dataset = load_training_dataset(
        TrainingUnit("jarvis_tensor", "elastic"), manifest_path=manifest
    )
    tensor = dataset[0].target_cartesian
    assert tensor[0, 0, 0, 0] == 1.0
    assert tensor[1, 1, 1, 1] == 0.0


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
    candidates = load_structure_candidates(
        TrainingUnit("matten", "elastic"), manifest_path=manifest
    )
    assert len(candidates) == 5
    assert candidates[3].sample_id == "matten-3"


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
