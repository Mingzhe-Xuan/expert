from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch

from data.prepare_dtnet_dielectric import (
    download_official,
    normalize_record,
    official_dtnet_split,
    prepare_dtnet_dataset,
)
from src.data import TrainingUnit, load_five_structure_smoke, load_training_dataset


def _record(index: int) -> dict:
    electronic = [[2.0, 1.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 4.0]]
    ionic = [[1.0, 0.0, 0.0], [0.0, 1.5, 0.0], [0.0, 0.0, 2.0]]
    total = [
        [electronic[row][column] + ionic[row][column] for column in range(3)]
        for row in range(3)
    ]
    return {
        "structure": {
            "lattice": {
                "matrix": [[4.0, 0.0, 0.0], [0.0, 4.2, 0.0], [0.0, 0.0, 4.4]],
                "pbc": [True, True, True],
            },
            "sites": [
                {
                    "species": [{"element": "Si", "occu": 1}],
                    "abc": [0.0, 0.0, 0.0],
                },
                {
                    "species": [{"element": "O", "occu": 1}],
                    "abc": [1.25, 0.25, 0.25],
                },
            ],
        },
        "electronic": electronic,
        "ionic": ionic,
        "total": total,
        "band_gap": 1.0 + index,
    }


def _prepare_fixture(tmp_path: Path, count: int = 20):
    raw_dir = tmp_path / "raw" / "dtnet"
    processed_dir = tmp_path / "processed" / "dtnet"
    manifest_dir = tmp_path / "manifests"
    raw_dir.mkdir(parents=True)
    raw = raw_dir / "mp_dielectric.json"
    raw.write_text(
        json.dumps({f"mp-{index:02d}": _record(index) for index in range(count)}),
        encoding="utf-8",
    )
    digest = hashlib.sha256(raw.read_bytes()).hexdigest()
    output = processed_dir / "dielectric.jsonl"
    manifest = manifest_dir / "dtnet_dielectric.json"
    result = prepare_dtnet_dataset(
        raw,
        output,
        manifest,
        expected_input_sha256=digest,
        source_url="https://example.invalid/mp_dielectric.json",
        source_commit="fixture",
    )
    return raw, output, manifest, result


def test_official_split_reproduces_dtnet_sklearn_order() -> None:
    split = official_dtnet_split([f"mp-{index:02d}" for index in range(20)])
    assert split == {
        "train": [
            "mp-17", "mp-00", "mp-06", "mp-05", "mp-15", "mp-12", "mp-13", "mp-11",
            "mp-10", "mp-03", "mp-07", "mp-01", "mp-08", "mp-09", "mp-16", "mp-19",
        ],
        "validation": ["mp-18", "mp-04"],
        "test": ["mp-14", "mp-02"],
    }


def test_converter_preserves_raw_tensors_and_builds_symmetric_total_target(tmp_path) -> None:
    _, output, manifest_path, manifest = _prepare_fixture(tmp_path)
    assert manifest["split_counts"] == {"train": 16, "validation": 2, "test": 2}
    assert manifest["quality_summary"]["antisymmetric_max_abs"]["total"] == {
        "records_above_1e-8": 20,
        "maximum_abs": 1.0,
    }
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 20
    assert rows[0]["dielectric_raw"]["total"][0][1] == 1.0
    assert rows[0]["dielectric_raw"]["total"][1][0] == 0.0
    assert rows[0]["dielectric_symmetric"]["total"][0][1] == 0.5
    assert rows[0]["fractional_coordinates"][1] == [0.25, 0.25, 0.25]

    dataset = load_training_dataset(
        TrainingUnit("dtnet", "dielectric"), manifest_path=manifest_path
    )
    assert len(dataset) == 20
    sample = dataset.by_id("mp-00")
    assert sample.target_unit == "dimensionless"
    assert torch.equal(sample.atomic_numbers, torch.tensor([14, 8]))
    assert sample.target_cartesian[0, 1] == sample.target_cartesian[1, 0] == 0.5
    assert sample.source["dielectric_raw"]["total"][0][1] == 1.0
    smoke = load_five_structure_smoke(
        TrainingUnit("dtnet", "dielectric"), manifest_path=manifest_path
    )
    assert [sample.sample_id for sample in smoke] == [
        *manifest["splits"]["train"][:3],
        manifest["splits"]["validation"][0],
        manifest["splits"]["test"][0],
    ]


@pytest.mark.parametrize(
    "mutation,message",
    [
        (lambda row: row["structure"]["sites"][0]["species"][0].update(occu=0.5), "partial"),
        (lambda row: row["structure"]["lattice"].update(matrix=[[0.0] * 3] * 3), "invertible"),
        (lambda row: row["total"][0].__setitem__(0, 99.0), "inconsistent"),
        (
            lambda row: (
                row["electronic"][0].__setitem__(0, 101.0),
                row["total"][0].__setitem__(0, 102.0),
            ),
            "published.*filter",
        ),
        (lambda row: row.update(band_gap=float("nan")), "finite"),
    ],
)
def test_normalizer_rejects_invalid_source_records(mutation, message) -> None:
    row = _record(0)
    mutation(row)
    with pytest.raises(ValueError, match=message):
        normalize_record("mp-1", row)
    with pytest.raises(ValueError, match="Materials Project"):
        normalize_record("material-1", _record(0))


def test_converter_and_loader_fail_closed_on_hash_tampering_and_duplicate_keys(tmp_path) -> None:
    raw, output, manifest, _ = _prepare_fixture(tmp_path)
    with pytest.raises(ValueError, match="input SHA-256"):
        prepare_dtnet_dataset(raw, output, manifest, expected_input_sha256="0" * 64)
    with pytest.raises(ValueError, match="must be distinct"):
        prepare_dtnet_dataset(raw, raw, manifest, expected_input_sha256=None)

    output.write_text(output.read_text(encoding="utf-8") + "{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="size mismatch"):
        load_training_dataset(TrainingUnit("dtnet", "dielectric"), manifest_path=manifest)

    duplicate = tmp_path / "raw" / "dtnet" / "duplicate.json"
    duplicate.write_text(
        '{"mp-1":' + json.dumps(_record(1)) + ',"mp-1":' + json.dumps(_record(1)) + "}",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate JSON key"):
        prepare_dtnet_dataset(
            duplicate,
            tmp_path / "processed" / "dtnet" / "duplicate.jsonl",
            tmp_path / "manifests" / "duplicate.json",
            expected_input_sha256=hashlib.sha256(duplicate.read_bytes()).hexdigest(),
        )


def test_download_promotes_only_a_matching_payload(tmp_path) -> None:
    source = tmp_path / "source.json"
    source.write_bytes(b"official-dtnet-fixture")
    expected = hashlib.sha256(source.read_bytes()).hexdigest()
    destination = tmp_path / "download" / "mp_dielectric.json"
    download_official(destination, url=source.as_uri(), expected_sha256=expected)
    assert destination.read_bytes() == source.read_bytes()

    destination.write_bytes(b"keep-existing")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        download_official(destination, url=source.as_uri(), expected_sha256="0" * 64)
    assert destination.read_bytes() == b"keep-existing"
