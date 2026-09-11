from __future__ import annotations

import copy

import pytest
import spglib
import torch

from src.data import StructureCandidate
from src.evaluation import (
    audit_backbone_graph_automorphism,
    classify_candidate,
    point_group_smoke_schedule,
    select_point_group_fixtures,
    validate_point_group_fixture_manifest,
)
from src.configs import enumerate_architecture_configs
from src.graphs import PeriodicGraph, build_periodic_graph
from src.symmetry import canonicalize_structure
from src.symmetry import PointGroupRegistry


def _candidate(sample_id: str, atoms: int = 1) -> StructureCandidate:
    fractional = torch.stack(
        [torch.tensor([0.071 + 0.03 * index, 0.183, 0.297]) for index in range(atoms)]
    )
    return StructureCandidate(
        sample_id=sample_id,
        source_dataset="jarvis_tensor__dielectric",
        source_manifest_sha256="a" * 64,
        lattice=torch.tensor(
            [[4.1, 0.0, 0.0], [0.3, 4.7, 0.0], [0.2, 0.4, 5.3]],
            dtype=torch.float64,
        ),
        fractional_positions=fractional.double(),
        atomic_numbers=torch.full((atoms,), 14, dtype=torch.long),
    )


def test_selector_is_complete_deterministic_and_checksum_protected() -> None:
    registry = PointGroupRegistry()
    candidates = [_candidate(f"fixture-{index}") for index, _ in enumerate(registry)]
    mapping = {}
    for candidate, group in zip(candidates, registry):
        space_group = spglib.get_spacegroup_type(group.representative_hall_number)
        assert space_group is not None
        mapping[candidate.sample_id] = (
            group.symbol,
            int(space_group.number),
            group.representative_hall_number,
        )
    larger_duplicate = _candidate("larger-duplicate", atoms=2)
    first_group = next(iter(registry))
    first_type = spglib.get_spacegroup_type(first_group.representative_hall_number)
    assert first_type is not None
    mapping[larger_duplicate.sample_id] = (
        first_group.symbol,
        int(first_type.number),
        first_group.representative_hall_number,
    )

    manifest = select_point_group_fixtures(
        [larger_duplicate, *reversed(candidates)],
        classifier=lambda candidate: mapping[candidate.sample_id],
    )
    validate_point_group_fixture_manifest(manifest, redetect=False)
    assert len(manifest["fixtures"]) == 32
    assert manifest["fixtures"][0]["sample_id"] == "fixture-0"
    assert [row["point_group"] for row in manifest["fixtures"]] == [
        group.symbol for group in registry
    ]

    tampered = copy.deepcopy(manifest)
    tampered["fixtures"][0]["atomic_numbers"][0] = 8
    with pytest.raises(ValueError, match="checksum"):
        validate_point_group_fixture_manifest(tampered, redetect=False)


def test_selector_rejects_missing_group_and_default_classifier_redetects_p1() -> None:
    registry = PointGroupRegistry()
    candidates = [_candidate(f"fixture-{index}") for index, _ in enumerate(registry)]
    mapping = {
        candidate.sample_id: (
            group.symbol,
            int(spglib.get_spacegroup_type(group.representative_hall_number).number),
            group.representative_hall_number,
        )
        for candidate, group in zip(candidates, registry)
    }
    with pytest.raises(ValueError, match="do not cover"):
        select_point_group_fixtures(
            candidates[:-1], classifier=lambda candidate: mapping[candidate.sample_id]
        )

    asymmetric = StructureCandidate(
        sample_id="asymmetric",
        source_dataset="matten__elastic",
        source_manifest_sha256="b" * 64,
        lattice=torch.tensor(
            [[4.1, 0.0, 0.0], [0.3, 4.7, 0.0], [0.2, 0.4, 5.3]],
            dtype=torch.float64,
        ),
        fractional_positions=torch.tensor(
            [[0.071, 0.183, 0.297], [0.337, 0.419, 0.613], [0.727, 0.811, 0.139]],
            dtype=torch.float64,
        ),
        atomic_numbers=torch.tensor([14, 8, 6]),
    )
    symbol, space_group, hall = classify_candidate(asymmetric)
    assert symbol == "1"
    assert space_group == 1
    assert hall == 1


def test_58_row_schedule_covers_every_pg_with_pge_and_all_26_configs() -> None:
    schedule = point_group_smoke_schedule()
    configs = enumerate_architecture_configs()
    assert len(schedule) == 58
    assert [row["fixture_index"] for row in schedule[:32]] == list(range(32))
    assert all("PGE" in configs[row["variant_index"]].branch for row in schedule[:32])
    assert {row["variant_index"] for row in schedule[32:]} == set(range(26))
    assert {row["backbone"] for row in schedule} == {
        "mace",
        "grace",
        "dpa4",
        "equiformerv2",
    }
    assert {row["task"] for row in schedule} == {"dielectric", "elastic", "bec"}


def test_periodic_edge_multiset_automorphism_detects_missing_image() -> None:
    cell = 2.0 * torch.eye(3, dtype=torch.float64)
    positions = torch.zeros(1, 3, dtype=torch.float64)
    numbers = torch.tensor([14])
    canonical = canonicalize_structure(positions, cell, numbers)
    graph = build_periodic_graph(positions, cell, numbers, 2.1)
    audit_backbone_graph_automorphism(graph, canonical.symmetry)

    keep = torch.arange(graph.num_edges - 1)
    broken = PeriodicGraph(
        positions=graph.positions,
        cell=graph.cell,
        atomic_numbers=graph.atomic_numbers,
        node_batch=graph.node_batch,
        edge_index=graph.edge_index[:, keep],
        cell_shifts=graph.cell_shifts[keep],
        edge_vectors=graph.edge_vectors[keep],
        edge_distances=graph.edge_distances[keep],
        cutoff=graph.cutoff,
    )
    with pytest.raises(AssertionError, match="not symmetry closed"):
        audit_backbone_graph_automorphism(broken, canonical.symmetry)
