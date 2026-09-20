from __future__ import annotations

from dataclasses import replace

import pytest
import torch

import src.symmetry.parent_detection as parent_detection

from src.cli.reduced_cgcnn_full_pg_train import ARCHITECTURE
from src.cli.reduced_protocol import canonical_expert_point_groups
from src.features import CGCNN_SOURCE_LAYOUT, cgcnn_node_features
from src.graphs import build_periodic_graph
from src.heads import TARGET_LAYOUTS
from src.models import CGCNNFeatureTensorModel
from src.symmetry import (
    canonicalize_structure,
    discover_material_parent_routing,
    load_parent_routing_cache,
    parent_routing_coverage,
    routing_from_payload,
    routing_to_payload,
    save_parent_routing_cache,
    PointGroupAncestorDAG,
)
from src.training import FrozenFeatureExample, collate_frozen_examples
from src.training.benchmark import _predict


def _distorted_tetragonal_parent_fixture():
    positions = torch.zeros((1, 3), dtype=torch.float64)
    cell = torch.diag(torch.tensor([3.0, 3.01, 3.2], dtype=torch.float64))
    atomic_numbers = torch.tensor([14], dtype=torch.long)
    canonical = canonicalize_structure(positions, cell, atomic_numbers)
    routing = discover_material_parent_routing(
        "material-1",
        canonical.canonical_positions,
        canonical.canonical_cell,
        atomic_numbers,
        canonical.symmetry,
    )
    return canonical, routing


def test_relaxed_detection_builds_checked_material_parent_dag() -> None:
    canonical, routing = _distorted_tetragonal_parent_fixture()
    assert canonical.symmetry.current_point_group == "mmm"
    assert len(routing.dag.embeddings) == 1
    parent = routing.dag.embeddings[0]
    assert parent.child_hall_number == canonical.symmetry.hall_number
    assert routing.detection_symprecs[parent.parent_hall_number] == 1.0e-2
    assert routing.residuals[canonical.symmetry.hall_number] == 0.0
    assert routing.residuals[parent.parent_hall_number] > 0.0
    assert parent.checksum == parent.payload_checksum()
    restored = routing_from_payload(routing_to_payload(routing))
    assert restored == routing
    coverage = parent_routing_coverage((routing,))
    assert coverage["coverage_percent"] == 100.0
    assert coverage["parent_point_group_counts"] == {"4/mmm": 1}


def test_invalid_relaxed_parent_candidate_is_rejected_fail_closed(monkeypatch) -> None:
    positions = torch.zeros((1, 3), dtype=torch.float64)
    cell = torch.diag(torch.tensor([3.0, 3.01, 3.2], dtype=torch.float64))
    atomic_numbers = torch.tensor([14], dtype=torch.long)
    canonical = canonicalize_structure(positions, cell, atomic_numbers)

    def reject_candidate(**_):
        raise ValueError("affine operation group is not multiplication closed")

    monkeypatch.setattr(parent_detection, "_embedding", reject_candidate)
    routing = discover_material_parent_routing(
        "invalid-candidate",
        canonical.canonical_positions,
        canonical.canonical_cell,
        atomic_numbers,
        canonical.symmetry,
    )
    assert routing.dag.embeddings == ()
    assert routing.residuals == {canonical.symmetry.hall_number: 0.0}
    assert parent_routing_coverage((routing,))["coverage_percent"] == 0.0


def test_parent_detection_prioritizes_cached_hall_setting(monkeypatch) -> None:
    positions = torch.zeros((1, 3), dtype=torch.float64)
    cell = torch.diag(torch.tensor([3.0, 3.01, 3.2], dtype=torch.float64))
    atomic_numbers = torch.tensor([14], dtype=torch.long)
    canonical = canonicalize_structure(positions, cell, atomic_numbers)
    original = parent_detection.spglib.get_symmetry_dataset
    calls = []

    def record_call(spglib_cell, **kwargs):
        calls.append(kwargs.get("hall_number"))
        if kwargs.get("hall_number") is None and kwargs["symprec"] == 1.0e-5:
            raise AssertionError("automatic base Hall selection must not run first")
        return original(spglib_cell, **kwargs)

    monkeypatch.setattr(parent_detection.spglib, "get_symmetry_dataset", record_call)
    routing = discover_material_parent_routing(
        "cached-hall-first",
        canonical.canonical_positions,
        canonical.canonical_cell,
        atomic_numbers,
        canonical.symmetry,
    )
    assert calls[0] == canonical.symmetry.hall_number
    assert routing.dag.current_hall_number == canonical.symmetry.hall_number


def test_unreproduced_cached_child_falls_back_to_current_only(monkeypatch) -> None:
    positions = torch.zeros((1, 3), dtype=torch.float64)
    cell = torch.diag(torch.tensor([3.0, 3.01, 3.2], dtype=torch.float64))
    atomic_numbers = torch.tensor([14], dtype=torch.long)
    canonical = canonicalize_structure(positions, cell, atomic_numbers)
    original = parent_detection.spglib.get_symmetry_dataset
    cubic_cell = (
        torch.eye(3, dtype=torch.float64).numpy(),
        torch.zeros((1, 3), dtype=torch.float64).numpy(),
        atomic_numbers.numpy(),
    )
    mismatched = original(cubic_cell, symprec=1.0e-5, angle_tolerance=-1.0)
    calls = []

    def fail_cached_child(spglib_cell, **kwargs):
        calls.append(kwargs.get("hall_number"))
        return mismatched

    monkeypatch.setattr(parent_detection.spglib, "get_symmetry_dataset", fail_cached_child)
    routing = discover_material_parent_routing(
        "unreproduced-child",
        canonical.canonical_positions,
        canonical.canonical_cell,
        atomic_numbers,
        canonical.symmetry,
    )
    assert calls == [canonical.symmetry.hall_number]
    assert routing.dag.embeddings == ()
    assert routing.residuals == {canonical.symmetry.hall_number: 0.0}
    assert routing.detection_symprecs == {canonical.symmetry.hall_number: 1.0e-5}


def test_inconsistent_cached_hall_metadata_remains_fatal() -> None:
    positions = torch.zeros((1, 3), dtype=torch.float64)
    cell = torch.diag(torch.tensor([3.0, 3.01, 3.2], dtype=torch.float64))
    atomic_numbers = torch.tensor([14], dtype=torch.long)
    canonical = canonicalize_structure(positions, cell, atomic_numbers)
    inconsistent = replace(canonical.symmetry, current_point_group="1")
    try:
        discover_material_parent_routing(
            "inconsistent-cache",
            canonical.canonical_positions,
            canonical.canonical_cell,
            atomic_numbers,
            inconsistent,
        )
    except ValueError as error:
        assert "internally inconsistent" in str(error)
    else:
        raise AssertionError("inconsistent cached Hall metadata must remain fatal")


def test_parent_routing_cache_is_exact_and_rejects_dataset_drift(tmp_path) -> None:
    _, routing = _distorted_tetragonal_parent_fixture()
    path = tmp_path / "parents.pt"
    digest = "a" * 64
    save_parent_routing_cache(
        path, (routing,), sample_ids=("material-1",), dataset_sha256=digest
    )
    assert load_parent_routing_cache(
        path, sample_ids=("material-1",), dataset_sha256=digest
    ) == (routing,)
    try:
        load_parent_routing_cache(
            path, sample_ids=("material-1",), dataset_sha256="b" * 64
        )
    except ValueError as error:
        assert "dataset_sha256 mismatch" in str(error)
    else:
        raise AssertionError("dataset drift must invalidate a parent routing cache")


def test_cgcnn_batch_routes_current_and_parent_experts_and_backpropagates() -> None:
    canonical, routing = _distorted_tetragonal_parent_fixture()
    graph = build_periodic_graph(
        canonical.canonical_positions.float(), canonical.canonical_cell.float(),
        canonical.atomic_numbers, cutoff=4.0,
    )
    example = FrozenFeatureExample(
        sample_id="material-1",
        features=cgcnn_node_features(graph.atomic_numbers),
        graph=graph,
        symmetry=replace(
            canonical.symmetry,
            canonical_frame=canonical.symmetry.canonical_frame.float(),
            rotations=canonical.symmetry.rotations.float(),
            translations=canonical.symmetry.translations.float(),
        ),
        target_coefficients=torch.zeros((1, TARGET_LAYOUTS["dielectric"].dimension)),
        target_cartesian=torch.zeros((1, 3, 3)),
        parent_dag=routing.dag,
        parent_residuals=routing.residuals,
    )
    groups = canonical_expert_point_groups((example,), (example,), (example,))
    assert groups == ("mmm", "4/mmm")
    model = CGCNNFeatureTensorModel(ARCHITECTURE, "dielectric", groups)
    batch = collate_frozen_examples((example,), CGCNN_SOURCE_LAYOUT, device="cpu")
    prediction = _predict(model, batch)
    assert prediction.raw_cartesian.shape == (1, 3, 3)
    prediction.raw_cartesian.square().sum().backward()
    assert model.downstream.routing_gate.log_sigma.grad is not None


def test_current_only_batch_keeps_legacy_three_argument_forward() -> None:
    class LegacyModel(torch.nn.Module):
        def forward(self, features, graph, symmetries):
            return len(symmetries)

    canonical, _ = _distorted_tetragonal_parent_fixture()
    graph = build_periodic_graph(
        canonical.canonical_positions.float(), canonical.canonical_cell.float(),
        canonical.atomic_numbers, cutoff=4.0,
    )
    example = FrozenFeatureExample(
        "material-1",
        cgcnn_node_features(graph.atomic_numbers),
        graph,
        replace(
            canonical.symmetry,
            canonical_frame=canonical.symmetry.canonical_frame.float(),
            rotations=canonical.symmetry.rotations.float(),
            translations=canonical.symmetry.translations.float(),
        ),
        torch.zeros((1, TARGET_LAYOUTS["dielectric"].dimension)),
        torch.zeros((1, 3, 3)),
    )
    batch = collate_frozen_examples((example,), CGCNN_SOURCE_LAYOUT, device="cpu")
    assert _predict(LegacyModel(), batch) == 1


def test_point_group_number_routes_all_offline_ancestors_and_backpropagates() -> None:
    canonical, _ = _distorted_tetragonal_parent_fixture()
    graph = build_periodic_graph(
        canonical.canonical_positions.float(), canonical.canonical_cell.float(),
        canonical.atomic_numbers, cutoff=4.0,
    )
    dag = PointGroupAncestorDAG.from_path()
    current_number = dag.number(canonical.symmetry.current_point_group)
    active_numbers = dag.ancestors(current_number)
    groups = dag.symbols(active_numbers)
    example = FrozenFeatureExample(
        sample_id="material-static-pg",
        features=cgcnn_node_features(graph.atomic_numbers),
        graph=graph,
        symmetry=replace(
            canonical.symmetry,
            canonical_frame=canonical.symmetry.canonical_frame.float(),
            rotations=canonical.symmetry.rotations.float(),
            translations=canonical.symmetry.translations.float(),
        ),
        target_coefficients=torch.zeros((1, TARGET_LAYOUTS["dielectric"].dimension)),
        target_cartesian=torch.zeros((1, 3, 3)),
        point_group_number=current_number,
    )
    model = CGCNNFeatureTensorModel(
        ARCHITECTURE,
        "dielectric",
        groups,
        point_group_parent_dag=dag,
    )
    batch = collate_frozen_examples((example,), CGCNN_SOURCE_LAYOUT, device="cpu")
    assert model.downstream.active_expert_counts(
        batch.symmetries, point_group_numbers=batch.point_group_numbers
    ) == (len(active_numbers),)
    prediction = _predict(model, batch)
    prediction.raw_cartesian.square().sum().backward()
    for number in active_numbers:
        expert = model.downstream.pg_experts[f"pg{number:02d}"]
        assert any(parameter.grad is not None for parameter in expert.parameters())
    assert model.downstream.routing_gate.log_sigma.grad is None


def test_point_group_number_must_match_cached_symmetry() -> None:
    canonical, _ = _distorted_tetragonal_parent_fixture()
    graph = build_periodic_graph(
        canonical.canonical_positions.float(), canonical.canonical_cell.float(),
        canonical.atomic_numbers, cutoff=4.0,
    )
    dag = PointGroupAncestorDAG.from_path()
    example = FrozenFeatureExample(
        "mismatch",
        cgcnn_node_features(graph.atomic_numbers),
        graph,
        replace(
            canonical.symmetry,
            canonical_frame=canonical.symmetry.canonical_frame.float(),
            rotations=canonical.symmetry.rotations.float(),
            translations=canonical.symmetry.translations.float(),
        ),
        torch.zeros((1, TARGET_LAYOUTS["dielectric"].dimension)),
        torch.zeros((1, 3, 3)),
        point_group_number=dag.number("m-3m"),
    )
    groups = dag.symbols(dag.ancestors(dag.number("m-3m")))
    model = CGCNNFeatureTensorModel(
        ARCHITECTURE, "dielectric", groups, point_group_parent_dag=dag
    )
    batch = collate_frozen_examples((example,), CGCNN_SOURCE_LAYOUT, device="cpu")
    with pytest.raises(ValueError, match="disagrees"):
        _predict(model, batch)
