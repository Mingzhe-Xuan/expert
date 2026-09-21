from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from src.cli.reduced_cgcnn_full_pg_train import ARCHITECTURE
from src.cli.reduced_cgcnn_parent_dag_train import _shared_edge_ids
from src.experts.modules import (
    ContinuousEdgeGate,
    material_point_group_stick_breaking_weights,
)
from src.graphs import build_periodic_graph
from src.features import CGCNN_SOURCE_LAYOUT, cgcnn_node_features
from src.heads import TARGET_LAYOUTS
from src.models import CGCNNFeatureTensorModel
from src.symmetry import (
    ParentDAGSpec,
    ParentEmbeddingSpec,
    PointGroupAncestorDAG,
    SymmetryRecord,
    build_point_group_parent_dag,
    canonicalize_structure,
    load_parent_routing_cache,
    route_material_on_point_group_dag,
    save_parent_routing_cache,
    validate_point_group_parent_dag,
)
from src.training import FrozenFeatureExample, collate_frozen_examples
from src.training.benchmark import _predict


I = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
X = ((1.0, 0.0, 0.0), (0.0, -1.0, 0.0), (0.0, 0.0, -1.0))
Y = ((-1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, -1.0))
Z = ((-1.0, 0.0, 0.0), (0.0, -1.0, 0.0), (0.0, 0.0, 1.0))


def _edge(parent: int, child: int) -> ParentEmbeddingSpec:
    candidate = ParentEmbeddingSpec(
        parent_point_group_number=parent,
        child_point_group_number=child,
        parent_rotations=(I, X, Y, Z),
        child_rotation_variants=((I,),),
        edge_id=f"pg{parent:02d}-to-pg{child:02d}",
        asset_sha256="a" * 64,
        convention_id="test-relative-pg-v1",
        version=1,
        checksum="0" * 64,
    )
    return replace(candidate, checksum=candidate.payload_checksum())


def test_edge_gate_and_stick_breaking_follow_exact_boundary_formula() -> None:
    dag = ParentDAGSpec(
        "path-weights",
        1,
        (_edge(3, 2), _edge(2, 1), _edge(4, 1)),
    )
    residuals = {edge.checksum: value for edge, value in zip(dag.embeddings, (0.3, 0.1, 0.2))}
    gate = ContinuousEdgeGate(tuple(edge.edge_id for edge in dag.embeddings), initial_sigma=0.5)
    gates = {
        edge.checksum: gate.value(edge.edge_id, residuals[edge.checksum])
        for edge in dag.embeddings
    }
    weights = material_point_group_stick_breaking_weights(dag, residuals, gate)
    by_pair = {
        (edge.parent_point_group_number, edge.child_point_group_number): gates[edge.checksum]
        for edge in dag.embeddings
    }
    a32, a21, a41 = by_pair[(3, 2)], by_pair[(2, 1)], by_pair[(4, 1)]
    expected = {
        3: 0.6 * (1.0 - a32),
        2: 0.6 * a32 * (1.0 - a21),
        1: 0.6 * a32 * a21 + 0.4 * a41,
        4: 0.4 * (1.0 - a41),
    }
    assert gate.value(dag.embeddings[0].edge_id, 0.0).item() == 0.0
    assert torch.allclose(torch.stack(tuple(weights.values())).sum(), torch.tensor(1.0))
    for number, value in expected.items():
        assert torch.allclose(weights[number], value)
    weights[3].backward()
    assert any(parameter.grad is not None for parameter in gate.parameters())


@pytest.mark.parametrize("current", range(1, 33))
def test_offline_asset_builds_every_complete_point_group_path(current: int) -> None:
    class_dag = PointGroupAncestorDAG.from_path()
    dag = build_point_group_parent_dag("sample", current, class_dag)
    validate_point_group_parent_dag(dag, class_dag)
    assert dag.current_point_group_number == current
    assert set(dag.current_to_root_paths()) == set(class_dag.maximal_paths(current))
    assert all(edge.child_rotation_variants for edge in dag.embeddings)


def _orthorhombic_routing():
    positions = torch.zeros((1, 3), dtype=torch.float64)
    cell = torch.diag(torch.tensor([2.8, 3.1, 3.4], dtype=torch.float64))
    numbers = torch.tensor([14])
    canonical = canonicalize_structure(positions, cell, numbers)
    assert canonical.symmetry.current_point_group == "mmm"
    graph = build_periodic_graph(
        canonical.canonical_positions,
        canonical.canonical_cell,
        numbers,
        cutoff=3.5,
    )
    class_dag = PointGroupAncestorDAG.from_path()
    dag = build_point_group_parent_dag(
        "orthorhombic", class_dag.number("mmm"), class_dag
    )
    routing = route_material_on_point_group_dag(
        "orthorhombic",
        graph.edge_vectors,
        graph.edge_index,
        graph.atomic_numbers,
        canonical.symmetry,
        dag,
        class_dag,
    )
    return routing, class_dag, graph, canonical.symmetry


def test_point_group_distance_uses_relative_vectors_and_oriented_edge_minimum() -> None:
    routing, _, _, _ = _orthorhombic_routing()
    assert routing.residuals
    assert all(value >= 0.0 for value in routing.residuals.values())
    assert any(value > 0.0 for value in routing.residuals.values())


def test_point_group_routing_rejects_strict_current_pg_mismatch() -> None:
    routing, class_dag, _, _ = _orthorhombic_routing()
    graph = build_periodic_graph(
        torch.zeros((1, 3), dtype=torch.float64),
        torch.eye(3, dtype=torch.float64) * 3.0,
        torch.tensor([14]),
        cutoff=3.1,
    )
    wrong = SymmetryRecord(
        torch.eye(3), "1", 1, 1, torch.eye(3).unsqueeze(0), torch.zeros((1, 3))
    )
    with pytest.raises(ValueError, match="current point group"):
        route_material_on_point_group_dag(
            "orthorhombic",
            graph.edge_vectors,
            graph.edge_index,
            graph.atomic_numbers,
            wrong,
            routing.dag,
            class_dag,
        )


def test_point_group_routing_cache_round_trip_and_stale_schema_rejection(tmp_path) -> None:
    routing, class_dag, _, _ = _orthorhombic_routing()
    path = tmp_path / "point-group-routing.pt"
    save_parent_routing_cache(
        path,
        (routing,),
        sample_ids=("orthorhombic",),
        dataset_sha256="a" * 64,
        class_dag=class_dag,
    )
    loaded = load_parent_routing_cache(
        path,
        sample_ids=("orthorhombic",),
        dataset_sha256="a" * 64,
        class_dag=class_dag,
    )
    assert loaded[0].residuals == routing.residuals
    payload = torch.load(path, map_location="cpu", weights_only=True)
    payload["schema_version"] = 2
    torch.save(payload, path)
    with pytest.raises(ValueError, match="schema_version"):
        load_parent_routing_cache(
            path,
            sample_ids=("orthorhombic",),
            dataset_sha256="a" * 64,
            class_dag=class_dag,
        )


def test_shared_edge_ids_are_asset_level_not_material_level() -> None:
    first, class_dag, _, _ = _orthorhombic_routing()
    second_dag = build_point_group_parent_dag(
        "other", first.dag.current_point_group_number, class_dag
    )
    values = _shared_edge_ids(
        {
            "train": (SimpleNamespace(parent_dag=first.dag),),
            "validation": (SimpleNamespace(parent_dag=second_dag),),
        }
    )
    assert len(values) == len(first.dag.embeddings)
    assert all(value.startswith("pg") for value in values)


def test_cgcnn_point_group_edge_route_backpropagates_through_shared_gates() -> None:
    routing, class_dag, graph, symmetry = _orthorhombic_routing()
    active = class_dag.ancestors(routing.dag.current_point_group_number)
    groups = class_dag.symbols(active)
    example = FrozenFeatureExample(
        "orthorhombic",
        cgcnn_node_features(graph.atomic_numbers).float(),
        graph.to("cpu", dtype=torch.float32),
        replace(
            symmetry,
            canonical_frame=symmetry.canonical_frame.float(),
            rotations=symmetry.rotations.float(),
            translations=symmetry.translations.float(),
            fractional_rotations=symmetry.fractional_rotations.float(),
        ),
        torch.zeros((1, TARGET_LAYOUTS["dielectric"].dimension)),
        torch.zeros((1, 3, 3)),
        parent_dag=routing.dag,
        parent_residuals=routing.residuals,
    )
    model = CGCNNFeatureTensorModel(
        ARCHITECTURE,
        "dielectric",
        groups,
        material_edge_ids=tuple(edge.edge_id for edge in routing.dag.embeddings),
    )
    prediction = _predict(
        model, collate_frozen_examples((example,), CGCNN_SOURCE_LAYOUT, device="cpu")
    )
    prediction.raw_cartesian.square().sum().backward()
    assert any(parameter.grad is not None for parameter in model.downstream.edge_gate.parameters())
