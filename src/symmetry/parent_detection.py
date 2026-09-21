from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment
import torch

from .contracts import ParentDAGSpec, ParentEmbeddingSpec, SymmetryRecord
from .point_group_dag import PointGroupAncestorDAG
from .registry import DEFAULT_REGISTRY_PATH, PointGroupRegistry


PARENT_DAG_CONVENTION = "point-group-relative-edge-stick-breaking-v1"


@dataclass(frozen=True, slots=True)
class MaterialParentRouting:
    dag: ParentDAGSpec
    residuals: Mapping[str, float]

    def __post_init__(self) -> None:
        edge_ids = {embedding.checksum for embedding in self.dag.embeddings}
        if set(self.residuals) != edge_ids:
            raise ValueError("parent residuals must cover exactly the point-group cover edges")
        if any(not np.isfinite(value) or value < 0 for value in self.residuals.values()):
            raise ValueError("parent edge residuals must be finite and non-negative")


def _matrix_tuple(matrix: torch.Tensor) -> tuple[tuple[float, float, float], ...]:
    return tuple(tuple(float(value) for value in row) for row in matrix.tolist())


@lru_cache(maxsize=4)
def _all_point_group_edge_templates(
    asset_path: str, expected_sha256: str
) -> tuple[ParentEmbeddingSpec, ...]:
    source = Path(asset_path)
    encoded = source.read_bytes()
    asset_sha256 = hashlib.sha256(encoded).hexdigest()
    if asset_sha256 != expected_sha256:
        raise ValueError("point-group registry and class DAG asset hashes disagree")
    payload = json.loads(encoded.decode("utf-8"))
    registry = PointGroupRegistry(source)
    embeddings = []
    for edge in payload["class_cover_edges"]:
        parent = int(edge["parent_number"])
        child = int(edge["child_number"])
        raw_parent = payload["point_groups"][str(parent)]
        instances = [
            value
            for value in raw_parent["subgroup_instances"]
            if int(value["point_group_number"]) == child and bool(value["is_maximal"])
        ]
        if not instances:
            raise ValueError(f"offline edge {parent}->{child} lacks oriented subgroup instances")
        parent_rotations = registry[parent].cartesian_rotations
        variants = tuple(
            tuple(
                _matrix_tuple(parent_rotations[int(index)])
                for index in instance["operation_indices"]
            )
            for instance in instances
        )
        candidate = ParentEmbeddingSpec(
            parent_point_group_number=parent,
            child_point_group_number=child,
            parent_rotations=tuple(_matrix_tuple(value) for value in parent_rotations),
            child_rotation_variants=variants,
            edge_id=f"pg{parent:02d}-to-pg{child:02d}",
            asset_sha256=asset_sha256,
            convention_id=PARENT_DAG_CONVENTION,
            version=1,
            checksum="0" * 64,
        )
        values = {
            name: getattr(candidate, name)
            for name in candidate.__dataclass_fields__
            if name != "checksum"
        }
        embeddings.append(
            ParentEmbeddingSpec(**values, checksum=candidate.payload_checksum())
        )
    return tuple(embeddings)


def build_point_group_parent_dag(
    material_id: str,
    current_point_group_number: int,
    class_dag: PointGroupAncestorDAG,
    *,
    asset_path: str | Path = DEFAULT_REGISTRY_PATH,
) -> ParentDAGSpec:
    """Build the complete oriented PG-cover DAG directly from the frozen subgroup asset."""

    if current_point_group_number not in class_dag.symbols_by_number:
        raise ValueError("current point-group number is absent from the offline DAG")
    active = set(class_dag.ancestors(current_point_group_number))
    templates = _all_point_group_edge_templates(
        str(Path(asset_path).resolve()), class_dag.asset_sha256
    )
    embeddings = tuple(
        edge
        for edge in templates
        if edge.parent_point_group_number in active
        and edge.child_point_group_number in active
        and edge.parent_point_group_number
        in class_dag.parents_by_child[edge.child_point_group_number]
    )
    dag = ParentDAGSpec(material_id, current_point_group_number, embeddings)
    validate_point_group_parent_dag(dag, class_dag)
    return dag


def validate_point_group_parent_dag(
    dag: ParentDAGSpec, class_dag: PointGroupAncestorDAG
) -> None:
    """Require exact offline class-path coverage and asset-bound oriented edge contracts."""

    active = set(class_dag.ancestors(dag.current_point_group_number))
    expected_edges = {
        (parent, child)
        for child in active
        for parent in class_dag.parents_by_child[child]
        if parent in active
    }
    actual_edges = {
        (edge.parent_point_group_number, edge.child_point_group_number)
        for edge in dag.embeddings
    }
    if actual_edges != expected_edges:
        raise ValueError("point-group parent DAG does not contain the exact offline cover edges")
    if set(dag.current_to_root_paths()) != set(
        class_dag.maximal_paths(dag.current_point_group_number)
    ):
        raise ValueError("point-group parent DAG does not cover every offline maximal path")
    if any(edge.asset_sha256 != class_dag.asset_sha256 for edge in dag.embeddings):
        raise ValueError("point-group parent edge asset hash mismatch")


def _rotation_key(rotation: np.ndarray, tolerance: float = 1.0e-7) -> tuple[int, ...]:
    return tuple(np.rint(rotation.reshape(-1) / tolerance).astype(np.int64).tolist())


def _relative_edge_residual(
    embedding: ParentEmbeddingSpec,
    edge_vectors: np.ndarray,
    edge_species: np.ndarray,
) -> float:
    """Measure parent-minus-child rotations on translation-free relative edge vectors."""

    if not len(edge_vectors):
        raise ValueError("point-group distance requires at least one relative edge vector")
    parent = [np.asarray(value, dtype=np.float64) for value in embedding.parent_rotations]
    grouped = {
        tuple(int(value) for value in pair): np.flatnonzero(
            np.all(edge_species == pair, axis=1)
        )
        for pair in np.unique(edge_species, axis=0)
    }
    variant_residuals = []
    for variant in embedding.child_rotation_variants:
        child_keys = {_rotation_key(np.asarray(value)) for value in variant}
        additional = [value for value in parent if _rotation_key(value) not in child_keys]
        squared = []
        for rotation in additional:
            moved = edge_vectors @ rotation.T
            for indices in grouped.values():
                reference = edge_vectors[indices]
                costs = np.sum(
                    np.square(moved[indices, None, :] - reference[None, :, :]), axis=-1
                )
                rows, columns = linear_sum_assignment(costs)
                squared.extend(costs[rows, columns].tolist())
        if not squared:
            raise ValueError("a point-group cover edge must add at least one rotation")
        variant_residuals.append(float(np.sqrt(np.mean(squared))))
    return min(variant_residuals)


def route_material_on_point_group_dag(
    material_id: str,
    edge_vectors: torch.Tensor,
    edge_index: torch.Tensor,
    atomic_numbers: torch.Tensor,
    symmetry: SymmetryRecord,
    candidate_dag: ParentDAGSpec,
    class_dag: PointGroupAncestorDAG,
) -> MaterialParentRouting:
    """Compute fixed-topology point-group edge distances from relative graph vectors."""

    if candidate_dag.material_id != material_id:
        raise ValueError("candidate point-group DAG material ID mismatch")
    current = class_dag.number(symmetry.current_point_group)
    if candidate_dag.current_point_group_number != current:
        raise ValueError("candidate DAG current point group disagrees with strict symmetry")
    validate_point_group_parent_dag(candidate_dag, class_dag)
    vectors = edge_vectors.detach().cpu().double().numpy()
    indices = edge_index.detach().cpu().numpy()
    species = atomic_numbers.detach().cpu().numpy()
    edge_species = np.stack((species[indices[0]], species[indices[1]]), axis=1)
    residuals = {
        embedding.checksum: _relative_edge_residual(embedding, vectors, edge_species)
        for embedding in candidate_dag.embeddings
    }
    return MaterialParentRouting(candidate_dag, residuals)


def routing_to_payload(routing: MaterialParentRouting) -> dict[str, object]:
    return {
        "material_id": routing.dag.material_id,
        "current_point_group_number": routing.dag.current_point_group_number,
        "residuals": dict(routing.residuals),
    }


def routing_from_payload(
    payload: Mapping[str, object], class_dag: PointGroupAncestorDAG
) -> MaterialParentRouting:
    dag = build_point_group_parent_dag(
        str(payload["material_id"]),
        int(payload["current_point_group_number"]),
        class_dag,
    )
    return MaterialParentRouting(
        dag,
        {str(key): float(value) for key, value in payload["residuals"].items()},  # type: ignore[union-attr]
    )


def parent_detection_config_sha256(class_dag: PointGroupAncestorDAG) -> str:
    payload = {
        "convention": PARENT_DAG_CONVENTION,
        "asset_sha256": class_dag.asset_sha256,
        "topology": "offline-complete-point-group-cover-paths",
        "residual": "minimum-oriented-parent-minus-child-relative-edge-rms-v1",
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def save_parent_routing_cache(
    path: str | Path,
    routings: Sequence[MaterialParentRouting],
    *,
    sample_ids: Sequence[str],
    dataset_sha256: str,
    class_dag: PointGroupAncestorDAG,
) -> None:
    if [item.dag.material_id for item in routings] != list(sample_ids):
        raise ValueError("parent routing cache must preserve the exact sample order")
    for routing in routings:
        validate_point_group_parent_dag(routing.dag, class_dag)
    payload = {
        "schema_version": 3,
        "convention_id": PARENT_DAG_CONVENTION,
        "dataset_sha256": dataset_sha256,
        "class_dag_sha256": class_dag.asset_sha256,
        "config_sha256": parent_detection_config_sha256(class_dag),
        "sample_ids": list(sample_ids),
        "routings": [routing_to_payload(item) for item in routings],
    }
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_parent_routing_cache(
    path: str | Path,
    *,
    sample_ids: Sequence[str],
    dataset_sha256: str,
    class_dag: PointGroupAncestorDAG,
) -> tuple[MaterialParentRouting, ...]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    expected = {
        "schema_version": 3,
        "convention_id": PARENT_DAG_CONVENTION,
        "dataset_sha256": dataset_sha256,
        "class_dag_sha256": class_dag.asset_sha256,
        "config_sha256": parent_detection_config_sha256(class_dag),
        "sample_ids": list(sample_ids),
    }
    for name, value in expected.items():
        if payload.get(name) != value:
            raise ValueError(f"parent routing cache {name} mismatch")
    return tuple(routing_from_payload(value, class_dag) for value in payload["routings"])


def parent_routing_coverage(
    routings: Sequence[MaterialParentRouting], class_dag: PointGroupAncestorDAG
) -> dict[str, object]:
    if not routings:
        raise ValueError("parent routing coverage requires at least one sample")
    parent_sets = [
        {edge.parent_point_group_number for edge in item.dag.embeddings}
        for item in routings
    ]
    path_sets = [item.dag.current_to_root_embedding_paths() for item in routings]
    counts: dict[str, int] = {}
    for values in parent_sets:
        for number in values:
            symbol = class_dag.symbols_by_number[number]
            counts[symbol] = counts.get(symbol, 0) + 1
    path_lengths = [len(path) + 1 for paths in path_sets for path in paths]
    return {
        "samples": len(routings),
        "coverage_percent": 100.0 * sum(bool(value) for value in parent_sets) / len(routings),
        "mean_parent_count": sum(map(len, parent_sets)) / len(parent_sets),
        "max_parent_count": max(map(len, parent_sets), default=0),
        "mean_path_count": sum(map(len, path_sets)) / len(path_sets),
        "max_path_count": max(map(len, path_sets)),
        "mean_path_node_count": sum(path_lengths) / len(path_lengths),
        "max_path_node_count": max(path_lengths),
        "parent_point_group_counts": dict(sorted(counts.items())),
    }
