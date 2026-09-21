"""Canonical symmetry and physical-parent embedding contracts."""

from .contracts import (
    ParentDAGSpec,
    ParentEmbeddingSpec,
    SymmetryRecord,
    validate_parent_embedding,
)
from .parent_detection import (
    MaterialParentRouting,
    build_point_group_parent_dag,
    load_parent_routing_cache,
    parent_detection_config_sha256,
    parent_routing_coverage,
    routing_from_payload,
    routing_to_payload,
    route_material_on_point_group_dag,
    save_parent_routing_cache,
    validate_point_group_parent_dag,
)
from .canonicalization import CanonicalizationResult, canonicalize_structure
from .registry import PointGroup, PointGroupRegistry, canonical_point_group_symbol
from .point_group_dag import (
    POINT_GROUP_DAG_CONVENTION,
    PointGroupAncestorDAG,
    load_point_group_number_cache,
    save_point_group_number_cache,
)

__all__ = [
    "ParentDAGSpec",
    "ParentEmbeddingSpec",
    "CanonicalizationResult",
    "PointGroup",
    "PointGroupRegistry",
    "SymmetryRecord",
    "MaterialParentRouting",
    "build_point_group_parent_dag",
    "load_parent_routing_cache",
    "parent_detection_config_sha256",
    "parent_routing_coverage",
    "routing_from_payload",
    "routing_to_payload",
    "route_material_on_point_group_dag",
    "save_parent_routing_cache",
    "validate_point_group_parent_dag",
    "canonical_point_group_symbol",
    "POINT_GROUP_DAG_CONVENTION",
    "PointGroupAncestorDAG",
    "load_point_group_number_cache",
    "save_point_group_number_cache",
    "canonicalize_structure",
    "validate_parent_embedding",
]
