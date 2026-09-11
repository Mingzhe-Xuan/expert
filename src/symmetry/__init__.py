"""Canonical symmetry and physical-parent embedding contracts."""

from .contracts import (
    ParentDAGSpec,
    ParentEmbeddingSpec,
    SymmetryRecord,
    validate_parent_embedding,
)
from .canonicalization import CanonicalizationResult, canonicalize_structure
from .registry import PointGroup, PointGroupRegistry, canonical_point_group_symbol

__all__ = [
    "ParentDAGSpec",
    "ParentEmbeddingSpec",
    "CanonicalizationResult",
    "PointGroup",
    "PointGroupRegistry",
    "SymmetryRecord",
    "canonical_point_group_symbol",
    "canonicalize_structure",
    "validate_parent_embedding",
]
