from __future__ import annotations

from dataclasses import replace

import pytest

from src.symmetry import ParentDAGSpec, ParentEmbeddingSpec, validate_parent_embedding


I = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
X = ((1.0, 0.0, 0.0), (0.0, -1.0, 0.0), (0.0, 0.0, -1.0))
Y = ((-1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, -1.0))
Z = ((-1.0, 0.0, 0.0), (0.0, -1.0, 0.0), (0.0, 0.0, 1.0))


def _embedding(parent: int = 3, child: int = 2) -> ParentEmbeddingSpec:
    candidate = ParentEmbeddingSpec(
        parent_point_group_number=parent,
        child_point_group_number=child,
        parent_rotations=(I, X, Y, Z),
        child_rotation_variants=((I, X), (I, Y), (I, Z)),
        edge_id=f"pg{parent:02d}-to-pg{child:02d}",
        asset_sha256="a" * 64,
        convention_id="test-point-group-edge-v1",
        version=1,
        checksum="0" * 64,
    )
    return replace(candidate, checksum=candidate.payload_checksum())


def test_point_group_edge_validates_orientations_and_checksum() -> None:
    embedding = _embedding()
    validate_parent_embedding(embedding)
    assert len(embedding.child_rotation_variants) == 3
    with pytest.raises(ValueError, match="checksum"):
        validate_parent_embedding(replace(embedding, edge_id="changed"))


def test_point_group_edge_requires_closed_unique_strict_subgroups() -> None:
    with pytest.raises(ValueError, match="multiplication closed|inverse closed"):
        replace(
            _embedding(),
            parent_rotations=(I, ((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0))),
            checksum="0" * 64,
        )
    with pytest.raises(ValueError, match="strict parent subset"):
        replace(
            _embedding(),
            child_rotation_variants=((I, X, Y, Z),),
            checksum="0" * 64,
        )
    with pytest.raises(ValueError, match="variants must be unique"):
        replace(
            _embedding(),
            child_rotation_variants=((I, X), (I, X)),
            checksum="0" * 64,
        )


def test_parent_dag_enumerates_complete_class_paths() -> None:
    lower = _embedding(3, 2)
    upper = replace(
        _embedding(4, 3),
        edge_id="pg04-to-pg03",
        checksum="0" * 64,
    )
    upper = replace(upper, checksum=upper.payload_checksum())
    dag = ParentDAGSpec("sample", 2, (upper, lower))
    assert dag.current_to_root_paths() == ((2, 3, 4),)
    assert dag.current_to_root_embedding_paths() == ((lower, upper),)
    with pytest.raises(ValueError, match="duplicate"):
        ParentDAGSpec("sample", 2, (lower, lower))


def test_parent_dag_rejects_disconnected_edges() -> None:
    with pytest.raises(ValueError, match="lead to the current"):
        ParentDAGSpec("sample", 2, (_embedding(4, 3), _embedding(5, 2)))
