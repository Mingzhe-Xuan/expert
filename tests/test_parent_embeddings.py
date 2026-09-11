from __future__ import annotations

from dataclasses import replace

import pytest

from src.symmetry import ParentDAGSpec, ParentEmbeddingSpec, validate_parent_embedding


IDENTITY_OPERATION = (
    (
        ((1, 0, 0), (0, 1, 0), (0, 0, 1)),
        (0.0, 0.0, 0.0),
    ),
)


def _embedding(parent: int = 2, child: int = 1, variant: str = "identity"):
    candidate = ParentEmbeddingSpec(
        parent_hall_number=parent,
        child_hall_number=child,
        parent_setting=f"hall-{parent}",
        child_setting=f"hall-{child}",
        basis_transform=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        origin_shift=(0.0, 0.0, 0.0),
        supercell_transform=((1, 0, 0), (0, 1, 0), (0, 0, 1)),
        operations=IDENTITY_OPERATION,
        parent_atomic_numbers=(6, 8),
        child_atomic_numbers=(6, 6, 8),
        atom_correspondence=(0, 0, 1),
        wyckoff_splitting=("1a->1a+1b", "1b->1c"),
        domain_variant=variant,
        convention_id="hall-common-cell-v1",
        version=1,
        checksum="0" * 64,
    )
    return replace(candidate, checksum=candidate.payload_checksum())


def test_embedding_validates_species_preserving_wyckoff_split_and_checksum() -> None:
    embedding = _embedding()
    validate_parent_embedding(embedding)
    assert embedding.atom_correspondence == (0, 0, 1)
    with pytest.raises(ValueError, match="checksum"):
        validate_parent_embedding(replace(embedding, origin_shift=(0.5, 0.0, 0.0)))
    with pytest.raises(ValueError, match="preserve species"):
        replace(
            embedding,
            child_atomic_numbers=(6, 7, 8),
            checksum="0" * 64,
        )


def test_affine_operations_require_complete_group() -> None:
    quarter_turn = ((0, -1, 0), (1, 0, 0), (0, 0, 1))
    with pytest.raises(ValueError, match="inverse closed|multiplication closed"):
        replace(
            _embedding(),
            operations=(IDENTITY_OPERATION[0], (quarter_turn, (0.0, 0.0, 0.0))),
            checksum="0" * 64,
        )
    reflection = ((-1, 0, 0), (0, 1, 0), (0, 0, 1))
    candidate = replace(
        _embedding(),
        operations=(IDENTITY_OPERATION[0], (reflection, (0.5, 0.0, 0.0))),
        checksum="0" * 64,
    )
    valid = replace(candidate, checksum=candidate.payload_checksum())
    validate_parent_embedding(valid)


def test_parent_dag_allows_orientation_variants_and_connected_chains() -> None:
    direct = _embedding(2, 1, "domain-a")
    alternate = _embedding(2, 1, "domain-b")
    ancestor = _embedding(3, 2, "domain-a")
    dag = ParentDAGSpec("sample", 1, (ancestor, direct, alternate))
    assert len(dag.embeddings) == 3
    with pytest.raises(ValueError, match="duplicate"):
        ParentDAGSpec("sample", 1, (direct, direct))


def test_parent_dag_rejects_cycle_and_disconnected_candidates() -> None:
    with pytest.raises(ValueError, match="cycle"):
        ParentDAGSpec(
            "sample",
            1,
            (_embedding(3, 2), _embedding(2, 3), _embedding(4, 1)),
        )
    with pytest.raises(ValueError, match="lead to the current"):
        ParentDAGSpec("sample", 1, (_embedding(3, 2), _embedding(4, 1)))
