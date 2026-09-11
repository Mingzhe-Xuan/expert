from __future__ import annotations

import torch

from src.heads import TARGET_LAYOUTS
from src.irreps import IrrepLayout, IrrepTerm
from src.symmetry import PointGroupRegistry


NATURAL_LMAX4 = IrrepLayout(
    tuple(
        IrrepTerm(1, degree, "e" if degree % 2 == 0 else "o", f"l{degree}")
        for degree in range(5)
    )
)


def _fractional_key(matrix: torch.Tensor) -> tuple[int, ...]:
    return tuple(int(value) for value in matrix.reshape(-1))


def test_registry_loads_exactly_32_validated_groups() -> None:
    registry = PointGroupRegistry()
    assert len(registry) == 32
    assert [group.number for group in registry] == list(range(1, 33))
    assert registry["m -3 m"].number == 32
    for group in registry:
        assert group.fractional_rotations.shape == (group.order, 3, 3)
        assert group.cartesian_rotations.shape == (group.order, 3, 3)
        identity = torch.eye(3, dtype=torch.float64)
        products = group.cartesian_rotations.transpose(1, 2) @ group.cartesian_rotations
        assert torch.allclose(products, identity.expand_as(products), atol=1e-10, rtol=1e-10)


def test_lmax4_representations_are_group_homomorphisms() -> None:
    registry = PointGroupRegistry()
    for group in registry:
        representation = group.representation(NATURAL_LMAX4)
        index = {
            _fractional_key(rotation): operation_index
            for operation_index, rotation in enumerate(group.fractional_rotations)
        }
        sample_count = min(group.order, 4)
        for left_index in range(sample_count):
            for right_index in range(sample_count):
                product = (
                    group.fractional_rotations[left_index]
                    @ group.fractional_rotations[right_index]
                )
                product_index = index[_fractional_key(product)]
                assert torch.allclose(
                    representation[left_index] @ representation[right_index],
                    representation[product_index],
                    atol=2e-8,
                    rtol=2e-8,
                ), group.symbol


def test_target_invariant_projectors_and_bases_for_all_groups() -> None:
    registry = PointGroupRegistry()
    for group in registry:
        for task, layout in TARGET_LAYOUTS.items():
            representation = group.representation(layout)
            projector = group.invariant_projector(layout)
            basis = group.invariant_basis(layout)
            assert torch.allclose(projector @ projector, projector, atol=2e-8, rtol=2e-8)
            assert torch.allclose(projector, projector.T, atol=1e-12, rtol=1e-12)
            assert basis.shape[0] == layout.dimension
            assert basis.shape[1] == int(torch.linalg.matrix_rank(projector, atol=1e-8))
            assert torch.allclose(
                basis.T @ basis,
                torch.eye(basis.shape[1], dtype=basis.dtype),
                atol=2e-8,
                rtol=2e-8,
            )
            assert torch.allclose(
                representation @ basis,
                basis.expand(group.order, -1, -1),
                atol=2e-8,
                rtol=2e-8,
            ), (group.symbol, task)
            checksum = group.invariant_basis_checksum(layout)
            assert len(checksum) == 64
            assert checksum == group.invariant_basis_checksum(layout)


def test_known_fixed_space_dimensions() -> None:
    registry = PointGroupRegistry()
    for task, layout in TARGET_LAYOUTS.items():
        assert registry["1"].invariant_basis(layout).shape[1] == layout.dimension
        assert registry["-1"].invariant_basis(layout).shape[1] == layout.dimension
    assert registry["m-3m"].invariant_basis(TARGET_LAYOUTS["dielectric"]).shape[1] == 1
    assert registry["m-3m"].invariant_basis(TARGET_LAYOUTS["elastic"]).shape[1] == 3
    assert registry["m-3m"].invariant_basis(TARGET_LAYOUTS["bec"]).shape[1] == 1
