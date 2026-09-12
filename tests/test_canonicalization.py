from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from src.heads import rotate_cartesian
from src.symmetry import PointGroupRegistry, canonicalize_structure
from e3nn import o3


def _p1_structure():
    cell = torch.tensor(
        [[3.0, 0.0, 0.0], [0.2, 4.0, 0.0], [0.1, 0.3, 5.0]],
        dtype=torch.float64,
    )
    fractional = torch.tensor(
        [[0.11, 0.22, 0.33], [0.27, 0.45, 0.62], [0.76, 0.18, 0.54]],
        dtype=torch.float64,
    )
    return fractional @ cell, cell, torch.tensor([6, 7, 8])


def _cubic_structure():
    cell = 4.0 * torch.eye(3, dtype=torch.float64)
    fractional = torch.tensor(
        [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]], dtype=torch.float64
    )
    return fractional @ cell, cell, torch.tensor([55, 17])


def test_hall_setting_is_repeatable_and_atom_order_independent() -> None:
    positions, cell, species = _p1_structure()
    first = canonicalize_structure(positions, cell, species)
    second = canonicalize_structure(positions, cell, species)
    permutation = torch.tensor([2, 0, 1])
    reordered = canonicalize_structure(positions[permutation], cell, species[permutation])
    assert (
        first.symmetry.current_space_group,
        first.symmetry.current_point_group,
        first.symmetry.hall_number,
        first.hall_symbol,
        first.setting_choice,
    ) == (
        second.symmetry.current_space_group,
        second.symmetry.current_point_group,
        second.symmetry.hall_number,
        second.hall_symbol,
        second.setting_choice,
    ) == (
        reordered.symmetry.current_space_group,
        reordered.symmetry.current_point_group,
        reordered.symmetry.hall_number,
        reordered.hall_symbol,
        reordered.setting_choice,
    )
    assert first.spglib_version == "2.6.0"
    assert torch.equal(first.input_to_canonical, second.input_to_canonical)
    assert first.canonical_to_input_sites.tolist() == [0, 1, 2]


def test_canonical_frame_round_trip_and_proper_rotation_covariance() -> None:
    positions, cell, species = _p1_structure()
    positions = positions.requires_grad_()
    result = canonicalize_structure(positions, cell, species)
    restored_positions = result.canonical_positions @ result.canonical_to_input.T
    restored_cell = result.canonical_cell @ result.canonical_to_input.T
    assert torch.allclose(restored_positions, positions, atol=1e-12, rtol=1e-12)
    assert torch.allclose(restored_cell, cell, atol=1e-12, rtol=1e-12)
    result.canonical_positions.square().sum().backward()
    assert positions.grad is not None and torch.isfinite(positions.grad).all()

    rotation = o3.rand_matrix(dtype=torch.float64)
    rotated = canonicalize_structure(
        positions.detach() @ rotation.T, cell @ rotation.T, species
    )
    assert rotated.symmetry.hall_number == result.symmetry.hall_number
    assert torch.allclose(rotated.canonical_cell, result.canonical_cell, atol=2e-10, rtol=2e-10)
    assert torch.allclose(
        rotated.canonical_positions, result.canonical_positions.detach(), atol=2e-10, rtol=2e-10
    )


def test_float32_canonical_frame_uses_dtype_aware_inverse_tolerance() -> None:
    positions, cell, species = _p1_structure()
    rotation = o3.rand_matrix(dtype=torch.float32)
    result = canonicalize_structure(
        positions.float() @ rotation.T,
        cell.float() @ rotation.T,
        species,
    )
    identity = torch.eye(3, dtype=torch.float32)
    assert torch.allclose(
        result.canonical_to_input @ result.input_to_canonical,
        identity,
        atol=1.0e-6,
        rtol=1.0e-6,
    )
    with pytest.raises(ValueError, match="canonical frame matrices are not inverses"):
        replace(
            result,
            canonical_to_input=result.canonical_to_input + 1.0e-3,
        )


@pytest.mark.parametrize("task,shape", [("dielectric", (2, 3, 3)), ("elastic", (2, 3, 3, 3, 3))])
def test_tensor_frame_round_trip(task: str, shape: tuple[int, ...]) -> None:
    positions, cell, species = _p1_structure()
    result = canonicalize_structure(positions, cell, species)
    tensor = torch.randn(shape, dtype=torch.float64)
    canonical = rotate_cartesian(tensor, result.input_to_canonical, task)
    restored = rotate_cartesian(canonical, result.canonical_to_input, task)
    assert torch.allclose(restored, tensor, atol=2e-10, rtol=2e-10)


def test_full_operations_produce_species_preserving_bijections() -> None:
    positions, cell, species = _cubic_structure()
    result = canonicalize_structure(positions, cell, species)
    symmetry = result.symmetry
    assert (symmetry.current_space_group, symmetry.current_point_group) == (221, "m-3m")
    assert symmetry.audit_permutations is not None
    fractional = torch.linalg.solve(cell.T, positions.T).T
    # For this two-species structure every operation must fix each unique-species site.
    assert torch.equal(symmetry.audit_permutations, torch.tensor([[0, 1]]).expand(48, -1))
    assert symmetry.rotations.shape == (48, 3, 3)
    identity = torch.eye(3, dtype=torch.float64)
    assert torch.allclose(
        symmetry.rotations.transpose(1, 2) @ symmetry.rotations,
        identity.expand(48, -1, -1),
        atol=1e-10,
        rtol=1e-10,
    )
    registry = PointGroupRegistry()
    unique = {
        tuple(torch.round(rotation * 1e10).to(torch.int64).flatten().tolist())
        for rotation in symmetry.rotations
    }
    assert len(unique) == registry[symmetry.current_point_group].order
    assert fractional.shape == (2, 3)


def test_invalid_canonicalization_input_fails_closed() -> None:
    positions, cell, species = _p1_structure()
    with pytest.raises(ValueError, match="invertible"):
        canonicalize_structure(positions, torch.zeros_like(cell), species)
    with pytest.raises(TypeError, match="torch.long"):
        canonicalize_structure(positions, cell, species.to(torch.int32))
