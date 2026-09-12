from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np
from scipy.optimize import linear_sum_assignment
import spglib
import torch

from .contracts import SymmetryRecord
from .registry import canonical_point_group_symbol


def _orthogonalize(matrix: np.ndarray, *, force_proper: bool = False) -> np.ndarray:
    left, _, right = np.linalg.svd(matrix)
    rotation = left @ right
    if force_proper and np.linalg.det(rotation) < 0:
        left[:, -1] *= -1
        rotation = left @ right
    return rotation


def _cartesian_operations(
    fractional_rotations: np.ndarray,
    cell: np.ndarray,
    input_to_canonical: np.ndarray,
) -> np.ndarray:
    inverse_transpose = np.linalg.inv(cell.T)
    operations = []
    for fractional in fractional_rotations:
        input_cartesian = cell.T @ fractional @ inverse_transpose
        canonical = input_to_canonical @ input_cartesian @ input_to_canonical.T
        operations.append(_orthogonalize(canonical))
    return np.stack(operations)


def _periodic_costs(
    transformed: np.ndarray, reference: np.ndarray, cell: np.ndarray
) -> np.ndarray:
    delta = transformed[:, None, :] - reference[None, :, :]
    shifts = np.asarray(tuple(product((-1, 0, 1), repeat=3)), dtype=np.float64)
    vectors = (delta[:, :, None, :] - shifts[None, None, :, :]) @ cell
    return np.min(np.sum(vectors * vectors, axis=-1), axis=-1)


def _operation_permutations(
    fractional_positions: np.ndarray,
    atomic_numbers: np.ndarray,
    cell: np.ndarray,
    rotations: np.ndarray,
    translations: np.ndarray,
    symprec: float,
) -> np.ndarray:
    permutations = []
    species = np.unique(atomic_numbers)
    tolerance = max(5.0 * symprec, 1.0e-7)
    for rotation, translation in zip(rotations, translations):
        moved = fractional_positions @ rotation.T + translation
        moved -= np.floor(moved)
        permutation = np.full(len(fractional_positions), -1, dtype=np.int64)
        for atomic_number in species:
            indices = np.flatnonzero(atomic_numbers == atomic_number)
            costs = _periodic_costs(moved[indices], fractional_positions[indices], cell)
            rows, columns = linear_sum_assignment(costs)
            assigned_distances = np.sqrt(costs[rows, columns])
            if np.any(assigned_distances > tolerance):
                raise ValueError(
                    "spglib operation does not close on a species-preserving site mapping"
                )
            permutation[indices[rows]] = indices[columns]
        if sorted(permutation.tolist()) != list(range(len(fractional_positions))):
            raise ValueError("symmetry operation site mapping is not bijective")
        permutations.append(permutation)
    return np.stack(permutations)


@dataclass(frozen=True, slots=True)
class CanonicalizationResult:
    canonical_positions: torch.Tensor
    canonical_cell: torch.Tensor
    atomic_numbers: torch.Tensor
    symmetry: SymmetryRecord
    input_to_canonical: torch.Tensor
    canonical_to_input: torch.Tensor
    canonical_to_input_sites: torch.Tensor
    hall_symbol: str
    setting_choice: str
    transformation_matrix: torch.Tensor
    origin_shift: torch.Tensor
    spglib_version: str

    def __post_init__(self) -> None:
        node_count = self.canonical_positions.shape[0]
        if self.canonical_positions.shape != (node_count, 3):
            raise ValueError("canonical_positions must have shape [num_nodes, 3]")
        if self.canonical_cell.shape != (3, 3):
            raise ValueError("canonical_cell must have shape [3, 3]")
        if self.atomic_numbers.shape != (node_count,):
            raise ValueError("atomic_numbers must align with canonical_positions")
        if self.canonical_to_input_sites.shape != (node_count,):
            raise ValueError("canonical_to_input_sites must have shape [num_nodes]")
        identity = torch.eye(
            3, dtype=self.input_to_canonical.dtype, device=self.input_to_canonical.device
        )
        frame_tolerance = max(
            1.0e-8,
            8.0 * torch.finfo(self.input_to_canonical.dtype).eps,
        )
        if not torch.allclose(
            self.canonical_to_input @ self.input_to_canonical,
            identity,
            atol=frame_tolerance,
            rtol=frame_tolerance,
        ):
            raise ValueError("canonical frame matrices are not inverses")
        if not self.hall_symbol or not self.spglib_version:
            raise ValueError("Hall symbol and spglib version must be frozen")


def canonicalize_structure(
    positions: torch.Tensor,
    cell: torch.Tensor,
    atomic_numbers: torch.Tensor,
    *,
    symprec: float = 1.0e-5,
    angle_tolerance: float = -1.0,
) -> CanonicalizationResult:
    """Detect a deterministic Hall setting and rotate into spglib's idealized frame.

    The operation is site-order preserving. Discrete symmetry discovery and the frame are
    treated as fixed; gradients through canonical positions still flow to input positions.
    """

    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("positions must have shape [num_nodes, 3]")
    if cell.shape != (3, 3) or abs(float(torch.linalg.det(cell))) < 1.0e-12:
        raise ValueError("cell must be an invertible [3, 3] matrix")
    if atomic_numbers.shape != (positions.shape[0],) or atomic_numbers.dtype != torch.long:
        raise TypeError("atomic_numbers must be torch.long with shape [num_nodes]")
    if positions.dtype != cell.dtype or not positions.dtype.is_floating_point:
        raise TypeError("positions and cell must share a floating dtype")
    if positions.device != cell.device or positions.device != atomic_numbers.device:
        raise ValueError("positions, cell and atomic_numbers must share a device")
    if symprec <= 0:
        raise ValueError("symprec must be positive")

    cell_numpy = cell.detach().cpu().double().numpy()
    fractional = torch.linalg.solve(cell.T, positions.T).T
    fractional_numpy = fractional.detach().cpu().double().numpy()
    fractional_numpy -= np.floor(fractional_numpy)
    species_numpy = atomic_numbers.detach().cpu().numpy()
    spglib_cell = (cell_numpy, fractional_numpy, species_numpy)
    initial = spglib.get_symmetry_dataset(
        spglib_cell, symprec=symprec, angle_tolerance=angle_tolerance
    )
    if initial is None:
        raise ValueError("spglib could not detect a symmetry dataset")
    # Explicitly repeat with the detected Hall number so setting selection is frozen.
    dataset = spglib.get_symmetry_dataset(
        spglib_cell,
        symprec=symprec,
        angle_tolerance=angle_tolerance,
        hall_number=int(initial.hall_number),
    )
    if dataset is None or int(dataset.hall_number) != int(initial.hall_number):
        raise ValueError("spglib could not reproduce the explicitly selected Hall setting")

    input_to_canonical_numpy = _orthogonalize(
        np.asarray(dataset.std_rotation_matrix, dtype=np.float64), force_proper=True
    )
    input_to_canonical = torch.as_tensor(
        input_to_canonical_numpy, dtype=positions.dtype, device=positions.device
    )
    canonical_to_input = input_to_canonical.T
    canonical_positions = positions @ input_to_canonical.T
    canonical_cell = cell @ input_to_canonical.T

    rotations_numpy = np.asarray(dataset.rotations, dtype=np.int64)
    translations_numpy = np.asarray(dataset.translations, dtype=np.float64)
    permutations_numpy = _operation_permutations(
        fractional_numpy,
        species_numpy,
        cell_numpy,
        rotations_numpy,
        translations_numpy,
        symprec,
    )
    canonical_rotations = _cartesian_operations(
        rotations_numpy, cell_numpy, input_to_canonical_numpy
    )
    symmetry = SymmetryRecord(
        canonical_frame=canonical_to_input,
        current_point_group=canonical_point_group_symbol(str(dataset.pointgroup)),
        current_space_group=int(dataset.number),
        hall_number=int(dataset.hall_number),
        rotations=torch.as_tensor(
            canonical_rotations, dtype=positions.dtype, device=positions.device
        ),
        translations=torch.as_tensor(
            translations_numpy, dtype=positions.dtype, device=positions.device
        ),
        audit_permutations=torch.as_tensor(
            permutations_numpy, dtype=torch.long, device=positions.device
        ),
    )
    return CanonicalizationResult(
        canonical_positions=canonical_positions,
        canonical_cell=canonical_cell,
        atomic_numbers=atomic_numbers,
        symmetry=symmetry,
        input_to_canonical=input_to_canonical,
        canonical_to_input=canonical_to_input,
        canonical_to_input_sites=torch.arange(
            positions.shape[0], dtype=torch.long, device=positions.device
        ),
        hall_symbol=str(dataset.hall),
        setting_choice=str(dataset.choice),
        transformation_matrix=torch.as_tensor(
            np.asarray(dataset.transformation_matrix),
            dtype=positions.dtype,
            device=positions.device,
        ),
        origin_shift=torch.as_tensor(
            np.asarray(dataset.origin_shift),
            dtype=positions.dtype,
            device=positions.device,
        ),
        spglib_version=str(spglib.__version__),
    )
