"""Parity-aware crystallographic standardization, independent of legacy frames."""

import torch
from ...symmetry import canonicalize_structure


def standardize_o3(positions, cell, atomic_numbers, *, symprec=1e-5):
    """Return canonical metadata and input->standard O(3) matrix.

    Row-lattice polar decomposition C=S Q makes S frame invariant even when det(Q)<0.
    Spglib sees this same S for proper and improper externally transformed inputs.
    The returned matrix includes the parity needed by e3nn.D_from_matrix.
    """
    if cell.shape != (3, 3) or not torch.isfinite(cell).all():
        raise ValueError("finite 3x3 cell required")
    left, singular, right = torch.linalg.svd(cell.detach().double())
    if singular.min() <= 1e-10:
        raise ValueError("cell must be nonsingular")
    frame = (left @ right).to(cell)
    canonical = canonicalize_structure(
        positions @ frame.T, cell @ frame.T, atomic_numbers, symprec=symprec
    )
    return canonical, canonical.input_to_canonical @ frame
