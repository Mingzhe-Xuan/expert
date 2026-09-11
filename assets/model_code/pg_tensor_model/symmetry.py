from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import spglib
import torch
from e3nn import o3
from scipy.optimize import linear_sum_assignment


TARGET_IRREPS = {
    "dielectric": o3.Irreps("1x0e+1x2e"),
    "elastic": o3.Irreps("2x0e+2x2e+1x4e"),
}


def _canonical_symbol(symbol: str) -> str:
    return symbol.replace(" ", "")


@dataclass(frozen=True)
class PointGroupRecord:
    number: int
    symbol: str
    schoenflies: str
    order: int
    fractional_rotations: torch.Tensor
    cartesian_rotations: torch.Tensor


class PointGroupDAG:
    """The 32-class point-group DAG generated in docs/ref.

    This is a class-level DAG. It intentionally does not pretend to contain
    material-specific Hall embeddings, translations, or Wyckoff mappings.
    """

    def __init__(self, path: str | Path):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        self.records: dict[str, PointGroupRecord] = {}
        self.number_to_symbol: dict[int, str] = {}
        for value in data["point_groups"].values():
            symbol = _canonical_symbol(value["hm_symbol"])
            frac = np.asarray(value["operation_matrices"], dtype=np.float64)
            cart = _fractional_group_to_cartesian(frac)
            record = PointGroupRecord(
                number=int(value["number"]),
                symbol=symbol,
                schoenflies=value["schoenflies"],
                order=int(value["order"]),
                fractional_rotations=torch.from_numpy(frac),
                cartesian_rotations=torch.from_numpy(cart),
            )
            self.records[symbol] = record
            self.number_to_symbol[record.number] = symbol

        self.parents: dict[str, set[str]] = {key: set() for key in self.records}
        self.children: dict[str, set[str]] = {key: set() for key in self.records}
        for edge in data["class_cover_edges"]:
            parent = _canonical_symbol(edge["parent_symbol"])
            child = _canonical_symbol(edge["child_symbol"])
            self.parents[child].add(parent)
            self.children[parent].add(child)

    @property
    def symbols(self) -> list[str]:
        return [self.number_to_symbol[i] for i in range(1, 33)]

    def ancestors(self, symbol: str) -> list[str]:
        symbol = _canonical_symbol(symbol)
        found = {symbol}
        frontier = [symbol]
        while frontier:
            child = frontier.pop()
            for parent in self.parents[child]:
                if parent not in found:
                    found.add(parent)
                    frontier.append(parent)
        return sorted(found, key=lambda item: self.records[item].number)

    def root_to_current_paths(self, current: str) -> list[list[str]]:
        current = _canonical_symbol(current)
        active = set(self.ancestors(current))
        roots = sorted(
            (node for node in active if not (self.parents[node] & active)),
            key=lambda item: self.records[item].number,
        )
        paths: list[list[str]] = []

        def visit(node: str, path: list[str]) -> None:
            if node == current:
                paths.append(path.copy())
                return
            for child in sorted(
                self.children[node] & active,
                key=lambda item: self.records[item].number,
            ):
                if child not in path:
                    visit(child, [*path, child])

        for root in roots:
            visit(root, [root])
        return paths or [[current]]


def _fractional_group_to_cartesian(rotations: np.ndarray) -> np.ndarray:
    # G = sum R^T R is positive definite and invariant under the finite group.
    metric = sum(rotation.T @ rotation for rotation in rotations)
    basis = np.linalg.cholesky(metric).T
    basis_inv = np.linalg.inv(basis)
    cartesian = np.stack([basis @ rotation @ basis_inv for rotation in rotations])
    # Remove numerical drift before passing matrices to e3nn.
    clean = []
    for rotation in cartesian:
        u, _, vt = np.linalg.svd(rotation)
        orthogonal = u @ vt
        if np.linalg.det(orthogonal) * np.linalg.det(rotation) < 0:
            u[:, -1] *= -1
            orthogonal = u @ vt
        clean.append(orthogonal)
    return np.stack(clean)


def invariant_basis(irreps: o3.Irreps, rotations: torch.Tensor) -> torch.Tensor:
    rotations = rotations.to(dtype=torch.float64)
    projector = torch.stack([irreps.D_from_matrix(r) for r in rotations]).mean(0)
    projector = 0.5 * (projector + projector.T)
    eigenvalues, eigenvectors = torch.linalg.eigh(projector)
    basis = eigenvectors[:, eigenvalues > 0.5]
    # QR fixes accumulated numerical non-orthogonality; column signs are fixed
    # so parameter reports and checkpoints are deterministic.
    basis = torch.linalg.qr(basis, mode="reduced").Q
    if basis.numel():
        pivots = basis.abs().argmax(dim=0)
        signs = torch.sign(basis[pivots, torch.arange(basis.shape[1])])
        basis = basis * torch.where(signs == 0, torch.ones_like(signs), signs)
    return basis.to(dtype=torch.get_default_dtype())


def per_l_invariant_bases(
    lmax: int, rotations: torch.Tensor
) -> tuple[torch.Tensor, ...]:
    return tuple(
        invariant_basis(o3.Irreps([(1, o3.Irrep(ell, (-1) ** ell))]), rotations)
        for ell in range(lmax + 1)
    )


def infer_point_group(
    positions: torch.Tensor,
    cell: torch.Tensor,
    atomic_numbers: torch.Tensor,
    symprec: float,
) -> str:
    fractional = torch.linalg.solve(cell.T, positions.T).T
    dataset = spglib.get_symmetry_dataset(
        (
            cell.detach().cpu().double().numpy(),
            fractional.detach().cpu().double().numpy(),
            atomic_numbers.detach().cpu().numpy(),
        ),
        symprec=symprec,
    )
    if dataset is None:
        raise ValueError("spglib could not identify the input structure")
    return _canonical_symbol(str(dataset.pointgroup))


def _periodic_squared_distances(
    left: torch.Tensor, right: torch.Tensor, cell: torch.Tensor
) -> torch.Tensor:
    delta = left[:, None, :] - right[None, :, :]
    shifts = torch.cartesian_prod(
        *[torch.arange(-1, 2, device=delta.device, dtype=delta.dtype)] * 3
    )
    candidates = (delta[..., None, :] - shifts) @ cell
    return candidates.square().sum(-1).min(-1).values


def symmetry_residual(
    positions: torch.Tensor,
    cell: torch.Tensor,
    atomic_numbers: torch.Tensor,
    fractional_rotations: torch.Tensor,
    cell_weight: float = 0.5,
    position_weight: float = 0.5,
) -> torch.Tensor:
    """Approximate operation residual for a class-level reference embedding.

    A best translation and same-species Hungarian assignment are selected for
    every rotation. The discrete choice is treated as fixed in backpropagation.
    Material-specific embedded operations should be supplied by preprocessing
    in production work.
    """

    dtype, device = positions.dtype, positions.device
    frac = torch.linalg.solve(cell.T, positions.T).T
    frac = frac - torch.floor(frac)
    rotations = fractional_rotations.to(device=device, dtype=dtype)
    metric = cell @ cell.T
    metric_scale = metric.square().mean().clamp_min(1.0e-12)
    species = atomic_numbers.detach().cpu().numpy()
    anchor_species = species[0]
    anchor_targets = np.flatnonzero(species == anchor_species).tolist()
    operation_costs = []

    for rotation in rotations:
        transformed = frac @ rotation.T
        candidate_costs = []
        for target_index in anchor_targets:
            translation = frac[target_index] - transformed[0]
            moved = transformed + translation
            total = positions.new_zeros(())
            for z in np.unique(species):
                indices = np.flatnonzero(species == z)
                idx = torch.as_tensor(indices, device=device, dtype=torch.long)
                costs = _periodic_squared_distances(moved[idx], frac[idx], cell)
                rows, cols = linear_sum_assignment(costs.detach().cpu().numpy())
                total = total + costs[
                    torch.as_tensor(rows, device=device),
                    torch.as_tensor(cols, device=device),
                ].sum()
            candidate_costs.append(total / max(len(species), 1))
        position_term = torch.stack(candidate_costs).min() / metric.diag().mean().clamp_min(1e-12)
        transformed_metric = rotation.T @ metric @ rotation
        cell_term = (transformed_metric - metric).square().mean() / metric_scale
        operation_costs.append(cell_weight * cell_term + position_weight * position_term)
    return torch.stack(operation_costs).mean().clamp_min(0).sqrt()


def active_parameter_names_for_edges(paths: Iterable[list[str]]) -> set[tuple[str, str]]:
    return {
        (parent, child)
        for path in paths
        for parent, child in zip(path[:-1], path[1:])
    }
