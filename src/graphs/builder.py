from __future__ import annotations

from itertools import product
import math
from typing import Iterable

import torch

from .contracts import PeriodicGraph


def _image_limits(cell: torch.Tensor, cutoff: float) -> tuple[int, int, int]:
    inverse = torch.linalg.inv(cell)
    # For v = q @ cell and ||v|| < cutoff, |q_i| is bounded by the norm of
    # inverse[:, i]. Wrapped pair deltas contribute less than one more unit.
    bounds = cutoff * torch.linalg.vector_norm(inverse, dim=0)
    return tuple(math.ceil(float(bound)) + 1 for bound in bounds)


def build_periodic_graph(
    positions: torch.Tensor,
    cell: torch.Tensor,
    atomic_numbers: torch.Tensor,
    cutoff: float,
) -> PeriodicGraph:
    """Deterministically enumerate every directed periodic image edge below cutoff."""

    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("positions must have shape [num_nodes, 3]")
    if cell.shape != (3, 3):
        raise ValueError("cell must have shape [3, 3]")
    if atomic_numbers.shape != (positions.shape[0],) or atomic_numbers.dtype != torch.long:
        raise TypeError("atomic_numbers must be torch.long with shape [num_nodes]")
    if cutoff <= 0 or not math.isfinite(cutoff):
        raise ValueError("cutoff must be finite and positive")
    if positions.device != cell.device or positions.device != atomic_numbers.device:
        raise ValueError("positions, cell and atomic_numbers must share a device")
    if positions.dtype != cell.dtype or not positions.dtype.is_floating_point:
        raise TypeError("positions and cell must share a floating dtype")
    if abs(float(torch.linalg.det(cell))) < 1.0e-12:
        raise ValueError("cell must be invertible")

    fractional = torch.linalg.solve(cell.T, positions.T).T
    fractional = fractional - torch.floor(fractional)
    limits = _image_limits(cell, cutoff)
    shifts = tuple(
        product(
            range(-limits[0], limits[0] + 1),
            range(-limits[1], limits[1] + 1),
            range(-limits[2], limits[2] + 1),
        )
    )
    edge_sources: list[int] = []
    edge_targets: list[int] = []
    accepted_shifts: list[tuple[int, int, int]] = []
    vectors: list[torch.Tensor] = []
    distances: list[torch.Tensor] = []
    cutoff_tensor = positions.new_tensor(cutoff)

    # Target/source/lexicographic-shift ordering is frozen as graph convention v1.
    for target in range(positions.shape[0]):
        for source in range(positions.shape[0]):
            delta = fractional[source] - fractional[target]
            for shift in shifts:
                if source == target and shift == (0, 0, 0):
                    continue
                shift_tensor = positions.new_tensor(shift)
                vector = (delta + shift_tensor) @ cell
                distance = torch.linalg.vector_norm(vector)
                if bool(distance < cutoff_tensor):
                    edge_sources.append(source)
                    edge_targets.append(target)
                    accepted_shifts.append(shift)
                    vectors.append(vector)
                    distances.append(distance)

    if vectors:
        edge_index = torch.tensor(
            (edge_sources, edge_targets), dtype=torch.long, device=positions.device
        )
        cell_shifts = torch.tensor(
            accepted_shifts, dtype=torch.long, device=positions.device
        )
        edge_vectors = torch.stack(vectors)
        edge_distances = torch.stack(distances)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long, device=positions.device)
        cell_shifts = torch.empty((0, 3), dtype=torch.long, device=positions.device)
        edge_vectors = positions.new_empty((0, 3))
        edge_distances = positions.new_empty((0,))

    return PeriodicGraph(
        positions=positions,
        cell=cell.unsqueeze(0),
        atomic_numbers=atomic_numbers,
        node_batch=torch.zeros(positions.shape[0], dtype=torch.long, device=positions.device),
        edge_index=edge_index,
        cell_shifts=cell_shifts,
        edge_vectors=edge_vectors,
        edge_distances=edge_distances,
        cutoff=cutoff,
    )


def collate_periodic_graphs(graphs: Iterable[PeriodicGraph]) -> PeriodicGraph:
    """Collate graphs without losing multiedges or per-crystal cell indices."""

    items = tuple(graphs)
    if not items:
        raise ValueError("at least one graph is required")
    first = items[0]
    if any(graph.positions.device != first.positions.device for graph in items):
        raise ValueError("all graphs must share a device before collation")
    if any(graph.positions.dtype != first.positions.dtype for graph in items):
        raise ValueError("all graphs must share a floating dtype before collation")
    if any(graph.cutoff != first.cutoff for graph in items):
        raise ValueError("all graphs must share an exact cutoff")
    if any(graph.boundary_convention != first.boundary_convention for graph in items):
        raise ValueError("all graphs must share a boundary convention")

    positions = []
    cells = []
    atomic_numbers = []
    node_batches = []
    edge_indices = []
    cell_shifts = []
    edge_vectors = []
    edge_distances = []
    node_offset = 0
    graph_offset = 0
    for graph in items:
        positions.append(graph.positions)
        cells.append(graph.cell)
        atomic_numbers.append(graph.atomic_numbers)
        node_batches.append(graph.node_batch + graph_offset)
        edge_indices.append(graph.edge_index + node_offset)
        cell_shifts.append(graph.cell_shifts)
        edge_vectors.append(graph.edge_vectors)
        edge_distances.append(graph.edge_distances)
        node_offset += graph.num_nodes
        graph_offset += graph.num_graphs

    return PeriodicGraph(
        positions=torch.cat(positions),
        cell=torch.cat(cells),
        atomic_numbers=torch.cat(atomic_numbers),
        node_batch=torch.cat(node_batches),
        edge_index=torch.cat(edge_indices, dim=1),
        cell_shifts=torch.cat(cell_shifts),
        edge_vectors=torch.cat(edge_vectors),
        edge_distances=torch.cat(edge_distances),
        cutoff=first.cutoff,
        boundary_convention=first.boundary_convention,
    )
