from __future__ import annotations

from itertools import product

import pytest
import torch

from src.graphs import build_periodic_graph, collate_periodic_graphs


def _edge_keys(graph) -> set[tuple[int, int, int, int, int]]:
    return {
        (
            int(graph.edge_index[0, edge]),
            int(graph.edge_index[1, edge]),
            *[int(value) for value in graph.cell_shifts[edge]],
        )
        for edge in range(graph.num_edges)
    }


def test_single_atom_cubic_complete_shell_and_strict_boundary() -> None:
    positions = torch.zeros(1, 3, dtype=torch.float64)
    cell = torch.eye(3, dtype=torch.float64)
    species = torch.tensor([14])
    graph = build_periodic_graph(positions, cell, species, cutoff=1.01)
    expected_shifts = {
        (-1, 0, 0),
        (1, 0, 0),
        (0, -1, 0),
        (0, 1, 0),
        (0, 0, -1),
        (0, 0, 1),
    }
    assert graph.num_edges == 6
    assert {tuple(map(int, shift)) for shift in graph.cell_shifts} == expected_shifts
    assert torch.allclose(graph.edge_distances, torch.ones(6, dtype=torch.float64))

    boundary = build_periodic_graph(positions, cell, species, cutoff=1.0)
    assert boundary.num_edges == 0
    assert boundary.edge_index.shape == (2, 0)
    assert boundary.edge_vectors.shape == (0, 3)


def test_distinct_sites_retain_periodic_image_multiedges() -> None:
    graph = build_periodic_graph(
        torch.tensor([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]], dtype=torch.float64),
        torch.eye(3, dtype=torch.float64),
        torch.tensor([1, 1]),
        cutoff=0.6,
    )
    assert _edge_keys(graph) == {
        (1, 0, -1, 0, 0),
        (1, 0, 0, 0, 0),
        (0, 1, 0, 0, 0),
        (0, 1, 1, 0, 0),
    }


def test_skew_cell_adaptive_bounds_match_large_brute_force() -> None:
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [0.71, 0.17, 0.22]], dtype=torch.float64
    )
    cell = torch.tensor(
        [[1.0, 0.0, 0.0], [0.92, 0.31, 0.0], [0.25, 0.13, 0.77]],
        dtype=torch.float64,
    )
    cutoff = 0.52
    graph = build_periodic_graph(positions, cell, torch.tensor([6, 8]), cutoff)
    fractional = torch.linalg.solve(cell.T, positions.T).T
    fractional = fractional - torch.floor(fractional)
    brute = set()
    for target in range(2):
        for source in range(2):
            for shift in product(range(-5, 6), repeat=3):
                if source == target and shift == (0, 0, 0):
                    continue
                vector = (fractional[source] - fractional[target] + torch.tensor(shift)) @ cell
                if torch.linalg.vector_norm(vector) < cutoff:
                    brute.add((source, target, *shift))
    assert _edge_keys(graph) == brute


def test_mixed_size_collation_and_dtype_transfer() -> None:
    first = build_periodic_graph(
        torch.zeros(1, 3, dtype=torch.float64),
        torch.eye(3, dtype=torch.float64),
        torch.tensor([1]),
        cutoff=0.6,
    )
    second = build_periodic_graph(
        torch.tensor([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]], dtype=torch.float64),
        torch.eye(3, dtype=torch.float64),
        torch.tensor([6, 8]),
        cutoff=0.6,
    )
    batch = collate_periodic_graphs((first, second))
    assert (batch.num_graphs, batch.num_nodes, batch.num_edges) == (2, 3, 4)
    assert batch.node_batch.tolist() == [0, 1, 1]
    assert torch.equal(batch.edge_index, second.edge_index + 1)
    moved = batch.to("cpu", dtype=torch.float32)
    assert moved.positions.dtype == torch.float32
    assert moved.edge_vectors.dtype == torch.float32
    assert moved.edge_index.dtype == torch.long
    assert moved.cell_shifts.dtype == torch.long


@pytest.mark.parametrize(
    "rotation",
    [
        torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
        torch.diag(torch.tensor([-1.0, 1.0, 1.0])),
    ],
)
def test_graph_geometry_is_o3_covariant(rotation: torch.Tensor) -> None:
    positions = torch.tensor(
        [[0.1, 0.2, 0.3], [0.6, 0.3, 0.9]], dtype=torch.float64
    )
    cell = torch.tensor(
        [[1.2, 0.0, 0.0], [0.2, 1.1, 0.0], [0.1, 0.3, 1.3]],
        dtype=torch.float64,
    )
    graph = build_periodic_graph(positions, cell, torch.tensor([14, 8]), cutoff=0.9)
    rotation = rotation.to(dtype=torch.float64)
    transformed = build_periodic_graph(
        positions @ rotation.T,
        cell @ rotation.T,
        torch.tensor([14, 8]),
        cutoff=0.9,
    )
    assert torch.equal(transformed.edge_index, graph.edge_index)
    assert torch.equal(transformed.cell_shifts, graph.cell_shifts)
    assert torch.allclose(transformed.edge_vectors, graph.edge_vectors @ rotation.T, atol=1e-12)
    assert torch.allclose(transformed.edge_distances, graph.edge_distances, atol=1e-12)
