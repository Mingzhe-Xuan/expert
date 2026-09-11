from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class PeriodicGraph:
    """A directed periodic multigraph retaining every explicit image edge."""

    positions: torch.Tensor
    cell: torch.Tensor
    atomic_numbers: torch.Tensor
    node_batch: torch.Tensor
    edge_index: torch.Tensor
    cell_shifts: torch.Tensor
    edge_vectors: torch.Tensor
    edge_distances: torch.Tensor
    cutoff: float
    boundary_convention: str = "half_open_distance_lt_cutoff"

    def __post_init__(self) -> None:
        node_count = self.positions.shape[0]
        if self.positions.ndim != 2 or self.positions.shape[1] != 3:
            raise ValueError("positions must have shape [num_nodes, 3]")
        if self.cell.ndim != 3 or self.cell.shape[1:] != (3, 3):
            raise ValueError("cell must have shape [num_graphs, 3, 3]")
        if self.atomic_numbers.shape != (node_count,):
            raise ValueError("atomic_numbers must have shape [num_nodes]")
        if self.atomic_numbers.dtype != torch.long:
            raise TypeError("atomic_numbers must use torch.long")
        if self.node_batch.shape != (node_count,) or self.node_batch.dtype != torch.long:
            raise TypeError("node_batch must be torch.long with shape [num_nodes]")
        if self.edge_index.ndim != 2 or self.edge_index.shape[0] != 2:
            raise ValueError("edge_index must have shape [2, num_edges]")
        if self.edge_index.dtype != torch.long:
            raise TypeError("edge_index must use torch.long")
        edge_count = self.edge_index.shape[1]
        if self.cell_shifts.shape != (edge_count, 3):
            raise ValueError("cell_shifts must have shape [num_edges, 3]")
        if self.cell_shifts.dtype != torch.long:
            raise TypeError("cell_shifts must use torch.long")
        if self.edge_vectors.shape != (edge_count, 3):
            raise ValueError("edge_vectors must have shape [num_edges, 3]")
        if self.edge_distances.shape != (edge_count,):
            raise ValueError("edge_distances must have shape [num_edges]")
        if self.cutoff <= 0:
            raise ValueError("cutoff must be positive")
        if not self.boundary_convention:
            raise ValueError("boundary_convention must be explicit")
        tensors = (
            self.cell,
            self.atomic_numbers,
            self.node_batch,
            self.edge_index,
            self.cell_shifts,
            self.edge_vectors,
            self.edge_distances,
        )
        if any(tensor.device != self.positions.device for tensor in tensors):
            raise ValueError("all PeriodicGraph tensors must share a device")
        if node_count and self.node_batch.numel():
            if int(self.node_batch.min()) < 0 or int(self.node_batch.max()) >= self.cell.shape[0]:
                raise ValueError("node_batch contains an invalid graph index")
        if edge_count:
            if int(self.edge_index.min()) < 0 or int(self.edge_index.max()) >= node_count:
                raise ValueError("edge_index contains an invalid node index")
            expected = torch.linalg.vector_norm(self.edge_vectors, dim=-1)
            if not torch.allclose(expected, self.edge_distances, rtol=1e-5, atol=1e-7):
                raise ValueError("edge_distances do not match edge_vectors")
            if bool((self.edge_distances >= self.cutoff).any()):
                raise ValueError("an edge violates the strict cutoff convention")

    @property
    def num_graphs(self) -> int:
        return self.cell.shape[0]

    @property
    def num_nodes(self) -> int:
        return self.positions.shape[0]

    @property
    def num_edges(self) -> int:
        return self.edge_index.shape[1]

    def to(
        self,
        device: torch.device | str,
        dtype: torch.dtype | None = None,
    ) -> "PeriodicGraph":
        """Move all tensors together; integer topology retains integer dtype."""

        floating_dtype = self.positions.dtype if dtype is None else dtype
        if not floating_dtype.is_floating_point:
            raise TypeError("PeriodicGraph floating tensors require a floating dtype")
        return PeriodicGraph(
            positions=self.positions.to(device=device, dtype=floating_dtype),
            cell=self.cell.to(device=device, dtype=floating_dtype),
            atomic_numbers=self.atomic_numbers.to(device=device),
            node_batch=self.node_batch.to(device=device),
            edge_index=self.edge_index.to(device=device),
            cell_shifts=self.cell_shifts.to(device=device),
            edge_vectors=self.edge_vectors.to(device=device, dtype=floating_dtype),
            edge_distances=self.edge_distances.to(device=device, dtype=floating_dtype),
            cutoff=self.cutoff,
            boundary_convention=self.boundary_convention,
        )
