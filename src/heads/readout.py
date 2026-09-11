from __future__ import annotations

import math

import torch
from torch import nn

from ..graphs import PeriodicGraph
from ..irreps import IrrepLayout, IrrepTerm
from ..symmetry import SymmetryRecord
from ..symmetry.registry import _layout_irreps
from ..tensor_products import build_tensor_product
from e3nn import o3
from .contracts import TARGET_LAYOUTS, TensorPrediction
from .transforms import (
    apply_bec_asr,
    irreps_to_cartesian,
    project_to_symmetry_operations,
    rotate_cartesian,
)


def _readout_edge_layout(lmax: int = 2) -> IrrepLayout:
    return IrrepLayout(
        tuple(
            IrrepTerm(1, degree, "e" if degree % 2 == 0 else "o", f"edge_l{degree}")
            for degree in range(lmax + 1)
        )
    )


class TensorReadout(nn.Module):
    """Exactly one selected TP placement followed by task-aware pooling/conversion."""

    def __init__(
        self,
        hidden_layout: IrrepLayout,
        task: str,
        backend: str,
        *,
        mmax: int = 2,
        cutoff: float = 6.0,
    ) -> None:
        super().__init__()
        if task not in TARGET_LAYOUTS:
            raise ValueError(f"unsupported target task {task!r}")
        self.hidden_layout = hidden_layout
        self.target_layout = TARGET_LAYOUTS[task]
        self.task = task
        self.backend = backend
        self.scope = "node" if task == "bec" else "global"
        self.cutoff = cutoff
        self.edge_layout = _readout_edge_layout()
        self.tensor_product = build_tensor_product(
            backend, hidden_layout, self.edge_layout, self.target_layout, mmax=mmax
        )
        self.self_connection = o3.Linear(
            _layout_irreps(hidden_layout), _layout_irreps(self.target_layout)
        )
        self.radial = nn.Sequential(nn.Linear(1, 8), nn.SiLU(), nn.Linear(8, 1))

    def coefficient_forward(
        self,
        features: torch.Tensor,
        graph: PeriodicGraph,
        symmetries: tuple[SymmetryRecord, ...],
    ) -> torch.Tensor:
        if features.shape != (graph.num_nodes, self.hidden_layout.dimension):
            raise ValueError("readout features do not match graph/hidden layout")
        if len(symmetries) != graph.num_graphs:
            raise ValueError("one SymmetryRecord is required per graph")
        node_coefficients = self.self_connection(features)
        if graph.num_edges:
            source, target = graph.edge_index
            edge_features = o3.spherical_harmonics(
                _layout_irreps(self.edge_layout),
                graph.edge_vectors,
                normalize=True,
                normalization="component",
            )
            messages = self.tensor_product(
                features[source], edge_features, graph.edge_vectors
            )
            normalized_distance = (graph.edge_distances / self.cutoff).unsqueeze(-1)
            envelope = 0.5 * (torch.cos(math.pi * normalized_distance) + 1.0)
            messages = messages * self.radial(normalized_distance) * envelope
            aggregate = torch.zeros_like(node_coefficients).index_add(0, target, messages)
            degree = torch.bincount(target, minlength=graph.num_nodes).to(features.dtype)
            node_coefficients = node_coefficients + aggregate / degree.clamp_min(1).sqrt()[:, None]
        if self.scope == "node":
            return node_coefficients
        pooled = node_coefficients.new_zeros((graph.num_graphs, self.target_layout.dimension))
        pooled.index_add_(0, graph.node_batch, node_coefficients)
        counts = torch.bincount(graph.node_batch, minlength=graph.num_graphs).to(features.dtype)
        pooled = pooled / counts.clamp_min(1).unsqueeze(-1)
        constrained = []
        for graph_index, symmetry in enumerate(symmetries):
            constrained.append(
                project_to_symmetry_operations(
                    pooled[graph_index : graph_index + 1],
                    self.task,
                    symmetry.rotations,
                )
            )
        return torch.cat(constrained)

    def forward(
        self,
        features: torch.Tensor,
        graph: PeriodicGraph,
        symmetries: tuple[SymmetryRecord, ...],
    ) -> TensorPrediction:
        coefficients = self.coefficient_forward(features, graph, symmetries)
        canonical = irreps_to_cartesian(coefficients, self.task)
        if self.scope == "global":
            raw = torch.cat(
                [
                    rotate_cartesian(
                        canonical[index : index + 1],
                        symmetries[index].canonical_frame,
                        self.task,
                    )
                    for index in range(graph.num_graphs)
                ]
            )
            return TensorPrediction(
                raw_cartesian=raw,
                irrep_coefficients=coefficients,
                task=self.task,
                scope="global",
                canonical_cartesian=canonical,
            )
        raw = torch.empty_like(canonical)
        for graph_index, symmetry in enumerate(symmetries):
            node_mask = graph.node_batch == graph_index
            raw[node_mask] = rotate_cartesian(
                canonical[node_mask], symmetry.canonical_frame, "bec"
            )
        return TensorPrediction(
            raw_cartesian=raw,
            irrep_coefficients=coefficients,
            task="bec",
            scope="node",
            node_batch=graph.node_batch,
            canonical_cartesian=canonical,
            asr_cartesian=apply_bec_asr(raw, graph.node_batch),
            symmetry_control_cartesian=None,
        )
