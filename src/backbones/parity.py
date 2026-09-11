from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

import torch
from torch import nn
from e3nn import o3

from ..graphs import PeriodicGraph, build_periodic_graph, collate_periodic_graphs
from ..irreps import IrrepLayout, IrrepTerm, O3FeatureBatch
from ..symmetry.registry import _layout_irreps


@dataclass(frozen=True, slots=True)
class SO3Term:
    multiplicity: int
    degree: int
    copy_label: str

    def __post_init__(self) -> None:
        if self.multiplicity < 1 or self.degree < 0 or not self.copy_label:
            raise ValueError("SO(3) terms require positive multiplicity, degree, and label")

    @property
    def dimension(self) -> int:
        return self.multiplicity * (2 * self.degree + 1)


@dataclass(frozen=True, slots=True)
class SO3Layout:
    terms: tuple[SO3Term, ...]
    component_order: str = "e3nn_real_y_lm_m_minus_l_to_l"

    def __post_init__(self) -> None:
        if not self.terms:
            raise ValueError("an SO(3) layout cannot be empty")
        labels = [term.copy_label for term in self.terms]
        if len(labels) != len(set(labels)) or not self.component_order:
            raise ValueError("SO(3) copy labels/order must be explicit and unique")

    @property
    def dimension(self) -> int:
        return sum(term.dimension for term in self.terms)


@dataclass(frozen=True, slots=True)
class SO3FeatureBatch:
    node_features: torch.Tensor
    node_layout: SO3Layout
    node_batch: torch.Tensor
    edge_geometry: Mapping[str, torch.Tensor] | None = None

    def __post_init__(self) -> None:
        if self.node_features.ndim != 2 or self.node_features.shape[1] != self.node_layout.dimension:
            raise ValueError("SO(3) node features do not match the declared layout")
        if self.node_batch.shape != (self.node_features.shape[0],):
            raise ValueError("SO(3) node_batch must align with nodes")
        if self.node_batch.dtype != torch.long or self.node_batch.device != self.node_features.device:
            raise TypeError("SO(3) node_batch must be torch.long on the feature device")
        geometry = {} if self.edge_geometry is None else dict(self.edge_geometry)
        object.__setattr__(self, "edge_geometry", MappingProxyType(geometry))


def invert_periodic_graph(graph: PeriodicGraph) -> PeriodicGraph:
    """Invert fractional sites modulo one while retaining each right-handed cell."""

    inverted = []
    for graph_index in range(graph.num_graphs):
        mask = graph.node_batch == graph_index
        cell = graph.cell[graph_index]
        fractional = torch.linalg.solve(cell.T, graph.positions[mask].T).T
        inverted_positions = torch.remainder(-fractional, 1.0) @ cell
        inverted.append(
            build_periodic_graph(
                inverted_positions,
                cell,
                graph.atomic_numbers[mask],
                graph.cutoff,
            )
        )
    return collate_periodic_graphs(inverted)


class InversionPairedReynolds(nn.Module):
    """Turn a frozen SO(3) extractor into an O(3) interface using two real calls."""

    def __init__(self, extractor: nn.Module, source_layout: SO3Layout) -> None:
        super().__init__()
        self.extractor = extractor
        self.source_layout = source_layout
        self.output_layout = IrrepLayout(
            tuple(
                output
                for term in source_layout.terms
                for output in (
                    IrrepTerm(term.multiplicity, term.degree, "e", f"{term.copy_label}_even"),
                    IrrepTerm(term.multiplicity, term.degree, "o", f"{term.copy_label}_odd"),
                )
            ),
            component_order=source_layout.component_order,
        )
        self.extractor.requires_grad_(False)
        self.extractor.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        self.extractor.eval()
        return self

    def forward(self, graph: PeriodicGraph) -> O3FeatureBatch:
        inverted_graph = invert_periodic_graph(graph)
        with torch.no_grad():
            direct = self.extractor(graph)
            inverted = self.extractor(inverted_graph)
        for value, name in ((direct, "direct"), (inverted, "inverted")):
            if not isinstance(value, SO3FeatureBatch):
                raise TypeError(f"{name} SO(3) extractor output has the wrong contract")
            if value.node_layout != self.source_layout:
                raise ValueError(f"{name} SO(3) extractor layout mismatch")
            if not torch.equal(value.node_batch, graph.node_batch):
                raise ValueError(f"{name} SO(3) extractor changed node ordering/batch mapping")
        blocks = []
        offset = 0
        for term in self.source_layout.terms:
            stop = offset + term.dimension
            first = direct.node_features[:, offset:stop]
            second = inverted.node_features[:, offset:stop]
            blocks.extend((0.5 * (first + second), 0.5 * (first - second)))
            offset = stop
        features = torch.cat(blocks, dim=-1)
        return O3FeatureBatch(
            node_features=features,
            node_layout=self.output_layout,
            node_batch=graph.node_batch,
            edge_geometry=(
                direct.edge_geometry
                if direct.edge_geometry
                else {
                    "edge_index": graph.edge_index,
                    "cell_shifts": graph.cell_shifts,
                    "edge_vectors": graph.edge_vectors,
                    "edge_distances": graph.edge_distances,
                }
            ),
        )


class O3InterfaceProjector(nn.Module):
    """Trainable equivariant map from a checkpoint layout to the shared hidden layout."""

    def __init__(self, source_layout: IrrepLayout, target_layout: IrrepLayout) -> None:
        super().__init__()
        self.source_layout = source_layout
        self.target_layout = target_layout
        self.linear = o3.Linear(_layout_irreps(source_layout), _layout_irreps(target_layout))

    def forward(self, features: O3FeatureBatch) -> O3FeatureBatch:
        if features.node_layout != self.source_layout:
            raise ValueError("checkpoint feature layout does not match interface projector")
        return O3FeatureBatch(
            node_features=self.linear(features.node_features),
            node_layout=self.target_layout,
            node_batch=features.node_batch,
            edge_features=features.edge_features,
            edge_layout=features.edge_layout,
            edge_geometry=features.edge_geometry,
        )
