from __future__ import annotations

import copy
from typing import Mapping

import spglib
import torch
from torch import nn

from ..configs import ArchitectureConfig
from ..graphs import PeriodicGraph
from ..heads import TensorPrediction, TensorReadout
from ..irreps import IrrepLayout, IrrepTerm, O3FeatureBatch
from ..symmetry import ParentDAGSpec, PointGroupRegistry, SymmetryRecord
from ..symmetry.registry import canonical_point_group_symbol
from .modules import (
    A1PointGroupExpert,
    ContinuousResidualGate,
    FullPointGroupExpert,
    O3Adaptation,
    RoutedO3Expert,
    active_hall_numbers,
    hierarchical_fusion,
)


def default_hidden_layout(config: ArchitectureConfig) -> IrrepLayout:
    multiplicities = (8, 2, 2, 2, 2) if config.pg_hidden_mode == "a1_only" else (4, 1, 1, 1, 1)
    return IrrepLayout(
        tuple(
            IrrepTerm(
                multiplicity,
                degree,
                "e" if degree % 2 == 0 else "o",
                f"hidden_l{degree}",
            )
            for degree, multiplicity in enumerate(multiplicities)
        )
    )


def _expert_key(number: int) -> str:
    return f"pg{number:02d}"


def _hall_point_group(hall_number: int) -> str:
    value = spglib.get_spacegroup_type(hall_number)
    if value is None:
        raise ValueError(f"invalid Hall number {hall_number}")
    return canonical_point_group_symbol(value.pointgroup_international)


def _extract_graph(graph: PeriodicGraph, graph_index: int) -> tuple[torch.Tensor, PeriodicGraph]:
    node_indices = torch.nonzero(graph.node_batch == graph_index, as_tuple=False).flatten()
    mapping = torch.full(
        (graph.num_nodes,), -1, dtype=torch.long, device=graph.positions.device
    )
    mapping[node_indices] = torch.arange(node_indices.numel(), device=graph.positions.device)
    edge_mask = graph.node_batch[graph.edge_index[0]] == graph_index
    edge_index = mapping[graph.edge_index[:, edge_mask]]
    local = PeriodicGraph(
        positions=graph.positions[node_indices],
        cell=graph.cell[graph_index : graph_index + 1],
        atomic_numbers=graph.atomic_numbers[node_indices],
        node_batch=torch.zeros(node_indices.numel(), dtype=torch.long, device=graph.positions.device),
        edge_index=edge_index,
        cell_shifts=graph.cell_shifts[edge_mask],
        edge_vectors=graph.edge_vectors[edge_mask],
        edge_distances=graph.edge_distances[edge_mask],
        cutoff=graph.cutoff,
        boundary_convention=graph.boundary_convention,
    )
    return node_indices, local


class PointGroupTensorModel(nn.Module):
    """Five-branch downstream dispatcher beginning at real backbone O(3) features."""

    def __init__(
        self,
        architecture: ArchitectureConfig,
        task: str,
        *,
        hidden_layout: IrrepLayout | None = None,
        expert_point_groups: tuple[str, ...] | None = None,
        cutoff: float = 6.0,
    ) -> None:
        super().__init__()
        self.architecture = architecture
        self.task = task
        self.hidden_layout = hidden_layout or default_hidden_layout(architecture)
        self.registry = PointGroupRegistry()
        selected = (
            tuple(group.symbol for group in self.registry)
            if expert_point_groups is None
            else tuple(dict.fromkeys(canonical_point_group_symbol(value) for value in expert_point_groups))
        )
        for symbol in selected:
            self.registry[symbol]
        self.expert_point_groups = selected

        self.adaptation = (
            O3Adaptation(
                self.hidden_layout,
                architecture.adaptation_backend,
                mmax=architecture.o2_mmax,
                cutoff=cutoff,
            )
            if architecture.adaptation_backend != "none"
            else None
        )
        self.o3_experts: nn.ModuleDict | None = None
        self.pg_experts: nn.ModuleDict | None = None
        if architecture.o3e_backend != "none":
            template = RoutedO3Expert(
                self.hidden_layout,
                architecture.o3e_backend,
                mmax=architecture.o2_mmax,
                cutoff=cutoff,
            )
            self.o3_experts = nn.ModuleDict(
                {
                    _expert_key(self.registry[symbol].number): copy.deepcopy(template)
                    for symbol in selected
                }
            )
        if architecture.pg_hidden_mode != "none":
            expert_type = (
                A1PointGroupExpert
                if architecture.pg_hidden_mode == "a1_only"
                else FullPointGroupExpert
            )
            self.pg_experts = nn.ModuleDict(
                {
                    _expert_key(self.registry[symbol].number): expert_type(
                        self.registry[symbol], self.hidden_layout
                    )
                    for symbol in selected
                }
            )
        self.routing_gate = (
            ContinuousResidualGate()
            if self.o3_experts is not None or self.pg_experts is not None
            else None
        )
        self.readout = TensorReadout(
            self.hidden_layout,
            task,
            architecture.readout_backend,
            mmax=architecture.o2_mmax,
            cutoff=cutoff,
        )

    def _active_halls(
        self,
        symmetry: SymmetryRecord,
        parent_dag: ParentDAGSpec | None,
    ) -> tuple[int, ...]:
        if parent_dag is None:
            return (symmetry.hall_number,)
        if parent_dag.current_hall_number != symmetry.hall_number:
            raise ValueError("ParentDAGSpec current Hall number disagrees with symmetry record")
        return active_hall_numbers(parent_dag)

    def _apply_experts(
        self,
        features: torch.Tensor,
        graph: PeriodicGraph,
        symmetries: tuple[SymmetryRecord, ...],
        parent_dags: tuple[ParentDAGSpec | None, ...],
        residuals: tuple[Mapping[int, torch.Tensor | float] | None, ...],
    ) -> torch.Tensor:
        output = torch.empty_like(features)
        for graph_index, symmetry in enumerate(symmetries):
            node_indices, local_graph = _extract_graph(graph, graph_index)
            local_features = features[node_indices]
            halls = self._active_halls(symmetry, parent_dags[graph_index])
            supplied = residuals[graph_index]
            if supplied is None:
                if len(halls) != 1:
                    raise ValueError("parent-active routing requires residuals for every Hall node")
                supplied = {halls[0]: local_features.new_zeros(())}
            weights = self.routing_gate(halls, supplied)
            branches = {}
            for hall in halls:
                symbol = _hall_point_group(hall)
                group = self.registry[symbol]
                if symbol not in self.expert_point_groups:
                    raise ValueError(f"point-group expert {symbol!r} was not instantiated")
                key = _expert_key(group.number)
                if self.o3_experts is not None:
                    branches[hall] = self.o3_experts[key](local_features, local_graph)
                else:
                    branches[hall] = self.pg_experts[key](local_features)
            output[node_indices] = hierarchical_fusion(
                branches, weights, self.hidden_layout
            )
        return output

    def forward(
        self,
        backbone_features: O3FeatureBatch,
        graph: PeriodicGraph,
        symmetries: tuple[SymmetryRecord, ...],
        *,
        parent_dags: tuple[ParentDAGSpec | None, ...] | None = None,
        parent_residuals: tuple[
            Mapping[int, torch.Tensor | float] | None, ...
        ]
        | None = None,
    ) -> TensorPrediction:
        if backbone_features.node_layout != self.hidden_layout:
            raise ValueError("backbone O(3) layout does not match downstream hidden layout")
        if not torch.equal(backbone_features.node_batch, graph.node_batch):
            raise ValueError("backbone and graph node_batch mappings disagree")
        if len(symmetries) != graph.num_graphs:
            raise ValueError("one symmetry record is required per graph")
        parent_dags = parent_dags or (None,) * graph.num_graphs
        parent_residuals = parent_residuals or (None,) * graph.num_graphs
        if len(parent_dags) != graph.num_graphs or len(parent_residuals) != graph.num_graphs:
            raise ValueError("parent routing inputs must align with the graph batch")
        features = backbone_features.node_features
        if self.adaptation is not None:
            features = self.adaptation(features, graph)
        if self.routing_gate is not None:
            features = self._apply_experts(
                features, graph, symmetries, parent_dags, parent_residuals
            )
        return self.readout(features, graph, symmetries)

    def active_nonbackbone_parameter_count(
        self,
        symmetries: tuple[SymmetryRecord, ...],
        parent_dags: tuple[ParentDAGSpec | None, ...] | None = None,
    ) -> int:
        parent_dags = parent_dags or (None,) * len(symmetries)
        modules: list[nn.Module] = [self.readout]
        if self.adaptation is not None:
            modules.append(self.adaptation)
        if self.routing_gate is not None:
            modules.append(self.routing_gate)
            expert_container = self.o3_experts or self.pg_experts
            keys = set()
            for symmetry, dag in zip(symmetries, parent_dags):
                for hall in self._active_halls(symmetry, dag):
                    group = self.registry[_hall_point_group(hall)]
                    keys.add(_expert_key(group.number))
            modules.extend(expert_container[key] for key in sorted(keys))
        parameters = {id(parameter): parameter for module in modules for parameter in module.parameters()}
        return sum(parameter.numel() for parameter in parameters.values())
