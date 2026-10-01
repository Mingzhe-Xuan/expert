from __future__ import annotations

import copy
from contextlib import nullcontext
from numbers import Integral
from typing import Mapping, Sequence

import torch
from torch import nn

from ..configs import ArchitectureConfig
from ..graphs import PeriodicGraph, collate_periodic_graphs
from ..heads import TensorPrediction, TensorReadout
from ..irreps import IrrepLayout, IrrepTerm, O3FeatureBatch
from ..symmetry import (
    ParentDAGSpec,
    PointGroupAncestorDAG,
    PointGroupRegistry,
    SymmetryRecord,
)
from ..symmetry.registry import canonical_point_group_symbol
from .modules import (
    A1PointGroupExpert,
    ContinuousEdgeGate,
    ContinuousResidualGate,
    FullPointGroupExpert,
    O3Adaptation,
    RoutedO3Expert,
    active_parent_point_group_numbers,
    material_point_group_stick_breaking_weights,
)
from .optimized import GroupedFullPointGroupExpert, VectorizedMaterialWeights
from .chain_routing import HierarchicalChainRouter, VectorizedChainRouter


def hidden_layout_from_multiplicities(multiplicities: Sequence[int]) -> IrrepLayout:
    supplied = tuple(multiplicities)
    if len(supplied) != 5 or any(
        isinstance(value, bool) or not isinstance(value, Integral) or value <= 0
        for value in supplied
    ):
        raise ValueError("hidden multiplicities must contain five positive integers")
    values = tuple(int(value) for value in supplied)
    return IrrepLayout(
        tuple(
            IrrepTerm(
                multiplicity,
                degree,
                "e" if degree % 2 == 0 else "o",
                f"hidden_l{degree}",
            )
            for degree, multiplicity in enumerate(values)
        )
    )


def default_hidden_layout(config: ArchitectureConfig) -> IrrepLayout:
    return hidden_layout_from_multiplicities((8, 2, 2, 2, 2))


def _expert_key(number: int) -> str:
    return f"pg{number:02d}"


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
        point_group_parent_dag: PointGroupAncestorDAG | None = None,
        material_edge_ids: tuple[str, ...] | None = None,
        cutoff: float = 6.0,
        grouped_pg_gates: bool = True,
        vectorized_pg_routing: bool = True,
        pg_weighting: str = "legacy",
        chain_temperature: float = 1.0,
        initial_sigma: float = 0.08,
        sigma_floor: float = 1e-8,
        pg_expert_depth: int = 2,
    ) -> None:
        super().__init__()
        self.architecture = architecture
        self.task = task
        if isinstance(pg_expert_depth, bool) or not isinstance(pg_expert_depth, int) or pg_expert_depth < 1:
            raise ValueError("PG expert depth must be a positive integer")
        if pg_expert_depth != 2 and architecture.pg_hidden_mode != "full_pg":
            raise ValueError("custom PG expert depth requires full_pg")
        self.pg_expert_depth = pg_expert_depth
        self.grouped_pg_gates = grouped_pg_gates
        self.vectorized_pg_routing = vectorized_pg_routing
        if pg_weighting not in {"legacy", "within_cross_chain"}:
            raise ValueError("pg_weighting must be legacy or within_cross_chain")
        self.pg_weighting = pg_weighting
        self._material_weights = VectorizedMaterialWeights()
        if point_group_parent_dag is not None and material_edge_ids is not None:
            raise ValueError("static and residual-weighted PG routing configurations are exclusive")
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
        self.point_group_parent_dag = point_group_parent_dag
        self.material_edge_ids = material_edge_ids
        self._cuda_expert_streams: dict[tuple[int, int], torch.cuda.Stream] = {}
        self.last_dispatch_stats: dict[str, int | bool] = {}

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
                else (GroupedFullPointGroupExpert if grouped_pg_gates else FullPointGroupExpert)
            )
            self.pg_experts = nn.ModuleDict(
                {
                    _expert_key(self.registry[symbol].number): expert_type(
                        self.registry[symbol], self.hidden_layout,
                        **({"depth": pg_expert_depth} if architecture.pg_hidden_mode == "full_pg" else {}),
                    )
                    for symbol in selected
                }
            )
        self.routing_gate = (
            ContinuousResidualGate()
            if self.o3_experts is not None or self.pg_experts is not None
            else None
        )
        self.uses_chain_router = (
            pg_weighting == "within_cross_chain" and architecture.pg_hidden_mode == "full_pg"
            and self.o3_experts is None and material_edge_ids is not None
        )
        self.edge_gate = None
        if material_edge_ids is not None:
            if self.uses_chain_router:
                router_type = VectorizedChainRouter if vectorized_pg_routing else HierarchicalChainRouter
                self.edge_gate = router_type(tuple(sorted(material_edge_ids)), initial_sigma=initial_sigma,
                    sigma_floor=sigma_floor, temperature=chain_temperature)
            else:
                self.edge_gate = ContinuousEdgeGate(tuple(sorted(material_edge_ids)), initial_sigma=initial_sigma)
        self.readout = TensorReadout(
            self.hidden_layout,
            task,
            architecture.readout_backend,
            mmax=architecture.o2_mmax,
            cutoff=cutoff,
        )

    def pg_routing_metadata(self):
        if not self.uses_chain_router:
            return None
        return {"algorithm": "near-current-stick-breaking-immediate-parent-softmax-v1",
                "temperature": self.edge_gate.temperature, "sigma_floor": self.edge_gate.sigma_floor,
                "edge_ids": list(self.edge_gate.edge_ids), "residual_gradients": "detached_metadata"}

    def _active_parent_point_groups(
        self,
        symmetry: SymmetryRecord,
        parent_dag: ParentDAGSpec | None,
    ) -> tuple[int, ...]:
        if parent_dag is None:
            return (self.registry[symmetry.current_point_group].number,)
        current = self.registry[symmetry.current_point_group].number
        if parent_dag.current_point_group_number != current:
            raise ValueError("ParentDAGSpec current point group disagrees with symmetry record")
        return active_parent_point_group_numbers(parent_dag)

    def _apply_experts(
        self,
        features: torch.Tensor,
        graph: PeriodicGraph,
        symmetries: tuple[SymmetryRecord, ...],
        parent_dags: tuple[ParentDAGSpec | None, ...],
        residuals: tuple[Mapping[str, torch.Tensor | float] | None, ...],
        point_group_numbers: tuple[int | None, ...],
    ) -> torch.Tensor:
        node_indices_by_graph = []
        local_graphs = []
        weights_by_graph = []
        expert_buckets: dict[int, list[int]] = {}
        for graph_index, symmetry in enumerate(symmetries):
            if self.vectorized_pg_routing and self.o3_experts is None:
                node_indices = torch.nonzero(graph.node_batch == graph_index, as_tuple=False).flatten()
                local_graph = None
            else:
                node_indices, local_graph = _extract_graph(graph, graph_index)
            node_indices_by_graph.append(node_indices)
            local_graphs.append(local_graph)
            local_features = features[node_indices]
            point_group_number = point_group_numbers[graph_index]
            if point_group_number is not None:
                if self.point_group_parent_dag is None:
                    raise ValueError("point-group number routing requires an offline parent DAG")
                if parent_dags[graph_index] is not None or residuals[graph_index] is not None:
                    raise ValueError("static and residual-weighted PG routing inputs cannot mix")
                current = self.registry[symmetry.current_point_group].number
                if int(point_group_number) != current:
                    raise ValueError("stored point-group number disagrees with symmetry record")
                active_ids = self.point_group_parent_dag.ancestors(current)
                weight = local_features.new_tensor(1.0 / len(active_ids))
                weights = {number: weight for number in active_ids}
            else:
                if self.point_group_parent_dag is not None:
                    raise ValueError("offline parent-DAG model requires a point-group number")
                active_numbers = self._active_parent_point_groups(
                    symmetry, parent_dags[graph_index]
                )
                supplied = residuals[graph_index]
                if supplied is None:
                    if len(active_numbers) != 1:
                        raise ValueError("parent-active routing requires point-group edge residuals")
                    supplied = {active_numbers[0]: local_features.new_zeros(())}
                material_dag = parent_dags[graph_index]
                if material_dag is None:
                    weights = self.routing_gate(active_numbers, supplied)
                else:
                    if self.edge_gate is None:
                        raise ValueError("material PG routing requires configured cover-edge IDs")
                    if self.uses_chain_router:
                        weights = self.edge_gate(material_dag, supplied).alpha
                    else:
                        weight_function = (
                            self._material_weights if self.vectorized_pg_routing and self.o3_experts is None
                            else material_point_group_stick_breaking_weights
                        )
                        weights = weight_function(material_dag, supplied, self.edge_gate)
                active_ids = tuple(sorted(weights))
            stacked_weights = torch.stack(
                [torch.as_tensor(weights[active_id]).to(local_features) for active_id in active_ids]
            )
            if bool((stacked_weights < 0).any()) or not torch.allclose(
                stacked_weights.sum(),
                stacked_weights.new_ones(()),
                atol=1.0e-6,
                rtol=1.0e-6,
            ):
                raise ValueError("fusion weights must be non-negative and normalized")
            weights_by_graph.append(weights)
            for active_id in active_ids:
                group = self.registry[active_id]
                symbol = group.symbol
                if symbol not in self.expert_point_groups:
                    raise ValueError(f"point-group expert {symbol!r} was not instantiated")
                expert_buckets.setdefault(active_id, []).append(graph_index)

        output = torch.zeros_like(features)
        pending = []
        current_stream = torch.cuda.current_stream(features.device) if features.is_cuda else None
        for active_id in sorted(expert_buckets):
            graph_indices = expert_buckets[active_id]
            node_indices = torch.cat(
                [node_indices_by_graph[graph_index] for graph_index in graph_indices]
            )
            expert_features = features[node_indices]
            expert_graph = (
                collate_periodic_graphs(
                    local_graphs[graph_index] for graph_index in graph_indices
                )
                if self.o3_experts is not None
                else None
            )
            key = _expert_key(active_id)

            stream = None
            if current_stream is not None:
                device_index = features.device.index
                if device_index is None:
                    device_index = torch.cuda.current_device()
                stream_key = (device_index, active_id)
                stream = self._cuda_expert_streams.get(stream_key)
                if stream is None:
                    stream = torch.cuda.Stream(device=features.device)
                    self._cuda_expert_streams[stream_key] = stream
                stream.wait_stream(current_stream)
            context = torch.cuda.stream(stream) if stream is not None else nullcontext()
            with context:
                if self.o3_experts is not None:
                    expert_output = self.o3_experts[key](expert_features, expert_graph)
                else:
                    expert_output = self.pg_experts[key](expert_features)
            pending.append(
                (active_id, graph_indices, node_indices, expert_output, stream)
            )

        for active_id, graph_indices, node_indices, expert_output, stream in pending:
            if current_stream is not None:
                current_stream.wait_stream(stream)
                expert_output.record_stream(current_stream)
            node_weights = torch.cat(
                [
                    torch.as_tensor(weights_by_graph[graph_index][active_id])
                    .to(expert_output)
                    .expand(node_indices_by_graph[graph_index].numel())
                    for graph_index in graph_indices
                ]
            ).unsqueeze(-1)
            output = output.index_add(0, node_indices, expert_output * node_weights)
        self.last_dispatch_stats = {
            "expert_buckets": len(expert_buckets),
            "max_structures_per_expert": max(map(len, expert_buckets.values())),
            "asynchronous_cuda": current_stream is not None and len(expert_buckets) > 1,
            "cuda_streams": len(
                {id(stream) for *_, stream in pending if stream is not None}
            ),
        }
        return output

    def forward(
        self,
        backbone_features: O3FeatureBatch,
        graph: PeriodicGraph,
        symmetries: tuple[SymmetryRecord, ...],
        *,
        parent_dags: tuple[ParentDAGSpec | None, ...] | None = None,
        parent_residuals: tuple[
            Mapping[str, torch.Tensor | float] | None, ...
        ]
        | None = None,
        point_group_numbers: tuple[int | None, ...] | None = None,
    ) -> TensorPrediction:
        if backbone_features.node_layout != self.hidden_layout:
            raise ValueError("backbone O(3) layout does not match downstream hidden layout")
        if not torch.equal(backbone_features.node_batch, graph.node_batch):
            raise ValueError("backbone and graph node_batch mappings disagree")
        if len(symmetries) != graph.num_graphs:
            raise ValueError("one symmetry record is required per graph")
        parent_dags = parent_dags or (None,) * graph.num_graphs
        parent_residuals = parent_residuals or (None,) * graph.num_graphs
        point_group_numbers = point_group_numbers or (None,) * graph.num_graphs
        if (
            len(parent_dags) != graph.num_graphs
            or len(parent_residuals) != graph.num_graphs
            or len(point_group_numbers) != graph.num_graphs
        ):
            raise ValueError("parent routing inputs must align with the graph batch")
        features = backbone_features.node_features
        if self.adaptation is not None:
            features = self.adaptation(features, graph)
        if self.routing_gate is not None:
            features = self._apply_experts(
                features,
                graph,
                symmetries,
                parent_dags,
                parent_residuals,
                point_group_numbers,
            )
        return self.readout(features, graph, symmetries)

    def active_nonbackbone_parameter_count(
        self,
        symmetries: tuple[SymmetryRecord, ...],
        parent_dags: tuple[ParentDAGSpec | None, ...] | None = None,
        point_group_numbers: tuple[int | None, ...] | None = None,
    ) -> int:
        parent_dags = parent_dags or (None,) * len(symmetries)
        point_group_numbers = point_group_numbers or (None,) * len(symmetries)
        modules: list[nn.Module] = [self.readout]
        if self.adaptation is not None:
            modules.append(self.adaptation)
        if self.routing_gate is not None:
            if self.edge_gate is not None and any(dag is not None for dag in parent_dags):
                modules.append(self.edge_gate)
            elif self.point_group_parent_dag is None:
                modules.append(self.routing_gate)
            expert_container = self.o3_experts or self.pg_experts
            keys = set()
            for symmetry, dag, point_group_number in zip(
                symmetries, parent_dags, point_group_numbers
            ):
                if point_group_number is not None:
                    if self.point_group_parent_dag is None:
                        raise ValueError("point-group number routing requires an offline parent DAG")
                    if self.registry[symmetry.current_point_group].number != point_group_number:
                        raise ValueError("stored point-group number disagrees with symmetry record")
                    numbers = self.point_group_parent_dag.ancestors(point_group_number)
                    keys.update(_expert_key(number) for number in numbers)
                else:
                    if self.point_group_parent_dag is not None:
                        raise ValueError("offline parent-DAG model requires a point-group number")
                    for number in self._active_parent_point_groups(symmetry, dag):
                        keys.add(_expert_key(number))
            modules.extend(expert_container[key] for key in sorted(keys))
        parameters = {id(parameter): parameter for module in modules for parameter in module.parameters()}
        return sum(parameter.numel() for parameter in parameters.values())

    def active_expert_counts(
        self,
        symmetries: tuple[SymmetryRecord, ...],
        parent_dags: tuple[ParentDAGSpec | None, ...] | None = None,
        point_group_numbers: tuple[int | None, ...] | None = None,
    ) -> tuple[int, ...]:
        """Return the number of deduplicated routed expert branches per sample."""

        parent_dags = parent_dags or (None,) * len(symmetries)
        point_group_numbers = point_group_numbers or (None,) * len(symmetries)
        if len(parent_dags) != len(symmetries) or len(point_group_numbers) != len(symmetries):
            raise ValueError("parent DAGs must align with symmetries")
        if self.routing_gate is None:
            return (0,) * len(symmetries)
        counts = []
        for symmetry, dag, point_group_number in zip(
            symmetries, parent_dags, point_group_numbers
        ):
            if point_group_number is not None:
                if self.point_group_parent_dag is None:
                    raise ValueError("point-group number routing requires an offline parent DAG")
                if self.registry[symmetry.current_point_group].number != point_group_number:
                    raise ValueError("stored point-group number disagrees with symmetry record")
                counts.append(len(self.point_group_parent_dag.ancestors(point_group_number)))
            else:
                if self.point_group_parent_dag is not None:
                    raise ValueError("offline parent-DAG model requires a point-group number")
                counts.append(
                    len(
                        {
                            number
                            for number in self._active_parent_point_groups(symmetry, dag)
                        }
                    )
                )
        return tuple(counts)
