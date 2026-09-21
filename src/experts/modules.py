from __future__ import annotations

import math
from types import MappingProxyType
from typing import Mapping

import torch
from torch import nn

from ..graphs import PeriodicGraph
from ..irreps import IrrepLayout, build_subduction_plan
from ..symmetry import ParentDAGSpec, PointGroup
from ..symmetry.registry import _layout_irreps
from ..tensor_products import build_tensor_product
from e3nn import o3


def _edge_layout(lmax: int) -> IrrepLayout:
    from ..irreps import IrrepTerm

    return IrrepLayout(
        tuple(
            IrrepTerm(1, degree, "e" if degree % 2 == 0 else "o", f"edge_l{degree}")
            for degree in range(lmax + 1)
        )
    )


class O3MessageBlock(nn.Module):
    """One radial cutoff PBC message block using a selected O(3) TP backend."""

    def __init__(
        self,
        layout: IrrepLayout,
        backend: str,
        *,
        edge_lmax: int = 2,
        mmax: int = 2,
        cutoff: float = 6.0,
    ) -> None:
        super().__init__()
        self.layout = layout
        self.edge_layout = _edge_layout(edge_lmax)
        self.backend = backend
        self.cutoff = cutoff
        self.tensor_product = build_tensor_product(
            backend, layout, self.edge_layout, layout, mmax=mmax
        )
        self.self_connection = o3.Linear(_layout_irreps(layout), _layout_irreps(layout))
        self.radial = nn.Sequential(nn.Linear(1, 8), nn.SiLU(), nn.Linear(8, 1))
        self.residual_scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, features: torch.Tensor, graph: PeriodicGraph) -> torch.Tensor:
        if features.shape != (graph.num_nodes, self.layout.dimension):
            raise ValueError("node features do not match graph/layout")
        self_term = self.self_connection(features)
        if graph.num_edges == 0:
            return self.residual_scale * features + self_term
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
        radial = self.radial(normalized_distance) * envelope
        messages = messages * radial
        aggregated = torch.zeros_like(features).index_add(0, target, messages)
        degree = torch.bincount(target, minlength=graph.num_nodes).to(features.dtype)
        aggregated = aggregated / degree.clamp_min(1).sqrt().unsqueeze(-1)
        return self.residual_scale * features + self_term + aggregated


class O3Adaptation(nn.Module):
    def __init__(self, layout: IrrepLayout, backend: str, *, mmax: int = 2, cutoff: float = 6.0):
        super().__init__()
        self.block = O3MessageBlock(layout, backend, mmax=mmax, cutoff=cutoff)

    def forward(self, features: torch.Tensor, graph: PeriodicGraph) -> torch.Tensor:
        return self.block(features, graph)


class RoutedO3Expert(nn.Module):
    """Two independent O(3) message blocks for one routed active expert."""

    def __init__(self, layout: IrrepLayout, backend: str, *, mmax: int = 2, cutoff: float = 6.0):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                O3MessageBlock(layout, backend, mmax=mmax, cutoff=cutoff),
                O3MessageBlock(layout, backend, mmax=mmax, cutoff=cutoff),
            ]
        )

    def forward(self, features: torch.Tensor, graph: PeriodicGraph) -> torch.Tensor:
        for block in self.blocks:
            features = block(features, graph)
        return features


class _A1Block(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.linear1 = nn.Linear(width, width)
        self.linear2 = nn.Linear(width, width)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return values + self.linear2(torch.nn.functional.silu(self.linear1(values)))


class A1PointGroupExpert(nn.Module):
    """Two-block expert acting only on provenance-preserving invariant coordinates."""

    def __init__(self, group: PointGroup, layout: IrrepLayout, *, bypass_c1: bool = True) -> None:
        super().__init__()
        self.group_symbol = group.symbol
        self.layout = layout
        basis = group.invariant_basis(layout)
        self.register_buffer("basis", basis)
        if group.symbol == "1" and bypass_c1:
            self.blocks = nn.ModuleList((nn.Identity(), nn.Identity()))
        else:
            self.blocks = nn.ModuleList((_A1Block(basis.shape[1]), _A1Block(basis.shape[1])))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.shape[-1] != self.layout.dimension:
            raise ValueError("features do not match A1 expert layout")
        values = features @ self.basis.to(features)
        for block in self.blocks:
            values = block(values)
        return values @ self.basis.to(features).T


class _FiniteGroupBlock(nn.Module):
    def __init__(self, representation: torch.Tensor, copy_slices: tuple[slice, ...]) -> None:
        super().__init__()
        dimension = representation.shape[1]
        self.register_buffer("representation", representation)
        self.register_buffer("invariant_projector", representation.mean(0))
        self.weight = nn.Parameter(torch.empty(dimension, dimension))
        self.bias = nn.Parameter(torch.zeros(dimension))
        self.gate_gain = nn.Parameter(torch.ones(len(copy_slices)))
        self.gate_bias = nn.Parameter(torch.zeros(len(copy_slices)))
        self.copy_slices = copy_slices
        nn.init.xavier_uniform_(self.weight)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        representation = self.representation.to(features)
        equivariant_weight = torch.einsum(
            "gij,jk,glk->il", representation, self.weight, representation
        ) / representation.shape[0]
        invariant_bias = self.bias @ self.invariant_projector.to(features)
        hidden = features @ equivariant_weight + invariant_bias
        gated = []
        for index, copy_slice in enumerate(self.copy_slices):
            block = hidden[:, copy_slice]
            block_representation = representation[:, copy_slice, copy_slice]
            trivial = block.shape[1] == 1 and torch.allclose(
                block_representation, torch.ones_like(block_representation), atol=1e-8, rtol=1e-8
            )
            if trivial:
                gated.append(torch.nn.functional.silu(block))
            else:
                norm = torch.linalg.vector_norm(block, dim=-1, keepdim=True)
                gate = torch.sigmoid(self.gate_gain[index] * norm + self.gate_bias[index])
                gated.append(block * gate)
        return features + torch.cat(gated, dim=-1)


class FullPointGroupExpert(nn.Module):
    """Two independent blocks retaining all finite-group irreducible carriers."""

    def __init__(self, group: PointGroup, layout: IrrepLayout, *, bypass_c1: bool = True) -> None:
        super().__init__()
        self.group_symbol = group.symbol
        self.layout = layout
        plan = build_subduction_plan(group, layout)
        self.register_buffer("subduction", plan.matrix)
        self.register_buffer("pg_representation", plan.representation)
        self.copy_metadata = plan.copies
        slices = tuple(slice(copy.start, copy.stop) for copy in plan.copies)
        if group.symbol == "1" and bypass_c1:
            self.blocks = nn.ModuleList((nn.Identity(), nn.Identity()))
        else:
            self.blocks = nn.ModuleList(
                (
                    _FiniteGroupBlock(plan.representation, slices),
                    _FiniteGroupBlock(plan.representation, slices),
                )
            )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.shape[-1] != self.layout.dimension:
            raise ValueError("features do not match Full-PG expert layout")
        values = features @ self.subduction.to(features)
        for block in self.blocks:
            values = block(values)
        return values @ self.subduction.to(features).T


class ContinuousResidualGate(nn.Module):
    """Differentiable normalized gates over a deduplicated validated active set."""

    def __init__(self, initial_sigma: float = 0.08) -> None:
        super().__init__()
        if initial_sigma <= 0:
            raise ValueError("initial_sigma must be positive")
        self.log_sigma = nn.Parameter(torch.tensor(math.log(math.expm1(initial_sigma))))

    @property
    def sigma(self) -> torch.Tensor:
        return torch.nn.functional.softplus(self.log_sigma).clamp_min(1.0e-8)

    def forward(
        self, active_ids: tuple[int, ...], residuals: Mapping[int, torch.Tensor | float]
    ) -> Mapping[int, torch.Tensor]:
        deduplicated = tuple(dict.fromkeys(active_ids))
        if not deduplicated or set(deduplicated) != set(residuals):
            raise ValueError("residuals must cover exactly the deduplicated active set")
        reference = next(iter(residuals.values()))
        if not isinstance(reference, torch.Tensor):
            reference = self.log_sigma
        values = torch.stack(
            [torch.as_tensor(residuals[item], dtype=reference.dtype, device=reference.device) for item in deduplicated]
        )
        if bool((values < 0).any()) or not torch.isfinite(values).all():
            raise ValueError("symmetry residuals must be finite and non-negative")
        weights = torch.softmax(-values / self.sigma.to(values), dim=0)
        return MappingProxyType({item: weight for item, weight in zip(deduplicated, weights)})


class ContinuousEdgeGate(nn.Module):
    """Edge-specific positive scales for analytic parent-to-child breaking gates."""

    def __init__(self, edge_ids: tuple[str, ...], initial_sigma: float = 0.08) -> None:
        super().__init__()
        if initial_sigma <= 0 or len(edge_ids) != len(set(edge_ids)):
            raise ValueError("edge gate requires positive sigma and unique edge IDs")
        initial = math.log(math.expm1(initial_sigma))
        self.log_sigmas = nn.ParameterDict(
            {f"edge_{edge_id}": nn.Parameter(torch.tensor(initial)) for edge_id in edge_ids}
        )

    def value(self, edge_id: str, residual: torch.Tensor | float) -> torch.Tensor:
        """Evaluate one residual instance with its shared offline edge scale."""

        key = f"edge_{edge_id}"
        if key not in self.log_sigmas:
            raise ValueError(f"unconfigured offline edge ID: {edge_id}")
        parameter = self.log_sigmas[key]
        value = torch.as_tensor(residual, dtype=parameter.dtype, device=parameter.device)
        if not torch.isfinite(value) or bool(value < 0):
            raise ValueError("edge residuals must be finite and non-negative")
        sigma = torch.nn.functional.softplus(parameter).clamp_min(1.0e-8)
        return -torch.expm1(-torch.square(value / sigma))

    def forward(
        self, residuals: Mapping[str, torch.Tensor | float]
    ) -> Mapping[str, torch.Tensor]:
        return MappingProxyType(
            {edge_id: self.value(edge_id, residual) for edge_id, residual in residuals.items()}
        )


def active_parent_point_group_numbers(parent_dag: ParentDAGSpec) -> tuple[int, ...]:
    values = {parent_dag.current_point_group_number}
    for embedding in parent_dag.embeddings:
        values.add(embedding.parent_point_group_number)
        values.add(embedding.child_point_group_number)
    return tuple(sorted(values))


def material_point_group_stick_breaking_weights(
    parent_dag: ParentDAGSpec,
    residuals: Mapping[str, torch.Tensor | float],
    gate: ContinuousEdgeGate,
) -> Mapping[int, torch.Tensor]:
    """Apply analytic edge gates, per-path stick-breaking, and length path priors."""

    active = active_parent_point_group_numbers(parent_dag)
    if set(residuals) != {embedding.checksum for embedding in parent_dag.embeddings}:
        raise ValueError("edge residuals must cover exactly the point-group DAG edges")
    edge_gates = {
        embedding.checksum: gate.value(
            embedding.edge_id, residuals[embedding.checksum]
        )
        for embedding in parent_dag.embeddings
    }
    paths = parent_dag.current_to_root_embedding_paths()
    if not paths:
        raise ValueError("point-group DAG must contain at least one current-to-root path")
    total_length = sum(len(path) + 1 for path in paths)
    reference = next(iter(edge_gates.values()), None)
    if reference is None:
        reference = torch.zeros(())
    combined = {
        hall: torch.zeros((), dtype=reference.dtype, device=reference.device)
        for hall in active
    }
    for path in paths:
        path_prior = reference.new_tensor((len(path) + 1) / total_length)
        remaining = reference.new_ones(())
        for embedding in reversed(path):
            edge_gate = edge_gates[embedding.checksum]
            combined[embedding.parent_point_group_number] = (
                combined[embedding.parent_point_group_number]
                + path_prior * remaining * (1.0 - edge_gate)
            )
            remaining = remaining * edge_gate
        combined[parent_dag.current_point_group_number] = (
            combined[parent_dag.current_point_group_number] + path_prior * remaining
        )
    normalizer = torch.stack(tuple(combined.values())).sum()
    if not torch.isfinite(normalizer) or bool(normalizer <= 0):
        raise ValueError("point-group path weights must have a finite positive normalizer")
    return MappingProxyType({hall: value / normalizer for hall, value in combined.items()})


def hierarchical_fusion(
    branches: Mapping[int, torch.Tensor],
    weights: Mapping[int, torch.Tensor],
    layout: IrrepLayout,
) -> torch.Tensor:
    if not branches or set(branches) != set(weights):
        raise ValueError("fusion branches and weights must have identical non-empty keys")
    values = list(branches.values())
    if any(value.shape != values[0].shape or value.shape[-1] != layout.dimension for value in values):
        raise ValueError("all fusion branches must share the public O(3) layout and shape")
    stacked_weights = torch.stack([torch.as_tensor(weights[key]).to(values[0]) for key in branches])
    if bool((stacked_weights < 0).any()) or not torch.allclose(
        stacked_weights.sum(), stacked_weights.new_ones(()), atol=1e-6, rtol=1e-6
    ):
        raise ValueError("fusion weights must be non-negative and normalized")
    return torch.stack(
        [branches[key] * stacked_weights[index] for index, key in enumerate(branches)]
    ).sum(0)
