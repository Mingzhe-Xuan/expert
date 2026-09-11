from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Mapping

import torch
from e3nn import o3
from e3nn.io import CartesianTensor
from torch import nn

from .config import ModelConfig
from .layers import (
    A1PointGroupBlock,
    O2ReducedMessagePassing,
    O2ReducedReadout,
    SharedRouter,
    gaussian_radial_basis,
)
from .symmetry import (
    TARGET_IRREPS,
    PointGroupDAG,
    active_parameter_names_for_edges,
    infer_point_group,
    invariant_basis,
    per_l_invariant_bases,
    symmetry_residual,
)


@dataclass
class CrystalInput:
    positions: torch.Tensor
    cell: torch.Tensor
    atomic_numbers: torch.Tensor
    node_features: torch.Tensor
    edge_index: torch.Tensor | None = None
    edge_vectors: torch.Tensor | None = None
    current_point_group: str | None = None
    parent_residuals: Mapping[str, float | torch.Tensor] | None = None
    canonical_to_input: torch.Tensor | None = None


@dataclass
class ModelOutput:
    tensor: torch.Tensor
    irrep_coefficients: torch.Tensor
    current_point_group: str
    active_point_groups: tuple[str, ...]
    branch_weights: dict[str, torch.Tensor] = field(default_factory=dict)
    residuals: dict[str, torch.Tensor] = field(default_factory=dict)


def _module_key(number: int) -> str:
    return f"pg{number:02d}"


class GroupBranch(nn.Module):
    def __init__(
        self,
        symbol: str,
        number: int,
        rotations: torch.Tensor,
        config: ModelConfig,
    ) -> None:
        super().__init__()
        self.symbol = symbol
        self.number = number
        self.multiplicities = config.multiplicities
        bases = per_l_invariant_bases(config.lmax, rotations)
        for ell, basis in enumerate(bases):
            self.register_buffer(f"node_basis_{ell}", basis)
        self.a1_multiplicities = tuple(basis.shape[1] for basis in bases)
        self.node_width = sum(
            mul * a1 for mul, a1 in zip(config.multiplicities, self.a1_multiplicities)
        )
        self.edge_width = config.radial_channels * sum(self.a1_multiplicities)
        self.blocks = nn.ModuleList()
        if symbol != "1":
            for _ in range(config.pg_layers):
                self.blocks.append(
                    A1PointGroupBlock(self.node_width, self.edge_width, config.route_width)
                )

        self.target_bases: dict[str, torch.Tensor] = {}
        self.readouts = nn.ModuleDict()
        for task, irreps in TARGET_IRREPS.items():
            basis = invariant_basis(irreps, rotations)
            self.register_buffer(f"target_basis_{task}", basis)
            self.target_bases[task] = basis
            self.readouts[task] = nn.Linear(self.node_width, basis.shape[1])

    def node_to_a1(self, features: torch.Tensor) -> torch.Tensor:
        outputs = []
        offset = 0
        for ell, multiplicity in enumerate(self.multiplicities):
            dimension = 2 * ell + 1
            width = multiplicity * dimension
            block = features[:, offset : offset + width].reshape(
                features.shape[0], multiplicity, dimension
            )
            basis = getattr(self, f"node_basis_{ell}").to(block)
            outputs.append(torch.einsum("nmd,da->nma", block, basis).flatten(1))
            offset += width
        return torch.cat(outputs, dim=-1)

    def edge_to_a1(
        self, edge_vectors: torch.Tensor, compressed_radial: torch.Tensor
    ) -> torch.Tensor:
        outputs = []
        for ell in range(len(self.multiplicities)):
            harmonics = o3.spherical_harmonics(
                ell, edge_vectors, normalize=True, normalization="component"
            )
            basis = getattr(self, f"node_basis_{ell}").to(harmonics)
            invariant_harmonics = harmonics @ basis
            outputs.append(
                torch.einsum("ea,eq->eaq", invariant_harmonics, compressed_radial).flatten(1)
            )
        return torch.cat(outputs, dim=-1)

    def forward(
        self,
        shared_features: torch.Tensor,
        compressed_radial: torch.Tensor,
        route_latent: torch.Tensor,
        edge_index: torch.Tensor,
        edge_vectors: torch.Tensor,
        task: str,
    ) -> torch.Tensor:
        hidden = self.node_to_a1(shared_features)
        if self.blocks:
            edge_features = self.edge_to_a1(edge_vectors, compressed_radial)
            for block in self.blocks:
                hidden = block(hidden, edge_features, route_latent, edge_index)
        pooled = hidden.mean(dim=0, keepdim=True)
        independent = self.readouts[task](pooled)
        target_basis = getattr(self, f"target_basis_{task}").to(independent)
        return independent @ target_basis.T


class HierarchicalGate(nn.Module):
    def __init__(self, dag: PointGroupDAG, initial_sigma: float) -> None:
        super().__init__()
        inverse_softplus = math.log(math.expm1(initial_sigma))
        self.log_sigma = nn.ParameterDict()
        for parent, children in dag.children.items():
            for child in children:
                self.log_sigma[f"{dag.records[parent].number}_{dag.records[child].number}"] = (
                    nn.Parameter(torch.tensor(inverse_softplus))
                )
        self.dag = dag

    def sigma(self, parent: str, child: str) -> torch.Tensor:
        key = f"{self.dag.records[parent].number}_{self.dag.records[child].number}"
        return torch.nn.functional.softplus(self.log_sigma[key]).clamp_min(1.0e-6)

    def forward(
        self, current: str, residuals: Mapping[str, torch.Tensor]
    ) -> tuple[dict[str, torch.Tensor], set[tuple[str, str]]]:
        paths = self.dag.root_to_current_paths(current)
        beta = {symbol: next(iter(residuals.values())).new_zeros(()) for symbol in self.dag.ancestors(current)}
        prior = 1.0 / len(paths)
        for path in paths:
            if len(path) == 1:
                beta[path[0]] = beta[path[0]] + prior
                continue
            gates = []
            for parent, child in zip(path[:-1], path[1:]):
                ratio = residuals[parent] / self.sigma(parent, child)
                gates.append(1.0 - torch.exp(-ratio.square()))
            prefix = gates[0].new_ones(())
            for index, symbol in enumerate(path):
                if index < len(gates):
                    contribution = prefix * (1.0 - gates[index])
                    prefix = prefix * gates[index]
                else:
                    contribution = prefix
                beta[symbol] = beta[symbol] + prior * contribution
        normalizer = torch.stack(list(beta.values())).sum().clamp_min(1.0e-12)
        return (
            {symbol: value / normalizer for symbol, value in beta.items()},
            active_parameter_names_for_edges(paths),
        )


class PointGroupTensorModel(nn.Module):
    """Complete downstream forward, beginning at backbone-compatible O(3) features."""

    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or ModelConfig()
        self.hidden_irreps = o3.Irreps(self.config.hidden_irreps)
        edge_irreps = o3.Irreps.spherical_harmonics(self.config.lmax)
        self.dag = PointGroupDAG(self.config.dag_path)
        self.shared_adaptation = O2ReducedMessagePassing(
            self.hidden_irreps,
            edge_irreps,
            self.hidden_irreps,
            self.config.radial_basis,
            self.config.radial_channels,
            self.config.o2_mmax,
            self.config.cutoff,
        )
        self.router = SharedRouter(self.config.multiplicities[0], self.config.route_width)
        # PG edge features reuse the same 8 -> 2 compression as the shared layer.
        self.pg_radial_compression = nn.Linear(
            self.config.radial_basis, self.config.radial_channels, bias=False
        )
        self.branches = nn.ModuleDict()
        self.symbol_to_key: dict[str, str] = {}
        for symbol in self.dag.symbols:
            record = self.dag.records[symbol]
            key = _module_key(record.number)
            self.symbol_to_key[symbol] = key
            self.branches[key] = GroupBranch(
                symbol, record.number, record.cartesian_rotations, self.config
            )
        self.hierarchical_gate = HierarchicalGate(self.dag, self.config.gate_sigma)
        self.final_readouts = nn.ModuleDict(
            {
                task: O2ReducedReadout(irreps, self.hidden_irreps, self.config.o2_mmax)
                for task, irreps in TARGET_IRREPS.items()
            }
        )
        self.cartesian = {
            "dielectric": CartesianTensor("ij=ji"),
            "elastic": CartesianTensor("ijkl=ijlk=jikl=klij"),
        }

    def _build_graph(self, crystal: CrystalInput) -> tuple[torch.Tensor, torch.Tensor]:
        if crystal.edge_index is not None or crystal.edge_vectors is not None:
            if crystal.edge_index is None or crystal.edge_vectors is None:
                raise ValueError("edge_index and edge_vectors must be supplied together")
            return crystal.edge_index, crystal.edge_vectors
        positions, cell = crystal.positions, crystal.cell
        frac = torch.linalg.solve(cell.T, positions.T).T
        edges: list[tuple[int, int, torch.Tensor]] = []
        translations = torch.cartesian_prod(
            *[torch.arange(-1, 2, device=positions.device, dtype=positions.dtype)] * 3
        )
        for target in range(positions.shape[0]):
            candidates = []
            for source in range(positions.shape[0]):
                vectors = (frac[source] + translations - frac[target]) @ cell
                distances = vectors.norm(dim=-1)
                for vector, distance, shift in zip(vectors, distances, translations):
                    if distance < self.config.cutoff and not (
                        target == source and bool((shift == 0).all())
                    ):
                        candidates.append((float(distance.detach()), source, vector))
            candidates.sort(key=lambda item: item[0])
            for _, source, vector in candidates[: self.config.max_neighbors]:
                edges.append((source, target, vector))
        if not edges:
            return (
                torch.empty((2, 0), dtype=torch.long, device=positions.device),
                positions.new_empty((0, 3)),
            )
        edge_index = torch.tensor(
            [[source, target] for source, target, _ in edges],
            dtype=torch.long,
            device=positions.device,
        ).T
        edge_vectors = torch.stack([vector for _, _, vector in edges])
        return edge_index, edge_vectors

    def _residuals(self, crystal: CrystalInput, active: list[str]) -> dict[str, torch.Tensor]:
        supplied = crystal.parent_residuals or {}
        output = {}
        for symbol in active:
            if symbol in supplied:
                output[symbol] = torch.as_tensor(
                    supplied[symbol], device=crystal.positions.device, dtype=crystal.positions.dtype
                )
            elif symbol == crystal.current_point_group:
                output[symbol] = crystal.positions.new_zeros(())
            else:
                output[symbol] = symmetry_residual(
                    crystal.positions,
                    crystal.cell,
                    crystal.atomic_numbers,
                    self.dag.records[symbol].fractional_rotations,
                    self.config.residual_cell_weight,
                    self.config.residual_position_weight,
                )
        return output

    def forward(self, crystal: CrystalInput, task: str) -> ModelOutput:
        if task not in TARGET_IRREPS:
            raise ValueError(f"unknown task {task!r}; expected dielectric or elastic")
        if crystal.node_features.shape != (crystal.positions.shape[0], self.hidden_irreps.dim):
            raise ValueError(
                f"node_features must have shape [N, {self.hidden_irreps.dim}] for "
                f"{self.hidden_irreps}, got {tuple(crystal.node_features.shape)}"
            )
        current = crystal.current_point_group or infer_point_group(
            crystal.positions,
            crystal.cell,
            crystal.atomic_numbers,
            self.config.symprec,
        )
        current = current.replace(" ", "")
        if current not in self.dag.records:
            raise ValueError(f"point group {current!r} is not crystallographic")
        crystal.current_point_group = current
        edge_index, edge_vectors = self._build_graph(crystal)
        shared = self.shared_adaptation(crystal.node_features, edge_index, edge_vectors)
        scalar_channels = shared[:, : self.config.multiplicities[0]]
        route_latent = self.router(scalar_channels, edge_index)
        distances = edge_vectors.norm(dim=-1)
        radial = gaussian_radial_basis(
            distances, self.config.cutoff, self.config.radial_basis
        )
        compressed_radial = self.pg_radial_compression(radial)
        active = self.dag.ancestors(current)
        residuals = self._residuals(crystal, active)
        weights, _ = self.hierarchical_gate(current, residuals)
        branch_coefficients = []
        for symbol in active:
            branch = self.branches[self.symbol_to_key[symbol]]
            coefficients = branch(
                shared,
                compressed_radial,
                route_latent,
                edge_index,
                edge_vectors,
                task,
            )
            branch_coefficients.append(weights[symbol] * coefficients)
        fused = torch.stack(branch_coefficients).sum(dim=0)
        output_coefficients = self.final_readouts[task](
            fused, shared, edge_index, edge_vectors
        )
        if crystal.canonical_to_input is not None:
            frame = crystal.canonical_to_input.to(output_coefficients)
            output_coefficients = output_coefficients @ TARGET_IRREPS[task].D_from_matrix(frame).T
        tensor = self.cartesian[task].to_cartesian(output_coefficients)
        return ModelOutput(
            tensor=tensor,
            irrep_coefficients=output_coefficients,
            current_point_group=current,
            active_point_groups=tuple(active),
            branch_weights=weights,
            residuals=residuals,
        )
