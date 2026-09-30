"""Algorithm sections 5--7: near-current stick breaking and chain softmax."""

from dataclasses import dataclass
import math
from typing import Mapping

import torch
from torch import nn

from ...symmetry import ParentDAGSpec, PointGroupAncestorDAG
from ...symmetry.parent_detection import validate_point_group_parent_dag


@dataclass(frozen=True)
class RoutingWeights:
    groups: tuple[tuple[int, ...], ...]
    omega: tuple[torch.Tensor, ...]
    energies: torch.Tensor
    pi: torch.Tensor
    alpha: Mapping[int, torch.Tensor]


class HierarchicalChainRouter(nn.Module):
    """Shared trainable edge scales; no per-PG or per-chain trainable logits."""

    def __init__(
        self,
        edge_ids,
        *,
        initial_sigma=0.08,
        sigma_floor=1e-8,
        temperature=1.0,
        class_dag=None,
    ):
        super().__init__()
        if (
            not all(
                math.isfinite(v) and v > 0
                for v in (initial_sigma, sigma_floor, temperature)
            )
            or initial_sigma <= sigma_floor
        ):
            raise ValueError(
                "positive finite scales/temperature and sigma > floor required"
            )
        self.edge_ids = tuple(edge_ids)
        if len(set(self.edge_ids)) != len(self.edge_ids):
            raise ValueError("edge IDs must be unique")
        self.edge_indices = {key: index for index, key in enumerate(self.edge_ids)}
        self.class_dag = class_dag or PointGroupAncestorDAG.from_path()
        self.sigma_floor = sigma_floor
        self.temperature = temperature
        # Stable inverse softplus, including large requested initial scales.
        value = initial_sigma - sigma_floor
        initial = value + math.log(-math.expm1(-value))
        self.log_scales = nn.Parameter(torch.full((len(self.edge_ids),), initial))

    @property
    def scales(self):
        return torch.nn.functional.softplus(self.log_scales) + self.sigma_floor

    def forward(self, dag: ParentDAGSpec, residuals: Mapping) -> RoutingWeights:
        validate_point_group_parent_dag(dag, self.class_dag)
        if set(residuals) != {edge.checksum for edge in dag.embeddings}:
            raise ValueError("residual keys must exactly cover material DAG edges")
        energies = {}
        for edge in dag.embeddings:
            if edge.edge_id not in self.edge_indices:
                raise ValueError(f"unregistered edge ID: {edge.edge_id}")
            residual = torch.as_tensor(
                residuals[edge.checksum],
                device=self.log_scales.device,
                dtype=self.log_scales.dtype,
            ).detach()
            if residual.ndim or not torch.isfinite(residual) or residual < 0:
                raise ValueError("residual must be a finite nonnegative scalar")
            energy = (residual / self.scales[self.edge_indices[edge.edge_id]]).square()
            if not torch.isfinite(energy):
                raise ValueError("normalized residual energy overflow")
            energies[edge.checksum] = energy
        groups, omega, first_energies = [], [], []
        for path in dag.current_to_root_embedding_paths():
            remaining = self.log_scales.new_ones(())
            ancestors = []
            for edge in path:
                energy = energies[edge.checksum]
                stop = torch.exp(-energy)
                ancestors.append(remaining * stop)
                remaining = remaining * -torch.expm1(-energy)
            groups.append(
                (
                    dag.current_point_group_number,
                    *(edge.parent_point_group_number for edge in path),
                )
            )
            omega.append(torch.stack((remaining, *ancestors)))
            first_energies.append(
                energies[path[0].checksum] if path else self.log_scales.new_zeros(())
            )
        energy_vector = torch.stack(first_energies)
        pi = torch.softmax(-energy_vector / self.temperature, dim=0)
        alpha = {}
        for chain, conditional, prior in zip(groups, omega, pi):
            for group, weight in zip(chain, conditional):
                alpha[group] = alpha.get(group, 0) + prior * weight
        for weights in (*omega, pi, torch.stack(tuple(alpha.values()))):
            if (
                not torch.isfinite(weights).all()
                or bool((weights < 0).any())
                or not torch.allclose(
                    weights.sum(), weights.new_ones(()), atol=1e-6, rtol=1e-6
                )
            ):
                raise ValueError(
                    "routing weights must be finite nonnegative and normalized"
                )
        return RoutingWeights(tuple(groups), tuple(omega), energy_vector, pi, alpha)
