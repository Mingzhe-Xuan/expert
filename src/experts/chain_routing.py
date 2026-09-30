"""Algorithm sections 5--7: near-current stick breaking and chain softmax."""

from dataclasses import dataclass
import math
from collections import OrderedDict
import copy
from typing import Mapping

import torch
from torch import nn

from ..symmetry import ParentDAGSpec, PointGroupAncestorDAG
from ..symmetry.parent_detection import validate_point_group_parent_dag


def migrate_legacy_edge_scales(state_dict, router, *, prefix="edge_gate."):
    """Explicit model-state migration; optimizer state must be initialized anew.

    Preserve positive sigmas, not the legacy logits: the wrapper uses softplus + floor
    instead of clamp_min. Changing routing intentionally does not preserve predictions.
    """
    migrated = copy.copy(state_dict)
    target = prefix + "log_scales"
    if target in migrated:
        raise ValueError("state already contains chain-router scales")
    keys = [prefix + "log_sigmas.edge_" + edge for edge in router.edge_ids]
    supplied = {key for key in migrated if key.startswith(prefix + "log_sigmas.")}
    if supplied != set(keys):
        raise ValueError("legacy checkpoint edge IDs differ from configured router")
    if keys:
        values = torch.stack([migrated.pop(key) for key in keys])
        positive = torch.nn.functional.softplus(values).clamp_min(1e-8) - router.sigma_floor
        if not torch.isfinite(positive).all() or bool((positive <= 0).any()):
            raise ValueError("legacy sigma cannot be represented above the requested floor")
        migrated[target] = positive + torch.log(-torch.expm1(-positive))
    else:
        migrated[target] = router.log_scales.detach().clone()
    return migrated


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


class VectorizedChainRouter(HierarchicalChainRouter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._plans = OrderedDict()

    def _apply(self, fn, recurse=True):
        self._plans.clear()
        return super()._apply(fn, recurse=recurse)

    def _plan(self, dag):
        key = id(dag)
        if key in self._plans and self._plans[key][0] is dag:
            self._plans.move_to_end(key)
            return self._plans[key][1]
        validate_point_group_parent_dag(dag, self.class_dag)
        paths = dag.current_to_root_embedding_paths()
        groups = tuple((dag.current_point_group_number, *(e.parent_point_group_number for e in path)) for path in paths)
        keys = tuple(e.checksum for e in dag.embeddings)
        positions = {checksum: i for i, checksum in enumerate(keys)}
        if any(e.edge_id not in self.edge_indices for e in dag.embeddings):
            raise ValueError("unregistered edge ID")
        length = max(map(len, paths))
        indices = torch.full((len(paths), length), len(keys), dtype=torch.long)
        unique = tuple(dict.fromkeys(pg for chain in groups for pg in chain))
        pg_indices = torch.zeros((len(paths), length + 1), dtype=torch.long)
        for i, (path, chain) in enumerate(zip(paths, groups)):
            indices[i, :len(path)] = torch.tensor([positions[e.checksum] for e in path], dtype=torch.long)
            pg_indices[i, :len(chain)] = torch.tensor([unique.index(pg) for pg in chain])
        device = self.log_scales.device
        plan = (keys, groups, unique, indices.to(device), pg_indices.to(device),
                torch.tensor([self.edge_indices[e.edge_id] for e in dag.embeddings], device=device, dtype=torch.long))
        self._plans[key] = (dag, plan)
        while len(self._plans) > 8192:
            self._plans.popitem(last=False)
        return plan

    def forward(self, dag, residuals, *, scales=None):
        keys, groups, unique, indices, pg_indices, edge_indices = self._plan(dag)
        if set(residuals) != set(keys):
            raise ValueError("residual keys must exactly cover material DAG edges")
        values = [torch.as_tensor(residuals[k], dtype=self.log_scales.dtype).detach() for k in keys]
        if any(v.ndim for v in values):
            raise ValueError("residual must be a finite nonnegative scalar")
        # Float metadata is packed on CPU before a single device transfer.
        if values and len({v.device for v in values}) > 1:
            values = [v.to(self.log_scales) for v in values]
        residual = (torch.stack(values).to(self.log_scales) if values else self.log_scales.new_empty(0))
        if not torch.isfinite(residual).all() or bool((residual < 0).any()):
            raise ValueError("residual must be a finite nonnegative scalar")
        scales = self.scales if scales is None else scales
        energies = (residual / scales.index_select(0, edge_indices)).square()
        if not torch.isfinite(energies).all():
            raise ValueError("normalized residual energy overflow")
        gates = torch.cat((-torch.expm1(-energies), energies.new_ones(1)))
        stops = torch.cat((torch.exp(-energies), energies.new_zeros(1)))
        selected = gates[indices]
        prefix = torch.cat((energies.new_ones((len(groups), 1)), selected.cumprod(dim=1)), dim=1)
        conditional = torch.cat((prefix[:, -1:], prefix[:, :-1] * stops[indices]), dim=1)
        first = (torch.cat((energies, energies.new_zeros(1)))[indices[:, 0]]
                 if indices.shape[1] else energies.new_zeros(len(groups)))
        pi = torch.softmax(-first / self.temperature, dim=0)
        alpha_values = energies.new_zeros(len(unique)).scatter_add(
            0, pg_indices.flatten(), (conditional * pi[:, None]).flatten())
        all_weights = torch.cat((conditional.flatten(), pi, alpha_values))
        sums = torch.cat((conditional.sum(1), pi.sum().reshape(1), alpha_values.sum().reshape(1)))
        if (not torch.isfinite(all_weights).all() or bool((all_weights < 0).any())
                or not torch.allclose(sums, torch.ones_like(sums), atol=1e-6, rtol=1e-6)):
            raise ValueError("routing weights must be finite nonnegative and normalized")
        return RoutingWeights(groups, tuple(conditional[i, :len(chain)] for i, chain in enumerate(groups)),
                              first, pi, dict(zip(unique, alpha_values.unbind())))
