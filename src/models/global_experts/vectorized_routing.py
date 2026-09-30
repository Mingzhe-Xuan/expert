"""Padded chain-prefix products with exact zero-gate behavior and static plan reuse."""
from collections import OrderedDict

import torch

from ...symmetry.parent_detection import validate_point_group_parent_dag
from .routing import HierarchicalChainRouter, RoutingWeights


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
