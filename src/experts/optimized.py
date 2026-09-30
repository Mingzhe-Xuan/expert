"""Checkpoint-compatible execution optimizations shared by standalone and global PG models."""
from collections import OrderedDict
from types import MappingProxyType

import torch
from torch import nn

from .modules import FullPointGroupExpert, active_parent_point_group_numbers


class GroupedFiniteGroupBlock(nn.Module):
    def __init__(self, reference):
        super().__init__()
        for name in ("representation", "invariant_projector"):
            self.register_buffer(name, getattr(reference, name))
        for name in ("weight", "bias", "gate_gain", "gate_bias"):
            self.register_parameter(name, getattr(reference, name))
        self.copy_slices = reference.copy_slices
        self.trivial_copies = reference.trivial_copies
        groups = {}
        for index, (part, trivial) in enumerate(zip(self.copy_slices, self.trivial_copies)):
            groups.setdefault((part.stop - part.start, trivial), []).append(index)
        self.groups = tuple(groups)
        order = []
        for k, ((width, trivial), copies) in enumerate(groups.items()):
            coordinates = [j for i in copies for j in range(self.copy_slices[i].start, self.copy_slices[i].stop)]
            order.extend(coordinates)
            self.register_buffer(f"coordinates_{k}", torch.tensor(coordinates), persistent=False)
            self.register_buffer(f"copies_{k}", torch.tensor(copies), persistent=False)
        self.register_buffer("restore", torch.argsort(torch.tensor(order)), persistent=False)

    def forward(self, features):
        representation = self.representation.to(features)
        weight = torch.einsum("gij,jk,glk->il", representation, self.weight, representation) / len(representation)
        hidden = features @ weight + self.bias @ self.invariant_projector.to(features)
        outputs = []
        for k, (width, trivial) in enumerate(self.groups):
            copies = getattr(self, f"copies_{k}")
            values = hidden.index_select(-1, getattr(self, f"coordinates_{k}"))
            values = values.reshape(*hidden.shape[:-1], len(copies), width)
            if trivial:
                gated = torch.nn.functional.silu(values)
            else:
                norm = torch.linalg.vector_norm(values, dim=-1, keepdim=True)
                gain = self.gate_gain.index_select(0, copies).unsqueeze(-1)
                bias = self.gate_bias.index_select(0, copies).unsqueeze(-1)
                gated = values * torch.sigmoid(gain * norm + bias)
            outputs.append(gated.flatten(-2))
        return features + torch.cat(outputs, dim=-1).index_select(-1, self.restore)


class GroupedFullPointGroupExpert(FullPointGroupExpert):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.blocks = nn.ModuleList([
            block if isinstance(block, nn.Identity) else GroupedFiniteGroupBlock(block)
            for block in self.blocks
        ])


class VectorizedMaterialWeights:
    """Legacy root-first, length-prior routing; cache topology, never autograd values."""

    def __init__(self, capacity=8192):
        self.capacity = capacity
        self.plans = OrderedDict()

    def __call__(self, dag, residuals, gate):
        if set(residuals) != {edge.checksum for edge in dag.embeddings}:
            raise ValueError("residuals must cover exactly the material cover edges")
        if not dag.embeddings:
            return MappingProxyType({dag.current_point_group_number: torch.ones(())})
        parameters = []
        for edge in dag.embeddings:
            key = f"edge_{edge.edge_id}"
            if key not in gate.log_sigmas:
                raise ValueError(f"unconfigured offline edge ID: {edge.edge_id}")
            parameters.append(gate.log_sigmas[key])
        scales = torch.stack(parameters)
        values = torch.stack([torch.as_tensor(residuals[edge.checksum],
                              dtype=scales.dtype, device=scales.device)
                              for edge in dag.embeddings])
        if not torch.isfinite(values).all() or bool((values < 0).any()):
            raise ValueError("edge residuals must be finite and non-negative")
        gates = -torch.expm1(-(values / torch.nn.functional.softplus(scales).clamp_min(1e-8)).square())
        key = (id(dag), str(scales.device), scales.dtype)
        if key not in self.plans:
            numbers = active_parent_point_group_numbers(dag)
            lookup = {edge.checksum: i for i, edge in enumerate(dag.embeddings)}
            paths = dag.current_to_root_embedding_paths()
            width = max(map(len, paths))
            indices, destinations, priors = [], [], []
            total_length = sum(len(path) + 1 for path in paths)
            for path in paths:
                edges = tuple(reversed(path))
                indices.append([lookup[edge.checksum] for edge in edges] +
                               [len(dag.embeddings)] * (width - len(edges)))
                destinations.append([numbers.index(edge.parent_point_group_number) for edge in edges] +
                                    [0] * (width - len(edges)) +
                                    [numbers.index(dag.current_point_group_number)])
                priors.append((len(path) + 1) / total_length)
            # Retain the immutable DAG so Python cannot reuse its id for another plan.
            self.plans[key] = (dag, numbers,
                torch.tensor(indices, device=scales.device),
                torch.tensor(destinations, device=scales.device), scales.new_tensor(priors))
            while len(self.plans) > self.capacity:
                self.plans.popitem(last=False)
        self.plans.move_to_end(key)
        _, numbers, indices, destinations, priors = self.plans[key]
        path_gates = torch.cat((gates, gates.new_ones(1)))[indices]
        prefix = torch.cat((path_gates.new_ones((len(priors), 1)), path_gates.cumprod(-1)), -1)
        stops = torch.cat((1 - path_gates, path_gates.new_ones((len(priors), 1))), -1)
        contributions = priors[:, None] * prefix * stops
        combined = gates.new_zeros(len(numbers)).scatter_add(0, destinations.flatten(), contributions.flatten())
        normalizer = combined.sum()
        if not torch.isfinite(normalizer) or bool(normalizer <= 0):
            raise ValueError("invalid material routing normalizer")
        return MappingProxyType(dict(zip(numbers, combined / normalizer)))
