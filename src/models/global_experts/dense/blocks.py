"""Always-active O(3) tensor-product residual message passing."""

import math
import torch
from torch import nn

from ....symmetry import registry as _registry
from e3nn import o3


class InvariantGate(nn.Module):
    """One scalar gate per irrep copy, shared across its magnetic components."""

    def __init__(self, irreps, width):
        super().__init__()
        self.irreps = o3.Irreps(irreps)
        count = self.irreps.num_irreps
        even = sum(mul for mul, ir in self.irreps if ir.l == 0 and ir.p == 1)
        self.net = nn.Sequential(nn.Linear(count + even, width), nn.SiLU(),
                                 nn.Linear(width, count), nn.Sigmoid())

    def forward(self, x):
        blocks = [x[:, sl].reshape(len(x), mul, ir.dim)
                  for (mul, ir), sl in zip(self.irreps, self.irreps.slices())]
        invariants = [torch.sqrt(b.square().mean(-1) + 1e-8) for b in blocks]
        invariants += [b[..., 0] for b, (_, ir) in zip(blocks, self.irreps)
                       if ir.l == 0 and ir.p == 1]
        gates = self.net(torch.cat(invariants, dim=-1)).split(
            [mul for mul, _ in self.irreps], dim=-1)
        return torch.cat([(b * g[..., None]).flatten(1)
                          for b, g in zip(blocks, gates)], dim=-1)


class DenseO3Block(nn.Module):
    def __init__(self, irreps, edge_irreps, radial_width, gate_width, initial_logit):
        super().__init__()
        self.tp = o3.FullyConnectedTensorProduct(irreps, edge_irreps, irreps)
        self.radial = nn.Sequential(nn.Linear(1, radial_width), nn.SiLU(),
                                    nn.Linear(radial_width, 1))
        self.gate = InvariantGate(irreps, gate_width)
        self.output = o3.Linear(irreps, irreps)
        self.residual_logit = nn.Parameter(torch.tensor(float(initial_logit)))

    def forward(self, x, edge_index, harmonics, distance, envelope, degree):
        receiver, sender = edge_index
        message = self.tp(x[sender], harmonics) * self.radial(distance) * envelope
        aggregate = torch.zeros_like(x).index_add(0, receiver, message) / degree
        return x + self.residual_logit.tanh() * self.output(self.gate(aggregate))


class DenseO3Branch(nn.Module):
    def __init__(self, irreps, *, hidden_irreps=None, depth=3, radial_width=8,
                 gate_width=32, edge_lmax=2, cutoff=6.0, initial_logit=0.0):
        super().__init__()
        if min(depth, radial_width, gate_width) < 1 or edge_lmax < 0:
            raise ValueError("invalid dense dimensions")
        if not math.isfinite(cutoff) or cutoff <= 0 or not math.isfinite(initial_logit):
            raise ValueError("invalid dense cutoff or initial logit")
        self.irreps = o3.Irreps(irreps)
        hidden = o3.Irreps(hidden_irreps or irreps)
        self.cutoff = cutoff
        self.edge_irreps = o3.Irreps.spherical_harmonics(edge_lmax)
        self.input = nn.Identity() if hidden == self.irreps else o3.Linear(self.irreps, hidden)
        self.output = nn.Identity() if hidden == self.irreps else o3.Linear(hidden, self.irreps)
        self.blocks = nn.ModuleList([
            DenseO3Block(hidden, self.edge_irreps, radial_width, gate_width, initial_logit)
            for _ in range(depth)])

    def forward(self, features, edge_index, edge_vectors):
        if features.ndim != 2 or features.shape[-1] != self.irreps.dim:
            raise ValueError("dense features do not match carrier")
        if edge_index.ndim != 2 or edge_index.shape[0] != 2 or edge_vectors.shape != (edge_index.shape[1], 3):
            raise ValueError("invalid dense edge geometry")
        if not edge_index.shape[1]:
            return features
        harmonics = o3.spherical_harmonics(self.edge_irreps, edge_vectors, True,
                                          normalization="component")
        distance = edge_vectors.norm(dim=-1, keepdim=True) / self.cutoff
        envelope = torch.where(distance <= 1, 0.5 * (torch.cos(math.pi * distance) + 1), 0.0)
        degree = torch.bincount(edge_index[0], minlength=len(features)).to(features)
        degree = degree.clamp_min(1).sqrt().unsqueeze(-1)
        initial = self.input(features)
        x = initial
        for block in self.blocks:
            x = block(x, edge_index, harmonics, distance, envelope, degree)
        # External identity is retained even when internal multiplicities change.
        return features + self.output(x - initial)
