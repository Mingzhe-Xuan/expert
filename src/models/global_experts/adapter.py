"""Shared identity-residual O(3) Adapter from algorithm section 17.2."""

import math
import torch
from torch import nn

from ...experts.modules import _edge_layout
from ...symmetry.registry import _layout_irreps
from ...tensor_products import build_tensor_product
from e3nn import o3


class IdentityMessageAdapter(nn.Module):
    def __init__(
        self,
        layout,
        *,
        backend="full_o3",
        mmax=2,
        edge_lmax=2,
        cutoff=6.0,
        radial_width=8,
        initial_logit=0.0
    ):
        super().__init__()
        if not math.isfinite(cutoff) or cutoff <= 0 or radial_width < 1:
            raise ValueError("Adapter cutoff and radial width must be positive")
        self.layout, self.cutoff = layout, cutoff
        self.edge_layout = _edge_layout(edge_lmax)
        self.tensor_product = build_tensor_product(
            backend, layout, self.edge_layout, layout, mmax=mmax
        )
        self.radial = nn.Sequential(
            nn.Linear(1, radial_width), nn.SiLU(), nn.Linear(radial_width, 1)
        )
        self.residual_logit = nn.Parameter(torch.tensor(float(initial_logit)))

    def forward(self, features, edge_index, edge_vectors):
        if features.ndim != 2 or features.shape[-1] != self.layout.dimension:
            raise ValueError("Adapter features do not match carrier")
        if edge_vectors.shape != (edge_index.shape[1], 3):
            raise ValueError("edge vectors and topology differ")
        if edge_index.shape[1] == 0:
            return features
        receiver, sender = edge_index
        harmonics = o3.spherical_harmonics(
            _layout_irreps(self.edge_layout),
            edge_vectors,
            True,
            normalization="component",
        )
        messages = self.tensor_product(features[sender], harmonics, edge_vectors)
        distance = edge_vectors.norm(dim=-1, keepdim=True) / self.cutoff
        envelope = torch.where(
            distance <= 1, 0.5 * (torch.cos(math.pi * distance) + 1), 0.0
        )
        messages = messages * self.radial(distance) * envelope
        summed = torch.zeros_like(features).index_add(0, receiver, messages)
        degree = torch.bincount(receiver, minlength=len(features)).to(features)
        summed = summed / degree.clamp_min(1).sqrt().unsqueeze(-1)
        return features + self.residual_logit.tanh() * summed
