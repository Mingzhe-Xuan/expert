"""Optional invariant attention for the pinned official GMTNet TP stack."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class EquivariantAttentionConv(nn.Module):
    """Reuse an official TP layer, adding one invariant scalar per edge."""

    def __init__(self, layer: nn.Module) -> None:
        super().__init__()
        self.in_irreps = layer.in_irreps
        self.out_irreps = layer.out_irreps
        self.sh_irreps = layer.sh_irreps
        self.residual = layer.residual
        self.tp = layer.tp
        self.fc = layer.fc
        self._irreps = self.tp.irreps_in1
        edge_width = self.fc[0].in_features
        invariant_width = sum(mul for mul, _ in self._irreps)
        self.attn_mlp = nn.Sequential(
            nn.Linear(2 * invariant_width + edge_width, edge_width),
            nn.SiLU(),
            nn.Linear(edge_width, 1),
        ).to(device=self.fc[0].weight.device, dtype=self.fc[0].weight.dtype)
        self.train(layer.training)

    def invariant_features(self, features: torch.Tensor) -> torch.Tensor:
        blocks = []
        for (mul, ir), section in zip(self._irreps, self._irreps.slices()):
            block = features[:, section].reshape(features.shape[0], mul, ir.dim)
            # Only 0e is a signed O(3) invariant; 0o changes under reflection.
            blocks.append(block.squeeze(-1) if ir.l == 0 and ir.p == 1
                          else torch.linalg.vector_norm(block, dim=-1))
        return torch.cat(blocks, dim=-1)

    def attention_weights(self, node_attr, edge_index, edge_attr, out_nodes=None):
        from torch_geometric.utils import softmax

        receiver, sender = edge_index
        invariants = self.invariant_features(node_attr)
        logits = self.attn_mlp(torch.cat(
            (invariants[receiver], invariants[sender], edge_attr), dim=-1
        ))
        return softmax(logits, receiver, num_nodes=out_nodes or node_attr.shape[0])

    def forward(self, node_attr, edge_index, edge_attr, edge_sh, out_nodes=None, reduce='mean'):
        from torch_geometric.utils import scatter

        if reduce not in ('mean', 'sum', 'add'):
            raise ValueError("equivariant attention supports mean/sum/add aggregation")
        # Official TP code gathers edge_index[1] and aggregates into edge_index[0].
        receiver, sender = edge_index
        messages = self.tp(node_attr[sender], edge_sh, self.fc(edge_attr))
        weights = self.attention_weights(node_attr, edge_index, edge_attr, out_nodes)
        out = scatter(messages * weights, receiver, dim=0,
                      dim_size=out_nodes or node_attr.shape[0], reduce='sum')
        if self.residual:
            out = out + F.pad(node_attr, (0, out.shape[-1] - node_attr.shape[-1]))
        return out


def configure_equivariant_attention(model: nn.Module, *, use_equiv_attn: bool = False):
    """Configure a freshly built official model without changing its forward API.

    False is an exact no-op. True preserves TP/fc weights and adds attention keys.
    Repeated True calls are idempotent; disabling an already enabled model is rejected.
    """
    layers = [(name, getattr(model.equi_update, name))
              for name in ('nlayer_1', 'nlayer_2', 'nlayer_3')]
    if not use_equiv_attn:
        if any(isinstance(layer, EquivariantAttentionConv) for _, layer in layers):
            raise ValueError("construct a fresh model to disable equivariant attention")
        return model
    for name, layer in layers:
        if not isinstance(layer, EquivariantAttentionConv):
            setattr(model.equi_update, name, EquivariantAttentionConv(layer))
    return model


def build_gmtnet(official_model, args, *, use_equiv_attn=None):
    """Build from an official module and legacy args; optionally enable attention."""
    enabled = getattr(args, 'use_equiv_attn', False) if use_equiv_attn is None else use_equiv_attn
    return configure_equivariant_attention(
        official_model.GMTNet(args), use_equiv_attn=enabled
    )
