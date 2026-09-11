from __future__ import annotations

import math

import torch
from e3nn import o3
from torch import nn


def gaussian_radial_basis(
    distances: torch.Tensor, cutoff: float, count: int
) -> torch.Tensor:
    centers = torch.linspace(0.0, cutoff, count, device=distances.device, dtype=distances.dtype)
    width = cutoff / max(count - 1, 1)
    basis = torch.exp(-0.5 * ((distances[:, None] - centers) / width) ** 2)
    envelope = 0.5 * (torch.cos(math.pi * distances / cutoff) + 1.0)
    envelope = torch.where(distances < cutoff, envelope, torch.zeros_like(envelope))
    return basis * envelope[:, None]


def edge_frames(vectors: torch.Tensor) -> torch.Tensor:
    z_axis = vectors / vectors.norm(dim=-1, keepdim=True).clamp_min(1.0e-12)
    ez = torch.tensor([0.0, 0.0, 1.0], device=vectors.device, dtype=vectors.dtype)
    ex = torch.tensor([1.0, 0.0, 0.0], device=vectors.device, dtype=vectors.dtype)
    reference = torch.where((z_axis[:, 2].abs() < 0.9)[:, None], ez, ex)
    x_axis = torch.linalg.cross(reference, z_axis)
    x_axis = x_axis / x_axis.norm(dim=-1, keepdim=True).clamp_min(1.0e-12)
    y_axis = torch.linalg.cross(z_axis, x_axis)
    return torch.stack((x_axis, y_axis, z_axis), dim=-1)


def m_band_projector(irreps: o3.Irreps, mmax: int) -> torch.Tensor:
    """Basis-independent projector onto local SO(2) modes |m| <= mmax."""
    blocks = []
    angle = torch.tensor(1.0e-4, dtype=torch.float64)
    for multiplicity, irrep in irreps:
        d_plus = irrep.D_from_matrix(o3.matrix_z(angle))
        d_minus = irrep.D_from_matrix(o3.matrix_z(-angle))
        generator = (d_plus - d_minus) / (2.0 * angle)
        m_squared, vectors = torch.linalg.eigh(-generator @ generator)
        keep = m_squared <= (mmax + 0.25) ** 2
        one = vectors[:, keep] @ vectors[:, keep].T
        blocks.extend([one] * multiplicity)
    return torch.block_diag(*blocks).to(dtype=torch.get_default_dtype())


class O2ReducedMessagePassing(nn.Module):
    """Edge-frame O(2)-bandlimited, globally O(3)-equivariant TP layer.

    The implementation rotates features into any frame whose local z-axis is
    the edge direction, applies an O(2)-stable |m| cutoff, evaluates a fixed-CG
    e3nn tensor product, and rotates back. It uses O(3)-coupling weights as a
    conservative reference parameterization; a specialized O(2) kernel may
    later reduce FLOPs without changing this public interface.
    """

    def __init__(
        self,
        irreps_in: o3.Irreps,
        irreps_edge: o3.Irreps,
        irreps_out: o3.Irreps,
        radial_basis_count: int,
        radial_channels: int,
        mmax: int,
        cutoff: float,
    ) -> None:
        super().__init__()
        self.irreps_in = o3.Irreps(irreps_in)
        self.irreps_edge = o3.Irreps(irreps_edge)
        self.irreps_out = o3.Irreps(irreps_out)
        self.mmax = mmax
        self.cutoff = cutoff
        self.radial_basis_count = radial_basis_count
        self.tp = o3.FullyConnectedTensorProduct(
            self.irreps_in,
            self.irreps_edge,
            self.irreps_out,
            internal_weights=False,
            shared_weights=False,
        )
        self.radial_compression = nn.Linear(radial_basis_count, radial_channels, bias=False)
        self.radial_to_tp = nn.Linear(radial_channels, self.tp.weight_numel, bias=False)
        self.self_connection = o3.Linear(self.irreps_in, self.irreps_out)
        self.register_buffer("in_projector", m_band_projector(self.irreps_in, mmax))
        self.register_buffer("edge_projector", m_band_projector(self.irreps_edge, mmax))
        self.register_buffer("out_projector", m_band_projector(self.irreps_out, mmax))

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_vectors: torch.Tensor,
    ) -> torch.Tensor:
        source, target = edge_index
        if edge_vectors.shape[0] == 0:
            return self.self_connection(node_features)
        frames = edge_frames(edge_vectors)
        d_in = self.irreps_in.D_from_matrix(frames)
        d_edge = self.irreps_edge.D_from_matrix(frames)
        d_out = self.irreps_out.D_from_matrix(frames)
        source_global = node_features[source]
        edge_global = o3.spherical_harmonics(
            self.irreps_edge,
            edge_vectors,
            normalize=True,
            normalization="component",
        )
        source_local = torch.einsum("eij,ei->ej", d_in, source_global)
        edge_local = torch.einsum("eij,ei->ej", d_edge, edge_global)
        source_local = source_local @ self.in_projector.T
        edge_local = edge_local @ self.edge_projector.T
        distances = edge_vectors.norm(dim=-1)
        radial = gaussian_radial_basis(distances, self.cutoff, self.radial_basis_count)
        weights = self.radial_to_tp(self.radial_compression(radial))
        message_local = self.tp(source_local, edge_local, weights) @ self.out_projector.T
        message_global = torch.einsum("eij,ej->ei", d_out, message_local)
        output = self.self_connection(node_features)
        output = output.index_add(0, target, message_global)
        degree = torch.bincount(target, minlength=node_features.shape[0]).to(output.dtype)
        return output / degree.clamp_min(1.0).sqrt()[:, None]


class SharedRouter(nn.Module):
    def __init__(self, scalar_channels: int, route_width: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(2 * scalar_channels, route_width),
            nn.SiLU(),
            nn.Linear(route_width, route_width),
        )

    def forward(self, scalars: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        source, target = edge_index
        return self.network(torch.cat((scalars[source], scalars[target]), dim=-1))


class ScalarRMSNorm(nn.Module):
    def __init__(self, width: int, epsilon: float = 1.0e-8) -> None:
        super().__init__()
        self.gain = nn.Parameter(torch.ones(width))
        self.epsilon = epsilon

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        scale = features.square().mean(dim=-1, keepdim=True).add(self.epsilon).rsqrt()
        return features * scale * self.gain


class A1PointGroupBlock(nn.Module):
    """A dense low-rank-routed A1 x A1 -> A1 path table."""

    def __init__(self, node_width: int, edge_width: int, route_width: int) -> None:
        super().__init__()
        shape = (node_width, node_width, edge_width)
        self.path_bias = nn.Parameter(torch.empty(shape))
        self.path_router = nn.Parameter(torch.empty((*shape, route_width)))
        self.self_interaction = nn.Linear(node_width, node_width, bias=False)
        self.eq_mlp = nn.Sequential(
            nn.Linear(node_width, node_width),
            nn.SiLU(),
            nn.Linear(node_width, node_width),
        )
        self.norm = ScalarRMSNorm(node_width)
        nn.init.xavier_uniform_(self.path_bias.flatten(1))
        nn.init.normal_(self.path_router, std=1.0 / math.sqrt(max(node_width * edge_width, 1)))

    def forward(
        self,
        node_features: torch.Tensor,
        edge_features: torch.Tensor,
        route_latent: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        source, target = edge_index
        source_features = node_features[source]
        base = torch.einsum(
            "es,er,osr->eo", source_features, edge_features, self.path_bias
        )
        routed = torch.einsum(
            "es,er,eq,osrq->eo",
            source_features,
            edge_features,
            route_latent,
            self.path_router,
        )
        messages = base + routed
        aggregated = torch.zeros_like(node_features).index_add(0, target, messages)
        degree = torch.bincount(target, minlength=node_features.shape[0]).to(node_features.dtype)
        hidden = self.self_interaction(node_features) + aggregated / degree.clamp_min(1).sqrt()[:, None]
        return self.norm(hidden + self.eq_mlp(hidden))


class O2ReducedReadout(nn.Module):
    def __init__(self, target_irreps: o3.Irreps, carrier_irreps: o3.Irreps, mmax: int) -> None:
        super().__init__()
        self.target_irreps = o3.Irreps(target_irreps)
        self.carrier_irreps = o3.Irreps(carrier_irreps)
        self.tp = o3.FullyConnectedTensorProduct(
            self.target_irreps, self.carrier_irreps, self.target_irreps
        )
        self.fallback = o3.Linear(self.target_irreps, self.target_irreps)
        self.register_buffer("target_projector", m_band_projector(self.target_irreps, mmax))
        self.register_buffer("carrier_projector", m_band_projector(self.carrier_irreps, mmax))

    def forward(
        self,
        coefficients: torch.Tensor,
        carrier: torch.Tensor,
        edge_index: torch.Tensor,
        edge_vectors: torch.Tensor,
    ) -> torch.Tensor:
        if edge_vectors.shape[0] == 0:
            return self.fallback(coefficients)
        source, _ = edge_index
        frames = edge_frames(edge_vectors)
        d_target = self.target_irreps.D_from_matrix(frames)
        d_carrier = self.carrier_irreps.D_from_matrix(frames)
        coeff_edges = coefficients.expand(edge_vectors.shape[0], -1)
        coeff_local = torch.einsum("eij,ei->ej", d_target, coeff_edges)
        carrier_local = torch.einsum("eij,ei->ej", d_carrier, carrier[source])
        coeff_local = coeff_local @ self.target_projector.T
        carrier_local = carrier_local @ self.carrier_projector.T
        output_local = self.tp(coeff_local, carrier_local) @ self.target_projector.T
        output_global = torch.einsum("eij,ej->ei", d_target, output_local)
        return output_global.mean(dim=0, keepdim=True) + self.fallback(coefficients)
