from __future__ import annotations

from typing import Any

import torch
from torch import nn

from ..graphs import PeriodicGraph
from ..irreps import IrrepLayout, O3FeatureBatch
from .parity import InversionPairedReynolds, O3InterfaceProjector, SO3FeatureBatch, SO3Layout, SO3Term
from .resources import BackboneResourceRegistry, require_distribution_version


DPA4_DISTRIBUTION = "deepmd-kit"
DPA4_VERSION = "3.2.0"
DPA4_CHANNELS = 64
DPA4_LMAX = 4
DPA4_COEFFICIENTS = (DPA4_LMAX + 1) ** 2
DPA4_SO3_LAYOUT = SO3Layout(
    tuple(SO3Term(DPA4_CHANNELS, degree, f"dpa4_l{degree}") for degree in range(DPA4_LMAX + 1))
)


def flatten_dpa4_latent(latent: torch.Tensor) -> torch.Tensor:
    """Convert DeePMD's [node, lm, singleton, channel] state to e3nn copy-major blocks."""

    expected_tail = (DPA4_COEFFICIENTS, 1, DPA4_CHANNELS)
    if latent.ndim != 4 or tuple(latent.shape[1:]) != expected_tail:
        raise ValueError(
            "DPA4 equivariant latent must have shape "
            f"[num_nodes, {DPA4_COEFFICIENTS}, 1, {DPA4_CHANNELS}]"
        )
    blocks = []
    offset = 0
    for degree in range(DPA4_LMAX + 1):
        width = 2 * degree + 1
        block = latent[:, offset : offset + width, 0, :]
        blocks.append(block.transpose(1, 2).contiguous().reshape(latent.shape[0], -1))
        offset += width
    return torch.cat(blocks, dim=-1)


def _descriptor_from_model(model: nn.Module) -> nn.Module:
    atomic_model = getattr(model, "atomic_model", None)
    descriptor = getattr(atomic_model, "descriptor", None)
    if descriptor is None:
        raise ValueError("DPA4 checkpoint model lacks model.atomic_model.descriptor")
    return descriptor


def _schema_tensor(schema: Any, name: str) -> torch.Tensor:
    value = getattr(schema, name, None)
    if value is None and isinstance(schema, dict):
        value = schema.get(name)
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"DPA4 neighbor schema lacks tensor {name!r}")
    return value


def _neighbor_schema_on_device(
    schema: Any,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Normalize DeePMD's host/device-dependent neighbor-list schema."""

    return (
        _schema_tensor(schema, "edge_index").to(device=device, dtype=torch.long),
        _schema_tensor(schema, "edge_vec").to(device=device, dtype=dtype),
        _schema_tensor(schema, "edge_mask").to(device=device, dtype=torch.bool),
    )


class _DPA4SO3Extractor(nn.Module):
    """Frozen official DeePMD model exposing SeZM's final SO(3) node state."""

    def __init__(self, checkpoint: str, *, device: str) -> None:
        super().__init__()
        try:
            from deepmd.pt.infer.inference import Tester
        except ImportError as exc:
            raise ImportError("DPA4 adapter requires the pinned DeePMD PyTorch runtime") from exc
        tester = Tester(checkpoint)
        model = tester.model.to(device)
        descriptor = _descriptor_from_model(model)
        if int(getattr(descriptor, "lmax", -1)) != DPA4_LMAX:
            raise ValueError("DPA4 descriptor lmax drifted from the frozen lmax=4 contract")
        if int(getattr(descriptor, "channels", -1)) != DPA4_CHANNELS:
            raise ValueError("DPA4 descriptor channels drifted from the frozen 64-channel contract")
        self.tester = tester
        self.model = model
        self.descriptor = descriptor
        self.model.requires_grad_(False)
        self.model.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        self.model.eval()
        return self

    def _encode_one(self, graph: PeriodicGraph, graph_index: int) -> dict[str, torch.Tensor]:
        mask = graph.node_batch == graph_index
        parameter = next(self.model.parameters())
        positions = graph.positions[mask].to(device=parameter.device, dtype=parameter.dtype)
        cell = graph.cell[graph_index].to(device=parameter.device, dtype=parameter.dtype)
        atomic_numbers = graph.atomic_numbers[mask].to(parameter.device)
        if bool(((atomic_numbers < 1) | (atomic_numbers > 118)).any()):
            raise ValueError("DPA4 OMat24 type map only accepts atomic numbers 1..118")
        atype = (atomic_numbers - 1).unsqueeze(0)
        schema = self.model.build_neighbor_list(
            positions.unsqueeze(0), atype, cell.reshape(1, 9)
        )
        edge_index_all, edge_vectors_all, edge_mask = _neighbor_schema_on_device(
            schema,
            device=parameter.device,
            dtype=parameter.dtype,
        )
        with torch.no_grad():
            _, latent = self.descriptor.forward_with_edges(
                extended_coord=positions.unsqueeze(0),
                extended_atype=atype,
                edge_index=edge_index_all,
                edge_vec=edge_vectors_all,
                edge_mask=edge_mask,
            )
        features = flatten_dpa4_latent(latent)
        if features.shape[0] != positions.shape[0]:
            raise ValueError("DPA4 feature tap changed the local node count")
        edge_index = edge_index_all[:, edge_mask]
        edge_vectors = edge_vectors_all[edge_mask]
        source, target = edge_index
        cartesian_shifts = edge_vectors - (positions[source] - positions[target])
        cell_shifts = torch.round(cartesian_shifts @ torch.linalg.inv(cell)).to(torch.long)
        if not torch.allclose(
            cell_shifts.to(cartesian_shifts.dtype) @ cell,
            cartesian_shifts,
            atol=2e-4,
            rtol=2e-4,
        ):
            raise ValueError("DPA4 edge vectors are not integer periodic images")
        return {
            "features": features,
            "positions": positions,
            "cell": cell.unsqueeze(0),
            "atomic_numbers": atomic_numbers,
            "edge_index": edge_index,
            "cell_shifts": cell_shifts,
            "edge_vectors": edge_vectors,
            "edge_distances": torch.linalg.vector_norm(edge_vectors, dim=-1),
        }

    def forward(self, graph: PeriodicGraph) -> SO3FeatureBatch:
        encoded = [self._encode_one(graph, index) for index in range(graph.num_graphs)]
        offsets: list[int] = []
        offset = 0
        for item in encoded:
            offsets.append(offset)
            offset += item["features"].shape[0]
        device = encoded[0]["features"].device
        node_batch = torch.cat(
            [
                torch.full((item["features"].shape[0],), index, dtype=torch.long, device=device)
                for index, item in enumerate(encoded)
            ]
        )
        geometry = {
            "positions": torch.cat([item["positions"] for item in encoded]),
            "cell": torch.cat([item["cell"] for item in encoded]),
            "atomic_numbers": torch.cat([item["atomic_numbers"] for item in encoded]),
            "edge_index": torch.cat(
                [item["edge_index"] + offsets[index] for index, item in enumerate(encoded)], dim=1
            ),
            "cell_shifts": torch.cat([item["cell_shifts"] for item in encoded]),
            "edge_vectors": torch.cat([item["edge_vectors"] for item in encoded]),
            "edge_distances": torch.cat([item["edge_distances"] for item in encoded]),
        }
        return SO3FeatureBatch(
            torch.cat([item["features"] for item in encoded]),
            DPA4_SO3_LAYOUT,
            node_batch,
            geometry,
        )


class DPA4BackboneAdapter(nn.Module):
    """DPA4-Plus tap with two-forward O(3) parity completion and trainable projection."""

    def __init__(
        self,
        target_layout: IrrepLayout,
        *,
        device: str = "cpu",
        registry: BackboneResourceRegistry | None = None,
    ) -> None:
        super().__init__()
        resource = (registry or BackboneResourceRegistry())["dpa4"]
        checkpoint = resource.verify()
        require_distribution_version(DPA4_DISTRIBUTION, DPA4_VERSION)
        extractor = _DPA4SO3Extractor(str(checkpoint), device=device)
        self.resource = resource
        self.parity = InversionPairedReynolds(extractor, DPA4_SO3_LAYOUT)
        self.interface = O3InterfaceProjector(self.parity.output_layout, target_layout)
        parameter = next(extractor.model.parameters())
        self.interface.to(device=parameter.device, dtype=parameter.dtype)

    @property
    def extractor(self) -> _DPA4SO3Extractor:
        return self.parity.extractor

    def train(self, mode: bool = True):
        super().train(mode)
        self.extractor.eval()
        return self

    def forward(self, graph: PeriodicGraph) -> O3FeatureBatch:
        return self.interface(self.forward_source(graph))

    def forward_source(self, graph: PeriodicGraph) -> O3FeatureBatch:
        """Return parity-completed frozen features before the trainable interface."""

        return self.parity(graph)

    @property
    def trainable_interface_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.interface.parameters())
