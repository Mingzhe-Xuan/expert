from __future__ import annotations

from typing import Any

import torch
from torch import nn

from ..graphs import PeriodicGraph
from ..irreps import IrrepLayout, O3FeatureBatch
from .parity import (
    InversionPairedReynolds,
    O3InterfaceProjector,
    SO3FeatureBatch,
    SO3Layout,
    SO3Term,
)
from .resources import BackboneResourceRegistry, require_distribution_version


EQUIFORMER_DISTRIBUTION = "fairchem-core"
EQUIFORMER_VERSION = "1.10.0"
EQUIFORMER_CHANNELS = 128
EQUIFORMER_LMAX = 4
EQUIFORMER_COEFFICIENTS = (EQUIFORMER_LMAX + 1) ** 2
EQUIFORMER_SO3_LAYOUT = SO3Layout(
    tuple(
        SO3Term(EQUIFORMER_CHANNELS, degree, f"equiformerv2_final_l{degree}")
        for degree in range(EQUIFORMER_LMAX + 1)
    )
)


def flatten_equiformer_embedding(embedding: torch.Tensor) -> torch.Tensor:
    """Convert fairchem's [node, lm, channel] tensor to copy-major e3nn blocks."""

    expected_tail = (EQUIFORMER_COEFFICIENTS, EQUIFORMER_CHANNELS)
    if embedding.ndim != 3 or tuple(embedding.shape[1:]) != expected_tail:
        raise ValueError(
            "EquiformerV2 final embedding must have shape "
            f"[num_nodes, {EQUIFORMER_COEFFICIENTS}, {EQUIFORMER_CHANNELS}]"
        )
    blocks = []
    offset = 0
    for degree in range(EQUIFORMER_LMAX + 1):
        width = 2 * degree + 1
        block = embedding[:, offset : offset + width, :]
        blocks.append(block.transpose(1, 2).contiguous().reshape(embedding.shape[0], -1))
        offset += width
    return torch.cat(blocks, dim=-1)


def _graph_tensor(graph: Any, name: str) -> torch.Tensor:
    value = getattr(graph, name, None)
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"EquiformerV2 graph lacks tensor {name!r}")
    return value


class _EquiformerV2SO3Extractor(nn.Module):
    """Frozen fairchem backbone exposing its final normalized SO(3) node state."""

    def __init__(self, checkpoint: str, *, device: str) -> None:
        super().__init__()
        try:
            from fairchem.core.common.utils import load_model_and_weights_from_checkpoint
            from fairchem.core.preprocessing import AtomsToGraphs
        except ImportError as exc:
            raise ImportError("EquiformerV2 adapter requires the pinned fairchem runtime") from exc

        model = load_model_and_weights_from_checkpoint(checkpoint).to(device)
        backbone = getattr(model, "backbone", None)
        if backbone is None:
            raise ValueError("EquiformerV2 checkpoint does not expose HydraModel.backbone")
        if list(getattr(backbone, "lmax_list", [])) != [EQUIFORMER_LMAX]:
            raise ValueError("EquiformerV2 checkpoint lmax drifted from [4]")
        if list(getattr(backbone, "mmax_list", [])) != [2]:
            raise ValueError("EquiformerV2 checkpoint mmax drifted from [2]")
        if int(getattr(backbone, "sphere_channels", -1)) != EQUIFORMER_CHANNELS:
            raise ValueError("EquiformerV2 checkpoint sphere channels drifted from 128")
        if int(getattr(backbone, "num_layers", -1)) != 8:
            raise ValueError("EquiformerV2 checkpoint layer count drifted from 8")
        if float(getattr(backbone, "max_radius", -1.0)) != 12.0:
            raise ValueError("EquiformerV2 checkpoint cutoff drifted from 12 angstrom")
        self.model = model
        self.backbone = backbone
        self.converter = AtomsToGraphs(r_edges=False, r_pbc=True)
        self.model.requires_grad_(False)
        self.model.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        self.model.eval()
        return self

    def _encode_one(self, graph: PeriodicGraph, graph_index: int) -> dict[str, torch.Tensor]:
        try:
            from ase import Atoms
            from torch_geometric.data import Batch
        except ImportError as exc:
            raise ImportError("EquiformerV2 adapter requires ASE and torch-geometric") from exc

        mask = graph.node_batch == graph_index
        parameter = next(self.backbone.parameters())
        atoms = Atoms(
            numbers=graph.atomic_numbers[mask].detach().cpu().numpy(),
            positions=graph.positions[mask].detach().cpu().numpy(),
            cell=graph.cell[graph_index].detach().cpu().numpy(),
            pbc=True,
        )
        data = Batch.from_data_list([self.converter.convert(atoms)]).to(parameter.device)
        self.backbone.eval()
        with torch.no_grad():
            result = self.backbone(data)
        node_embedding = result.get("node_embedding")
        raw = getattr(node_embedding, "embedding", None)
        if not isinstance(raw, torch.Tensor):
            raise ValueError("EquiformerV2 backbone did not return node_embedding.embedding")
        features = flatten_equiformer_embedding(raw)
        if features.shape[0] != int(mask.sum()):
            raise ValueError("EquiformerV2 feature tap changed the local node count")

        used_graph = result.get("graph")
        edge_index = _graph_tensor(used_graph, "edge_index").long()
        edge_vectors = _graph_tensor(used_graph, "edge_distance_vec")
        edge_distances = _graph_tensor(used_graph, "edge_distance")
        cell_shifts = _graph_tensor(used_graph, "cell_offsets").long()
        positions = data.pos
        cell = data.cell.reshape(-1, 3, 3)[0]
        source, target = edge_index
        cartesian_shifts = edge_vectors - (positions[source] - positions[target])
        if not torch.allclose(
            cell_shifts.to(edge_vectors.dtype) @ cell,
            cartesian_shifts,
            atol=2e-5,
            rtol=2e-5,
        ):
            raise ValueError("EquiformerV2 edge vectors disagree with periodic cell offsets")
        return {
            "features": features,
            "positions": positions,
            "cell": cell.unsqueeze(0),
            "atomic_numbers": data.atomic_numbers.long(),
            "edge_index": edge_index,
            "cell_shifts": cell_shifts,
            "edge_vectors": edge_vectors,
            "edge_distances": edge_distances,
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
            EQUIFORMER_SO3_LAYOUT,
            node_batch,
            geometry,
        )


class EquiformerV2BackboneAdapter(nn.Module):
    """OMat24 EquiformerV2 tap with paired-inversion O(3) completion."""

    def __init__(
        self,
        target_layout: IrrepLayout,
        *,
        device: str = "cpu",
        registry: BackboneResourceRegistry | None = None,
    ) -> None:
        super().__init__()
        resource = (registry or BackboneResourceRegistry())["equiformerv2"]
        checkpoint = resource.verify()
        require_distribution_version(EQUIFORMER_DISTRIBUTION, EQUIFORMER_VERSION)
        extractor = _EquiformerV2SO3Extractor(str(checkpoint), device=device)
        self.resource = resource
        self.parity = InversionPairedReynolds(extractor, EQUIFORMER_SO3_LAYOUT)
        self.interface = O3InterfaceProjector(self.parity.output_layout, target_layout)
        parameter = next(extractor.backbone.parameters())
        self.interface.to(device=parameter.device, dtype=parameter.dtype)

    @property
    def extractor(self) -> _EquiformerV2SO3Extractor:
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
