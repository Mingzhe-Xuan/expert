from __future__ import annotations

import torch
from torch import nn
from e3nn import o3

from ..graphs import PeriodicGraph
from ..irreps import IrrepLayout, IrrepTerm, O3FeatureBatch
from .parity import O3InterfaceProjector
from .resources import BackboneResourceRegistry, require_distribution_version


MACE_DISTRIBUTION = "mace-torch"
MACE_VERSION = "0.3.16"
MACE_SOURCE_IRREPS = o3.Irreps("128x0e + 128x1o")


def irrep_layout_from_e3nn(irreps: o3.Irreps, prefix: str) -> IrrepLayout:
    return IrrepLayout(
        tuple(
            IrrepTerm(
                multiplicity=int(multiplicity),
                degree=int(irrep.l),
                parity="e" if irrep.p == 1 else "o",
                copy_label=f"{prefix}_{index}_l{irrep.l}{'e' if irrep.p == 1 else 'o'}",
            )
            for index, (multiplicity, irrep) in enumerate(irreps)
        )
    )


class MACEBackboneAdapter(nn.Module):
    """Frozen MACE-MP-0b3 feature tap with a trainable O(3) interface map."""

    def __init__(
        self,
        target_layout: IrrepLayout,
        *,
        device: str = "cpu",
        interface_dtype: str = "float32",
        registry: BackboneResourceRegistry | None = None,
    ) -> None:
        super().__init__()
        resource = (registry or BackboneResourceRegistry())["mace"]
        checkpoint = resource.verify()
        require_distribution_version(MACE_DISTRIBUTION, MACE_VERSION)
        try:
            from mace.calculators.foundations_models import mace_mp
        except ImportError as exc:
            raise ImportError("MACE adapter requires the revision-pinned mace-torch runtime") from exc

        calculator = mace_mp(
            model=str(checkpoint),
            device=device,
            default_dtype=interface_dtype,
        )
        backbone = calculator.models[0]
        source_irreps = o3.Irreps(backbone.products[0].linear.irreps_out)
        if source_irreps != MACE_SOURCE_IRREPS:
            raise ValueError(
                "MACE first-interaction tap layout drifted: "
                f"expected {MACE_SOURCE_IRREPS}, received {source_irreps}"
            )
        self.resource = resource
        self.calculator = calculator
        self.backbone = backbone
        self.source_irreps = source_irreps
        self.source_layout = irrep_layout_from_e3nn(source_irreps, "mace_first")
        self.interface = O3InterfaceProjector(self.source_layout, target_layout)
        parameter = next(self.backbone.parameters())
        self.interface.to(device=parameter.device, dtype=parameter.dtype)
        self.backbone.requires_grad_(False)
        self.backbone.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.eval()
        return self

    def _encode_one(self, graph: PeriodicGraph, graph_index: int):
        try:
            from ase import Atoms
        except ImportError as exc:
            raise ImportError("MACE adapter requires ASE") from exc
        mask = graph.node_batch == graph_index
        atoms = Atoms(
            numbers=graph.atomic_numbers[mask].detach().cpu().numpy(),
            positions=graph.positions[mask].detach().cpu().numpy(),
            cell=graph.cell[graph_index].detach().cpu().numpy(),
            pbc=True,
        )
        batch = self.calculator._atoms_to_batch(atoms)  # noqa: SLF001
        parameter = next(self.backbone.parameters())
        batch = batch.to(parameter.device)
        self.backbone.eval()
        with torch.no_grad():
            result = self.backbone(
                batch.to_dict(),
                training=False,
                compute_force=False,
                compute_stress=False,
                compute_virials=False,
                compute_displacement=False,
            )
            tapped = result["node_feats"][:, : self.source_irreps.dim]
        if tapped.shape != (int(mask.sum()), self.source_irreps.dim):
            raise ValueError("MACE first-interaction node feature shape drifted")
        sender, receiver = batch.edge_index
        mace_vectors = batch.positions[receiver] - batch.positions[sender] + batch.shifts
        cell = batch.cell.reshape(-1, 3, 3)[0]
        fractional_shifts = torch.round(batch.shifts @ torch.linalg.inv(cell)).to(torch.long)
        if not torch.allclose(
            fractional_shifts.to(batch.shifts.dtype) @ cell,
            batch.shifts,
            atol=2e-5,
            rtol=2e-5,
        ):
            raise ValueError("MACE Cartesian shifts are not integer lattice images")
        # The downstream graph convention stores target->source vectors. Swapping
        # MACE's sender/receiver rows retains its exact vector and periodic image.
        return {
            "features": tapped,
            "positions": batch.positions,
            "cell": cell.unsqueeze(0),
            "atomic_numbers": graph.atomic_numbers[mask].to(parameter.device),
            "edge_index": torch.stack((receiver, sender)),
            "cell_shifts": fractional_shifts,
            "edge_vectors": mace_vectors,
            "edge_distances": torch.linalg.vector_norm(mace_vectors, dim=-1),
        }

    def forward(self, graph: PeriodicGraph) -> O3FeatureBatch:
        encoded = [self._encode_one(graph, index) for index in range(graph.num_graphs)]
        node_offsets = []
        offset = 0
        for item in encoded:
            node_offsets.append(offset)
            offset += item["features"].shape[0]
        device = encoded[0]["features"].device
        node_batch = torch.cat(
            [
                torch.full(
                    (item["features"].shape[0],), index, dtype=torch.long, device=device
                )
                for index, item in enumerate(encoded)
            ]
        )
        edge_geometry = {
            "positions": torch.cat([item["positions"] for item in encoded]),
            "cell": torch.cat([item["cell"] for item in encoded]),
            "atomic_numbers": torch.cat([item["atomic_numbers"] for item in encoded]),
            "edge_index": torch.cat(
                [item["edge_index"] + node_offsets[index] for index, item in enumerate(encoded)],
                dim=1,
            ),
            "cell_shifts": torch.cat([item["cell_shifts"] for item in encoded]),
            "edge_vectors": torch.cat([item["edge_vectors"] for item in encoded]),
            "edge_distances": torch.cat([item["edge_distances"] for item in encoded]),
        }
        source = O3FeatureBatch(
            node_features=torch.cat([item["features"] for item in encoded]),
            node_layout=self.source_layout,
            node_batch=node_batch,
            edge_geometry=edge_geometry,
        )
        return self.interface(source)

    @property
    def trainable_interface_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.interface.parameters())
