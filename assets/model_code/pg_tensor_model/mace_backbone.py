from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Mapping

import torch
from e3nn import o3
from torch import nn

from .config import ModelConfig
from .model import CrystalInput, ModelOutput, PointGroupTensorModel

if TYPE_CHECKING:
    from ase import Atoms


@dataclass
class MACEBackboneOutput:
    """MACE features and the exact periodic graph used to compute them."""

    node_features: torch.Tensor
    positions: torch.Tensor
    cell: torch.Tensor
    atomic_numbers: torch.Tensor
    edge_index: torch.Tensor
    edge_vectors: torch.Tensor


class MACEBackbone(nn.Module):
    """Frozen MACE-MP checkpoint plus a trainable equivariant input adapter.

    MACE concatenates the node states from all interaction layers.  In
    ``medium-0b3`` the first state is ``128x0e + 128x1o`` while the final state
    is scalar-only.  We retain the first state and map its scalar/vector
    channels to the proposal's natural-parity layout.  Since the source has no
    l=2,3,4 irreps, those adapter outputs are exactly zero on entry; the shared
    tensor-product adaptation can create them from edge harmonics.
    """

    def __init__(
        self,
        config: ModelConfig | None = None,
        checkpoint: str = "medium-0b3",
        device: str = "cpu",
        default_dtype: str = "float32",
    ) -> None:
        super().__init__()
        try:
            from mace.calculators.foundations_models import mace_mp
        except ImportError as exc:  # pragma: no cover - dependency diagnostic
            raise ImportError(
                "MACEBackbone requires mace-torch; install assets/model_code/requirements.txt"
            ) from exc

        self.config = config or ModelConfig()
        self.checkpoint = checkpoint
        self.calculator = mace_mp(
            model=checkpoint,
            device=device,
            default_dtype=default_dtype,
        )
        self.mace_model = self.calculator.models[0]
        self.mace_model.requires_grad_(False)
        self.mace_model.eval()

        self.source_irreps = o3.Irreps(
            self.mace_model.products[0].linear.irreps_out
        )
        self.target_irreps = o3.Irreps(self.config.hidden_irreps)
        self.source_dimension = self.source_irreps.dim
        self.adapter = o3.Linear(self.source_irreps, self.target_irreps)
        self.adapter.to(device=device, dtype=next(self.mace_model.parameters()).dtype)

    def train(self, mode: bool = True) -> MACEBackbone:
        super().train(mode)
        # The checkpoint stays deterministic and frozen while the adapter trains.
        self.mace_model.eval()
        return self

    def forward(self, atoms: Atoms) -> MACEBackboneOutput:
        batch = self.calculator._atoms_to_batch(atoms)  # noqa: SLF001
        model_parameter = next(self.mace_model.parameters())
        batch = batch.to(model_parameter.device)
        self.mace_model.eval()
        with torch.no_grad():
            result = self.mace_model(
                batch.to_dict(),
                training=False,
                compute_force=False,
                compute_stress=False,
                compute_virials=False,
                compute_displacement=False,
            )
            first_interaction = result["node_feats"][:, : self.source_dimension]

        node_features = self.adapter(first_interaction)
        sender, receiver = batch.edge_index
        edge_vectors = (
            batch.positions[receiver] - batch.positions[sender] + batch.shifts
        )
        cell = batch.cell.reshape(-1, 3, 3)[0]
        atomic_numbers = torch.as_tensor(
            atoms.numbers, dtype=torch.long, device=batch.positions.device
        )
        return MACEBackboneOutput(
            node_features=node_features,
            positions=batch.positions,
            cell=cell,
            atomic_numbers=atomic_numbers,
            edge_index=batch.edge_index,
            edge_vectors=edge_vectors,
        )


class MACEPointGroupTensorModel(nn.Module):
    """End-to-end ASE ``Atoms`` -> MACE-MP -> point-group tensor model."""

    def __init__(
        self,
        config: ModelConfig | None = None,
        checkpoint: str = "medium-0b3",
        device: str = "cpu",
        default_dtype: str = "float32",
    ) -> None:
        super().__init__()
        config = config or ModelConfig()
        self.backbone = MACEBackbone(
            config=config,
            checkpoint=checkpoint,
            device=device,
            default_dtype=default_dtype,
        )
        self.tensor_model = PointGroupTensorModel(config).to(
            device=device, dtype=next(self.backbone.adapter.parameters()).dtype
        )

    def forward(
        self,
        atoms: Atoms,
        task: str,
        *,
        current_point_group: str | None = None,
        parent_residuals: Mapping[str, float | torch.Tensor] | None = None,
        canonical_to_input: torch.Tensor | None = None,
    ) -> ModelOutput:
        encoded = self.backbone(atoms)
        crystal = CrystalInput(
            positions=encoded.positions,
            cell=encoded.cell,
            atomic_numbers=encoded.atomic_numbers,
            node_features=encoded.node_features,
            edge_index=encoded.edge_index,
            edge_vectors=encoded.edge_vectors,
            current_point_group=current_point_group,
            parent_residuals=parent_residuals,
            canonical_to_input=canonical_to_input,
        )
        return self.tensor_model(crystal, task)
