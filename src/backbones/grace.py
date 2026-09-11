from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from e3nn import o3

from ..graphs import PeriodicGraph
from ..irreps import IrrepLayout, IrrepTerm, O3FeatureBatch
from .parity import O3InterfaceProjector
from .resources import BackboneResourceRegistry, require_distribution_version


GRACE_DISTRIBUTION = "tensorpotential"
GRACE_VERSION = "0.6.0"
GRACE_CHECKPOINT_METADATA_VERSION = "0.5.10"
GRACE_TAP = "AA"
GRACE_CHANNELS = 32
GRACE_AA_GROUPS = (
    *((0, 1, f"({degree},{degree})") for degree in range(5)),
    *((1, -1, history) for history in ("(1,0)", "(2,1)", "(3,2)", "(4,3)")),
    *((2, 1, history) for history in ("(1,1)", "(2,0)", "(2,2)", "(3,1)", "(3,3)", "(4,2)", "(4,4)")),
    *((3, -1, history) for history in ("(2,1)", "(3,0)", "(3,2)", "(4,1)", "(4,3)")),
    *((4, 1, history) for history in ("(2,2)", "(3,1)", "(3,3)", "(4,0)", "(4,2)", "(4,4)")),
)
GRACE_AA_COEFFICIENTS = sum(2 * degree + 1 for degree, _, _ in GRACE_AA_GROUPS)
GRACE_SOURCE_LAYOUT = IrrepLayout(
    tuple(
        IrrepTerm(
            GRACE_CHANNELS,
            degree,
            "e" if parity == 1 else "o",
            f"grace_AA_{index}_l{degree}_{history}",
        )
        for index, (degree, parity, history) in enumerate(GRACE_AA_GROUPS)
    )
)


def validate_grace_aa_metadata(rows: Sequence[tuple[int, int, int, str]]) -> None:
    """Validate exact contiguous (l,m,parity,history) groups at the frozen AA tap."""

    expected = tuple(
        (degree, order, parity, history)
        for degree, parity, history in GRACE_AA_GROUPS
        for order in range(-degree, degree + 1)
    )
    actual = tuple((int(l), int(m), int(parity), str(history)) for l, m, parity, history in rows)
    if actual != expected:
        raise ValueError("GRACE AA coupling metadata drifted from the frozen layout")


def convert_grace_aa(
    features: torch.Tensor,
    tp_to_e3nn: Sequence[torch.Tensor],
) -> torch.Tensor:
    """Convert [node, channel, TP-lm] AA features into labelled e3nn copy-major blocks."""

    if features.ndim != 3 or tuple(features.shape[1:]) != (
        GRACE_CHANNELS,
        GRACE_AA_COEFFICIENTS,
    ):
        raise ValueError(
            f"GRACE AA features must have shape [num_nodes, {GRACE_CHANNELS}, "
            f"{GRACE_AA_COEFFICIENTS}]"
        )
    if len(tp_to_e3nn) != 5:
        raise ValueError("GRACE basis conversion requires one matrix for each l=0..4")
    blocks = []
    offset = 0
    for degree, _, _ in GRACE_AA_GROUPS:
        width = 2 * degree + 1
        transform = tp_to_e3nn[degree].to(device=features.device, dtype=features.dtype)
        if transform.shape != (width, width):
            raise ValueError(f"invalid GRACE l={degree} basis transform shape")
        block = features[:, :, offset : offset + width] @ transform
        blocks.append(block.reshape(features.shape[0], -1))
        offset += width
    return torch.cat(blocks, dim=-1)


def _derive_tp_to_e3nn_matrices() -> tuple[torch.Tensor, ...]:
    """Fit and strictly validate the runtime's real-harmonic change of basis."""

    import tensorflow as tf
    from tensorpotential.functions.spherical_harmonics import SphericalHarmonics

    count = 64
    index = np.arange(count, dtype=np.float64) + 0.5
    z = 1.0 - 2.0 * index / count
    azimuth = index * np.pi * (3.0 - np.sqrt(5.0))
    radius = np.sqrt(1.0 - z * z)
    points = np.stack((radius * np.cos(azimuth), radius * np.sin(azimuth), z), axis=1)
    harmonics = SphericalHarmonics(4)
    harmonics.build(tf.float32)
    tp_values = harmonics(tf.constant(points, dtype=tf.float64)).numpy()
    e3nn_values = o3.spherical_harmonics(
        list(range(5)), torch.from_numpy(points), normalize=True, normalization="component"
    ).numpy()
    matrices = []
    for degree in range(5):
        start, stop = degree * degree, (degree + 1) * (degree + 1)
        matrix = np.linalg.lstsq(tp_values[:, start:stop], e3nn_values[:, start:stop], rcond=None)[0]
        residual = np.max(np.abs(tp_values[:, start:stop] @ matrix - e3nn_values[:, start:stop]))
        if residual > 2e-12 or not np.allclose(matrix.T @ matrix, np.eye(2 * degree + 1), atol=2e-12):
            raise ValueError(f"GRACE l={degree} harmonic convention could not be mapped to e3nn")
        matrices.append(torch.from_numpy(matrix))
    return tuple(matrices)


def _artifact_paths(resource: Any) -> dict[str, Path]:
    paths = {artifact.filename: artifact.path for artifact in resource.artifacts}
    required = {"model.yaml", "checkpoint.index", "checkpoint.data-00000-of-00001"}
    if set(paths) != required:
        raise ValueError("GRACE manifest must enumerate the exact extracted checkpoint artifacts")
    return paths


class GRACEBackboneAdapter(nn.Module):
    """Frozen GRACE AA feature tap with an explicit TensorPotential-to-e3nn basis map."""

    def __init__(
        self,
        target_layout: IrrepLayout,
        *,
        device: str = "cpu",
        registry: BackboneResourceRegistry | None = None,
    ) -> None:
        super().__init__()
        resource = (registry or BackboneResourceRegistry())["grace"]
        resource.verify()
        require_distribution_version(GRACE_DISTRIBUTION, GRACE_VERSION)
        paths = _artifact_paths(resource)
        try:
            import tensorflow as tf
            import yaml
            from tensorpotential import constants
            from tensorpotential.data.databuilder import GeometricalDataBuilder
            from tensorpotential.instructions import load_instructions
            from tensorpotential.tensorpot import TensorPotential
            from tensorpotential.tpmodel import ComputeFunction, extract_cutoff_and_elements
        except ImportError as exc:
            raise ImportError("GRACE adapter requires the pinned TensorPotential runtime") from exc

        metadata = yaml.safe_load(paths["model.yaml"].read_text(encoding="utf-8"))
        if metadata.get("metadata", {}).get("tensorpotential_version") != GRACE_CHECKPOINT_METADATA_VERSION:
            raise ValueError("GRACE checkpoint metadata version drifted")
        rho = metadata["instructions"]["rho"]
        if rho.get("ls_max") != [0, 0, 0, 0] or rho.get("n_out") != 17:
            raise ValueError("GRACE rho must remain the scalar-only 17-channel readout input")
        aa_yaml = metadata["instructions"][GRACE_TAP]
        if aa_yaml.get("lmax") != 4 or aa_yaml.get("Lmax") != 4:
            raise ValueError("GRACE AA tap must retain lmax=Lmax=4")

        instructions = load_instructions(str(paths["model.yaml"]))
        aa = instructions[GRACE_TAP]
        rows = tuple(
            (row.l, row.m, row.parity, row.hist)
            for row in aa.coupling_meta_data[["l", "m", "parity", "hist"]].itertuples(index=False)
        )
        if int(aa.n_out) != GRACE_CHANNELS:
            raise ValueError("GRACE AA channel count drifted")
        validate_grace_aa_metadata(rows)

        class FeatureCompute(ComputeFunction):
            specs: dict[str, dict] = {}

            def __call__(self, graph, input_data, training=False):
                for name, instruction in graph.items():
                    instruction(input_data, training=training)
                    if name == GRACE_TAP:
                        break
                return {"node_features": input_data[GRACE_TAP]}

        model = TensorPotential(
            instructions,
            model_compute_function=FeatureCompute(),
            param_dtype=tf.float32,
            eager_mode=True,
            jit_compile=False,
        )
        checkpoint_prefix = str(paths["checkpoint.index"]).removesuffix(".index")
        model.load_checkpoint(
            checkpoint_name=checkpoint_prefix,
            model_only=True,
            expect_partial=True,
            assert_existing_objects_matched=True,
        )
        cutoff, symbols, indices = extract_cutoff_and_elements(instructions)
        element_map = {str(symbol): int(index) for symbol, index in zip(symbols, indices)}
        self.data_builder = GeometricalDataBuilder(
            elements_map=element_map,
            cutoff=float(cutoff),
            float_dtype=np.float64,
        )
        self.constants = constants
        self.tf = tf
        self.resource = resource
        self.tp = model
        self.tp_to_e3nn = _derive_tp_to_e3nn_matrices()
        self.interface = O3InterfaceProjector(GRACE_SOURCE_LAYOUT, target_layout).to(device=device)
        self.device = torch.device(device)

    def train(self, mode: bool = True):
        super().train(mode)
        return self

    def _encode_one(self, graph: PeriodicGraph, graph_index: int) -> dict[str, torch.Tensor]:
        from ase import Atoms

        mask = graph.node_batch == graph_index
        atoms = Atoms(
            numbers=graph.atomic_numbers[mask].detach().cpu().numpy(),
            positions=graph.positions[mask].detach().cpu().numpy(),
            cell=graph.cell[graph_index].detach().cpu().numpy(),
            pbc=True,
        )
        sample = self.data_builder.extract_from_ase_atoms(atoms)
        data = self.data_builder.join_to_batch([sample])
        tensor_data = {key: self.tf.convert_to_tensor(value) for key, value in data.items()}
        result = self.tp.model.compute(tensor_data)
        raw = torch.from_numpy(result["node_features"].numpy()).to(self.device)
        features = convert_grace_aa(raw, self.tp_to_e3nn)
        n_atoms = int(data[self.constants.N_ATOMS_BATCH_REAL])
        n_edges = int(data[self.constants.N_NEIGHBORS_REAL])
        edge_vectors = torch.from_numpy(data[self.constants.BOND_VECTOR][:n_edges]).to(self.device)
        source = torch.from_numpy(data[self.constants.BOND_IND_J][:n_edges]).long().to(self.device)
        target = torch.from_numpy(data[self.constants.BOND_IND_I][:n_edges]).long().to(self.device)
        positions = torch.from_numpy(atoms.positions).to(self.device, dtype=edge_vectors.dtype)
        cell = torch.from_numpy(np.asarray(atoms.cell)).to(self.device, dtype=edge_vectors.dtype)
        cartesian_shifts = edge_vectors - (positions[source] - positions[target])
        cell_shifts = torch.round(cartesian_shifts @ torch.linalg.inv(cell)).long()
        if features.shape[0] != n_atoms or not torch.allclose(
            cell_shifts.to(edge_vectors.dtype) @ cell,
            cartesian_shifts,
            atol=2e-6,
            rtol=2e-6,
        ):
            raise ValueError("GRACE node count or periodic edge geometry drifted")
        return {
            "features": features,
            "positions": positions,
            "cell": cell.unsqueeze(0),
            "atomic_numbers": graph.atomic_numbers[mask].to(self.device),
            "edge_index": torch.stack((source, target)),
            "cell_shifts": cell_shifts,
            "edge_vectors": edge_vectors,
            "edge_distances": torch.linalg.vector_norm(edge_vectors, dim=-1),
        }

    def forward(self, graph: PeriodicGraph) -> O3FeatureBatch:
        encoded = [self._encode_one(graph, index) for index in range(graph.num_graphs)]
        offsets, offset = [], 0
        for item in encoded:
            offsets.append(offset)
            offset += item["features"].shape[0]
        node_batch = torch.cat(
            [
                torch.full((item["features"].shape[0],), index, dtype=torch.long, device=self.device)
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
        source = O3FeatureBatch(
            torch.cat([item["features"] for item in encoded]),
            GRACE_SOURCE_LAYOUT,
            node_batch,
            edge_geometry=geometry,
        )
        return self.interface(source)

    @property
    def trainable_interface_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.interface.parameters())
