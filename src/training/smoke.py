from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Callable, Sequence

import torch

from ..backbones import BackboneResourceRegistry
from ..configs import ArchitectureConfig
from ..data import TensorSample, TrainingUnit, load_five_structure_smoke
from ..experts import PointGroupTensorModel, default_hidden_layout
from ..graphs import PeriodicGraph, build_periodic_graph, collate_periodic_graphs
from ..heads import TARGET_LAYOUTS, cartesian_to_irreps, rotate_cartesian
from ..irreps import ConventionMetadata
from ..models import BackboneTensorModel, build_backbone_adapter
from ..symmetry import SymmetryRecord, canonicalize_structure
from .checkpoint import load_checkpoint, save_checkpoint
from .losses import coefficient_mse, physical_coefficient_metrics
from .normalization import CoefficientNormalizer


@dataclass(frozen=True, slots=True)
class PreparedTensorBatch:
    graph: PeriodicGraph
    symmetries: tuple[SymmetryRecord, ...]
    target_coefficients: torch.Tensor
    sample_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.target_coefficients.ndim != 2:
            raise ValueError("prepared targets must have shape [items, coefficients]")
        if len(self.symmetries) != self.graph.num_graphs:
            raise ValueError("prepared symmetries must align with graph batch")
        if len(self.sample_ids) != self.graph.num_graphs:
            raise ValueError("prepared sample IDs must align with graph batch")


def prepare_tensor_batch(
    samples: Sequence[TensorSample],
    *,
    cutoff: float,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> PreparedTensorBatch:
    """Canonicalize structures and targets together, retaining BEC site order."""

    if not samples:
        raise ValueError("cannot prepare an empty tensor batch")
    unit = samples[0].unit
    if any(sample.unit != unit for sample in samples):
        raise ValueError("a prepared batch cannot mix independent training units")
    graphs, symmetries, targets = [], [], []
    for sample in samples:
        canonical = canonicalize_structure(
            sample.cartesian_positions.to(device=device, dtype=dtype),
            sample.lattice.to(device=device, dtype=dtype),
            sample.atomic_numbers.to(device=device),
        )
        graphs.append(
            build_periodic_graph(
                canonical.canonical_positions,
                canonical.canonical_cell,
                canonical.atomic_numbers,
                cutoff,
            )
        )
        symmetries.append(canonical.symmetry)
        target = sample.target_cartesian.to(device=device, dtype=dtype)
        rotated = rotate_cartesian(target, canonical.input_to_canonical, unit.target)
        coefficients = cartesian_to_irreps(rotated, unit.target)
        targets.append(coefficients if unit.target == "bec" else coefficients.unsqueeze(0))
    batch = collate_periodic_graphs(graphs)
    target_coefficients = torch.cat(targets)
    expected = batch.num_nodes if unit.target == "bec" else batch.num_graphs
    if target_coefficients.shape != (expected, TARGET_LAYOUTS[unit.target].dimension):
        raise ValueError("canonical targets do not align with graph scope")
    return PreparedTensorBatch(
        batch,
        tuple(symmetries),
        target_coefficients,
        tuple(sample.sample_id for sample in samples),
    )


def default_convention_metadata() -> ConventionMetadata:
    asset = Path(__file__).resolve().parents[2] / "assets" / "docs" / "subgroup_chain.json"
    return ConventionMetadata(
        schema_version=1,
        real_harmonic_convention="e3nn_real_y_lm_m_minus_l_to_l_component",
        cg_convention="e3nn_wigner_3j_real_basis",
        path_ordering="degree_parity_copy_then_lexicographic_cg_path",
        copy_ordering="target_contract_v1",
        subduction_checksum=hashlib.sha256(asset.read_bytes()).hexdigest(),
    )


def build_real_system(
    backbone_family: str,
    architecture: ArchitectureConfig,
    unit: TrainingUnit,
    point_groups: tuple[str, ...],
    *,
    device: str,
) -> BackboneTensorModel:
    resource = BackboneResourceRegistry()[backbone_family]
    hidden = default_hidden_layout(architecture)
    adapter = build_backbone_adapter(backbone_family, hidden, device=device)
    downstream = PointGroupTensorModel(
        architecture,
        unit.target,
        hidden_layout=hidden,
        expert_point_groups=point_groups,
        cutoff=resource.cutoff_angstrom,
    ).to(device)
    return BackboneTensorModel(
        adapter, downstream, cutoff=resource.cutoff_angstrom
    ).to(device)


def _evaluate(system, batch, unit, normalizer):
    system.eval()
    with torch.no_grad():
        prediction = system(batch.graph, batch.symmetries)
        loss = coefficient_mse(
            prediction.irrep_coefficients,
            batch.target_coefficients,
            TARGET_LAYOUTS[unit.target],
            normalizer,
        ).total
    metrics = physical_coefficient_metrics(
        prediction.irrep_coefficients,
        batch.target_coefficients,
        TARGET_LAYOUTS[unit.target],
    )
    return float(loss), metrics


def run_five_structure_smoke(
    *,
    backbone_family: str,
    architecture: ArchitectureConfig,
    unit: TrainingUnit,
    checkpoint_path: str | Path,
    device: str = "cuda",
    samples: Sequence[TensorSample] | None = None,
    system_builder: Callable[..., BackboneTensorModel] = build_real_system,
) -> dict[str, object]:
    selected = tuple(load_five_structure_smoke(unit) if samples is None else samples)
    if len(selected) != 5 or any(sample.unit != unit for sample in selected):
        raise ValueError("smoke runner requires exactly five samples from one training unit")
    cutoff = BackboneResourceRegistry()[backbone_family].cutoff_angstrom
    train = prepare_tensor_batch(selected[:3], cutoff=cutoff, device=device)
    validation = prepare_tensor_batch(selected[3:4], cutoff=cutoff, device=device)
    test = prepare_tensor_batch(selected[4:], cutoff=cutoff, device=device)
    point_groups = tuple(
        dict.fromkeys(
            symmetry.current_point_group
            for batch in (train, validation, test)
            for symmetry in batch.symmetries
        )
    )
    system = system_builder(
        backbone_family, architecture, unit, point_groups, device=device
    )
    trainable = [parameter for parameter in system.parameters() if parameter.requires_grad]
    if not trainable:
        raise ValueError("system has no trainable interface/downstream parameters")
    optimizer = torch.optim.Adam(trainable, lr=1.0e-3)
    normalizer = CoefficientNormalizer.fit(
        train.target_coefficients,
        TARGET_LAYOUTS[unit.target],
        unit,
        split="train",
    )
    system.train()
    optimizer.zero_grad(set_to_none=True)
    prediction = system(train.graph, train.symmetries)
    train_loss = coefficient_mse(
        prediction.irrep_coefficients,
        train.target_coefficients,
        TARGET_LAYOUTS[unit.target],
        normalizer,
    ).total
    if not torch.isfinite(train_loss):
        raise ValueError("smoke training loss is non-finite")
    train_loss.backward()
    frozen_with_grad = [
        parameter
        for parameter in system.parameters()
        if not parameter.requires_grad and parameter.grad is not None
    ]
    if frozen_with_grad:
        raise AssertionError("a frozen backbone parameter received a gradient")
    gradients = [parameter.grad for parameter in trainable if parameter.grad is not None]
    if not gradients or any(not torch.isfinite(value).all() for value in gradients):
        raise AssertionError("trainable gradients are missing or non-finite")
    optimizer.step()
    validation_loss, validation_metrics = _evaluate(
        system, validation, unit, normalizer
    )
    convention = default_convention_metadata()
    save_checkpoint(
        checkpoint_path,
        model=system,
        optimizer=optimizer,
        architecture=architecture,
        unit=unit,
        convention=convention,
        normalizer=normalizer,
        step=1,
    )
    loaded = load_checkpoint(
        checkpoint_path,
        model=system,
        optimizer=optimizer,
        expected_architecture=architecture,
        expected_unit=unit,
        expected_convention=convention,
        expected_layout=TARGET_LAYOUTS[unit.target],
        map_location=device,
    )
    test_loss, test_metrics = _evaluate(system, test, unit, loaded.normalizer)
    active = system.downstream.active_nonbackbone_parameter_count(train.symmetries)
    if active >= 5_000_000:
        raise AssertionError("active non-backbone parameter budget was exceeded")
    return {
        "schema_version": 1,
        "status": "passed",
        "backbone": backbone_family,
        "training_unit": unit.namespace,
        "architecture": architecture.to_dict(),
        "sample_ids": {
            "train": list(train.sample_ids),
            "validation": list(validation.sample_ids),
            "test": list(test.sample_ids),
        },
        "train_loss": float(train_loss.detach()),
        "validation_loss": validation_loss,
        "test_loss": test_loss,
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "checkpoint": str(checkpoint_path),
        "conventions": asdict(convention),
        "convention_checksum": convention.checksum,
        "trainable_parameters": sum(parameter.numel() for parameter in trainable),
        "active_nonbackbone_parameters": active,
        "point_groups": list(point_groups),
    }


def write_smoke_report(path: str | Path, report: dict[str, object]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
