from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import os
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment
import spglib
import torch

from .contracts import ParentDAGSpec, ParentEmbeddingSpec, SymmetryRecord
from .registry import canonical_point_group_symbol


DEFAULT_PARENT_SYMPRECS = (1.0e-4, 1.0e-3, 1.0e-2, 5.0e-2, 1.0e-1)
PARENT_DAG_CONVENTION = "spglib-relaxed-common-cell-v3"


@dataclass(frozen=True, slots=True)
class MaterialParentRouting:
    dag: ParentDAGSpec
    residuals: Mapping[int, float]
    detection_symprecs: Mapping[int, float]

    def __post_init__(self) -> None:
        active = {self.dag.current_hall_number}
        active.update(embedding.parent_hall_number for embedding in self.dag.embeddings)
        if set(self.residuals) != active or set(self.detection_symprecs) != active:
            raise ValueError("parent routing metadata must cover exactly the active Hall nodes")
        if self.residuals[self.dag.current_hall_number] != 0.0:
            raise ValueError("the current Hall node must have zero residual")
        if any(not np.isfinite(value) or value < 0 for value in self.residuals.values()):
            raise ValueError("parent residuals must be finite and non-negative")


def _fractional_structure(
    positions: torch.Tensor, cell: torch.Tensor, atomic_numbers: torch.Tensor
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lattice = cell.detach().cpu().double().numpy()
    fractional = torch.linalg.solve(cell.T, positions.T).T.detach().cpu().double().numpy()
    fractional -= np.floor(fractional)
    species = atomic_numbers.detach().cpu().numpy().astype(np.int64, copy=False)
    return lattice, fractional, species


def _affine_key(rotation, translation, tolerance: float = 1.0e-6) -> tuple[int, ...]:
    rotation = np.asarray(rotation, dtype=np.int64)
    translation = np.asarray(translation, dtype=np.float64)
    translation -= np.floor(translation)
    translation[np.isclose(translation, 1.0, atol=tolerance)] = 0.0
    return (*rotation.reshape(-1).tolist(), *np.rint(translation / tolerance).astype(np.int64))


def _operation_keys(dataset) -> set[tuple[int, ...]]:
    return {
        _affine_key(rotation, translation)
        for rotation, translation in zip(dataset.rotations, dataset.translations)
    }


def _periodic_squared_costs(
    moved: np.ndarray, reference: np.ndarray, lattice: np.ndarray
) -> np.ndarray:
    delta = moved[:, None, :] - reference[None, :, :]
    delta -= np.rint(delta)
    vectors = delta @ lattice
    return np.sum(vectors * vectors, axis=-1)


def _parent_residual(
    dataset, lattice: np.ndarray, fractional: np.ndarray, species: np.ndarray
) -> float:
    squared = []
    for rotation, translation in zip(dataset.rotations, dataset.translations):
        moved = fractional @ np.asarray(rotation).T + np.asarray(translation)
        moved -= np.floor(moved)
        for atomic_number in np.unique(species):
            indices = np.flatnonzero(species == atomic_number)
            costs = _periodic_squared_costs(moved[indices], fractional[indices], lattice)
            rows, columns = linear_sum_assignment(costs)
            squared.extend(costs[rows, columns].tolist())
    site_rms = float(np.sqrt(np.mean(squared)))
    metric = lattice @ lattice.T
    metric_scale = max(float(np.linalg.norm(metric)), 1.0e-12)
    length_scale = abs(float(np.linalg.det(lattice))) ** (1.0 / 3.0)
    metric_residuals = [
        length_scale
        * float(
            np.linalg.norm(
                np.asarray(rotation).T @ metric @ np.asarray(rotation) - metric
            )
        )
        / metric_scale
        for rotation in dataset.rotations
    ]
    lattice_rms = float(np.sqrt(np.mean(np.square(metric_residuals))))
    return float(np.hypot(site_rms, lattice_rms))


def _operations(dataset):
    return tuple(
        (
            tuple(tuple(int(value) for value in row) for row in np.asarray(rotation)),
            tuple(float(value) for value in np.asarray(translation)),
        )
        for rotation, translation in zip(dataset.rotations, dataset.translations)
    )


def _embedding(
    *,
    parent_dataset,
    child_dataset,
    atomic_numbers: np.ndarray,
    detection_symprec: float,
) -> ParentEmbeddingSpec:
    parent_hall = int(parent_dataset.hall_number)
    child_hall = int(child_dataset.hall_number)
    splitting = tuple(
        sorted(
            {
                f"{parent}->{child}"
                for parent, child in zip(parent_dataset.wyckoffs, child_dataset.wyckoffs)
            }
        )
    )
    species = tuple(int(value) for value in atomic_numbers)
    candidate = ParentEmbeddingSpec(
        parent_hall_number=parent_hall,
        child_hall_number=child_hall,
        parent_setting=f"common-input-cell:hall-{parent_hall}:{parent_dataset.choice}",
        child_setting=f"common-input-cell:hall-{child_hall}:{child_dataset.choice}",
        basis_transform=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        origin_shift=(0.0, 0.0, 0.0),
        supercell_transform=((1, 0, 0), (0, 1, 0), (0, 0, 1)),
        operations=_operations(parent_dataset),
        parent_atomic_numbers=species,
        child_atomic_numbers=species,
        atom_correspondence=tuple(range(len(species))),
        wyckoff_splitting=splitting or ("unchanged",),
        domain_variant=f"relaxed-symprec-{detection_symprec:.8g}",
        convention_id=PARENT_DAG_CONVENTION,
        version=1,
        checksum="0" * 64,
    )
    return replace(candidate, checksum=candidate.payload_checksum())


def discover_material_parent_routing(
    material_id: str,
    positions: torch.Tensor,
    cell: torch.Tensor,
    atomic_numbers: torch.Tensor,
    symmetry: SymmetryRecord,
    *,
    symprecs: Sequence[float] = DEFAULT_PARENT_SYMPRECS,
    base_symprec: float = 1.0e-5,
    max_parents: int = 2,
    max_operation_index: int = 4,
) -> MaterialParentRouting:
    """Build a star-shaped material Hall DAG from verified relaxed-symmetry supergroups."""

    if not material_id or not symprecs or any(value <= base_symprec for value in symprecs):
        raise ValueError("parent detection requires an ID and symprecs above base_symprec")
    if tuple(symprecs) != tuple(sorted(set(symprecs))) or max_parents < 1:
        raise ValueError("parent symprecs must be strictly increasing and max_parents positive")
    lattice, fractional, species = _fractional_structure(positions, cell, atomic_numbers)
    spglib_cell = (lattice, fractional, species)
    child = spglib.get_symmetry_dataset(
        spglib_cell,
        symprec=base_symprec,
        angle_tolerance=-1.0,
        hall_number=symmetry.hall_number,
    )
    if child is None:
        child = spglib.get_symmetry_dataset(
            spglib_cell, symprec=base_symprec, angle_tolerance=-1.0
        )
    if child is None:
        raise ValueError("spglib could not reproduce the current structure for parent detection")
    child_group = canonical_point_group_symbol(str(child.pointgroup))
    if child_group != canonical_point_group_symbol(symmetry.current_point_group):
        raise ValueError("parent detection base point group disagrees with cached symmetry")
    if int(child.hall_number) != symmetry.hall_number:
        raise ValueError("parent detection base Hall setting disagrees with cached symmetry")

    child_keys = _operation_keys(child)
    child_order = len(child_keys)
    accepted: dict[str, tuple[object, float, float]] = {}
    for symprec in symprecs:
        candidate = spglib.get_symmetry_dataset(
            spglib_cell, symprec=float(symprec), angle_tolerance=-1.0
        )
        if candidate is None:
            continue
        parent_group = canonical_point_group_symbol(str(candidate.pointgroup))
        parent_keys = _operation_keys(candidate)
        if parent_group == child_group or not child_keys.issubset(parent_keys):
            continue
        if len(parent_keys) <= child_order or len(parent_keys) % child_order:
            continue
        if len(parent_keys) // child_order > max_operation_index:
            continue
        residual = _parent_residual(candidate, lattice, fractional, species)
        if residual > 5.0 * float(symprec) + 1.0e-8:
            continue
        previous = accepted.get(parent_group)
        if previous is None or residual < previous[1]:
            accepted[parent_group] = (candidate, residual, float(symprec))

    selected = []
    embeddings = []
    for candidate in sorted(
        accepted.values(), key=lambda item: (item[1], int(item[0].hall_number))
    ):
        dataset, _, symprec = candidate
        try:
            embedding = _embedding(
                parent_dataset=dataset,
                child_dataset=child,
                atomic_numbers=species,
                detection_symprec=symprec,
            )
        except ValueError:
            # Relaxed spglib output is only a proposal. A parent becomes active
            # only after the complete affine/species/checksum validator accepts it.
            continue
        selected.append(candidate)
        embeddings.append(embedding)
        if len(embeddings) == max_parents:
            break
    embeddings = tuple(embeddings)
    dag = ParentDAGSpec(material_id, symmetry.hall_number, embeddings)
    residuals = {symmetry.hall_number: 0.0}
    detection = {symmetry.hall_number: base_symprec}
    for (dataset, residual, symprec), embedding in zip(selected, embeddings):
        residuals[embedding.parent_hall_number] = residual
        detection[embedding.parent_hall_number] = symprec
    return MaterialParentRouting(dag, residuals, detection)


def _embedding_payload(embedding: ParentEmbeddingSpec) -> dict[str, object]:
    return {name: getattr(embedding, name) for name in embedding.__dataclass_fields__}


def routing_to_payload(routing: MaterialParentRouting) -> dict[str, object]:
    return {
        "material_id": routing.dag.material_id,
        "current_hall_number": routing.dag.current_hall_number,
        "embeddings": [_embedding_payload(value) for value in routing.dag.embeddings],
        "residuals": dict(routing.residuals),
        "detection_symprecs": dict(routing.detection_symprecs),
    }


def routing_from_payload(payload: Mapping[str, object]) -> MaterialParentRouting:
    embeddings = tuple(
        ParentEmbeddingSpec(**value) for value in payload["embeddings"]  # type: ignore[arg-type]
    )
    dag = ParentDAGSpec(
        str(payload["material_id"]), int(payload["current_hall_number"]), embeddings
    )
    return MaterialParentRouting(
        dag,
        {int(key): float(value) for key, value in payload["residuals"].items()},  # type: ignore[union-attr]
        {
            int(key): float(value)
            for key, value in payload["detection_symprecs"].items()  # type: ignore[union-attr]
        },
    )


def parent_detection_config_sha256(
    symprecs: Sequence[float] = DEFAULT_PARENT_SYMPRECS,
) -> str:
    payload = {
        "convention": PARENT_DAG_CONVENTION,
        "spglib": spglib.__version__,
        "symprecs": list(symprecs),
        "max_parents": 2,
        "max_operation_index": 4,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def save_parent_routing_cache(
    path: str | Path,
    routings: Sequence[MaterialParentRouting],
    *,
    sample_ids: Sequence[str],
    dataset_sha256: str,
) -> None:
    if [item.dag.material_id for item in routings] != list(sample_ids):
        raise ValueError("parent routing cache must preserve the exact sample order")
    payload = {
        "schema_version": 1,
        "dataset_sha256": dataset_sha256,
        "config_sha256": parent_detection_config_sha256(),
        "sample_ids": list(sample_ids),
        "routings": [routing_to_payload(item) for item in routings],
    }
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_parent_routing_cache(
    path: str | Path, *, sample_ids: Sequence[str], dataset_sha256: str
) -> tuple[MaterialParentRouting, ...]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    expected = {
        "schema_version": 1,
        "dataset_sha256": dataset_sha256,
        "config_sha256": parent_detection_config_sha256(),
        "sample_ids": list(sample_ids),
    }
    for name, value in expected.items():
        if payload.get(name) != value:
            raise ValueError(f"parent routing cache {name} mismatch")
    return tuple(routing_from_payload(value) for value in payload["routings"])


def parent_routing_coverage(
    routings: Sequence[MaterialParentRouting],
) -> dict[str, object]:
    if not routings:
        raise ValueError("parent routing coverage requires at least one sample")
    parent_counts = [len(item.dag.embeddings) for item in routings]
    groups: dict[str, int] = {}
    for item in routings:
        for embedding in item.dag.embeddings:
            group_type = spglib.get_spacegroup_type(embedding.parent_hall_number)
            if group_type is None:
                raise ValueError("parent routing contains an invalid Hall number")
            symbol = canonical_point_group_symbol(group_type.pointgroup_international)
            groups[symbol] = groups.get(symbol, 0) + 1
    return {
        "samples": len(routings),
        "samples_with_parent": sum(count > 0 for count in parent_counts),
        "coverage_percent": 100.0 * sum(count > 0 for count in parent_counts) / len(routings),
        "mean_parent_count": sum(parent_counts) / len(routings),
        "max_parent_count": max(parent_counts, default=0),
        "parent_point_group_counts": dict(sorted(groups.items())),
    }
