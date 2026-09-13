from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from itertools import product

import numpy as np
import spglib

from .records import AuditResult, DIELECTRIC_SUBTYPES, NormalizedRecord


@dataclass(frozen=True, slots=True)
class AuditThresholds:
    symprec: float = 1.0e-5
    angle_tolerance: float = -1.0
    structure_min_volume: float = 1.0e-6
    intrinsic_relative_tolerance: float = 1.0e-3
    point_group_relative_tolerance: float = 5.0e-3
    dielectric_eigen_tolerance: float = 1.0e-3
    elastic_eigen_relative_tolerance: float = 1.0e-8
    elastic_eigen_absolute_tolerance_gpa: float = 1.0e-6
    elastic_component_limit_gpa: float = 1500.0
    outlier_modified_z: float = 8.0


def _relative_residual(value: np.ndarray, reference: np.ndarray) -> float:
    return float(np.linalg.norm(value - reference) / max(np.linalg.norm(reference), 1.0))


def _intrinsic_projection(record: NormalizedRecord) -> np.ndarray:
    value = np.asarray(record.tensor, dtype=np.float64)
    if record.subtype in DIELECTRIC_SUBTYPES:
        return 0.5 * (value + value.T)
    # Average the elasticity tensor's minor and major symmetry orbit.
    variants = (
        value,
        value.swapaxes(0, 1),
        value.swapaxes(2, 3),
        value.swapaxes(0, 1).swapaxes(2, 3),
        value.transpose(2, 3, 0, 1),
        value.transpose(3, 2, 0, 1),
        value.transpose(2, 3, 1, 0),
        value.transpose(3, 2, 1, 0),
    )
    return np.mean(variants, axis=0)


def _symmetry_dataset(record: NormalizedRecord, thresholds: AuditThresholds) -> tuple[object, np.ndarray]:
    lattice = np.asarray(record.lattice, dtype=np.float64)
    fractional = np.asarray(record.fractional_positions, dtype=np.float64) % 1.0
    numbers = np.asarray(record.atomic_numbers, dtype=np.int32)
    dataset = spglib.get_symmetry_dataset(
        (lattice, fractional, numbers),
        symprec=thresholds.symprec,
        angle_tolerance=thresholds.angle_tolerance,
    )
    if dataset is None:
        raise ValueError("spglib could not classify structure")
    inverse_transpose = np.linalg.inv(lattice.T)
    rotations = []
    for fractional_rotation in np.asarray(dataset.rotations, dtype=np.float64):
        value = lattice.T @ fractional_rotation @ inverse_transpose
        left, _, right = np.linalg.svd(value)
        rotations.append(left @ right)
    return dataset, np.stack(rotations)


def _point_group_projection(tensor: np.ndarray, rotations: np.ndarray) -> np.ndarray:
    if tensor.ndim == 2:
        return np.mean([rotation @ tensor @ rotation.T for rotation in rotations], axis=0)
    return np.mean(
        [
            np.einsum("ia,jb,kc,ld,abcd->ijkl", r, r, r, r, tensor, optimize=True)
            for r in rotations
        ],
        axis=0,
    )


def _canonical_rotation(dataset: object) -> np.ndarray:
    value = np.asarray(dataset.std_rotation_matrix, dtype=np.float64)
    left, _, right = np.linalg.svd(value)
    rotation = left @ right
    if np.linalg.det(rotation) < 0:
        left[:, -1] *= -1
        rotation = left @ right
    return rotation


def _rotate_tensor(tensor: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    if tensor.ndim == 2:
        return rotation @ tensor @ rotation.T
    return np.einsum(
        "ia,jb,kc,ld,abcd->ijkl", rotation, rotation, rotation, rotation, tensor, optimize=True
    )


def _primitive_fingerprint(record: NormalizedRecord, thresholds: AuditThresholds) -> str:
    primitive = spglib.standardize_cell(
        (
            np.asarray(record.lattice, dtype=np.float64),
            np.asarray(record.fractional_positions, dtype=np.float64) % 1.0,
            np.asarray(record.atomic_numbers, dtype=np.int32),
        ),
        to_primitive=True,
        no_idealize=False,
        symprec=thresholds.symprec,
        angle_tolerance=thresholds.angle_tolerance,
    )
    if primitive is None:
        raise ValueError("spglib could not standardize structure")
    lattice, fractional, numbers = primitive
    lattice = np.asarray(lattice, dtype=np.float64)
    fractional = np.asarray(fractional, dtype=np.float64) % 1.0
    numbers = np.asarray(numbers, dtype=np.int32)
    pg = spglib.get_symmetry_dataset(
        (lattice, fractional, numbers),
        symprec=thresholds.symprec,
        angle_tolerance=thresholds.angle_tolerance,
    )
    if pg is None:
        raise ValueError("spglib could not classify standardized structure")
    distances: list[tuple[int, int, float]] = []
    shifts = np.asarray(tuple(product((-1, 0, 1), repeat=3)), dtype=np.float64)
    for left_index in range(len(numbers)):
        for right_index in range(left_index + 1, len(numbers)):
            delta = fractional[left_index] - fractional[right_index] + shifts
            distance = float(np.min(np.linalg.norm(delta @ lattice, axis=1)))
            pair = sorted((int(numbers[left_index]), int(numbers[right_index])))
            distances.append((pair[0], pair[1], round(distance, 3)))
    payload = {
        "space_group": int(pg.number),
        "composition": sorted(
            (int(number), int(np.sum(numbers == number))) for number in np.unique(numbers)
        ),
        "volume_per_atom": round(abs(float(np.linalg.det(lattice))) / len(numbers), 3),
        "standardized_metric": np.round(lattice @ lattice.T, 3).tolist(),
        "distances": sorted(distances),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _voigt_from_elastic(tensor: np.ndarray) -> np.ndarray:
    pairs = ((0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1))
    return np.asarray([[tensor[i, j, k, ell] for k, ell in pairs] for i, j in pairs])


def audit_record(
    record: NormalizedRecord, thresholds: AuditThresholds = AuditThresholds()
) -> AuditResult:
    reasons: list[str] = []
    warnings: list[str] = []
    metrics: dict[str, float | int | str] = {}
    arrays = (record.lattice, record.fractional_positions, record.atomic_numbers, record.tensor)
    if any(not np.all(np.isfinite(value)) for value in arrays):
        reasons.append("nonfinite_value")
        return AuditResult(record, False, reasons, warnings, metrics, None, None, None, None, None)
    lattice = np.asarray(record.lattice, dtype=np.float64)
    fractional = np.asarray(record.fractional_positions, dtype=np.float64)
    numbers = np.asarray(record.atomic_numbers)
    if lattice.shape != (3, 3) or fractional.ndim != 2 or fractional.shape[1:] != (3,):
        reasons.append("invalid_structure_shape")
    if len(fractional) < 1 or numbers.shape != (len(fractional),):
        reasons.append("invalid_structure_sites")
    if numbers.size and (np.min(numbers) < 1 or np.max(numbers) > 118):
        reasons.append("invalid_atomic_number")
    volume = abs(float(np.linalg.det(lattice))) if lattice.shape == (3, 3) else 0.0
    metrics["volume_angstrom3"] = volume
    metrics["atoms"] = int(len(fractional)) if fractional.ndim else 0
    if volume <= thresholds.structure_min_volume:
        reasons.append("singular_or_tiny_cell")
    if reasons:
        return AuditResult(record, False, reasons, warnings, metrics, None, None, None, None, None)
    try:
        dataset, rotations = _symmetry_dataset(record, thresholds)
        fingerprint = _primitive_fingerprint(record, thresholds)
    except (ValueError, np.linalg.LinAlgError):
        reasons.append("symmetry_detection_failed")
        return AuditResult(record, False, reasons, warnings, metrics, None, None, None, None, None)
    clean_intrinsic = _intrinsic_projection(record)
    intrinsic_residual = _relative_residual(np.asarray(record.tensor), clean_intrinsic)
    clean = _point_group_projection(clean_intrinsic, rotations)
    pg_residual = _relative_residual(clean_intrinsic, clean)
    metrics.update(
        {
            "point_group": str(dataset.pointgroup).replace(" ", ""),
            "space_group": int(dataset.number),
            "intrinsic_relative_residual": intrinsic_residual,
            "point_group_relative_residual": pg_residual,
            "tensor_frobenius": float(np.linalg.norm(clean_intrinsic)),
            "tensor_max_abs": float(np.max(np.abs(clean_intrinsic))),
        }
    )
    if intrinsic_residual > thresholds.intrinsic_relative_tolerance:
        reasons.append("intrinsic_symmetry_residual")
    elif intrinsic_residual > 1.0e-10:
        warnings.append("intrinsic_symmetry_projected")
    if pg_residual > thresholds.point_group_relative_tolerance:
        reasons.append("point_group_residual")
    elif pg_residual > 1.0e-10:
        warnings.append("point_group_projected")
    if record.subtype in DIELECTRIC_SUBTYPES:
        eigenvalues = np.linalg.eigvalsh(clean)
        minimum = float(eigenvalues.min())
        metrics.update(
            {
                "eigenvalue_min": minimum,
                "eigenvalue_max": float(eigenvalues.max()),
                "trace_mean": float(np.trace(clean) / 3.0),
            }
        )
        lower_bound = 1.0 if record.subtype != "dielectric_ionic" else 0.0
        if minimum < lower_bound - thresholds.dielectric_eigen_tolerance:
            reasons.append("dielectric_not_positive_semidefinite")
    else:
        voigt = _voigt_from_elastic(clean)
        kelvin_scale = np.diag([1.0, 1.0, 1.0, np.sqrt(2.0), np.sqrt(2.0), np.sqrt(2.0)])
        kelvin = kelvin_scale @ voigt @ kelvin_scale
        eigenvalues = np.linalg.eigvalsh(kelvin)
        minimum = float(eigenvalues.min())
        maximum = float(eigenvalues.max())
        metrics.update(
            {
                "kelvin_eigenvalue_min_gpa": minimum,
                "kelvin_eigenvalue_max_gpa": maximum,
                "kelvin_condition": float(maximum / minimum) if minimum > 0 else float("inf"),
            }
        )
        tolerance = max(
            thresholds.elastic_eigen_absolute_tolerance_gpa,
            thresholds.elastic_eigen_relative_tolerance * max(maximum, 1.0),
        )
        if minimum <= tolerance:
            reasons.append("elastic_not_positive_definite")
        if float(np.max(np.abs(clean_intrinsic))) > thresholds.elastic_component_limit_gpa:
            reasons.append("elastic_component_limit")
    canonical = _rotate_tensor(clean, _canonical_rotation(dataset))
    valid = not reasons
    return AuditResult(
        record=record,
        physical_valid=valid,
        reasons=reasons,
        warnings=warnings,
        metrics=metrics,
        clean_tensor=clean if valid else None,
        canonical_tensor=canonical if valid else None,
        structure_fingerprint=fingerprint,
        point_group=str(dataset.pointgroup).replace(" ", ""),
        space_group=int(dataset.number),
    )
