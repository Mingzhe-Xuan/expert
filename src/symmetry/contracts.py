from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math

import numpy as np

import torch


def _validate_rotation_matrix(matrix: torch.Tensor, name: str) -> None:
    if matrix.shape != (3, 3):
        raise ValueError(f"{name} must have shape [3, 3]")
    identity = torch.eye(3, dtype=matrix.dtype, device=matrix.device)
    if not torch.allclose(matrix.T @ matrix, identity, rtol=1e-5, atol=1e-6):
        raise ValueError(f"{name} must be orthogonal")
    determinant = torch.linalg.det(matrix).abs()
    if not torch.allclose(determinant, determinant.new_ones(()), rtol=1e-5, atol=1e-6):
        raise ValueError(f"{name} determinant must have magnitude one")


@dataclass(frozen=True, slots=True)
class SymmetryRecord:
    canonical_frame: torch.Tensor
    current_point_group: str
    current_space_group: int
    hall_number: int
    rotations: torch.Tensor
    translations: torch.Tensor
    audit_permutations: torch.Tensor | None = None

    def __post_init__(self) -> None:
        _validate_rotation_matrix(self.canonical_frame, "canonical_frame")
        if not self.current_point_group:
            raise ValueError("current_point_group must be non-empty")
        if not 1 <= self.current_space_group <= 230:
            raise ValueError("current_space_group must be in [1, 230]")
        if self.hall_number < 1:
            raise ValueError("hall_number must be positive")
        if self.rotations.ndim != 3 or self.rotations.shape[1:] != (3, 3):
            raise ValueError("rotations must have shape [num_operations, 3, 3]")
        operation_count = self.rotations.shape[0]
        if self.translations.shape != (operation_count, 3):
            raise ValueError("translations must have shape [num_operations, 3]")
        if self.audit_permutations is not None:
            if self.audit_permutations.ndim != 2:
                raise ValueError("audit_permutations must have shape [num_operations, num_nodes]")
            if self.audit_permutations.shape[0] != operation_count:
                raise ValueError("audit_permutations operation count mismatch")
            if self.audit_permutations.dtype != torch.long:
                raise TypeError("audit_permutations must use torch.long")


@dataclass(frozen=True, slots=True)
class ParentEmbeddingSpec:
    """One versioned, material-specific Hall-level parent embedding."""

    parent_hall_number: int
    child_hall_number: int
    parent_setting: str
    child_setting: str
    basis_transform: tuple[tuple[float, float, float], ...]
    origin_shift: tuple[float, float, float]
    supercell_transform: tuple[tuple[int, int, int], ...]
    operations: tuple[
        tuple[tuple[tuple[int, int, int], ...], tuple[float, float, float]], ...
    ]
    parent_atomic_numbers: tuple[int, ...]
    child_atomic_numbers: tuple[int, ...]
    atom_correspondence: tuple[int, ...]
    wyckoff_splitting: tuple[str, ...]
    domain_variant: str
    convention_id: str
    version: int
    checksum: str

    def __post_init__(self) -> None:
        if self.parent_hall_number < 1 or self.child_hall_number < 1:
            raise ValueError("Hall numbers must be positive")
        if self.parent_hall_number == self.child_hall_number:
            raise ValueError("a parent embedding must connect distinct Hall settings")
        if not self.parent_setting or not self.child_setting:
            raise ValueError("parent and child settings must be explicit")
        if len(self.basis_transform) != 3 or any(len(row) != 3 for row in self.basis_transform):
            raise ValueError("basis_transform must be 3x3")
        if len(self.supercell_transform) != 3 or any(
            len(row) != 3 for row in self.supercell_transform
        ):
            raise ValueError("supercell_transform must be 3x3")
        if not self.operations:
            raise ValueError("the full parent operation list is required")
        for rotation, translation in self.operations:
            if len(rotation) != 3 or any(len(row) != 3 for row in rotation):
                raise ValueError("every operation rotation must be 3x3")
            if len(translation) != 3:
                raise ValueError("every operation translation must have length 3")
        if not self.parent_atomic_numbers or not self.child_atomic_numbers:
            raise ValueError("parent and child species lists are required")
        if any(number < 1 for number in (*self.parent_atomic_numbers, *self.child_atomic_numbers)):
            raise ValueError("atomic numbers must be positive")
        if len(self.atom_correspondence) != len(self.child_atomic_numbers):
            raise ValueError("atom_correspondence must map every child site")
        if not self.atom_correspondence or min(self.atom_correspondence) < 0 or max(
            self.atom_correspondence
        ) >= len(self.parent_atomic_numbers):
            raise ValueError("atom_correspondence contains an invalid parent index")
        for child_index, parent_index in enumerate(self.atom_correspondence):
            if self.child_atomic_numbers[child_index] != self.parent_atomic_numbers[parent_index]:
                raise ValueError("atom correspondence must preserve species")
        if not self.wyckoff_splitting or any(not value for value in self.wyckoff_splitting):
            raise ValueError("Wyckoff splitting metadata must be non-empty")
        if not self.domain_variant or not self.convention_id or self.version < 1:
            raise ValueError("domain, convention and positive version are required")
        if len(self.checksum) != 64 or any(
            character not in "0123456789abcdef" for character in self.checksum
        ):
            raise ValueError("checksum must be a lowercase SHA-256 digest")
        validate_parent_embedding(self, validate_checksum=False)

    def payload_checksum(self) -> str:
        payload = {
            field: getattr(self, field)
            for field in self.__dataclass_fields__
            if field != "checksum"
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    def validate_checksum(self) -> None:
        if self.checksum != self.payload_checksum():
            raise ValueError("ParentEmbeddingSpec checksum mismatch")


@dataclass(frozen=True, slots=True)
class ParentDAGSpec:
    material_id: str
    current_hall_number: int
    embeddings: tuple[ParentEmbeddingSpec, ...]

    def __post_init__(self) -> None:
        if not self.material_id or self.current_hall_number < 1:
            raise ValueError("material_id and positive current_hall_number are required")
        nodes = {self.current_hall_number}
        identities: set[tuple[int, int, str, str, int]] = set()
        adjacency: dict[int, set[int]] = {}
        for embedding in self.embeddings:
            validate_parent_embedding(embedding)
            identity = (
                embedding.parent_hall_number,
                embedding.child_hall_number,
                embedding.domain_variant,
                embedding.convention_id,
                embedding.version,
            )
            if identity in identities:
                raise ValueError(f"duplicate parent embedding {identity}")
            identities.add(identity)
            nodes.update((embedding.parent_hall_number, embedding.child_hall_number))
            adjacency.setdefault(embedding.parent_hall_number, set()).add(
                embedding.child_hall_number
            )
        if self.embeddings and self.current_hall_number not in {
            embedding.child_hall_number for embedding in self.embeddings
        }:
            raise ValueError("the DAG must contain an embedding terminating at the current Hall setting")
        if adjacency.get(self.current_hall_number):
            raise ValueError("current Hall setting must be a terminal DAG node")

        visiting: set[int] = set()
        visited: set[int] = set()

        def visit(node: int) -> None:
            if node in visiting:
                raise ValueError("parent embedding graph contains a cycle")
            if node in visited:
                return
            visiting.add(node)
            for child in adjacency.get(node, ()):
                visit(child)
            visiting.remove(node)
            visited.add(node)

        for node in nodes:
            visit(node)
        reverse: dict[int, set[int]] = {}
        for parent, children in adjacency.items():
            for child in children:
                reverse.setdefault(child, set()).add(parent)
        connected = {self.current_hall_number}
        frontier = [self.current_hall_number]
        while frontier:
            child = frontier.pop()
            for parent in reverse.get(child, ()):
                if parent not in connected:
                    connected.add(parent)
                    frontier.append(parent)
        if connected != nodes:
            raise ValueError("every parent embedding node must lead to the current Hall setting")


def _affine_key(
    rotation: np.ndarray, translation: np.ndarray, tolerance: float = 1.0e-8
) -> tuple[int, ...]:
    normalized = translation - np.floor(translation)
    normalized[np.isclose(normalized, 1.0, atol=tolerance)] = 0.0
    quantized = np.rint(normalized / tolerance).astype(np.int64)
    return (*[int(value) for value in rotation.reshape(-1)], *quantized.tolist())


def validate_parent_embedding(
    embedding: ParentEmbeddingSpec, *, validate_checksum: bool = True
) -> None:
    """Validate affine-group, transform, species, and checksum invariants."""

    if validate_checksum:
        embedding.validate_checksum()
    basis = np.asarray(embedding.basis_transform, dtype=np.float64)
    supercell = np.asarray(embedding.supercell_transform, dtype=np.int64)
    if not np.isfinite(basis).all() or abs(float(np.linalg.det(basis))) < 1.0e-12:
        raise ValueError("basis_transform must be finite and invertible")
    if round(abs(float(np.linalg.det(supercell)))) < 1:
        raise ValueError("supercell_transform must be invertible")
    if not all(math.isfinite(value) for value in embedding.origin_shift):
        raise ValueError("origin_shift must be finite")

    parsed = []
    for rotation_value, translation_value in embedding.operations:
        rotation = np.asarray(rotation_value, dtype=np.int64)
        translation = np.asarray(translation_value, dtype=np.float64)
        determinant = round(float(np.linalg.det(rotation)))
        if abs(determinant) != 1 or not np.isfinite(translation).all():
            raise ValueError("affine operations require unimodular rotations and finite translations")
        parsed.append((rotation, translation))
    keys = {_affine_key(rotation, translation) for rotation, translation in parsed}
    if len(keys) != len(parsed):
        raise ValueError("affine operations must be unique modulo lattice translations")
    identity = _affine_key(np.eye(3, dtype=np.int64), np.zeros(3))
    if identity not in keys:
        raise ValueError("affine operation group must contain identity")
    for left_rotation, left_translation in parsed:
        inverse_rotation = np.rint(np.linalg.inv(left_rotation)).astype(np.int64)
        inverse_translation = -(inverse_rotation @ left_translation)
        if _affine_key(inverse_rotation, inverse_translation) not in keys:
            raise ValueError("affine operation group is not inverse closed")
        for right_rotation, right_translation in parsed:
            rotation = left_rotation @ right_rotation
            translation = left_rotation @ right_translation + left_translation
            if _affine_key(rotation, translation) not in keys:
                raise ValueError("affine operation group is not multiplication closed")
