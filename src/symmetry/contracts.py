from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

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
    atom_correspondence: tuple[int, ...]
    wyckoff_splitting: tuple[str, ...]
    domain_variant: str
    convention_id: str
    version: int
    checksum: str

    def __post_init__(self) -> None:
        if self.parent_hall_number < 1 or self.child_hall_number < 1:
            raise ValueError("Hall numbers must be positive")
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
        if not self.atom_correspondence or min(self.atom_correspondence) < 0:
            raise ValueError("atom_correspondence must contain non-negative indices")
        if len(set(self.atom_correspondence)) != len(self.atom_correspondence):
            raise ValueError("atom_correspondence must be one-to-one")
        if not self.domain_variant or not self.convention_id or self.version < 1:
            raise ValueError("domain, convention and positive version are required")
        if len(self.checksum) != 64:
            raise ValueError("checksum must be a SHA-256 digest")

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
        children: set[int] = set()
        edges: set[tuple[int, int]] = set()
        for embedding in self.embeddings:
            embedding.validate_checksum()
            edge = (embedding.parent_hall_number, embedding.child_hall_number)
            if edge in edges:
                raise ValueError(f"duplicate parent embedding edge {edge}")
            edges.add(edge)
            children.add(embedding.child_hall_number)
        if self.embeddings and self.current_hall_number not in children:
            raise ValueError("the DAG must contain an embedding terminating at the current Hall setting")
