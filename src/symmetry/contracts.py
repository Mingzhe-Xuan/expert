from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import numpy as np

import torch


COMMON_CELL_CONVENTION = "material-fractional-cell-canonical-cartesian-frame-v1"


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
    fractional_rotations: torch.Tensor | None = None
    common_cell_convention: str = COMMON_CELL_CONVENTION

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
        if self.fractional_rotations is not None:
            if self.fractional_rotations.shape != self.rotations.shape:
                raise ValueError("fractional_rotations must match rotations shape")
            rounded = self.fractional_rotations.round()
            if not torch.allclose(self.fractional_rotations, rounded):
                raise ValueError("fractional_rotations must contain integer matrices")
        if not self.common_cell_convention:
            raise ValueError("common_cell_convention must be explicit")


@dataclass(frozen=True, slots=True)
class ParentEmbeddingSpec:
    """One offline point-group cover edge with every concrete child orientation."""

    parent_point_group_number: int
    child_point_group_number: int
    parent_rotations: tuple[tuple[tuple[float, float, float], ...], ...]
    child_rotation_variants: tuple[
        tuple[tuple[tuple[float, float, float], ...], ...], ...
    ]
    edge_id: str
    asset_sha256: str
    convention_id: str
    version: int
    checksum: str

    def __post_init__(self) -> None:
        if not 1 <= self.parent_point_group_number <= 32:
            raise ValueError("parent point-group number must be in [1, 32]")
        if not 1 <= self.child_point_group_number <= 32:
            raise ValueError("child point-group number must be in [1, 32]")
        if self.parent_point_group_number == self.child_point_group_number:
            raise ValueError("a parent edge must connect distinct point groups")
        if not self.parent_rotations or not self.child_rotation_variants:
            raise ValueError("parent rotations and child orientation variants are required")
        if (
            not self.edge_id
            or "." in self.edge_id
            or len(self.asset_sha256) != 64
            or any(character not in "0123456789abcdef" for character in self.asset_sha256)
            or not self.convention_id
            or self.version < 1
        ):
            raise ValueError("edge, asset, convention and positive version are required")
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
    current_point_group_number: int
    embeddings: tuple[ParentEmbeddingSpec, ...]

    def __post_init__(self) -> None:
        if not self.material_id or not 1 <= self.current_point_group_number <= 32:
            raise ValueError("material_id and current point-group number are required")
        nodes = {self.current_point_group_number}
        identities: set[tuple[int, int]] = set()
        edge_ids: set[str] = set()
        adjacency: dict[int, set[int]] = {}
        for embedding in self.embeddings:
            validate_parent_embedding(embedding)
            identity = (
                embedding.parent_point_group_number,
                embedding.child_point_group_number,
            )
            if identity in identities:
                raise ValueError(f"duplicate parent embedding {identity}")
            identities.add(identity)
            if embedding.edge_id in edge_ids:
                raise ValueError(f"duplicate offline edge ID {embedding.edge_id}")
            edge_ids.add(embedding.edge_id)
            parent_node = embedding.parent_point_group_number
            child_node = embedding.child_point_group_number
            nodes.update((parent_node, child_node))
            adjacency.setdefault(parent_node, set()).add(child_node)
        if self.embeddings and self.current_point_group_number not in {
            embedding.child_point_group_number
            for embedding in self.embeddings
        }:
            raise ValueError("the DAG must contain an edge terminating at the current point group")
        if adjacency.get(self.current_point_group_number):
            raise ValueError("current point group must be a terminal DAG node")

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
        connected = {self.current_point_group_number}
        frontier = [self.current_point_group_number]
        while frontier:
            child = frontier.pop()
            for parent in reverse.get(child, ()):
                if parent not in connected:
                    connected.add(parent)
                    frontier.append(parent)
        if connected != nodes:
            raise ValueError("every parent node must lead to the current point group")

    def current_to_root_paths(self) -> tuple[tuple[int, ...], ...]:
        """Return deterministic maximal point-group paths from current to every root."""

        parents_by_child: dict[int, set[int]] = {}
        for embedding in self.embeddings:
            parents_by_child.setdefault(embedding.child_point_group_number, set()).add(
                embedding.parent_point_group_number
            )

        paths: list[tuple[int, ...]] = []

        def extend(node: int, prefix: tuple[int, ...]) -> None:
            parents = tuple(sorted(parents_by_child.get(node, ())))
            if not parents:
                paths.append(prefix)
                return
            for parent in parents:
                extend(parent, (*prefix, parent))

        extend(self.current_point_group_number, (self.current_point_group_number,))
        return tuple(paths)

    def current_to_root_embedding_paths(
        self,
    ) -> tuple[tuple[ParentEmbeddingSpec, ...], ...]:
        """Return every class-cover path, ordered current edge to root edge."""

        if not self.embeddings:
            return ((),)
        parents_by_child: dict[int, list[ParentEmbeddingSpec]] = {}
        for embedding in self.embeddings:
            child_node = embedding.child_point_group_number
            parents_by_child.setdefault(child_node, []).append(embedding)
        for values in parents_by_child.values():
            values.sort(key=lambda value: value.checksum)
        paths: list[tuple[ParentEmbeddingSpec, ...]] = []

        def extend(
            node: int, prefix: tuple[ParentEmbeddingSpec, ...]
        ) -> None:
            parents = parents_by_child.get(node, ())
            if not parents:
                paths.append(prefix)
                return
            for embedding in parents:
                parent_node = embedding.parent_point_group_number
                extend(parent_node, (*prefix, embedding))

        extend(self.current_point_group_number, ())
        return tuple(paths)


def _rotation_key(rotation: np.ndarray, tolerance: float = 1.0e-7) -> tuple[int, ...]:
    return tuple(np.rint(rotation.reshape(-1) / tolerance).astype(np.int64).tolist())


def validate_parent_embedding(
    embedding: ParentEmbeddingSpec, *, validate_checksum: bool = True
) -> None:
    """Validate point-group rotation subsets and checksum invariants."""

    if validate_checksum:
        embedding.validate_checksum()
    def validated_group(values, label):
        parsed = []
        for rotation_value in values:
            rotation = np.asarray(rotation_value, dtype=np.float64)
            if rotation.shape != (3, 3) or not np.isfinite(rotation).all():
                raise ValueError("point-group rotations must be finite 3x3 matrices")
            if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1.0e-6, rtol=0.0):
                raise ValueError("point-group rotations must be orthogonal")
            parsed.append(rotation)
        keys = {_rotation_key(rotation) for rotation in parsed}
        if len(keys) != len(parsed):
            raise ValueError(f"{label} rotations must be unique")
        identity = _rotation_key(np.eye(3))
        if identity not in keys:
            raise ValueError(f"{label} rotation group must contain identity")
        for left_rotation in parsed:
            if _rotation_key(left_rotation.T) not in keys:
                raise ValueError(f"{label} rotation group is not inverse closed")
            for right_rotation in parsed:
                if _rotation_key(left_rotation @ right_rotation) not in keys:
                    raise ValueError(f"{label} rotation group is not multiplication closed")
        return keys

    parent_keys = validated_group(embedding.parent_rotations, "parent")
    variant_keys = []
    for index, variant in enumerate(embedding.child_rotation_variants):
        child_keys = validated_group(variant, f"child variant {index}")
        if not child_keys < parent_keys:
            raise ValueError("every child rotation variant must be a strict parent subset")
        variant_keys.append(frozenset(child_keys))
    if len(set(variant_keys)) != len(variant_keys):
        raise ValueError("child orientation variants must be unique")
