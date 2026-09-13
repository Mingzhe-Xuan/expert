from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import TYPE_CHECKING

import torch

from .contracts import IrrepLayout

if TYPE_CHECKING:
    from ..symmetry.registry import PointGroup


SUBDUCTION_TOLERANCE = 5.0e-7
CG_TOLERANCE = 2.0e-8


def _deterministic_range_basis(
    projector: torch.Tensor, tolerance: float
) -> torch.Tensor:
    vectors: list[torch.Tensor] = []
    for column in projector.unbind(dim=1):
        residual = column.clone()
        for vector in vectors:
            residual -= torch.dot(vector, residual) * vector
        norm = torch.linalg.vector_norm(residual)
        if float(norm) <= tolerance:
            continue
        vector = residual / norm
        pivot = int(vector.abs().argmax())
        if float(vector[pivot]) < 0:
            vector = -vector
        vectors.append(vector)
    if not vectors:
        return projector.new_empty((projector.shape[0], 0))
    return torch.stack(vectors, dim=1)


def _symmetric_seeds(dimension: int, reference: torch.Tensor):
    for row in range(dimension):
        seed = reference.new_zeros((dimension, dimension))
        seed[row, row] = 1
        yield seed
    for row in range(dimension):
        for column in range(row + 1, dimension):
            seed = reference.new_zeros((dimension, dimension))
            seed[row, column] = seed[column, row] = 1
            yield seed


def _reynolds_commutant(
    representation: torch.Tensor, seed: torch.Tensor
) -> torch.Tensor:
    averaged = torch.einsum(
        "gij,jk,glk->il", representation, seed, representation
    ) / representation.shape[0]
    return 0.5 * (averaged + averaged.T)


def _eigenvalue_clusters(values: torch.Tensor, tolerance: float) -> list[slice]:
    starts = [0]
    for index in range(1, len(values)):
        scale = max(1.0, abs(float(values[index])), abs(float(values[index - 1])))
        if abs(float(values[index] - values[index - 1])) > tolerance * scale:
            starts.append(index)
    starts.append(len(values))
    return [slice(left, right) for left, right in zip(starts[:-1], starts[1:])]


def _symmetric_commutant_dimension(
    representation: torch.Tensor, tolerance: float
) -> int:
    columns = [
        _reynolds_commutant(representation, seed).reshape(-1)
        for seed in _symmetric_seeds(representation.shape[1], representation)
    ]
    matrix = torch.stack(columns, dim=1)
    return int(torch.linalg.matrix_rank(matrix, atol=tolerance, rtol=tolerance))


def _real_irreducible_subspaces(
    representation: torch.Tensor, tolerance: float
) -> tuple[torch.Tensor, ...]:
    """Split an orthogonal real representation using its symmetric commutant."""

    dimension = representation.shape[1]
    blocks = [torch.eye(dimension, dtype=representation.dtype, device=representation.device)]
    for seed in _symmetric_seeds(dimension, representation):
        commutant = _reynolds_commutant(representation, seed)
        refined = []
        changed = False
        for block in blocks:
            restricted = 0.5 * (block.T @ commutant @ block + block.T @ commutant.T @ block)
            values, vectors = torch.linalg.eigh(restricted)
            clusters = _eigenvalue_clusters(values, tolerance)
            if len(clusters) == 1:
                refined.append(block)
                continue
            changed = True
            for cluster in clusters:
                raw = block @ vectors[:, cluster]
                projector = raw @ raw.T
                refined.append(_deterministic_range_basis(projector, tolerance))
        blocks = refined
        if not changed and all(
            _symmetric_commutant_dimension(
                torch.einsum("ai,gab,bj->gij", block, representation, block), tolerance
            )
            == 1
            for block in blocks
        ):
            break

    for block in blocks:
        restricted = torch.einsum("ai,gab,bj->gij", block, representation, block)
        if _symmetric_commutant_dimension(restricted, tolerance) != 1:
            raise RuntimeError("failed to resolve a real finite-group irreducible subspace")
        leakage = representation @ block - torch.einsum("ai,gij->gaj", block, restricted)
        if float(leakage.abs().max()) > 10 * tolerance:
            raise RuntimeError("computed finite-group subspace is not invariant")

    def ordering_key(block: torch.Tensor):
        restricted = torch.einsum("ai,gab,bj->gij", block, representation, block)
        character = tuple(torch.round(restricted.diagonal(dim1=-2, dim2=-1).sum(-1) * 1e8).tolist())
        projector = block @ block.T
        pivot_signature = tuple(torch.round(projector.flatten() * 1e8).tolist())
        return (block.shape[1], character, pivot_signature)

    return tuple(sorted(blocks, key=ordering_key))


@dataclass(frozen=True, slots=True)
class FiniteIrrepCopy:
    source_label: str
    source_copy_index: int
    source_degree: int
    pg_copy_index: int
    start: int
    stop: int
    character_signature: tuple[float, ...]

    @property
    def dimension(self) -> int:
        return self.stop - self.start


@dataclass(frozen=True, slots=True)
class SubductionPlan:
    point_group: str
    source_layout: IrrepLayout
    matrix: torch.Tensor
    copies: tuple[FiniteIrrepCopy, ...]
    representation: torch.Tensor

    def __post_init__(self) -> None:
        dimension = self.source_layout.dimension
        if self.matrix.shape != (dimension, dimension):
            raise ValueError("subduction matrix must be square over the source layout")
        if self.representation.shape[1:] != (dimension, dimension):
            raise ValueError("PG representation shape does not match source layout")
        identity = torch.eye(dimension, dtype=self.matrix.dtype, device=self.matrix.device)
        if not torch.allclose(self.matrix.T @ self.matrix, identity, atol=SUBDUCTION_TOLERANCE, rtol=SUBDUCTION_TOLERANCE):
            raise ValueError("subduction matrix must be orthogonal")
        if not self.copies or self.copies[0].start != 0 or self.copies[-1].stop != dimension:
            raise ValueError("finite irrep copies must cover the full PG layout")
        if any(left.stop != right.start for left, right in zip(self.copies[:-1], self.copies[1:])):
            raise ValueError("finite irrep copy slices must be contiguous")

    def subduce(self, features: torch.Tensor) -> torch.Tensor:
        if features.shape[-1] != self.source_layout.dimension:
            raise ValueError("feature width does not match source layout")
        return features @ self.matrix

    def inverse(self, pg_features: torch.Tensor) -> torch.Tensor:
        if pg_features.shape[-1] != self.source_layout.dimension:
            raise ValueError("PG feature width does not match plan")
        return pg_features @ self.matrix.T

    @property
    def checksum(self) -> str:
        payload = {
            "schema_version": 1,
            "point_group": self.point_group,
            "source_layout": self.source_layout.to_spec(),
            "copies": [
                {
                    "source_label": copy.source_label,
                    "source_copy_index": copy.source_copy_index,
                    "source_degree": copy.source_degree,
                    "pg_copy_index": copy.pg_copy_index,
                    "start": copy.start,
                    "stop": copy.stop,
                    "character_signature": copy.character_signature,
                }
                for copy in self.copies
            ],
            "matrix": torch.round(self.matrix.detach().cpu() * 1e12).to(torch.int64).tolist(),
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def build_subduction_plan(
    group: "PointGroup",
    layout: IrrepLayout,
    *,
    dtype: torch.dtype = torch.float64,
    tolerance: float = SUBDUCTION_TOLERANCE,
) -> SubductionPlan:
    o3_representation = group.representation(layout, dtype=dtype)
    columns = []
    copies = []
    source_offset = 0
    pg_offset = 0
    for term in layout.terms:
        source_dimension = 2 * term.degree + 1
        for source_copy in range(term.multiplicity):
            source_slice = slice(source_offset, source_offset + source_dimension)
            restricted = o3_representation[:, source_slice, source_slice]
            subspaces = _real_irreducible_subspaces(restricted, tolerance)
            for pg_copy, local_basis in enumerate(subspaces):
                global_basis = o3_representation.new_zeros((layout.dimension, local_basis.shape[1]))
                global_basis[source_slice] = local_basis
                columns.append(global_basis)
                local_representation = torch.einsum(
                    "ai,gab,bj->gij", local_basis, restricted, local_basis
                )
                character = tuple(
                    round(float(value), 10)
                    for value in local_representation.diagonal(dim1=-2, dim2=-1).sum(-1)
                )
                copies.append(
                    FiniteIrrepCopy(
                        source_label=term.copy_label,
                        source_copy_index=source_copy,
                        source_degree=term.degree,
                        pg_copy_index=pg_copy,
                        start=pg_offset,
                        stop=pg_offset + local_basis.shape[1],
                        character_signature=character,
                    )
                )
                pg_offset += local_basis.shape[1]
            source_offset += source_dimension
    matrix = torch.cat(columns, dim=1)
    pg_representation = torch.einsum(
        "ai,gab,bj->gij", matrix, o3_representation, matrix
    )
    return SubductionPlan(group.symbol, layout, matrix, tuple(copies), pg_representation)


@dataclass(frozen=True, slots=True)
class CGPathBasis:
    matrices: torch.Tensor

    @property
    def path_count(self) -> int:
        return self.matrices.shape[0]

    @property
    def checksum(self) -> str:
        values = torch.round(self.matrices.detach().cpu() * 1e12).to(torch.int64).tolist()
        encoded = json.dumps(
            {"schema_version": 1, "path_order": "deterministic-projector-columns", "values": values},
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def finite_group_intertwiners(
    left: torch.Tensor,
    right: torch.Tensor,
    output: torch.Tensor,
    *,
    tolerance: float = CG_TOLERANCE,
) -> CGPathBasis:
    """Return an orthonormal deterministic basis of equivariant bilinear paths."""

    if left.ndim != 3 or right.ndim != 3 or output.ndim != 3:
        raise ValueError("finite-group representations must have shape [group, dim, dim]")
    if not (left.shape[0] == right.shape[0] == output.shape[0]):
        raise ValueError("representations must describe the same group operations")
    product_representation = torch.stack(
        [torch.kron(left[index], right[index]) for index in range(left.shape[0])]
    )
    hom_representation = torch.stack(
        [
            torch.kron(product_representation[index], output[index])
            for index in range(left.shape[0])
        ]
    )
    projector = hom_representation.mean(dim=0)
    projector = 0.5 * (projector + projector.T)
    basis = _deterministic_range_basis(projector, tolerance)
    matrices = basis.T.reshape(
        basis.shape[1], product_representation.shape[1], output.shape[1]
    )
    for matrix in matrices:
        residual = torch.einsum("gij,jk->gik", product_representation.transpose(1, 2), matrix) - torch.einsum(
            "ij,gjk->gik", matrix, output.transpose(1, 2)
        )
        if residual.numel() and float(residual.abs().max()) > 10 * tolerance:
            raise RuntimeError("computed finite-group CG path fails intertwining")
    return CGPathBasis(matrices)
