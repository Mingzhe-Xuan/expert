from __future__ import annotations

from dataclasses import dataclass
import functools
import hashlib
import json
import math

import torch
from torch import nn

from ..irreps import IrrepLayout, finite_group_intertwiners
from ..symmetry.registry import _layout_irreps
from e3nn import o3


def edge_frames(vectors: torch.Tensor) -> torch.Tensor:
    """Return deterministic local-to-global frames with local z on each edge axis."""

    if vectors.ndim != 2 or vectors.shape[1] != 3:
        raise ValueError("edge axes must have shape [batch, 3]")
    norms = torch.linalg.vector_norm(vectors, dim=-1)
    if bool((norms <= 1.0e-12).any()):
        raise ValueError("local O(2) tensor products require non-zero edge axes")
    z_axis = vectors / norms[:, None]
    z_reference = vectors.new_tensor([0.0, 0.0, 1.0]).expand_as(z_axis)
    x_reference = vectors.new_tensor([1.0, 0.0, 0.0]).expand_as(z_axis)
    reference = torch.where((z_axis[:, 2].abs() < 0.9)[:, None], z_reference, x_reference)
    x_axis = torch.linalg.cross(reference, z_axis)
    x_axis = x_axis / torch.linalg.vector_norm(x_axis, dim=-1, keepdim=True)
    y_axis = torch.linalg.cross(z_axis, x_axis)
    return torch.stack((x_axis, y_axis, z_axis), dim=-1)


class FullO3TensorProduct(nn.Module):
    """Unrestricted learned O(3) tensor product in the frozen real basis."""

    def __init__(self, left: IrrepLayout, right: IrrepLayout, output: IrrepLayout) -> None:
        super().__init__()
        self.left_layout = left
        self.right_layout = right
        self.output_layout = output
        self.tensor_product = o3.FullyConnectedTensorProduct(
            _layout_irreps(left), _layout_irreps(right), _layout_irreps(output)
        )

    def forward(
        self,
        left: torch.Tensor,
        right: torch.Tensor,
        axes: torch.Tensor | None = None,
        frames: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del axes, frames
        _validate_feature_pair(left, right, self.left_layout, self.right_layout)
        return self.tensor_product(left, right)

    @property
    def path_count(self) -> int:
        return self.tensor_product.weight_numel


def _m_band_basis(
    degree: int, parity: str, mmax: int, *, dtype: torch.dtype = torch.float64
) -> torch.Tensor:
    dimension = 2 * degree + 1
    if mmax >= degree:
        return torch.eye(dimension, dtype=dtype)
    angle = torch.tensor(1.0e-5, dtype=dtype)
    generator = (
        _stable_irrep_matrix(degree, parity, o3.matrix_z(angle))
        - _stable_irrep_matrix(degree, parity, o3.matrix_z(-angle))
    ) / (2 * angle)
    values, vectors = torch.linalg.eigh(-generator @ generator)
    keep = values <= (mmax + 0.25) ** 2
    projector = vectors[:, keep] @ vectors[:, keep].T
    # Deterministic projector-column Gram-Schmidt.
    columns = []
    for column in projector.unbind(1):
        residual = column.clone()
        for basis in columns:
            residual -= torch.dot(basis, residual) * basis
        norm = torch.linalg.vector_norm(residual)
        if float(norm) <= 1.0e-9:
            continue
        basis = residual / norm
        pivot = int(basis.abs().argmax())
        columns.append(-basis if float(basis[pivot]) < 0 else basis)
    return torch.stack(columns, dim=1)


@functools.lru_cache(maxsize=None)
def _float64_so3_generators(degree: int) -> torch.Tensor:
    """Construct e3nn's real-basis generators without default-dtype leakage."""

    lower = torch.arange(-degree, degree, dtype=torch.float64)
    raising = torch.diag(
        -torch.sqrt(degree * (degree + 1) - lower * (lower + 1)), diagonal=-1
    )
    upper = torch.arange(-degree + 1, degree + 1, dtype=torch.float64)
    lowering = torch.diag(
        torch.sqrt(degree * (degree + 1) - upper * (upper - 1)), diagonal=1
    )
    orders = torch.arange(-degree, degree + 1, dtype=torch.float64)
    complex_generators = torch.stack(
        (
            0.5 * (raising + lowering),
            torch.diag(1j * orders),
            -0.5j * (raising - lowering),
        )
    )
    change = torch.zeros(
        (2 * degree + 1, 2 * degree + 1), dtype=torch.complex128
    )
    for order in range(-degree, 0):
        change[degree + order, degree + abs(order)] = 1 / math.sqrt(2)
        change[degree + order, degree - abs(order)] = -1j / math.sqrt(2)
    change[degree, degree] = 1
    for order in range(1, degree + 1):
        sign = (-1) ** order
        change[degree + order, degree + order] = sign / math.sqrt(2)
        change[degree + order, degree - order] = 1j * sign / math.sqrt(2)
    change *= (-1j) ** degree
    generators = torch.conj(change.T) @ complex_generators @ change
    if not bool((generators.imag.abs() < 1.0e-12).all()):
        raise RuntimeError("real-basis SO(3) generators acquired an imaginary component")
    return generators.real


def _stable_irrep_matrix(
    degree: int, parity: str, operation: torch.Tensor
) -> torch.Tensor:
    """Evaluate an O(3) irrep in float64 independently of e3nn's default dtype."""

    operation = operation.to(dtype=torch.float64, device="cpu")
    determinant = torch.linalg.det(operation).sign()
    proper = determinant * operation
    alpha, beta, gamma = o3.matrix_to_angles(proper)
    generators = _float64_so3_generators(degree)
    matrix = (
        torch.matrix_exp(alpha * generators[1])
        @ torch.matrix_exp(beta * generators[0])
        @ torch.matrix_exp(gamma * generators[1])
    )
    if float(determinant) < 0:
        matrix = matrix * (1 if parity == "e" else -1)
    return matrix


def _sample_o2_representations(
    degree: int, parity: str, basis: torch.Tensor, cyclic_order: int
) -> torch.Tensor:
    reflection = torch.diag(torch.tensor([1.0, -1.0, 1.0], dtype=basis.dtype))
    matrices = []
    for index in range(cyclic_order):
        rotation = o3.matrix_z(
            torch.tensor(2 * math.pi * index / cyclic_order, dtype=basis.dtype)
        )
        for operation in (rotation, rotation @ reflection):
            full = _stable_irrep_matrix(degree, parity, operation)
            matrices.append(basis.T @ full @ basis)
    return torch.stack(matrices)


@dataclass(frozen=True, slots=True)
class O2PathMetadata:
    left_term: int
    right_term: int
    output_term: int
    local_path_count: int


class _O2DegreePaths(nn.Module):
    def __init__(
        self,
        metadata: O2PathMetadata,
        left_multiplicity: int,
        right_multiplicity: int,
        output_multiplicity: int,
        coupling: torch.Tensor,
    ) -> None:
        super().__init__()
        self.metadata = metadata
        self.register_buffer("coupling", coupling)
        self.weights = nn.Parameter(
            torch.empty(
                left_multiplicity,
                right_multiplicity,
                output_multiplicity,
                metadata.local_path_count,
            )
        )
        nn.init.normal_(
            self.weights,
            std=1.0
            / math.sqrt(
                max(left_multiplicity * right_multiplicity * metadata.local_path_count, 1)
            ),
        )


class O2TensorProduct(nn.Module):
    """Complete bandlimited local-O(2) bilinear tensor product.

    Every degree triple uses the full Hom-space intertwiner basis of a sufficiently large
    sampled dihedral group. Since all retained frequencies satisfy ``|m|<=mmax`` and the
    cyclic order exceeds the largest possible frequency sum, this equals the continuous
    O(2) selection-rule space rather than an aliased finite-group approximation.
    """

    def __init__(
        self,
        left: IrrepLayout,
        right: IrrepLayout,
        output: IrrepLayout,
        *,
        mmax: int = 2,
    ) -> None:
        super().__init__()
        if mmax < 0:
            raise ValueError("mmax must be non-negative")
        self.left_layout = left
        self.right_layout = right
        self.output_layout = output
        self.mmax = mmax
        self.left_irreps = _layout_irreps(left)
        self.right_irreps = _layout_irreps(right)
        self.output_irreps = _layout_irreps(output)
        self.paths = nn.ModuleList()

        for left_index, left_term in enumerate(left.terms):
            left_basis = _m_band_basis(left_term.degree, left_term.parity, mmax)
            for right_index, right_term in enumerate(right.terms):
                right_basis = _m_band_basis(right_term.degree, right_term.parity, mmax)
                for output_index, output_term in enumerate(output.terms):
                    output_basis = _m_band_basis(output_term.degree, output_term.parity, mmax)
                    cyclic_order = 2 * (
                        left_term.degree + right_term.degree + output_term.degree
                    ) + 3
                    left_rep = _sample_o2_representations(
                        left_term.degree, left_term.parity, left_basis, cyclic_order
                    )
                    right_rep = _sample_o2_representations(
                        right_term.degree, right_term.parity, right_basis, cyclic_order
                    )
                    output_rep = _sample_o2_representations(
                        output_term.degree, output_term.parity, output_basis, cyclic_order
                    )
                    local = finite_group_intertwiners(left_rep, right_rep, output_rep)
                    if not local.path_count:
                        continue
                    reduced = local.matrices.reshape(
                        local.path_count,
                        left_basis.shape[1],
                        right_basis.shape[1],
                        output_basis.shape[1],
                    )
                    coupling = torch.einsum(
                        "ai,bj,pijk,ck->pabc",
                        left_basis,
                        right_basis,
                        reduced,
                        output_basis,
                    )
                    metadata = O2PathMetadata(
                        left_index, right_index, output_index, local.path_count
                    )
                    self.paths.append(
                        _O2DegreePaths(
                            metadata,
                            left_term.multiplicity,
                            right_term.multiplicity,
                            output_term.multiplicity,
                            coupling,
                        )
                    )

        self._left_offsets = _term_offsets(left)
        self._right_offsets = _term_offsets(right)
        self._output_offsets = _term_offsets(output)

    def forward(
        self,
        left: torch.Tensor,
        right: torch.Tensor,
        axes: torch.Tensor | None = None,
        frames: torch.Tensor | None = None,
    ) -> torch.Tensor:
        _validate_feature_pair(left, right, self.left_layout, self.right_layout)
        if frames is None:
            if axes is None:
                raise ValueError("o2_tp requires edge axes or explicit local frames")
            frames = edge_frames(axes)
        if frames.shape != (left.shape[0], 3, 3):
            raise ValueError("frames must have shape [batch, 3, 3]")
        if axes is not None:
            normalized = axes / torch.linalg.vector_norm(axes, dim=-1, keepdim=True)
            if not torch.allclose(frames[:, :, 2], normalized, atol=1e-6, rtol=1e-6):
                raise ValueError("explicit frame z axes must align with edge axes")

        left_d = self.left_irreps.D_from_matrix(frames)
        right_d = self.right_irreps.D_from_matrix(frames)
        output_d = self.output_irreps.D_from_matrix(frames)
        left_local = torch.einsum("bij,bi->bj", left_d, left)
        right_local = torch.einsum("bij,bi->bj", right_d, right)
        output_blocks = [
            left.new_zeros((left.shape[0], term.multiplicity, 2 * term.degree + 1))
            for term in self.output_layout.terms
        ]
        for path in self.paths:
            metadata = path.metadata
            left_term = self.left_layout.terms[metadata.left_term]
            right_term = self.right_layout.terms[metadata.right_term]
            output_term = self.output_layout.terms[metadata.output_term]
            left_start = self._left_offsets[metadata.left_term]
            right_start = self._right_offsets[metadata.right_term]
            left_block = left_local[
                :, left_start : left_start + left_term.dimension
            ].reshape(left.shape[0], left_term.multiplicity, 2 * left_term.degree + 1)
            right_block = right_local[
                :, right_start : right_start + right_term.dimension
            ].reshape(right.shape[0], right_term.multiplicity, 2 * right_term.degree + 1)
            messages = torch.einsum(
                "bui,bvj,pijk,uvwp->bwk",
                left_block,
                right_block,
                path.coupling.to(left),
                path.weights,
            )
            if messages.shape[1:] != (
                output_term.multiplicity,
                2 * output_term.degree + 1,
            ):
                raise RuntimeError("local O(2) path produced an invalid output block")
            output_blocks[metadata.output_term] = (
                output_blocks[metadata.output_term] + messages
            )
        output_local = torch.cat([block.flatten(1) for block in output_blocks], dim=1)
        return torch.einsum("bij,bj->bi", output_d, output_local)

    @property
    def path_count(self) -> int:
        return sum(path.weights.numel() for path in self.paths)

    @property
    def coupling_checksum(self) -> str:
        payload = []
        for path in self.paths:
            payload.append(
                {
                    "metadata": {
                        "left_term": path.metadata.left_term,
                        "right_term": path.metadata.right_term,
                        "output_term": path.metadata.output_term,
                        "local_path_count": path.metadata.local_path_count,
                    },
                    "coupling": torch.round(path.coupling.detach().cpu() * 1e12)
                    .to(torch.int64)
                    .tolist(),
                }
            )
        encoded = json.dumps(
            {"schema_version": 1, "mmax": self.mmax, "paths": payload},
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _term_offsets(layout: IrrepLayout) -> tuple[int, ...]:
    offsets = []
    offset = 0
    for term in layout.terms:
        offsets.append(offset)
        offset += term.dimension
    return tuple(offsets)


def _validate_feature_pair(
    left: torch.Tensor,
    right: torch.Tensor,
    left_layout: IrrepLayout,
    right_layout: IrrepLayout,
) -> None:
    if left.ndim != 2 or right.ndim != 2 or left.shape[0] != right.shape[0]:
        raise ValueError("tensor-product inputs must share shape prefix [batch]")
    if left.shape[1] != left_layout.dimension or right.shape[1] != right_layout.dimension:
        raise ValueError("tensor-product feature widths do not match layouts")
    if left.device != right.device or left.dtype != right.dtype:
        raise ValueError("tensor-product inputs must share dtype and device")


def build_tensor_product(
    backend: str,
    left: IrrepLayout,
    right: IrrepLayout,
    output: IrrepLayout,
    *,
    mmax: int = 2,
) -> nn.Module:
    if backend == "full_o3":
        return FullO3TensorProduct(left, right, output)
    if backend == "o2_tp":
        return O2TensorProduct(left, right, output, mmax=mmax)
    raise ValueError(f"unsupported tensor-product backend {backend!r}")
