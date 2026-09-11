from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

# e3nn<=0.5 stores slice objects in its packaged, read-only Wigner constants. PyTorch
# 2.6 defaults to weights_only=True, so explicitly allow only this builtin container
# type instead of disabling safe checkpoint loading process-wide.
torch.serialization.add_safe_globals([slice])

from e3nn import o3

from ..irreps import IrrepLayout


DEFAULT_REGISTRY_PATH = (
    Path(__file__).resolve().parents[2] / "assets" / "docs" / "subgroup_chain.json"
)


def canonical_point_group_symbol(symbol: str) -> str:
    return symbol.replace(" ", "")


def _matrix_key(matrix: np.ndarray) -> tuple[int, ...]:
    return tuple(int(value) for value in matrix.reshape(-1))


def _fractional_to_cartesian(rotations: np.ndarray) -> np.ndarray:
    metric = sum(rotation.T @ rotation for rotation in rotations)
    basis = np.linalg.cholesky(metric).T
    basis_inverse = np.linalg.inv(basis)
    cartesian = []
    for rotation in rotations:
        transformed = basis @ rotation @ basis_inverse
        left, _, right = np.linalg.svd(transformed)
        orthogonal = left @ right
        if np.linalg.det(orthogonal) * np.linalg.det(transformed) < 0:
            left[:, -1] *= -1
            orthogonal = left @ right
        cartesian.append(orthogonal)
    return np.stack(cartesian)


def _layout_irreps(layout: IrrepLayout) -> o3.Irreps:
    return o3.Irreps(
        [
            (term.multiplicity, (term.degree, 1 if term.parity == "e" else -1))
            for term in layout.terms
        ]
    )


def _deterministic_range_basis(projector: torch.Tensor, tolerance: float) -> torch.Tensor:
    vectors: list[torch.Tensor] = []
    for column in projector.unbind(dim=1):
        residual = column.clone()
        for vector in vectors:
            residual = residual - torch.dot(vector, residual) * vector
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


@dataclass(frozen=True, slots=True)
class PointGroup:
    number: int
    symbol: str
    schoenflies: str
    order: int
    representative_hall_number: int
    fractional_rotations: torch.Tensor
    cartesian_rotations: torch.Tensor

    def representation(
        self, layout: IrrepLayout, *, dtype: torch.dtype = torch.float64
    ) -> torch.Tensor:
        irreps = _layout_irreps(layout)
        matrices = [
            irreps.D_from_matrix(rotation.to(dtype=dtype))
            for rotation in self.cartesian_rotations
        ]
        return torch.stack(matrices)

    def invariant_projector(
        self, layout: IrrepLayout, *, dtype: torch.dtype = torch.float64
    ) -> torch.Tensor:
        projector = self.representation(layout, dtype=dtype).mean(dim=0)
        return 0.5 * (projector + projector.T)

    def invariant_basis(
        self,
        layout: IrrepLayout,
        *,
        dtype: torch.dtype = torch.float64,
        tolerance: float = 1.0e-9,
    ) -> torch.Tensor:
        return _deterministic_range_basis(
            self.invariant_projector(layout, dtype=dtype), tolerance
        )

    def invariant_basis_checksum(self, layout: IrrepLayout) -> str:
        basis = self.invariant_basis(layout).detach().cpu().numpy()
        payload = {
            "point_group": self.symbol,
            "layout": layout.to_spec(),
            "shape": basis.shape,
            "basis": np.round(basis, decimals=12).tolist(),
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


class PointGroupRegistry:
    """Validated registry of exactly the 32 crystallographic point groups."""

    def __init__(self, path: str | Path = DEFAULT_REGISTRY_PATH) -> None:
        self.path = Path(path)
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != 1:
            raise ValueError("unsupported point-group registry schema")
        raw_groups = payload.get("point_groups")
        if not isinstance(raw_groups, dict) or set(raw_groups) != {
            str(index) for index in range(1, 33)
        }:
            raise ValueError("registry must contain exactly point groups 1..32")
        metadata = payload.get("metadata", {})
        if metadata.get("point_group_count") != 32:
            raise ValueError("registry metadata point_group_count mismatch")

        by_number: dict[int, PointGroup] = {}
        by_symbol: dict[str, PointGroup] = {}
        for number in range(1, 33):
            raw = raw_groups[str(number)]
            fractional = np.asarray(raw["operation_matrices"], dtype=np.int64)
            self._validate_fractional_group(fractional, int(raw["order"]), number)
            cartesian = _fractional_to_cartesian(fractional.astype(np.float64))
            identity = np.eye(3)
            if not np.allclose(
                np.swapaxes(cartesian, 1, 2) @ cartesian,
                identity,
                rtol=1.0e-10,
                atol=1.0e-10,
            ):
                raise ValueError(f"point group {number} Cartesian operations are not orthogonal")
            symbol = canonical_point_group_symbol(str(raw["hm_symbol"]))
            group = PointGroup(
                number=number,
                symbol=symbol,
                schoenflies=str(raw["schoenflies"]),
                order=int(raw["order"]),
                representative_hall_number=int(raw["representative_hall_number"]),
                fractional_rotations=torch.from_numpy(fractional.copy()),
                cartesian_rotations=torch.from_numpy(cartesian),
            )
            if symbol in by_symbol:
                raise ValueError(f"duplicate point-group symbol {symbol}")
            by_number[number] = group
            by_symbol[symbol] = group
        self._by_number = by_number
        self._by_symbol = by_symbol

    @staticmethod
    def _validate_fractional_group(
        rotations: np.ndarray, declared_order: int, number: int
    ) -> None:
        if rotations.shape != (declared_order, 3, 3):
            raise ValueError(f"point group {number} operation count/shape mismatch")
        keys = {_matrix_key(rotation) for rotation in rotations}
        if len(keys) != declared_order or _matrix_key(np.eye(3, dtype=np.int64)) not in keys:
            raise ValueError(f"point group {number} lacks unique operations or identity")
        for left in rotations:
            inverse = np.rint(np.linalg.inv(left)).astype(np.int64)
            if _matrix_key(inverse) not in keys:
                raise ValueError(f"point group {number} is not inverse closed")
            for right in rotations:
                if _matrix_key(left @ right) not in keys:
                    raise ValueError(f"point group {number} is not multiplication closed")

    def __len__(self) -> int:
        return len(self._by_number)

    def __iter__(self):
        return iter(self._by_number.values())

    def __getitem__(self, key: int | str) -> PointGroup:
        if isinstance(key, int):
            return self._by_number[key]
        return self._by_symbol[canonical_point_group_symbol(key)]
