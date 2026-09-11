from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch

from ..data import TrainingUnit
from ..irreps import IrrepLayout


@dataclass(frozen=True, slots=True)
class NormalizationBlock:
    copy_label: str
    copy_index: int
    start: int
    stop: int
    center: float
    scale: float


@dataclass(frozen=True, slots=True)
class CoefficientNormalizer:
    """Train-only, copy-aware equivariant coefficient normalization."""

    unit: TrainingUnit
    layout: IrrepLayout
    mode: str
    center: torch.Tensor
    scale: torch.Tensor
    blocks: tuple[NormalizationBlock, ...]

    def __post_init__(self) -> None:
        if self.mode not in {"rms", "variance"}:
            raise ValueError("normalization mode must be rms or variance")
        if self.center.shape != (self.layout.dimension,) or self.scale.shape != (
            self.layout.dimension,
        ):
            raise ValueError("normalizer vectors must match layout dimension")
        if self.center.device != self.scale.device or self.center.dtype != self.scale.dtype:
            raise ValueError("normalizer center and scale must share dtype/device")
        if not torch.isfinite(self.center).all() or not torch.isfinite(self.scale).all():
            raise ValueError("normalizer statistics must be finite")
        if bool((self.scale <= 0).any()):
            raise ValueError("normalizer scales must be positive")

    @classmethod
    def fit(
        cls,
        coefficients: torch.Tensor,
        layout: IrrepLayout,
        unit: TrainingUnit,
        *,
        split: str,
        mode: str = "rms",
        epsilon: float = 1.0e-8,
    ) -> "CoefficientNormalizer":
        if split != "train":
            raise ValueError("normalization statistics may only be fit on the train split")
        if mode not in {"rms", "variance"}:
            raise ValueError("normalization mode must be rms or variance")
        if coefficients.ndim != 2 or coefficients.shape[1] != layout.dimension:
            raise ValueError("coefficients must have shape [items, layout.dimension]")
        if not coefficients.shape[0] or not torch.isfinite(coefficients).all():
            raise ValueError("finite non-empty training coefficients are required")
        if epsilon <= 0:
            raise ValueError("epsilon must be positive")

        centers = torch.zeros(layout.dimension, dtype=coefficients.dtype, device=coefficients.device)
        scales = torch.empty_like(centers)
        blocks: list[NormalizationBlock] = []
        offset = 0
        for term in layout.terms:
            component_count = 2 * term.degree + 1
            for copy_index in range(term.multiplicity):
                start, stop = offset, offset + component_count
                values = coefficients[:, start:stop]
                center_value = values.mean() if mode == "variance" and term.degree == 0 else values.new_zeros(())
                centered = values - center_value
                scale_value = torch.sqrt(centered.square().mean()).clamp_min(epsilon)
                centers[start:stop] = center_value
                scales[start:stop] = scale_value
                blocks.append(
                    NormalizationBlock(
                        copy_label=term.copy_label,
                        copy_index=copy_index,
                        start=start,
                        stop=stop,
                        center=float(center_value.detach()),
                        scale=float(scale_value.detach()),
                    )
                )
                offset = stop
        return cls(unit, layout, mode, centers, scales, tuple(blocks))

    def transform(self, coefficients: torch.Tensor) -> torch.Tensor:
        self._validate_input(coefficients)
        return (coefficients - self.center.to(coefficients)) / self.scale.to(coefficients)

    def inverse_transform(self, normalized: torch.Tensor) -> torch.Tensor:
        self._validate_input(normalized)
        return normalized * self.scale.to(normalized) + self.center.to(normalized)

    def _validate_input(self, values: torch.Tensor) -> None:
        if values.shape[-1] != self.layout.dimension:
            raise ValueError("coefficient width does not match normalizer layout")

    def state_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "training_unit": self.unit.namespace,
            "mode": self.mode,
            "layout": list(self.layout.to_spec()),
            "center": self.center.detach().cpu().tolist(),
            "scale": self.scale.detach().cpu().tolist(),
        }

    @classmethod
    def from_state_dict(
        cls,
        state: Mapping[str, object],
        *,
        expected_unit: TrainingUnit,
        expected_layout: IrrepLayout,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str = "cpu",
    ) -> "CoefficientNormalizer":
        if state.get("schema_version") != 1:
            raise ValueError("unsupported normalizer schema")
        if state.get("training_unit") != expected_unit.namespace:
            raise ValueError("normalizer belongs to a different training unit")
        if state.get("layout") != list(expected_layout.to_spec()):
            raise ValueError("normalizer layout/copy ordering mismatch")
        center = torch.tensor(state["center"], dtype=dtype, device=device)
        scale = torch.tensor(state["scale"], dtype=dtype, device=device)
        # Reconstruct block metadata from the frozen layout and stored vectors.
        blocks = []
        offset = 0
        for term in expected_layout.terms:
            width = 2 * term.degree + 1
            for copy_index in range(term.multiplicity):
                blocks.append(
                    NormalizationBlock(
                        term.copy_label,
                        copy_index,
                        offset,
                        offset + width,
                        float(center[offset]),
                        float(scale[offset]),
                    )
                )
                offset += width
        return cls(expected_unit, expected_layout, str(state["mode"]), center, scale, tuple(blocks))
