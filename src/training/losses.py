from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

import torch

from ..irreps import IrrepLayout
from .normalization import CoefficientNormalizer


@dataclass(frozen=True, slots=True)
class CoefficientLoss:
    total: torch.Tensor
    by_copy: Mapping[str, torch.Tensor]

    def __post_init__(self) -> None:
        object.__setattr__(self, "by_copy", MappingProxyType(dict(self.by_copy)))


def _copy_slices(layout: IrrepLayout):
    offset = 0
    for term in layout.terms:
        width = 2 * term.degree + 1
        for copy_index in range(term.multiplicity):
            label = term.copy_label if term.multiplicity == 1 else f"{term.copy_label}#{copy_index}"
            yield label, slice(offset, offset + width)
            offset += width


def coefficient_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    layout: IrrepLayout,
    normalizer: CoefficientNormalizer | None = None,
) -> CoefficientLoss:
    """Compute equal-copy MSE without pooling the item/atom axis before loss."""

    if prediction.shape != target.shape or prediction.ndim != 2:
        raise ValueError("prediction and target must share shape [items, coefficients]")
    if prediction.shape[1] != layout.dimension:
        raise ValueError("coefficient width does not match layout")
    if normalizer is not None:
        if normalizer.layout != layout:
            raise ValueError("normalizer layout does not match loss layout")
        difference = (prediction - target) / normalizer.scale.to(prediction)
    else:
        difference = prediction - target
    losses = {
        label: difference[:, block].square().mean()
        for label, block in _copy_slices(layout)
    }
    return CoefficientLoss(torch.stack(tuple(losses.values())).mean(), losses)


def physical_coefficient_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    layout: IrrepLayout,
) -> dict[str, dict[str, float]]:
    """Return raw physical-unit MAE/RMSE for every labelled irrep copy."""

    if prediction.shape != target.shape or prediction.ndim != 2:
        raise ValueError("prediction and target must share shape [items, coefficients]")
    if prediction.shape[1] != layout.dimension:
        raise ValueError("coefficient width does not match layout")
    difference = prediction.detach() - target.detach()
    return {
        label: {
            "mae": float(difference[:, block].abs().mean()),
            "rmse": float(torch.sqrt(difference[:, block].square().mean())),
        }
        for label, block in _copy_slices(layout)
    }
