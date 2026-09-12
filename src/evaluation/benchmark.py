from __future__ import annotations

import math
from collections.abc import Sequence

import torch


DEFAULT_EWT_THRESHOLDS = (0.25, 0.10, 0.05)


def tensor_benchmark_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    thresholds: Sequence[float] = DEFAULT_EWT_THRESHOLDS,
) -> dict[str, float | int]:
    """Compute sample-mean Frobenius distance and GMTNet-style EwT percentages."""

    if prediction.shape != target.shape or prediction.ndim < 2:
        raise ValueError("prediction and target must share a leading sample axis")
    if prediction.shape[0] < 1:
        raise ValueError("benchmark metrics require at least one sample")
    if not torch.isfinite(prediction).all() or not torch.isfinite(target).all():
        raise ValueError("benchmark tensors must contain only finite values")
    parsed = tuple(float(value) for value in thresholds)
    if len(parsed) != len(set(parsed)) or any(
        not math.isfinite(value) or not 0.0 < value < 1.0 for value in parsed
    ):
        raise ValueError("EwT thresholds must be unique finite fractions in (0, 1)")

    difference = (prediction.detach() - target.detach()).reshape(prediction.shape[0], -1)
    flattened_target = target.detach().reshape(target.shape[0], -1)
    distance = torch.linalg.vector_norm(difference, dim=1)
    target_norm = torch.linalg.vector_norm(flattened_target, dim=1)
    relative = torch.where(
        target_norm > 0,
        distance / target_norm,
        torch.where(distance == 0, torch.zeros_like(distance), torch.full_like(distance, torch.inf)),
    )
    report: dict[str, float | int] = {
        "sample_count": int(prediction.shape[0]),
        "fnorm": float(distance.mean()),
    }
    for threshold in parsed:
        label = f"ewt_{int(round(100 * threshold))}"
        report[label] = 100.0 * float((relative < threshold).to(torch.float64).mean())
    return report
