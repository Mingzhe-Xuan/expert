from __future__ import annotations

import math
from collections.abc import Sequence

import torch


DEFAULT_EWT_THRESHOLDS = (0.25, 0.10, 0.05)
ELASTIC_VOIGT_PAIRS = ((0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1))


def _metric_view(values: torch.Tensor, task: str | None) -> torch.Tensor:
    if task is None or task == "dielectric":
        return values
    if task != "elastic":
        raise ValueError("benchmark task must be dielectric or elastic")
    if values.ndim != 5 or tuple(values.shape[1:]) != (3, 3, 3, 3):
        raise ValueError("elastic benchmark tensors must have shape [sample, 3, 3, 3, 3]")
    return torch.stack(
        [
            torch.stack(
                [values[:, i, j, k, ell] for k, ell in ELASTIC_VOIGT_PAIRS], dim=1
            )
            for i, j in ELASTIC_VOIGT_PAIRS
        ],
        dim=1,
    )


def tensor_benchmark_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    task: str | None = None,
    thresholds: Sequence[float] = DEFAULT_EWT_THRESHOLDS,
    relative_epsilon: float = 1.0e-5,
) -> dict[str, float | int]:
    """Compute sample-mean Frobenius distance and GMTNet-style EwT percentages."""

    if prediction.shape != target.shape or prediction.ndim < 2:
        raise ValueError("prediction and target must share a leading sample axis")
    if prediction.shape[0] < 1:
        raise ValueError("benchmark metrics require at least one sample")
    if not torch.isfinite(prediction).all() or not torch.isfinite(target).all():
        raise ValueError("benchmark tensors must contain only finite values")
    if not math.isfinite(relative_epsilon) or relative_epsilon <= 0:
        raise ValueError("relative epsilon must be finite and positive")
    parsed = tuple(float(value) for value in thresholds)
    if len(parsed) != len(set(parsed)) or any(
        not math.isfinite(value) or not 0.0 < value < 1.0 for value in parsed
    ):
        raise ValueError("EwT thresholds must be unique finite fractions in (0, 1)")

    prediction = _metric_view(prediction.detach(), task)
    target = _metric_view(target.detach(), task)
    difference = (prediction - target).reshape(prediction.shape[0], -1)
    flattened_target = target.reshape(target.shape[0], -1)
    distance = torch.linalg.vector_norm(difference, dim=1)
    target_norm = torch.linalg.vector_norm(flattened_target, dim=1)
    relative = distance / (target_norm + relative_epsilon)
    report: dict[str, float | int] = {
        "sample_count": int(prediction.shape[0]),
        "rmse": float(torch.sqrt(torch.mean(difference.square()))),
        "fnorm": float(distance.mean()),
    }
    for threshold in parsed:
        label = f"ewt_{int(round(100 * threshold))}"
        report[label] = 100.0 * float((relative < threshold).to(torch.float64).mean())
    return report
