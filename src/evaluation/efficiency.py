from __future__ import annotations

import time
from typing import Any, Callable, Sequence

import torch
from torch import nn

from ..configs import ArchitectureConfig


def profile_model_efficiency(
    model: nn.Module,
    end_to_end_forward: Callable[[], Any],
    downstream_forward: Callable[[Any], Any],
    *,
    architecture: ArchitectureConfig,
    task: str,
    point_groups: Sequence[str],
    active_nonbackbone_parameters: int,
    active_expert_counts: Sequence[int],
    device: torch.device | str,
) -> dict[str, object]:
    """Measure one real forward and active downstream operator FLOPs."""

    measured_device = torch.device(device)
    if not active_expert_counts:
        raise ValueError("efficiency profiling requires per-sample active expert counts")
    if active_nonbackbone_parameters < 0 or any(value < 0 for value in active_expert_counts):
        raise ValueError("efficiency parameter/expert counts must be non-negative")
    registered_total = sum(parameter.numel() for parameter in model.parameters())
    external_frozen = int(
        getattr(getattr(model, "adapter", None), "external_frozen_parameter_count", 0)
    )
    total_parameters = registered_total + external_frozen
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    was_training = model.training
    try:
        model.eval()
        if measured_device.type == "cuda":
            if not torch.cuda.is_available():
                raise ValueError("CUDA efficiency profiling requested without CUDA")
            torch.cuda.synchronize(measured_device)
            torch.cuda.reset_peak_memory_stats(measured_device)
            baseline_memory = torch.cuda.memory_allocated(measured_device)
        else:
            baseline_memory = None
        started = time.perf_counter()
        with torch.no_grad():
            cached = end_to_end_forward()
        if measured_device.type == "cuda":
            torch.cuda.synchronize(measured_device)
        latency_ms = (time.perf_counter() - started) * 1000.0
        peak_bytes = (
            torch.cuda.max_memory_allocated(measured_device) - baseline_memory
            if baseline_memory is not None
            else None
        )
        activities = [torch.profiler.ProfilerActivity.CPU]
        if measured_device.type == "cuda":
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        with torch.no_grad(), torch.profiler.profile(
            activities=activities,
            with_flops=True,
        ) as profiler:
            downstream_forward(cached)
        active_flops = sum(int(event.flops or 0) for event in profiler.key_averages())
    finally:
        model.train(was_training)
    counts = tuple(int(value) for value in active_expert_counts)
    return {
        "schema_version": 1,
        "architecture": architecture.variant_id,
        "task": task,
        "point_groups": list(point_groups),
        "pg_hidden_mode": architecture.pg_hidden_mode,
        "adaptation_backend": architecture.adaptation_backend,
        "o3e_backend": architecture.o3e_backend,
        "readout_backend": architecture.readout_backend,
        "total_parameters": total_parameters,
        "torch_registered_parameters": registered_total,
        "external_frozen_parameters": external_frozen,
        "trainable_parameters": trainable_parameters,
        "active_nonbackbone_parameters": int(active_nonbackbone_parameters),
        "active_expert_count_per_sample": list(counts),
        "active_expert_count_mean": sum(counts) / len(counts),
        "active_expert_count_max": max(counts),
        "active_downstream_flops": active_flops,
        "flops_scope": "torch_dispatched_active_downstream_forward",
        "end_to_end_forward_latency_ms": latency_ms,
        "latency_repetitions": 1,
        "peak_cuda_allocated_bytes": peak_bytes,
        "device": str(measured_device),
    }
