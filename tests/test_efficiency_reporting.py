from __future__ import annotations

import torch
from torch import nn

from src.configs import ArchitectureConfig
from src.evaluation import profile_model_efficiency


class _TinySystem(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.adapter = nn.Identity()
        self.linear = nn.Linear(4, 3)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.linear(values)


def test_cpu_efficiency_report_has_explicit_scopes_and_restores_training() -> None:
    model = _TinySystem()
    model.train()
    values = torch.ones(2, 4)
    architecture = ArchitectureConfig("B+R", "none", "none", "full_o3", "none")
    report = profile_model_efficiency(
        model,
        lambda: model(values),
        lambda cached: model.linear(values) + cached,
        architecture=architecture,
        task="dielectric",
        point_groups=("1",),
        active_nonbackbone_parameters=15,
        active_expert_counts=(0,),
        device="cpu",
    )
    assert model.training
    assert report["total_parameters"] == 15
    assert report["trainable_parameters"] == 15
    assert report["active_nonbackbone_parameters"] == 15
    assert report["active_expert_count_mean"] == report["active_expert_count_max"] == 0
    assert report["active_downstream_flops"] > 0
    assert report["flops_scope"] == "torch_dispatched_active_downstream_forward"
    assert report["end_to_end_forward_latency_ms"] > 0
    assert report["latency_repetitions"] == 1
    assert report["peak_cuda_allocated_bytes"] is None
    assert report["architecture"] == architecture.variant_id
    assert report["task"] == "dielectric"
    assert report["point_groups"] == ["1"]


def test_efficiency_report_counts_external_frozen_parameters() -> None:
    model = _TinySystem()
    model.adapter.external_frozen_parameter_count = 23
    report = profile_model_efficiency(
        model,
        lambda: model(torch.ones(1, 4)),
        lambda cached: cached + 1,
        architecture=ArchitectureConfig("B+R", "none", "none", "o2_tp", "none"),
        task="elastic",
        point_groups=("m-3m",),
        active_nonbackbone_parameters=15,
        active_expert_counts=(0,),
        device="cpu",
    )
    assert report["torch_registered_parameters"] == 15
    assert report["external_frozen_parameters"] == 23
    assert report["total_parameters"] == 38
