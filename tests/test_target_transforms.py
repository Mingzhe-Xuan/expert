from __future__ import annotations

import pytest
import torch

from src.heads import (
    TARGET_LAYOUTS,
    apply_bec_asr,
    cartesian_to_irreps,
    decanonicalize_cartesian,
    irreps_to_cartesian,
    project_bec_joint_symmetry,
    project_to_point_group,
    rotate_cartesian,
    target_representation,
)
from src.symmetry import PointGroupRegistry
from e3nn import o3


@pytest.mark.parametrize("task", ["dielectric", "elastic", "bec"])
def test_cartesian_irrep_round_trip_and_o3_covariance(task: str) -> None:
    torch.manual_seed(20260911)
    coefficients = torch.randn(3, TARGET_LAYOUTS[task].dimension, dtype=torch.float64)
    tensor = irreps_to_cartesian(coefficients, task)
    recovered = cartesian_to_irreps(tensor, task)
    assert torch.allclose(recovered, coefficients, atol=2e-10, rtol=2e-10)

    proper = o3.rand_matrix(dtype=torch.float64)
    improper = -proper
    for rotation in (proper, improper):
        rotated_tensor = rotate_cartesian(tensor, rotation, task)
        rotated_coefficients = coefficients @ target_representation(
            rotation, task, dtype=torch.float64
        ).T
        assert torch.allclose(
            cartesian_to_irreps(rotated_tensor, task),
            rotated_coefficients,
            atol=2e-8,
            rtol=2e-8,
        )


def test_cartesian_output_has_required_intrinsic_symmetries() -> None:
    dielectric = irreps_to_cartesian(torch.randn(2, 6, dtype=torch.float64), "dielectric")
    assert torch.allclose(dielectric, dielectric.transpose(-1, -2), atol=1e-12)

    elastic = irreps_to_cartesian(torch.randn(2, 21, dtype=torch.float64), "elastic")
    assert torch.allclose(elastic, elastic.transpose(-1, -2), atol=1e-12)
    assert torch.allclose(elastic, elastic.transpose(-3, -4), atol=1e-12)
    assert torch.allclose(elastic, elastic.permute(0, 3, 4, 1, 2), atol=1e-12)


def test_global_point_group_projection_for_all_groups() -> None:
    torch.manual_seed(17)
    registry = PointGroupRegistry()
    for group in registry:
        for task in ("dielectric", "elastic"):
            coefficients = torch.randn(2, TARGET_LAYOUTS[task].dimension, dtype=torch.float64)
            projected = project_to_point_group(coefficients, task, group)
            projected_twice = project_to_point_group(projected, task, group)
            assert torch.allclose(projected_twice, projected, atol=2e-8, rtol=2e-8)
            representation = group.representation(TARGET_LAYOUTS[task])
            assert torch.allclose(
                torch.einsum("gij,bj->gbi", representation, projected),
                projected.expand(group.order, -1, -1),
                atol=2e-8,
                rtol=2e-8,
            )
    with pytest.raises(ValueError, match="raw"):
        project_to_point_group(torch.zeros(2, 9), "bec", registry["1"])


def test_bec_asr_is_independent_per_crystal_and_differentiable() -> None:
    raw = torch.arange(45, dtype=torch.float64).reshape(5, 3, 3).requires_grad_()
    node_batch = torch.tensor([0, 0, 1, 1, 1])
    projected = apply_bec_asr(raw, node_batch)
    assert torch.allclose(projected[:2].sum(dim=0), torch.zeros(3, 3, dtype=raw.dtype))
    assert torch.allclose(projected[2:].sum(dim=0), torch.zeros(3, 3, dtype=raw.dtype))
    projected.square().sum().backward()
    assert raw.grad is not None and torch.isfinite(raw.grad).all()


def test_optional_joint_bec_projector_and_asr_commute() -> None:
    torch.manual_seed(4)
    raw = torch.randn(2, 3, 3, dtype=torch.float64)
    rotations = torch.stack((torch.eye(3, dtype=torch.float64), -torch.eye(3, dtype=torch.float64)))
    permutations = torch.tensor([[0, 1], [1, 0]])
    projected = project_bec_joint_symmetry(raw, rotations, permutations)
    assert torch.allclose(projected[1], projected[0], atol=1e-12)
    node_batch = torch.zeros(2, dtype=torch.long)
    asr_then_symmetry = project_bec_joint_symmetry(
        apply_bec_asr(raw, node_batch), rotations, permutations
    )
    symmetry_then_asr = apply_bec_asr(projected, node_batch)
    assert torch.allclose(asr_then_symmetry, symmetry_then_asr, atol=1e-12)


def test_decanonicalization_restores_frame_and_bec_site_order() -> None:
    tensor = torch.arange(18, dtype=torch.float64).reshape(2, 3, 3)
    rotation = torch.tensor(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=torch.float64,
    )
    restored = decanonicalize_cartesian(
        tensor,
        "bec",
        rotation,
        canonical_to_input_sites=torch.tensor([1, 0]),
    )
    expected = rotate_cartesian(tensor, rotation, "bec").flip(0)
    assert torch.equal(restored, expected)
