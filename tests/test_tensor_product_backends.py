from __future__ import annotations

import pytest
import torch

from src.irreps import IrrepLayout, IrrepTerm
from src.tensor_products import (
    FullO3TensorProduct,
    O2TensorProduct,
    build_tensor_product,
    edge_frames,
)
from src.tensor_products.backends import (
    _sample_o2_representations,
    _stable_irrep_matrix,
)
from e3nn import o3


SCALAR_E = IrrepLayout((IrrepTerm(1, 0, "e", "scalar_e"),))
SCALAR_O = IrrepLayout((IrrepTerm(1, 0, "o", "scalar_o"),))
POLAR = IrrepLayout((IrrepTerm(1, 1, "o", "polar"),))
AXIAL = IrrepLayout((IrrepTerm(1, 1, "e", "axial"),))
MIXED = IrrepLayout(
    (IrrepTerm(2, 0, "e", "scalars"), IrrepTerm(1, 1, "o", "polar"))
)


def _representation(layout: IrrepLayout, rotation: torch.Tensor) -> torch.Tensor:
    irreps = o3.Irreps(
        [(term.multiplicity, (term.degree, 1 if term.parity == "e" else -1)) for term in layout.terms]
    )
    return irreps.D_from_matrix(rotation)


@pytest.mark.parametrize("degree", range(5))
@pytest.mark.parametrize("parity", ["e", "o"])
def test_stable_irrep_matrix_matches_e3nn_float64_convention(
    degree: int, parity: str
) -> None:
    angle = torch.tensor(0.37, dtype=torch.float64)
    rotation = o3.matrix_z(angle)
    reflection = torch.diag(torch.tensor([1.0, -1.0, 1.0], dtype=torch.float64))
    irrep = o3.Irrep(degree, 1 if parity == "e" else -1)
    for operation in (rotation, rotation @ reflection):
        assert torch.allclose(
            _stable_irrep_matrix(degree, parity, operation),
            irrep.D_from_matrix(operation),
            atol=2.0e-10,
            rtol=2.0e-10,
        )


def test_sampled_o2_representations_obey_dihedral_group_law() -> None:
    basis = torch.eye(7, dtype=torch.float64)
    order = 11
    representation = _sample_o2_representations(3, "o", basis, order)
    rotations = representation[0::2]
    reflections = representation[1::2]
    for left in range(order):
        for right in range(order):
            assert torch.allclose(
                rotations[left] @ rotations[right],
                rotations[(left + right) % order],
                atol=2.0e-10,
                rtol=2.0e-10,
            )
            assert torch.allclose(
                reflections[left] @ reflections[right],
                rotations[(left - right) % order],
                atol=2.0e-10,
                rtol=2.0e-10,
            )


@pytest.mark.parametrize("backend", ["full_o3", "o2_tp"])
def test_backend_shape_gradients_and_parameter_audit(backend: str) -> None:
    torch.manual_seed(8)
    module = build_tensor_product(backend, MIXED, POLAR, MIXED, mmax=2).double()
    left = torch.randn(4, MIXED.dimension, dtype=torch.float64, requires_grad=True)
    right = torch.randn(4, POLAR.dimension, dtype=torch.float64, requires_grad=True)
    axes = torch.randn(4, 3, dtype=torch.float64)
    output = module(left, right, axes)
    assert output.shape == (4, MIXED.dimension)
    output.square().sum().backward()
    assert left.grad is not None and torch.isfinite(left.grad).all()
    assert right.grad is not None and torch.isfinite(right.grad).all()
    trainable = sum(parameter.numel() for parameter in module.parameters())
    assert trainable == module.path_count > 0
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in module.parameters())


@pytest.mark.parametrize("backend", ["full_o3", "o2_tp"])
@pytest.mark.parametrize("improper", [False, True])
def test_backends_are_globally_o3_equivariant(backend: str, improper: bool) -> None:
    torch.manual_seed(11)
    module = build_tensor_product(backend, POLAR, AXIAL, POLAR, mmax=1).double()
    left = torch.randn(5, 3, dtype=torch.float64)
    right = torch.randn(5, 3, dtype=torch.float64)
    axes = torch.randn(5, 3, dtype=torch.float64)
    rotation = o3.rand_matrix(dtype=torch.float64)
    if improper:
        rotation = -rotation
    output = module(left, right, axes)
    transformed = module(
        left @ _representation(POLAR, rotation).T,
        right @ _representation(AXIAL, rotation).T,
        axes @ rotation.T,
    )
    expected = output @ _representation(POLAR, rotation).T
    assert torch.allclose(transformed, expected, atol=3e-8, rtol=3e-8)


def test_o2_backend_is_invariant_to_full_local_o2_gauge() -> None:
    torch.manual_seed(21)
    module = O2TensorProduct(POLAR, POLAR, MIXED, mmax=1).double()
    left = torch.randn(4, 3, dtype=torch.float64)
    right = torch.randn(4, 3, dtype=torch.float64)
    axes = torch.randn(4, 3, dtype=torch.float64)
    frames = edge_frames(axes)
    rotation_gauge = o3.matrix_z(torch.tensor(0.731, dtype=torch.float64))
    reflection_gauge = torch.diag(torch.tensor([1.0, -1.0, 1.0], dtype=torch.float64))
    reference = module(left, right, axes, frames)
    for gauge in (rotation_gauge, reflection_gauge, rotation_gauge @ reflection_gauge):
        gauged = module(left, right, axes, frames @ gauge)
        assert torch.allclose(gauged, reference, atol=3e-8, rtol=3e-8)


def test_o2_mmax_controls_complete_local_path_space() -> None:
    axial = O2TensorProduct(POLAR, POLAR, SCALAR_E, mmax=0)
    full_m = O2TensorProduct(POLAR, POLAR, SCALAR_E, mmax=1)
    assert axial.path_count == 1
    # z*z and the transverse dot product are independent O(2) scalar paths.
    assert full_m.path_count == 2
    assert len(full_m.coupling_checksum) == 64
    assert full_m.coupling_checksum == O2TensorProduct(
        POLAR, POLAR, SCALAR_E, mmax=1
    ).coupling_checksum


@pytest.mark.parametrize("backend", ["full_o3", "o2_tp"])
def test_forbidden_scalar_parity_has_no_fake_path(backend: str) -> None:
    module = build_tensor_product(backend, SCALAR_E, SCALAR_E, SCALAR_O, mmax=0).double()
    left = torch.randn(3, 1, dtype=torch.float64)
    right = torch.randn(3, 1, dtype=torch.float64)
    axes = torch.randn(3, 3, dtype=torch.float64)
    output = module(left, right, axes)
    assert module.path_count == 0
    assert torch.equal(output, torch.zeros_like(output))


def test_o2_edge_axis_reversal_respects_polar_and_pseudo_parity() -> None:
    torch.manual_seed(34)
    module = O2TensorProduct(POLAR, AXIAL, POLAR, mmax=1).double()
    polar = torch.randn(3, 3, dtype=torch.float64)
    axial = torch.randn(3, 3, dtype=torch.float64)
    axes = torch.randn(3, 3, dtype=torch.float64)
    output = module(polar, axial, axes)
    inversion = -torch.eye(3, dtype=torch.float64)
    reversed_output = module(
        polar @ _representation(POLAR, inversion).T,
        axial @ _representation(AXIAL, inversion).T,
        -axes,
    )
    assert torch.allclose(
        reversed_output,
        output @ _representation(POLAR, inversion).T,
        atol=3e-8,
        rtol=3e-8,
    )


def test_invalid_backend_and_zero_axis_fail_closed() -> None:
    with pytest.raises(ValueError, match="unsupported"):
        build_tensor_product("none", SCALAR_E, SCALAR_E, SCALAR_E)
    module = O2TensorProduct(SCALAR_E, SCALAR_E, SCALAR_E, mmax=0)
    with pytest.raises(ValueError, match="non-zero"):
        module(torch.ones(1, 1), torch.ones(1, 1), torch.zeros(1, 3))
