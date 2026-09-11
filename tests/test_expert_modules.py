from __future__ import annotations

import pytest
import torch

from src.experts import (
    A1PointGroupExpert,
    ContinuousResidualGate,
    FullPointGroupExpert,
    O3Adaptation,
    RoutedO3Expert,
    hierarchical_fusion,
)
from src.graphs import build_periodic_graph
from src.irreps import IrrepLayout, IrrepTerm
from src.symmetry import PointGroupRegistry
from e3nn import o3


LAYOUT = IrrepLayout(
    (
        IrrepTerm(2, 0, "e", "scalars"),
        IrrepTerm(1, 1, "o", "polar"),
        IrrepTerm(1, 2, "e", "quadrupole"),
    )
)


def _representation(layout: IrrepLayout, rotation: torch.Tensor) -> torch.Tensor:
    return o3.Irreps(
        [(term.multiplicity, (term.degree, 1 if term.parity == "e" else -1)) for term in layout.terms]
    ).D_from_matrix(rotation)


def _graph(dtype=torch.float64):
    cell = torch.tensor(
        [[2.0, 0.0, 0.0], [0.2, 2.1, 0.0], [0.1, 0.3, 2.2]], dtype=dtype
    )
    positions = torch.tensor([[0.1, 0.2, 0.3], [1.0, 0.7, 0.8]], dtype=dtype)
    return build_periodic_graph(positions, cell, torch.tensor([6, 8]), cutoff=1.5)


@pytest.mark.parametrize("module_type", [O3Adaptation, RoutedO3Expert])
@pytest.mark.parametrize("backend", ["full_o3", "o2_tp"])
@pytest.mark.parametrize("improper", [False, True])
def test_o3_modules_use_both_backends_and_are_equivariant(
    module_type, backend: str, improper: bool
) -> None:
    torch.manual_seed(7)
    graph = _graph()
    module = module_type(LAYOUT, backend, mmax=2, cutoff=1.5).double()
    features = torch.randn(graph.num_nodes, LAYOUT.dimension, dtype=torch.float64)
    rotation = o3.rand_matrix(dtype=torch.float64)
    if improper:
        rotation = -rotation
    transformed_graph = build_periodic_graph(
        graph.positions @ rotation.T,
        graph.cell[0] @ rotation.T,
        graph.atomic_numbers,
        cutoff=1.5,
    )
    representation = _representation(LAYOUT, rotation)
    output = module(features, graph)
    transformed = module(features @ representation.T, transformed_graph)
    assert torch.allclose(transformed, output @ representation.T, atol=8e-8, rtol=8e-8)


@pytest.mark.parametrize("backend", ["full_o3", "o2_tp"])
def test_adaptation_empty_edge_fallback_and_gradients(backend: str) -> None:
    graph = build_periodic_graph(
        torch.zeros(1, 3, dtype=torch.float64),
        4 * torch.eye(3, dtype=torch.float64),
        torch.tensor([1]),
        cutoff=0.5,
    )
    module = O3Adaptation(LAYOUT, backend, cutoff=0.5).double()
    features = torch.randn(1, LAYOUT.dimension, dtype=torch.float64, requires_grad=True)
    output = module(features, graph)
    assert output.shape == features.shape
    output.square().sum().backward()
    assert features.grad is not None and torch.isfinite(features.grad).all()
    assert module.block.tensor_product.path_count > 0


def test_a1_and_full_pg_experts_are_equivariant_for_all_groups() -> None:
    torch.manual_seed(19)
    registry = PointGroupRegistry()
    features = torch.randn(2, LAYOUT.dimension, dtype=torch.float64)
    for group in registry:
        source_representation = group.representation(LAYOUT)
        for expert_type in (A1PointGroupExpert, FullPointGroupExpert):
            expert = expert_type(group, LAYOUT).double()
            output = expert(features)
            for operation in range(group.order):
                transformed = expert(features @ source_representation[operation].T)
                assert torch.allclose(
                    transformed,
                    output @ source_representation[operation].T,
                    atol=8e-8,
                    rtol=8e-8,
                ), (group.symbol, expert_type.__name__)


def test_pg_experts_have_two_independent_blocks_and_full_pg_keeps_nontrivial_carriers() -> None:
    group = PointGroupRegistry()["m-3m"]
    a1 = A1PointGroupExpert(group, LAYOUT, bypass_c1=False).double()
    full = FullPointGroupExpert(group, LAYOUT, bypass_c1=False).double()
    assert len(a1.blocks) == len(full.blocks) == 2
    for expert in (a1, full):
        first = {parameter.data_ptr() for parameter in expert.blocks[0].parameters()}
        second = {parameter.data_ptr() for parameter in expert.blocks[1].parameters()}
        assert first and second and first.isdisjoint(second)

    features = torch.zeros(1, LAYOUT.dimension, dtype=torch.float64)
    features[:, 2:5] = torch.tensor([[1.0, -2.0, 0.5]])
    assert torch.allclose(a1(features)[:, 2:5], torch.zeros(1, 3, dtype=torch.float64), atol=1e-10)
    assert torch.linalg.vector_norm(full(features)[:, 2:5]) > 0
    assert any(copy.dimension > 1 for copy in full.copy_metadata)


def test_c1_bypass_preserves_contract() -> None:
    group = PointGroupRegistry()["1"]
    features = torch.randn(3, LAYOUT.dimension, dtype=torch.float64)
    for expert_type in (A1PointGroupExpert, FullPointGroupExpert):
        expert = expert_type(group, LAYOUT, bypass_c1=True).double()
        assert len(expert.blocks) == 2
        assert torch.allclose(expert(features), features, atol=2e-8, rtol=2e-8)


def test_continuous_gate_deduplicates_and_has_parent_limits() -> None:
    gate = ContinuousResidualGate(initial_sigma=0.1).double()
    residual_current = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
    residual_parent = torch.tensor(10.0, dtype=torch.float64, requires_grad=True)
    weights = gate((1, 2, 2), {1: residual_current, 2: residual_parent})
    assert tuple(weights) == (1, 2)
    assert torch.allclose(torch.stack(tuple(weights.values())).sum(), torch.tensor(1.0, dtype=torch.float64))
    assert weights[1] > 0.999
    weights[1].backward()
    assert gate.log_sigma.grad is not None and torch.isfinite(gate.log_sigma.grad)
    assert residual_parent.grad is not None and torch.isfinite(residual_parent.grad)


def test_hierarchical_fusion_requires_common_o3_space() -> None:
    first = torch.randn(2, LAYOUT.dimension)
    second = torch.randn(2, LAYOUT.dimension)
    fused = hierarchical_fusion(
        {1: first, 2: second},
        {1: torch.tensor(0.25), 2: torch.tensor(0.75)},
        LAYOUT,
    )
    assert torch.allclose(fused, 0.25 * first + 0.75 * second)
    with pytest.raises(ValueError, match="layout"):
        hierarchical_fusion(
            {1: first, 2: second[:, :-1]},
            {1: torch.tensor(0.5), 2: torch.tensor(0.5)},
            LAYOUT,
        )


def test_active_nonbackbone_parameter_budget_is_well_below_five_million() -> None:
    a1_layout = IrrepLayout(
        tuple(
            IrrepTerm(multiplicity, degree, "e" if degree % 2 == 0 else "o", f"l{degree}")
            for degree, multiplicity in enumerate((8, 2, 2, 2, 2))
        )
    )
    full_layout = IrrepLayout(
        tuple(
            IrrepTerm(multiplicity, degree, "e" if degree % 2 == 0 else "o", f"l{degree}")
            for degree, multiplicity in enumerate((4, 1, 1, 1, 1))
        )
    )
    group = PointGroupRegistry()["2/m"]
    modules = (
        A1PointGroupExpert(group, a1_layout, bypass_c1=False),
        FullPointGroupExpert(group, full_layout, bypass_c1=False),
        RoutedO3Expert(full_layout, "o2_tp"),
    )
    assert sum(sum(parameter.numel() for parameter in module.parameters()) for module in modules) < 5_000_000
