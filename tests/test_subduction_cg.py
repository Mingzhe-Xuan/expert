from __future__ import annotations

import hashlib

import torch

from src.irreps import (
    ConventionMetadata,
    IrrepLayout,
    IrrepTerm,
    build_subduction_plan,
    finite_group_intertwiners,
)
from src.symmetry import PointGroupRegistry


NATURAL_LAYOUT = IrrepLayout(
    tuple(
        IrrepTerm(1, degree, "e" if degree % 2 == 0 else "o", f"l{degree}")
        for degree in range(5)
    )
)


def test_subduction_round_trip_and_equivariance_for_all_32_groups() -> None:
    torch.manual_seed(20260911)
    registry = PointGroupRegistry()
    features = torch.randn(3, NATURAL_LAYOUT.dimension, dtype=torch.float64)
    for group in registry:
        plan = build_subduction_plan(group, NATURAL_LAYOUT)
        pg_features = plan.subduce(features)
        assert torch.allclose(plan.inverse(pg_features), features, atol=2e-8, rtol=2e-8)
        source_representation = group.representation(NATURAL_LAYOUT)
        for operation in range(group.order):
            source_transformed = features @ source_representation[operation].T
            expected = pg_features @ plan.representation[operation].T
            assert torch.allclose(
                plan.subduce(source_transformed), expected, atol=2e-8, rtol=2e-8
            ), group.symbol
        assert len(plan.checksum) == 64
        assert plan.checksum == build_subduction_plan(group, NATURAL_LAYOUT).checksum


def test_full_pg_blocks_are_closed_and_keep_o3_provenance() -> None:
    plan = build_subduction_plan(PointGroupRegistry()["m-3m"], NATURAL_LAYOUT)
    assert {copy.source_label for copy in plan.copies} == {
        "l0",
        "l1",
        "l2",
        "l3",
        "l4",
    }
    # Cubic l=2 splits into non-trivial real irreps of dimensions 2 and 3.
    assert sorted(copy.dimension for copy in plan.copies if copy.source_degree == 2) == [2, 3]
    assert any(copy.dimension > 1 and copy.source_degree > 0 for copy in plan.copies)
    for current in plan.copies:
        outside = torch.cat(
            (
                plan.representation[:, current.start : current.stop, : current.start],
                plan.representation[:, current.start : current.stop, current.stop :],
            ),
            dim=-1,
        )
        assert outside.numel() == 0 or float(outside.abs().max()) < 2e-8


def test_subduction_tolerates_backend_scale_representation_residue() -> None:
    exact_group = PointGroupRegistry()["2/m"]
    generator = torch.Generator().manual_seed(20260913)
    exact = exact_group.representation(NATURAL_LAYOUT)
    approximate = exact + 2.5e-7 * torch.randn(
        exact.shape, dtype=exact.dtype, generator=generator
    )

    class ApproximatePointGroup:
        symbol = "2/m-backend-residue"

        @staticmethod
        def representation(layout, *, dtype=torch.float64):  # noqa: ANN001
            assert layout == NATURAL_LAYOUT
            return approximate.to(dtype=dtype)

    plan = build_subduction_plan(ApproximatePointGroup(), NATURAL_LAYOUT)
    assert plan.matrix.shape == (NATURAL_LAYOUT.dimension, NATURAL_LAYOUT.dimension)
    identity = torch.eye(NATURAL_LAYOUT.dimension, dtype=torch.float64)
    assert torch.allclose(plan.matrix.T @ plan.matrix, identity, atol=5e-7, rtol=5e-7)
    transformed = approximate @ plan.matrix
    restricted = torch.einsum("ai,gab,bj->gij", plan.matrix, approximate, plan.matrix)
    reconstructed = torch.einsum("ai,gij->gaj", plan.matrix, restricted)
    assert float((transformed - reconstructed).abs().max()) < 5e-6


def test_repeated_o3_copies_remain_separately_labelled() -> None:
    layout = IrrepLayout(
        (
            IrrepTerm(2, 1, "o", "vector_family"),
            IrrepTerm(1, 0, "e", "scalar"),
        )
    )
    plan = build_subduction_plan(PointGroupRegistry()["4mm"], layout)
    vector_sources = {(copy.source_label, copy.source_copy_index) for copy in plan.copies if copy.source_degree == 1}
    assert vector_sources == {("vector_family", 0), ("vector_family", 1)}


def test_finite_group_cg_paths_intertwine_and_have_stable_order() -> None:
    registry = PointGroupRegistry()
    vector = IrrepLayout((IrrepTerm(1, 1, "o", "polar"),))
    scalar = IrrepLayout((IrrepTerm(1, 0, "e", "scalar"),))
    checksums = []
    for group in registry:
        left = group.representation(vector)
        output = group.representation(scalar)
        paths = finite_group_intertwiners(left, left, output)
        assert paths.path_count >= 1
        product_representation = torch.stack(
            [torch.kron(left[index], left[index]) for index in range(group.order)]
        )
        for matrix in paths.matrices:
            assert torch.allclose(
                product_representation.transpose(1, 2) @ matrix,
                matrix @ output.transpose(1, 2),
                atol=2e-8,
                rtol=2e-8,
            )
        flattened = paths.matrices.flatten(1)
        assert torch.allclose(
            flattened @ flattened.T,
            torch.eye(paths.path_count, dtype=torch.float64),
            atol=2e-8,
            rtol=2e-8,
        )
        checksums.append(paths.checksum)
        assert paths.checksum == finite_group_intertwiners(left, left, output).checksum
    assert len(checksums) == 32


def test_cg_explicitly_returns_zero_paths_when_forbidden() -> None:
    group = PointGroupRegistry()["-1"]
    even = IrrepLayout((IrrepTerm(1, 0, "e", "even"),))
    odd = IrrepLayout((IrrepTerm(1, 0, "o", "odd"),))
    paths = finite_group_intertwiners(
        group.representation(even),
        group.representation(even),
        group.representation(odd),
    )
    assert paths.matrices.shape == (0, 1, 1)


def test_subduction_and_cg_checksums_enter_convention_metadata() -> None:
    group = PointGroupRegistry()["m-3m"]
    plan = build_subduction_plan(group, NATURAL_LAYOUT)
    scalar = IrrepLayout((IrrepTerm(1, 0, "e", "scalar"),))
    cg = finite_group_intertwiners(
        group.representation(scalar),
        group.representation(scalar),
        group.representation(scalar),
    )
    combined = hashlib.sha256(f"{plan.checksum}:{cg.checksum}".encode()).hexdigest()
    metadata = ConventionMetadata(
        schema_version=1,
        real_harmonic_convention=NATURAL_LAYOUT.component_order,
        cg_convention="finite-group-hom-reynolds-v1",
        path_ordering="deterministic-projector-columns",
        copy_ordering="o3-provenance-then-real-irrep",
        subduction_checksum=combined,
    )
    metadata.require_compatible({"convention_checksum": metadata.checksum})
