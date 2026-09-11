from __future__ import annotations

import sys
from pathlib import Path

import torch
from e3nn import o3

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pg_tensor_model import ModelConfig
from pg_tensor_model.layers import O2ReducedMessagePassing
from pg_tensor_model.symmetry import (
    TARGET_IRREPS,
    PointGroupDAG,
    invariant_basis,
    per_l_invariant_bases,
)


ATOL = 3.0e-4
RTOL = 3.0e-4


def _transform(features: torch.Tensor, irreps: o3.Irreps, matrix: torch.Tensor) -> torch.Tensor:
    """Apply an active O(3) transformation to row-major e3nn features."""
    return features @ irreps.D_from_matrix(matrix).T


def test_shared_o2_tp_is_equivariant_under_o3_rotations_and_reflections():
    """F(g.x) = D(g)F(x) for proper and improper O(3) elements."""
    torch.manual_seed(7)
    config = ModelConfig()
    irreps = o3.Irreps(config.hidden_irreps)
    edge_irreps = o3.Irreps.spherical_harmonics(config.lmax)
    layer = O2ReducedMessagePassing(
        irreps,
        edge_irreps,
        irreps,
        config.radial_basis,
        config.radial_channels,
        config.o2_mmax,
        config.cutoff,
    )
    features = torch.randn(4, irreps.dim)
    edge_index = torch.tensor(
        [[0, 1, 2, 3, 0, 2], [1, 2, 3, 0, 2, 0]], dtype=torch.long
    )
    edge_vectors = torch.randn(edge_index.shape[1], 3)
    baseline = layer(features, edge_index, edge_vectors)

    reflection = torch.diag(torch.tensor([-1.0, 1.0, 1.0]))
    transforms = (o3.rand_matrix(), o3.rand_matrix() @ reflection)
    for transform in transforms:
        transformed_output = layer(
            _transform(features, irreps, transform),
            edge_index,
            edge_vectors @ transform.T,
        )
        expected = _transform(baseline, irreps, transform)
        torch.testing.assert_close(transformed_output, expected, atol=ATOL, rtol=RTOL)


def test_all_crystallographic_pg_a1_bases_and_target_spaces_are_invariant():
    """For all 32 PGs, D(g)B_A1 = B_A1 and P_A1^2 = P_A1."""
    config = ModelConfig()
    dag = PointGroupDAG(config.dag_path)

    for symbol in dag.symbols:
        rotations = dag.records[symbol].cartesian_rotations

        # Hidden A1 blocks retain their (l, parity, subduction-copy) provenance.
        hidden_bases = per_l_invariant_bases(config.lmax, rotations)
        for ell, basis in enumerate(hidden_bases):
            irrep = o3.Irreps([(1, o3.Irrep(ell, (-1) ** ell))])
            projector = basis @ basis.T
            torch.testing.assert_close(
                projector @ projector, projector, atol=ATOL, rtol=RTOL
            )
            for rotation in rotations:
                representation = irrep.D_from_matrix(rotation).to(basis)
                torch.testing.assert_close(
                    representation @ basis, basis, atol=ATOL, rtol=RTOL
                )

        # The learned branch head predicts coordinates in this fixed target basis.
        for target_irreps in TARGET_IRREPS.values():
            basis = invariant_basis(target_irreps, rotations)
            arbitrary_coefficients = torch.randn(3, basis.shape[1])
            fixed_output = arbitrary_coefficients @ basis.T
            for rotation in rotations:
                transformed = _transform(fixed_output, target_irreps, rotation.to(fixed_output))
                torch.testing.assert_close(
                    transformed, fixed_output, atol=ATOL, rtol=RTOL
                )
