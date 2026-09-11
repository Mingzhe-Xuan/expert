from __future__ import annotations

import hashlib
import json

import pytest
import torch
from torch import nn

from src.backbones import (
    BACKBONE_FAMILIES,
    BackboneResourceRegistry,
    InversionPairedReynolds,
    O3InterfaceProjector,
    SO3FeatureBatch,
    SO3Layout,
    SO3Term,
    invert_periodic_graph,
)
from src.graphs import build_periodic_graph
from src.irreps import IrrepLayout, IrrepTerm
from src.symmetry.registry import _layout_irreps
from e3nn import o3


SO3_LAYOUT = SO3Layout(
    (
        SO3Term(1, 0, "scalar"),
        SO3Term(1, 1, "vector"),
        SO3Term(1, 2, "quadrupole"),
    )
)


def _graph(dtype: torch.dtype = torch.float64):
    cell = torch.tensor(
        [[2.3, 0.0, 0.0], [0.2, 2.5, 0.0], [0.1, 0.3, 2.7]], dtype=dtype
    )
    fractional = torch.tensor([[0.13, 0.27, 0.38], [0.61, 0.74, 0.82]], dtype=dtype)
    return build_periodic_graph(fractional @ cell, cell, torch.tensor([6, 8]), 1.6)


class AnalyticSO3Extractor(nn.Module):
    """Exact fixture extractor; real-checkpoint acceptance is a separate Guqq suite."""

    def __init__(self) -> None:
        super().__init__()
        self.calls = 0
        self.sentinel = nn.Parameter(torch.ones(()))

    def forward(self, graph) -> SO3FeatureBatch:
        self.calls += 1
        cell = graph.cell[graph.node_batch]
        fractional = torch.linalg.solve(cell.transpose(-1, -2), graph.positions.unsqueeze(-1)).squeeze(-1)
        vector = torch.sin(2 * torch.pi * fractional)[:, None, :] @ cell
        vector = vector.squeeze(1)
        scalar = graph.atomic_numbers.to(graph.positions.dtype).unsqueeze(-1)
        quadrupole = o3.spherical_harmonics(
            "1x2e", vector, normalize=True, normalization="component"
        )
        return SO3FeatureBatch(
            torch.cat((scalar, vector, quadrupole), dim=-1), SO3_LAYOUT, graph.node_batch
        )


def test_frozen_manifest_has_four_canonical_resources_and_gated_fails_closed() -> None:
    registry = BackboneResourceRegistry()
    assert tuple(resource.family for resource in registry) == BACKBONE_FAMILIES
    assert registry["mace"].sha256 == "2f2be696351ac9e94fbe01cdfb6f017679acdbd2db7645209ef55fec9826b012"
    assert registry["mace"].feature_layout == "128x0e + 128x1o"
    assert registry["grace"].parity_policy.startswith("explicit_o3")
    assert registry["dpa4"].parity_policy == "inversion_paired_reynolds"
    with pytest.raises(FileNotFoundError, match="gated|unavailable|status"):
        registry["equiformerv2"].verify()


def test_resource_verification_rejects_size_checksum_and_path_escape(tmp_path) -> None:
    source = json.loads(BackboneResourceRegistry().path.read_text(encoding="utf-8"))
    checkpoint = tmp_path / "checkpoint.bin"
    checkpoint.write_bytes(b"verified-checkpoint")
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    for record in source["selected_backbones"].values():
        record.update(
            local_path="checkpoint.bin",
            size_bytes=checkpoint.stat().st_size,
            sha256=digest,
            status="downloaded_and_checksum_verified",
        )
    manifest = tmp_path / "backbones.json"
    manifest.write_text(json.dumps(source), encoding="utf-8")
    registry = BackboneResourceRegistry(manifest, workspace_root=tmp_path)
    assert all(resource.verify() == checkpoint for resource in registry)

    source["selected_backbones"]["mace"]["sha256"] = "0" * 64
    manifest.write_text(json.dumps(source), encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256"):
        BackboneResourceRegistry(manifest, workspace_root=tmp_path)["mace"].verify()
    source["selected_backbones"]["mace"]["local_path"] = "../escape.model"
    manifest.write_text(json.dumps(source), encoding="utf-8")
    with pytest.raises(ValueError, match="escapes"):
        BackboneResourceRegistry(manifest, workspace_root=tmp_path)


def test_periodic_inversion_keeps_cell_species_order_and_is_an_involution() -> None:
    graph = _graph()
    inverted = invert_periodic_graph(graph)
    restored = invert_periodic_graph(inverted)
    assert torch.equal(inverted.cell, graph.cell)
    assert torch.equal(inverted.atomic_numbers, graph.atomic_numbers)
    assert torch.equal(inverted.node_batch, graph.node_batch)
    fractional = torch.linalg.solve(graph.cell[0].T, graph.positions.T).T.remainder(1.0)
    restored_fractional = torch.linalg.solve(restored.cell[0].T, restored.positions.T).T.remainder(1.0)
    assert torch.allclose(restored_fractional, fractional, atol=2e-12)


def test_two_call_reynolds_wrapper_is_o3_equivariant_and_frozen() -> None:
    torch.manual_seed(81)
    graph = _graph()
    extractor = AnalyticSO3Extractor().double()
    wrapper = InversionPairedReynolds(extractor, SO3_LAYOUT).double()
    wrapper.train()
    assert not wrapper.extractor.training
    assert all(not parameter.requires_grad for parameter in extractor.parameters())
    baseline = wrapper(graph)
    assert extractor.calls == 2
    assert tuple((term.degree, term.parity) for term in baseline.node_layout.terms) == (
        (0, "e"), (0, "o"), (1, "e"), (1, "o"), (2, "e"), (2, "o")
    )
    assert torch.allclose(baseline.node_features[:, 1:2], torch.zeros_like(baseline.node_features[:, 1:2]), atol=1e-12)
    assert torch.allclose(baseline.node_features[:, 2:5], torch.zeros_like(baseline.node_features[:, 2:5]), atol=1e-12)
    assert torch.allclose(baseline.node_features[:, -5:], torch.zeros_like(baseline.node_features[:, -5:]), atol=1e-12)

    for rotation in (o3.rand_matrix(dtype=torch.float64), -o3.rand_matrix(dtype=torch.float64)):
        transformed = build_periodic_graph(
            graph.positions @ rotation.T,
            graph.cell[0] @ rotation.T,
            graph.atomic_numbers,
            graph.cutoff,
        )
        actual = wrapper(transformed).node_features
        representation = _layout_irreps(wrapper.output_layout).D_from_matrix(rotation)
        assert torch.allclose(actual, baseline.node_features @ representation.T, atol=2e-8, rtol=2e-8)


def test_o3_interface_projection_preserves_mapping_and_backpropagates() -> None:
    graph = _graph()
    wrapper = InversionPairedReynolds(AnalyticSO3Extractor().double(), SO3_LAYOUT).double()
    source = wrapper(graph)
    target = IrrepLayout(
        (IrrepTerm(2, 0, "e", "hidden_scalar"), IrrepTerm(1, 1, "o", "hidden_vector"))
    )
    projector = O3InterfaceProjector(wrapper.output_layout, target).double()
    output = projector(source)
    assert output.node_layout == target
    assert torch.equal(output.node_batch, graph.node_batch)
    assert set(output.edge_geometry) == {
        "edge_index", "cell_shifts", "edge_vectors", "edge_distances"
    }
    output.node_features.square().sum().backward()
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in projector.parameters())
    assert all(parameter.grad is None for parameter in wrapper.extractor.parameters())
