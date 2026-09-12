from __future__ import annotations

import hashlib
import json
from dataclasses import replace

import pytest
import torch
from torch import nn

from src.backbones import (
    BACKBONE_FAMILIES,
    BackboneResourceRegistry,
    DPA4BackboneAdapter,
    DPA4_SO3_LAYOUT,
    EQUIFORMER_SO3_LAYOUT,
    EquiformerV2BackboneAdapter,
    GRACEBackboneAdapter,
    GRACE_SOURCE_LAYOUT,
    InversionPairedReynolds,
    MACEBackboneAdapter,
    O3InterfaceProjector,
    SO3FeatureBatch,
    SO3Layout,
    SO3Term,
    invert_periodic_graph,
    flatten_dpa4_latent,
    flatten_equiformer_embedding,
    convert_grace_aa,
    irrep_layout_from_e3nn,
    require_distribution_version,
)
from src.graphs import build_periodic_graph
from src.irreps import IrrepLayout, IrrepTerm
from src.symmetry.registry import _layout_irreps
from src.cli.mace_smoke import FLOAT32_EQUIVARIANCE_TOLERANCE, SMOKE_LAYOUT, _silicon_graph
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
        record.pop("config", None)
        record.pop("config_sha256", None)
        record.pop("artifacts", None)
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

    source["selected_backbones"]["mace"]["local_path"] = "checkpoint.bin"
    source["selected_backbones"]["dpa4"]["config"] = "../../escape.json"
    source["selected_backbones"]["dpa4"]["config_sha256"] = digest
    manifest.write_text(json.dumps(source), encoding="utf-8")
    with pytest.raises(ValueError, match="config path escapes"):
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


def test_mace_runtime_layout_and_resource_gate_precede_model_loading() -> None:
    require_distribution_version("mace-torch", "0.3.16")
    with pytest.raises(RuntimeError, match="incompatible"):
        require_distribution_version("mace-torch", "0.0.invalid")
    layout = irrep_layout_from_e3nn(o3.Irreps("128x0e + 128x1o"), "tap")
    assert layout.dimension == 512
    assert tuple((term.multiplicity, term.degree, term.parity) for term in layout.terms) == (
        (128, 0, "e"),
        (128, 1, "o"),
    )

    unavailable = replace(
        BackboneResourceRegistry()["mace"], status="blocked_for_test"
    )

    class UnavailableRegistry:
        def __getitem__(self, family):
            assert family == "mace"
            return unavailable

    with pytest.raises(FileNotFoundError, match="unavailable"):
        MACEBackboneAdapter(
            IrrepLayout((IrrepTerm(1, 0, "e", "target"),)),
            registry=UnavailableRegistry(),
        )


def test_mace_slurm_smoke_contract_is_small_and_predeclares_tolerance() -> None:
    graph = _silicon_graph("cpu")
    assert graph.num_nodes == 2 and graph.num_edges > 0
    assert graph.cutoff == 6.0 and SMOKE_LAYOUT.dimension == 10
    assert FLOAT32_EQUIVARIANCE_TOLERANCE == 3.0e-4


def test_dpa4_latent_flattening_is_copy_major_with_frozen_layout() -> None:
    latent = torch.arange(2 * 25 * 64, dtype=torch.float32).reshape(2, 25, 1, 64)
    flattened = flatten_dpa4_latent(latent)
    assert flattened.shape == (2, 1600)
    assert DPA4_SO3_LAYOUT.dimension == 1600
    assert tuple((term.multiplicity, term.degree) for term in DPA4_SO3_LAYOUT.terms) == (
        (64, 0), (64, 1), (64, 2), (64, 3), (64, 4)
    )
    # l=1 starts after the scalar block. Each channel/copy owns contiguous m=-1,0,1.
    expected_l1_copy_7 = latent[:, 1:4, 0, 7]
    assert torch.equal(flattened[:, 64 + 7 * 3 : 64 + 8 * 3], expected_l1_copy_7)
    # l=4 begins after 64 * (1 + 3 + 5 + 7) coefficients.
    l4_offset = 64 * 16
    expected_l4_copy_63 = latent[:, 16:25, 0, 63]
    assert torch.equal(flattened[:, l4_offset + 63 * 9 : l4_offset + 64 * 9], expected_l4_copy_63)


def test_dpa4_neighbor_schema_is_normalized_to_model_device_and_dtype() -> None:
    from src.backbones.dpa4 import _neighbor_schema_on_device

    schema = {
        "edge_index": torch.tensor([[0], [1]], dtype=torch.int32),
        "edge_vec": torch.ones((1, 3), dtype=torch.float64),
        "edge_mask": torch.tensor([1], dtype=torch.int8),
    }
    edge_index, edge_vectors, edge_mask = _neighbor_schema_on_device(
        schema,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    assert edge_index.dtype == torch.long
    assert edge_vectors.dtype == torch.float32
    assert edge_mask.dtype == torch.bool
    assert edge_index.device == edge_vectors.device == edge_mask.device


def test_dpa4_resource_gate_precedes_deepmd_import_and_runtime_is_pinned() -> None:
    registry = BackboneResourceRegistry()
    assert registry["dpa4"].required_runtime == "deepmd-kit==3.2.0; torch==2.11.*; CUDA==12.8"
    assert registry["dpa4"].config_path is not None
    unavailable = replace(registry["dpa4"], status="blocked_for_test")

    class UnavailableRegistry:
        def __getitem__(self, family):
            assert family == "dpa4"
            return unavailable

    with pytest.raises(FileNotFoundError, match="unavailable"):
        DPA4BackboneAdapter(
            IrrepLayout((IrrepTerm(1, 0, "e", "target"),)),
            registry=UnavailableRegistry(),
        )


def test_grace_aa_layout_and_copy_major_conversion_are_explicit() -> None:
    assert GRACE_SOURCE_LAYOUT.dimension == 4512
    counts = {
        degree: sum(term.multiplicity for term in GRACE_SOURCE_LAYOUT.terms if term.degree == degree)
        for degree in range(5)
    }
    assert counts == {0: 160, 1: 128, 2: 224, 3: 160, 4: 192}
    assert all(term.natural_parity for term in GRACE_SOURCE_LAYOUT.terms)

    raw = torch.arange(2 * 32 * 141, dtype=torch.float64).reshape(2, 32, 141)
    identity = tuple(torch.eye(2 * degree + 1, dtype=torch.float64) for degree in range(5))
    converted = convert_grace_aa(raw, identity)
    assert converted.shape == (2, 4512)
    # First l=1 history begins after five scalar-history blocks.
    assert torch.equal(converted[:, 5 * 32 : 5 * 32 + 3], raw[:, 0, 5:8])
    assert torch.equal(converted[:, 5 * 32 + 31 * 3 : 5 * 32 + 32 * 3], raw[:, 31, 5:8])
    with pytest.raises(ValueError, match="shape"):
        convert_grace_aa(raw[:, :, :-1], identity)


def test_grace_manifest_corrects_scalar_rho_tap_and_resource_gate_precedes_runtime() -> None:
    registry = BackboneResourceRegistry()
    resource = registry["grace"]
    assert resource.feature_tap.startswith("instruction:AA,")
    assert "4512" in resource.feature_layout
    assert tuple(artifact.filename for artifact in resource.artifacts) == (
        "model.yaml", "checkpoint.index", "checkpoint.data-00000-of-00001"
    )
    unavailable = replace(resource, status="blocked_for_test")

    class UnavailableRegistry:
        def __getitem__(self, family):
            assert family == "grace"
            return unavailable

    with pytest.raises(FileNotFoundError, match="unavailable"):
        GRACEBackboneAdapter(
            IrrepLayout((IrrepTerm(1, 0, "e", "target"),)),
            registry=UnavailableRegistry(),
        )


def test_grace_tensorflow_cpu_fallback_is_applied_before_runtime_ops(monkeypatch) -> None:
    from src.backbones.grace import (
        GRACE_GEOMETRY_FLOAT_DTYPE,
        _configure_tensorflow_device,
    )

    assert GRACE_GEOMETRY_FLOAT_DTYPE == "float64"

    calls = []

    class Config:
        @staticmethod
        def set_visible_devices(devices, device_type):
            calls.append((devices, device_type))

    class TensorFlow:
        config = Config()

    monkeypatch.setenv("EXPERT_GRACE_TF_DEVICE", "cpu")
    assert _configure_tensorflow_device(TensorFlow()) == "cpu"
    assert calls == [([], "GPU")]

    monkeypatch.setenv("EXPERT_GRACE_TF_DEVICE", "cuda")
    with pytest.raises(ValueError, match="must be 'auto' or 'cpu'"):
        _configure_tensorflow_device(TensorFlow())


def test_equiformerv2_embedding_flattening_is_copy_major_with_frozen_layout() -> None:
    embedding = torch.arange(2 * 25 * 128, dtype=torch.float32).reshape(2, 25, 128)
    flattened = flatten_equiformer_embedding(embedding)
    assert flattened.shape == (2, 3200)
    assert EQUIFORMER_SO3_LAYOUT.dimension == 3200
    assert tuple(
        (term.multiplicity, term.degree) for term in EQUIFORMER_SO3_LAYOUT.terms
    ) == ((128, 0), (128, 1), (128, 2), (128, 3), (128, 4))
    assert torch.equal(flattened[:, 128 + 19 * 3 : 128 + 20 * 3], embedding[:, 1:4, 19])
    l4_offset = 128 * 16
    assert torch.equal(
        flattened[:, l4_offset + 127 * 9 : l4_offset + 128 * 9],
        embedding[:, 16:25, 127],
    )
    with pytest.raises(ValueError, match="shape"):
        flatten_equiformer_embedding(embedding[:, :-1])


def test_equiformerv2_exact_gated_resource_fails_before_fairchem_import() -> None:
    registry = BackboneResourceRegistry()
    resource = registry["equiformerv2"]
    assert resource.repository == "facebook/OMAT24"
    assert resource.revision == "8a5a78c7ba7b250a17e85fe85943c4608499d895"
    assert resource.filename == "eqV2_31M_mp.pt"
    assert resource.required_runtime == "fairchem-core==1.10.0"
    assert resource.status == "blocked_on_huggingface_gated_access"
    unavailable = replace(resource, status="blocked_for_test")

    class UnavailableRegistry:
        def __getitem__(self, family):
            assert family == "equiformerv2"
            return unavailable

    with pytest.raises(FileNotFoundError, match="unavailable"):
        EquiformerV2BackboneAdapter(
            IrrepLayout((IrrepTerm(1, 0, "e", "target"),)),
            registry=UnavailableRegistry(),
        )
