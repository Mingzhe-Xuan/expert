from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from parameter_report import build_report
from pg_tensor_model import CrystalInput, ModelConfig, PointGroupTensorModel


def tiny_crystal(model: PointGroupTensorModel) -> CrystalInput:
    dtype = torch.get_default_dtype()
    positions = torch.tensor([[0.0, 0.0, 0.0], [1.5, 1.5, 1.5]], dtype=dtype)
    cell = 3.0 * torch.eye(3, dtype=dtype)
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    edge_vectors = torch.stack((positions[1] - positions[0], positions[0] - positions[1]))
    features = torch.randn(2, model.hidden_irreps.dim, dtype=dtype, requires_grad=True)
    return CrystalInput(
        positions=positions,
        cell=cell,
        atomic_numbers=torch.tensor([14, 14]),
        node_features=features,
        edge_index=edge_index,
        edge_vectors=edge_vectors,
        current_point_group="m-3m",
        parent_residuals={"m-3m": 0.0},
    )


def test_forward_shapes_and_backward():
    model = PointGroupTensorModel(ModelConfig())
    crystal = tiny_crystal(model)
    dielectric = model(crystal, "dielectric")
    assert dielectric.tensor.shape == (1, 3, 3)
    assert dielectric.irrep_coefficients.shape == (1, 6)
    assert dielectric.active_point_groups == ("m-3m",)
    dielectric.tensor.square().sum().backward()
    assert crystal.node_features.grad is not None


def test_elastic_shape():
    model = PointGroupTensorModel(ModelConfig())
    output = model(tiny_crystal(model), "elastic")
    assert output.tensor.shape == (1, 3, 3, 3, 3)
    assert output.irrep_coefficients.shape == (1, 21)


def test_parameter_report_has_every_point_group():
    model = PointGroupTensorModel(ModelConfig())
    report = build_report(model)
    assert len(report["point_groups"]) == 32
    assert report["point_groups"][0]["hm_symbol"] == "1"
    assert report["point_groups"][-1]["hm_symbol"] == "m-3m"
    assert report["point_groups"][0]["pg_block_1_parameters"] == 0
    assert report["point_groups"][1]["pg_block_1_parameters"] > 0
