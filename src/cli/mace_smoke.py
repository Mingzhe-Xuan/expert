from __future__ import annotations

import argparse
import importlib.metadata
from pathlib import Path

import torch
from e3nn import o3

from ..backbones import MACEBackboneAdapter
from ..graphs import build_periodic_graph
from ..irreps import IrrepLayout, IrrepTerm
from ..symmetry.registry import _layout_irreps
from .reporting import run_recorded_smoke


SMOKE_LAYOUT = IrrepLayout(
    (IrrepTerm(4, 0, "e", "smoke_scalar"), IrrepTerm(2, 1, "o", "smoke_vector"))
)
FLOAT32_EQUIVARIANCE_TOLERANCE = 3.0e-4


def _silicon_graph(device: str):
    dtype = torch.float32
    cell = 5.43 * torch.tensor(
        [[0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]],
        dtype=dtype,
        device=device,
    )
    positions = torch.tensor([[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]], dtype=dtype, device=device) @ cell
    return build_periodic_graph(positions, cell, torch.tensor([14, 14], device=device), 6.0)


def run(device: str) -> dict[str, object]:
    torch.manual_seed(20260911)
    graph = _silicon_graph(device)
    adapter = MACEBackboneAdapter(SMOKE_LAYOUT, device=device)
    adapter.train()
    baseline = adapter(graph)
    representation = _layout_irreps(SMOKE_LAYOUT)
    errors = {}
    for name, rotation in (
        ("proper", o3.rand_matrix(dtype=graph.positions.dtype, device=graph.positions.device)),
        ("improper", -o3.rand_matrix(dtype=graph.positions.dtype, device=graph.positions.device)),
    ):
        transformed_graph = build_periodic_graph(
            graph.positions @ rotation.T,
            graph.cell[0] @ rotation.T,
            graph.atomic_numbers,
            graph.cutoff,
        )
        actual = adapter(transformed_graph).node_features
        expected = baseline.node_features @ representation.D_from_matrix(rotation).T
        error = float((actual - expected).abs().max().detach().cpu())
        errors[name] = error
        if error >= FLOAT32_EQUIVARIANCE_TOLERANCE:
            raise AssertionError(f"MACE {name} equivariance error {error} exceeds tolerance")
    loss = baseline.node_features.square().mean()
    loss.backward()
    backbone_gradients = sum(
        parameter.numel() for parameter in adapter.backbone.parameters() if parameter.grad is not None
    )
    interface_gradients = [
        parameter.grad for parameter in adapter.interface.parameters() if parameter.requires_grad
    ]
    if backbone_gradients != 0:
        raise AssertionError("frozen MACE checkpoint received gradients")
    if not interface_gradients or any(
        gradient is None or not torch.isfinite(gradient).all() for gradient in interface_gradients
    ):
        raise AssertionError("MACE interface projector gradients are missing or non-finite")
    geometry = baseline.edge_geometry
    return {
        "status": "passed",
        "backbone": "mace",
        "checkpoint": str(adapter.resource.local_path),
        "checkpoint_sha256": adapter.resource.sha256,
        "runtime": f"mace-torch=={importlib.metadata.version('mace-torch')}",
        "source_irreps": str(adapter.source_irreps),
        "target_layout": list(SMOKE_LAYOUT.to_spec()),
        "nodes": int(baseline.node_features.shape[0]),
        "edges": int(geometry["edge_index"].shape[1]),
        "proper_max_abs_error": errors["proper"],
        "improper_max_abs_error": errors["improper"],
        "tolerance": FLOAT32_EQUIVARIANCE_TOLERANCE,
        "loss": float(loss.detach().cpu()),
        "frozen_backbone_parameters_with_grad": backbone_gradients,
        "trainable_interface_parameters": adapter.trainable_interface_parameter_count,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Real MACE checkpoint feature smoke")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--junit", type=Path, required=True)
    arguments = parser.parse_args()
    run_recorded_smoke(
        run,
        device=arguments.device,
        output=arguments.output,
        junit=arguments.junit,
        suite_name="mace_adapter_smoke",
    )


if __name__ == "__main__":
    main()
