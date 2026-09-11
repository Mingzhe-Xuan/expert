from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path

import torch
from e3nn import o3

from ..backbones import DPA4BackboneAdapter
from ..graphs import build_periodic_graph
from ..irreps import IrrepLayout, IrrepTerm
from ..symmetry.registry import _layout_irreps


SMOKE_LAYOUT = IrrepLayout(
    (IrrepTerm(4, 0, "e", "smoke_scalar"), IrrepTerm(2, 1, "o", "smoke_vector"))
)
FLOAT32_EQUIVARIANCE_TOLERANCE = 1.0e-3


def _silicon_graph(device: str):
    cell = 5.43 * torch.tensor(
        [[0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]],
        dtype=torch.float32,
        device=device,
    )
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]], dtype=torch.float32, device=device
    ) @ cell
    return build_periodic_graph(positions, cell, torch.tensor([14, 14], device=device), 6.0)


def run(device: str) -> dict[str, object]:
    torch.manual_seed(20260911)
    graph = _silicon_graph(device)
    adapter = DPA4BackboneAdapter(SMOKE_LAYOUT, device=device)
    adapter.train()
    baseline = adapter(graph)
    representation = _layout_irreps(SMOKE_LAYOUT)
    errors = {}
    for name, rotation in (
        ("proper", o3.rand_matrix(dtype=graph.positions.dtype, device=graph.positions.device)),
        ("improper", -o3.rand_matrix(dtype=graph.positions.dtype, device=graph.positions.device)),
    ):
        transformed = build_periodic_graph(
            graph.positions @ rotation.T,
            graph.cell[0] @ rotation.T,
            graph.atomic_numbers,
            graph.cutoff,
        )
        actual = adapter(transformed).node_features
        expected = baseline.node_features @ representation.D_from_matrix(rotation).T
        error = float((actual - expected).abs().max().detach().cpu())
        errors[name] = error
        if error >= FLOAT32_EQUIVARIANCE_TOLERANCE:
            raise AssertionError(f"DPA4 {name} equivariance error {error} exceeds tolerance")
    loss = baseline.node_features.square().mean()
    loss.backward()
    frozen_gradients = sum(
        parameter.numel()
        for parameter in adapter.extractor.model.parameters()
        if parameter.grad is not None
    )
    interface_gradients = [
        parameter.grad for parameter in adapter.interface.parameters() if parameter.requires_grad
    ]
    if frozen_gradients:
        raise AssertionError("frozen DPA4 checkpoint received gradients")
    if not interface_gradients or any(
        gradient is None or not torch.isfinite(gradient).all() for gradient in interface_gradients
    ):
        raise AssertionError("DPA4 interface gradients are missing or non-finite")
    geometry = baseline.edge_geometry
    return {
        "status": "passed",
        "backbone": "dpa4",
        "checkpoint": str(adapter.resource.local_path),
        "checkpoint_sha256": adapter.resource.sha256,
        "runtime": f"deepmd-kit=={importlib.metadata.version('deepmd-kit')}",
        "source_layout_dimension": DPA4_SO3_LAYOUT.dimension,
        "target_layout": list(SMOKE_LAYOUT.to_spec()),
        "nodes": int(baseline.node_features.shape[0]),
        "edges": int(geometry["edge_index"].shape[1]),
        "proper_max_abs_error": errors["proper"],
        "improper_max_abs_error": errors["improper"],
        "tolerance": FLOAT32_EQUIVARIANCE_TOLERANCE,
        "loss": float(loss.detach().cpu()),
        "frozen_backbone_parameters_with_grad": frozen_gradients,
        "trainable_interface_parameters": adapter.trainable_interface_parameter_count,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Real DPA4-Plus checkpoint feature smoke")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    result = run(arguments.device)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
