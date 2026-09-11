"""Run one end-to-end MACE-MP forward and backward pass."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from ase.build import bulk

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pg_tensor_model import MACEPointGroupTensorModel  # noqa: E402


def main() -> None:
    torch.manual_seed(0)
    model = MACEPointGroupTensorModel(
        checkpoint="medium-0b3", device="cpu", default_dtype="float32"
    )
    model.train()
    atoms = bulk("Si", "diamond", a=5.43)

    output = model(atoms, task="dielectric")
    loss = output.tensor.square().mean()
    loss.backward()

    backbone_parameters = list(model.backbone.mace_model.parameters())
    adapter_parameters = list(model.backbone.adapter.parameters())
    downstream_parameters = list(model.tensor_model.parameters())
    all_parameters = list(model.parameters())

    summary = {
        "checkpoint": model.backbone.checkpoint,
        "structure": "Si diamond primitive cell",
        "num_atoms": len(atoms),
        "current_point_group": output.current_point_group,
        "active_point_groups": list(output.active_point_groups),
        "output_shape": list(output.tensor.shape),
        "loss": float(loss.detach()),
        "mace_parameters": sum(p.numel() for p in backbone_parameters),
        "frozen_mace_parameters_with_grad": sum(
            p.numel() for p in backbone_parameters if p.grad is not None
        ),
        "adapter_trainable_parameters": sum(
            p.numel() for p in adapter_parameters if p.requires_grad
        ),
        "adapter_gradient_norm": float(
            torch.sqrt(
                sum(
                    p.grad.square().sum()
                    for p in adapter_parameters
                    if p.grad is not None
                )
            )
        ),
        "downstream_parameters_with_grad": sum(
            p.numel() for p in downstream_parameters if p.grad is not None
        ),
        "downstream_trainable_parameters": sum(
            p.numel() for p in downstream_parameters if p.requires_grad
        ),
        "total_trainable_parameters": sum(
            p.numel() for p in all_parameters if p.requires_grad
        ),
        "total_parameters_including_frozen_mace": sum(
            p.numel() for p in all_parameters
        ),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
