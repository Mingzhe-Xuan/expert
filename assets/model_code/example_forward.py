from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pg_tensor_model import CrystalInput, PointGroupTensorModel


def main() -> None:
    model = PointGroupTensorModel()
    # A real backbone adapter should produce features with this exact irrep layout.
    node_features = torch.randn(2, model.hidden_irreps.dim)
    crystal = CrystalInput(
        positions=torch.tensor([[0.0, 0.0, 0.0], [1.5, 1.5, 1.5]]),
        cell=3.0 * torch.eye(3),
        atomic_numbers=torch.tensor([14, 14]),
        node_features=node_features,
        current_point_group="m-3m",
        parent_residuals={"m-3m": 0.0},
    )
    for task in ("dielectric", "elastic"):
        output = model(crystal, task)
        print(task, tuple(output.tensor.shape), output.active_point_groups)


if __name__ == "__main__":
    main()
