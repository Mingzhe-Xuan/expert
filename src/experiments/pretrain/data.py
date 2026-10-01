"""Convert audited records into the repository's tensor contracts."""
from __future__ import annotations

import torch

from ...data import IndependentTensorDataset, TrainingUnit
from ...data.contracts import SplitManifest
from ...data.datasets import _make_sample
from .protocol import SPLITS


def dataset_from_records(rows, provenance):
    unit = TrainingUnit("jarvis_tensor", provenance["task"])
    split = SplitManifest(seed=20260911, source="curated_group_8_1_1",
                          **{name: tuple(r["record_id"] for r in rows if r["split"] == name)
                             for name in SPLITS},
                          sample_to_group={r["record_id"]: r["duplicate_group"] for r in rows})
    samples = [_make_sample(
        row["record_id"], unit,
        (torch.tensor(row["lattice_angstrom"], dtype=torch.float64),
         torch.tensor(row["fractional_coordinates"], dtype=torch.float64),
         torch.tensor(row["atomic_numbers"], dtype=torch.long)),
        torch.tensor(row["tensor"], dtype=torch.float64), row["unit"],
        {"manifest_sha256": provenance["dataset_sha256"], "point_group": row["point_group"],
         "space_group": row["space_group"], "provenance": row["provenance"]}) for row in rows]
    return IndependentTensorDataset(unit, samples, split)


OFFICIAL_PAIRS = ((0, 0), (1, 1), (2, 2), (0, 1), (1, 2), (0, 2))


def official_voigt(tensor):
    """Contract Cartesian stiffness to the original model's xx,yy,zz,xy,yz,xz."""
    return torch.stack([torch.stack([tensor[..., i, j, k, l] for k, l in OFFICIAL_PAIRS], -1)
                        for i, j in OFFICIAL_PAIRS], -2)


def standard_voigt(tensor):
    order = [0, 1, 2, 4, 5, 3]
    return tensor[..., order, :][..., :, order]
