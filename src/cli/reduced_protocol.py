from __future__ import annotations

from ..data import IndependentTensorDataset


REDUCED_POINT_GROUPS = ("2/m", "mm2", "mmm", "4/mmm", "-3m", "-43m", "m-3m")


def point_group_stratified_smoke_ids(
    dataset: IndependentTensorDataset,
) -> dict[str, tuple[str, ...]]:
    """Select the first record for every retained PG within every frozen split."""

    output = {}
    for split in ("train", "validation", "test"):
        selected = {}
        for sample_id in getattr(dataset.split_manifest, split):
            point_group = str(dataset.by_id(sample_id).source["point_group"])
            if point_group in REDUCED_POINT_GROUPS and point_group not in selected:
                selected[point_group] = sample_id
        missing = set(REDUCED_POINT_GROUPS) - set(selected)
        if missing:
            raise ValueError(f"{split} split lacks smoke representatives for {sorted(missing)}")
        output[split] = tuple(selected[group] for group in REDUCED_POINT_GROUPS)
    return output
