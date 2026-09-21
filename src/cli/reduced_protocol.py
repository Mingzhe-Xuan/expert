from __future__ import annotations

from collections.abc import Sequence

from ..data import IndependentTensorDataset
from ..symmetry import PointGroupRegistry
from ..symmetry.registry import canonical_point_group_symbol


REDUCED_POINT_GROUPS = ("2/m", "mm2", "mmm", "4/mmm", "-3m", "-43m", "m-3m")


def canonical_expert_point_groups(
    *example_splits: Sequence[object],
) -> tuple[str, ...]:
    """Return the deterministic expert domain used by cached canonical symmetries."""

    if not example_splits or any(not examples for examples in example_splits):
        raise ValueError("canonical expert discovery requires every split to be non-empty")
    observed = {
        canonical_point_group_symbol(example.symmetry.current_point_group)
        for examples in example_splits
        for example in examples
    }
    registry = PointGroupRegistry()
    for examples in example_splits:
        for example in examples:
            if getattr(example, "parent_dag", None) is None:
                continue
            for embedding in example.parent_dag.embeddings:
                observed.add(registry[embedding.parent_point_group_number].symbol)
    return tuple(
        group.symbol for group in registry if group.symbol in observed
    )


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
