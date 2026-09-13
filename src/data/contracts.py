from __future__ import annotations

from dataclasses import dataclass
import random
from types import MappingProxyType
from typing import Mapping, Sequence


LEGAL_TRAINING_UNITS = (
    ("jarvis_tensor", "dielectric"),
    ("jarvis_tensor", "elastic"),
    ("dtnet", "dielectric"),
    ("matten", "elastic"),
    ("jarvis_dfpt", "bec"),
)


@dataclass(frozen=True, slots=True)
class TrainingUnit:
    dataset: str
    target: str

    def __post_init__(self) -> None:
        if (self.dataset, self.target) not in LEGAL_TRAINING_UNITS:
            raise ValueError(f"unsupported independent training unit {self.dataset} × {self.target}")

    @property
    def namespace(self) -> str:
        return f"{self.dataset}__{self.target}"


@dataclass(frozen=True, slots=True)
class SplitManifest:
    seed: int
    source: str
    train: tuple[str, ...]
    validation: tuple[str, ...]
    test: tuple[str, ...]
    sample_to_group: Mapping[str, str]

    def __post_init__(self) -> None:
        if self.source not in {"published", "seeded_8_1_1"}:
            raise ValueError("split source must be published or seeded_8_1_1")
        splits = (self.train, self.validation, self.test)
        if any(not split for split in splits):
            raise ValueError("train, validation, and test splits must all be non-empty")
        flat = [sample for split in splits for sample in split]
        if len(flat) != len(set(flat)):
            raise ValueError("sample IDs must not overlap across splits")
        mapping = dict(self.sample_to_group)
        if set(mapping) != set(flat):
            raise ValueError("sample_to_group must cover exactly all split samples")
        split_groups = [{mapping[sample] for sample in split} for split in splits]
        if any(split_groups[left] & split_groups[right] for left in range(3) for right in range(left + 1, 3)):
            raise ValueError("duplicate/material groups must not cross splits")
        object.__setattr__(self, "sample_to_group", MappingProxyType(mapping))


def make_seeded_split(
    sample_ids: Sequence[str],
    group_ids: Sequence[str],
    seed: int = 20260911,
) -> SplitManifest:
    """Create a deterministic group-preserving 8:1:1 fallback split."""

    if len(sample_ids) != len(group_ids) or len(sample_ids) != len(set(sample_ids)):
        raise ValueError("sample_ids must be unique and align one-to-one with group_ids")
    groups = sorted(set(group_ids))
    if len(groups) < 3:
        raise ValueError("at least three independent groups are required")
    random.Random(seed).shuffle(groups)
    validation_count = max(1, len(groups) // 10)
    test_count = max(1, len(groups) // 10)
    if validation_count + test_count >= len(groups):
        raise ValueError("not enough independent groups for non-empty splits")
    train_count = len(groups) - validation_count - test_count
    split_by_group = {
        **{group: "train" for group in groups[:train_count]},
        **{
            group: "validation"
            for group in groups[train_count : train_count + validation_count]
        },
        **{group: "test" for group in groups[train_count + validation_count :]},
    }
    buckets: dict[str, list[str]] = {"train": [], "validation": [], "test": []}
    mapping = dict(zip(sample_ids, group_ids))
    for sample in sample_ids:
        buckets[split_by_group[mapping[sample]]].append(sample)
    return SplitManifest(
        seed=seed,
        source="seeded_8_1_1",
        train=tuple(buckets["train"]),
        validation=tuple(buckets["validation"]),
        test=tuple(buckets["test"]),
        sample_to_group=mapping,
    )
