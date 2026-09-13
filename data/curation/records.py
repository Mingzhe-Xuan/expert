from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np


DIELECTRIC_SUBTYPES = frozenset(
    {"dielectric_electronic", "dielectric_ionic", "dielectric_total"}
)
ELASTIC_SUBTYPES = frozenset({"elastic_stiffness"})
PROPERTY_SUBTYPES = DIELECTRIC_SUBTYPES | ELASTIC_SUBTYPES


@dataclass(frozen=True, slots=True)
class NormalizedRecord:
    record_id: str
    source_dataset: str
    source_id: str
    subtype: str
    lattice: np.ndarray
    fractional_positions: np.ndarray
    atomic_numbers: np.ndarray
    tensor: np.ndarray
    unit: str
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.record_id or not self.source_dataset or not self.source_id:
            raise ValueError("record and source identifiers must be non-empty")
        if self.subtype not in PROPERTY_SUBTYPES:
            raise ValueError(f"unsupported property subtype {self.subtype!r}")
        expected = (3, 3) if self.subtype in DIELECTRIC_SUBTYPES else (3, 3, 3, 3)
        if np.shape(self.tensor) != expected:
            raise ValueError(f"{self.subtype} tensor must have shape {expected}")
        if self.unit != ("dimensionless" if self.subtype in DIELECTRIC_SUBTYPES else "GPa"):
            raise ValueError("property subtype and unit are inconsistent")


@dataclass(slots=True)
class AuditResult:
    record: NormalizedRecord
    physical_valid: bool
    reasons: list[str]
    warnings: list[str]
    metrics: dict[str, float | int | str]
    clean_tensor: np.ndarray | None
    canonical_tensor: np.ndarray | None
    structure_fingerprint: str | None
    point_group: str | None
    space_group: int | None
    outlier: bool = False
    outlier_scores: dict[str, float] = field(default_factory=dict)
    duplicate_group: str | None = None
    duplicate_status: str = "unique"
    duplicate_members: list[dict[str, str]] = field(default_factory=list)
    recommended: bool = False
