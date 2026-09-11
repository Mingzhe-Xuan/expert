from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from types import MappingProxyType
from typing import Mapping

import torch


@dataclass(frozen=True, slots=True)
class IrrepTerm:
    """One labelled block of repeated real O(3) irreducible representations."""

    multiplicity: int
    degree: int
    parity: str
    copy_label: str

    def __post_init__(self) -> None:
        if self.multiplicity < 1:
            raise ValueError("multiplicity must be positive")
        if self.degree < 0:
            raise ValueError("degree must be non-negative")
        if self.parity not in {"e", "o"}:
            raise ValueError("parity must be 'e' or 'o'")
        if not self.copy_label:
            raise ValueError("copy_label must be non-empty")

    @property
    def dimension(self) -> int:
        return self.multiplicity * (2 * self.degree + 1)

    @property
    def natural_parity(self) -> bool:
        return self.parity == ("e" if self.degree % 2 == 0 else "o")


@dataclass(frozen=True, slots=True)
class IrrepLayout:
    """Ordered immutable carrier layout; order is part of the model contract."""

    terms: tuple[IrrepTerm, ...]
    component_order: str = "e3nn_real_y_lm_m_minus_l_to_l"

    def __post_init__(self) -> None:
        if not self.terms:
            raise ValueError("an irrep layout cannot be empty")
        labels = [term.copy_label for term in self.terms]
        if len(labels) != len(set(labels)):
            raise ValueError("copy_label values must be unique within a layout")
        if not self.component_order:
            raise ValueError("component_order must be explicit")

    @property
    def dimension(self) -> int:
        return sum(term.dimension for term in self.terms)

    def to_spec(self) -> tuple[dict[str, object], ...]:
        return tuple(asdict(term) for term in self.terms)


@dataclass(frozen=True, slots=True)
class ConventionMetadata:
    """Versioned numerical conventions embedded in every downstream checkpoint."""

    schema_version: int
    real_harmonic_convention: str
    cg_convention: str
    path_ordering: str
    copy_ordering: str
    subduction_checksum: str

    def __post_init__(self) -> None:
        if self.schema_version < 1:
            raise ValueError("schema_version must be positive")
        for name in (
            "real_harmonic_convention",
            "cg_convention",
            "path_ordering",
            "copy_ordering",
        ):
            if not getattr(self, name):
                raise ValueError(f"{name} must be non-empty")
        if len(self.subduction_checksum) != 64 or any(
            char not in "0123456789abcdef" for char in self.subduction_checksum
        ):
            raise ValueError("subduction_checksum must be a lowercase SHA-256 digest")

    @property
    def checksum(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def require_compatible(self, checkpoint: Mapping[str, object]) -> None:
        actual = checkpoint.get("convention_checksum")
        if actual != self.checksum:
            raise ValueError(
                "checkpoint convention mismatch: "
                f"expected {self.checksum}, received {actual!r}"
            )


@dataclass(frozen=True, slots=True)
class O3FeatureBatch:
    """Backbone features with an immutable O(3) layout and node mapping."""

    node_features: torch.Tensor
    node_layout: IrrepLayout
    node_batch: torch.Tensor
    edge_features: torch.Tensor | None = None
    edge_layout: IrrepLayout | None = None
    edge_geometry: Mapping[str, torch.Tensor] | None = None

    def __post_init__(self) -> None:
        if self.node_features.ndim != 2:
            raise ValueError("node_features must have shape [num_nodes, feature_dim]")
        if self.node_features.shape[1] != self.node_layout.dimension:
            raise ValueError("node feature width does not match node_layout")
        if self.node_batch.shape != (self.node_features.shape[0],):
            raise ValueError("node_batch must have shape [num_nodes]")
        if self.node_batch.dtype != torch.long:
            raise TypeError("node_batch must use torch.long")
        if self.node_batch.device != self.node_features.device:
            raise ValueError("node_batch and node_features must share a device")
        if (self.edge_features is None) != (self.edge_layout is None):
            raise ValueError("edge_features and edge_layout must be supplied together")
        if self.edge_features is not None:
            if self.edge_features.ndim != 2:
                raise ValueError("edge_features must have shape [num_edges, feature_dim]")
            if self.edge_features.shape[1] != self.edge_layout.dimension:
                raise ValueError("edge feature width does not match edge_layout")
        geometry = {} if self.edge_geometry is None else dict(self.edge_geometry)
        object.__setattr__(self, "edge_geometry", MappingProxyType(geometry))
