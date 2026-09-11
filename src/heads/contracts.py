from __future__ import annotations

from dataclasses import dataclass

import torch

from ..irreps import IrrepLayout, IrrepTerm


TARGET_LAYOUTS = {
    "dielectric": IrrepLayout(
        (IrrepTerm(1, 0, "e", "trace"), IrrepTerm(1, 2, "e", "traceless"))
    ),
    "elastic": IrrepLayout(
        (
            IrrepTerm(1, 0, "e", "scalar_bulk"),
            IrrepTerm(1, 0, "e", "scalar_shear"),
            IrrepTerm(1, 2, "e", "quadrupole_a"),
            IrrepTerm(1, 2, "e", "quadrupole_b"),
            IrrepTerm(1, 4, "e", "hexadecapole"),
        )
    ),
    "bec": IrrepLayout(
        (
            IrrepTerm(1, 0, "e", "isotropic"),
            IrrepTerm(1, 1, "e", "antisymmetric_axial"),
            IrrepTerm(1, 2, "e", "symmetric_traceless"),
        )
    ),
}


@dataclass(frozen=True, slots=True)
class TensorPrediction:
    raw_cartesian: torch.Tensor
    irrep_coefficients: torch.Tensor
    task: str
    scope: str
    node_batch: torch.Tensor | None = None
    canonical_cartesian: torch.Tensor | None = None
    asr_cartesian: torch.Tensor | None = None
    symmetry_control_cartesian: torch.Tensor | None = None

    def __post_init__(self) -> None:
        if self.task not in TARGET_LAYOUTS:
            raise ValueError(f"unsupported target task {self.task!r}")
        expected_scope = "node" if self.task == "bec" else "global"
        if self.scope != expected_scope:
            raise ValueError(f"{self.task} requires {expected_scope!r} scope")
        expected_rank = 4 if self.task == "elastic" else 2
        expected_shape = (3,) * expected_rank
        if self.raw_cartesian.shape[-expected_rank:] != expected_shape:
            raise ValueError(f"raw_cartesian must end in shape {expected_shape}")
        if self.irrep_coefficients.ndim != 2:
            raise ValueError("irrep_coefficients must have shape [items, coefficients]")
        if self.irrep_coefficients.shape[1] != TARGET_LAYOUTS[self.task].dimension:
            raise ValueError("irrep coefficient width does not match target layout")
        if self.raw_cartesian.shape[0] != self.irrep_coefficients.shape[0]:
            raise ValueError("Cartesian and irrep item counts must agree")
        if self.scope == "node":
            if self.node_batch is None or self.node_batch.shape != (self.raw_cartesian.shape[0],):
                raise ValueError("node-scope predictions require node_batch")
        elif self.node_batch is not None:
            raise ValueError("global predictions must not carry node_batch")
        for name in (
            "canonical_cartesian",
            "asr_cartesian",
            "symmetry_control_cartesian",
        ):
            value = getattr(self, name)
            if value is not None and value.shape != self.raw_cartesian.shape:
                raise ValueError(f"{name} must match raw_cartesian shape")
