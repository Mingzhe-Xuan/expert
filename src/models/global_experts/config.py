from dataclasses import asdict, dataclass
import math

from ...irreps import IrrepLayout, IrrepTerm
from ...symmetry import (
    registry as _registry,
)  # initialize the existing safe e3nn loader
from e3nn import o3


def layout_from_irreps(spec: str) -> IrrepLayout:
    return IrrepLayout(
        tuple(
            IrrepTerm(mul, ir.l, "e" if ir.p == 1 else "o", f"block_{i}")
            for i, (mul, ir) in enumerate(o3.Irreps(spec))
        )
    )


@dataclass(frozen=True)
class GlobalExpertsConfig:
    expert_irreps: str = "8x0e + 2x1o + 2x2e + 2x3o + 2x4e"
    adapter_backend: str = "full_o3"
    adapter_mmax: int = 2
    adapter_edge_lmax: int = 2
    adapter_cutoff: float = 6.0
    adapter_radial_width: int = 8
    adapter_initial_logit: float = 0.0
    initial_sigma: float = 0.08
    sigma_floor: float = 1e-8
    chain_temperature: float = 1.0
    branch_initial_logit: float = -4.0
    bypass_c1: bool = True
    auxiliary_enabled: bool = True
    use_equiv_attn: bool = False
    freeze_global: bool = False

    def __post_init__(self):
        layout_from_irreps(self.expert_irreps)
        if self.adapter_backend not in ("full_o3", "o2_tp"):
            raise ValueError("unsupported Adapter backend")
        if (
            self.adapter_edge_lmax < 0
            or self.adapter_mmax < 0
            or self.adapter_radial_width < 1
        ):
            raise ValueError("invalid Adapter dimensions")
        for value in (
            self.adapter_cutoff,
            self.initial_sigma,
            self.sigma_floor,
            self.chain_temperature,
        ):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(
                    "scales, cutoff, and temperature must be positive and finite"
                )
        if self.initial_sigma <= self.sigma_floor:
            raise ValueError("initial sigma must exceed its floor")
        if not all(
            math.isfinite(v)
            for v in (self.adapter_initial_logit, self.branch_initial_logit)
        ):
            raise ValueError("initial logits must be finite")

    def metadata(self):
        return asdict(self)
