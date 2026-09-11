from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelConfig:
    multiplicities: tuple[int, ...] = (8, 2, 2, 2, 2)
    lmax: int = 4
    o2_mmax: int = 2
    radial_basis: int = 8
    radial_channels: int = 2
    route_width: int = 8
    pg_layers: int = 2
    cutoff: float = 6.0
    max_neighbors: int = 64
    symprec: float = 1.0e-5
    gate_sigma: float = 0.08
    residual_cell_weight: float = 0.5
    residual_position_weight: float = 0.5
    dag_path: Path = (
        Path(__file__).resolve().parents[3]
        / "docs"
        / "ref"
        / "crystallographic_point_group_subgroups.json"
    )

    def __post_init__(self) -> None:
        if self.lmax != len(self.multiplicities) - 1:
            raise ValueError("lmax must equal len(multiplicities) - 1")
        if self.pg_layers != 2:
            raise ValueError("the frozen proposal uses exactly two PG blocks")
        if not 0 <= self.o2_mmax <= self.lmax:
            raise ValueError("o2_mmax must lie in [0, lmax]")

    @property
    def hidden_irreps(self) -> str:
        terms = []
        for ell, mul in enumerate(self.multiplicities):
            parity = "e" if ell % 2 == 0 else "o"
            terms.append(f"{mul}x{ell}{parity}")
        return "+".join(terms)
