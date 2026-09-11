from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import product
from typing import Iterator


ARCHITECTURE_BRANCHES = (
    "B+R",
    "B+A+R",
    "B+A+O3E+R",
    "B+PGE+R",
    "B+A+PGE+R",
)
TP_BACKENDS = ("full_o3", "o2_tp")
PG_HIDDEN_MODES = ("a1_only", "full_pg")


@dataclass(frozen=True, slots=True)
class ArchitectureConfig:
    """One legal member of the frozen five-branch architecture matrix."""

    branch: str
    adaptation_backend: str
    o3e_backend: str
    readout_backend: str
    pg_hidden_mode: str
    o2_mmax: int = 2

    def __post_init__(self) -> None:
        if self.branch not in ARCHITECTURE_BRANCHES:
            raise ValueError(f"unsupported architecture branch {self.branch!r}")
        has_adaptation = self.branch in {"B+A+R", "B+A+O3E+R", "B+A+PGE+R"}
        has_o3e = self.branch == "B+A+O3E+R"
        has_pge = self.branch in {"B+PGE+R", "B+A+PGE+R"}
        self._require_placement("adaptation_backend", self.adaptation_backend, has_adaptation)
        self._require_placement("o3e_backend", self.o3e_backend, has_o3e)
        self._require_placement("readout_backend", self.readout_backend, True)
        if has_pge:
            if self.pg_hidden_mode not in PG_HIDDEN_MODES:
                raise ValueError("PGE branches require pg_hidden_mode=a1_only or full_pg")
        elif self.pg_hidden_mode != "none":
            raise ValueError("non-PGE branches require pg_hidden_mode='none'")
        if not 0 <= self.o2_mmax <= 4:
            raise ValueError("o2_mmax must lie in [0, 4]")

    @staticmethod
    def _require_placement(name: str, value: str, exists: bool) -> None:
        if exists and value not in TP_BACKENDS:
            raise ValueError(f"{name} requires one of {TP_BACKENDS}")
        if not exists and value != "none":
            raise ValueError(f"{name} must be 'none' when the module is absent")

    @property
    def variant_id(self) -> str:
        fields = (
            self.branch.replace("+", "_plus_").lower(),
            f"a-{self.adaptation_backend}",
            f"o3e-{self.o3e_backend}",
            f"pg-{self.pg_hidden_mode}",
            f"r-{self.readout_backend}",
        )
        return "__".join(fields)

    def to_dict(self) -> dict[str, object]:
        return {"variant_id": self.variant_id, **asdict(self)}


def _branch_configs(branch: str) -> Iterator[ArchitectureConfig]:
    adaptation = TP_BACKENDS if branch in {"B+A+R", "B+A+O3E+R", "B+A+PGE+R"} else ("none",)
    o3e = TP_BACKENDS if branch == "B+A+O3E+R" else ("none",)
    pg = PG_HIDDEN_MODES if branch in {"B+PGE+R", "B+A+PGE+R"} else ("none",)
    for adaptation_backend, o3e_backend, pg_hidden_mode, readout_backend in product(
        adaptation, o3e, pg, TP_BACKENDS
    ):
        yield ArchitectureConfig(
            branch=branch,
            adaptation_backend=adaptation_backend,
            o3e_backend=o3e_backend,
            readout_backend=readout_backend,
            pg_hidden_mode=pg_hidden_mode,
        )


def enumerate_architecture_configs() -> tuple[ArchitectureConfig, ...]:
    configs = tuple(config for branch in ARCHITECTURE_BRANCHES for config in _branch_configs(branch))
    if len(configs) != 26 or len({config.variant_id for config in configs}) != 26:
        raise RuntimeError("the frozen architecture matrix must contain 26 unique variants")
    return configs
