"""Pinned official GMTNet dielectric-model adapter."""

from .runner import (
    GMTNET_OFFICIAL_COMMIT,
    GMTNetConfig,
    dpa4_invariant_node_embedding,
    run_gmtnet_benchmark,
)

__all__ = [
    "GMTNET_OFFICIAL_COMMIT",
    "GMTNetConfig",
    "dpa4_invariant_node_embedding",
    "run_gmtnet_benchmark",
]
