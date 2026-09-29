"""Pinned official GMTNet dielectric-model adapter."""

from .attention import build_gmtnet, configure_equivariant_attention

from .runner import (
    GMTNET_OFFICIAL_COMMIT,
    GMTNetConfig,
    dpa4_invariant_node_embedding,
    run_gmtnet_benchmark,
)

__all__ = [
    "build_gmtnet",
    "configure_equivariant_attention",
    "GMTNET_OFFICIAL_COMMIT",
    "GMTNetConfig",
    "dpa4_invariant_node_embedding",
    "run_gmtnet_benchmark",
]
