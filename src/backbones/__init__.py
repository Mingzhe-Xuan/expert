"""Frozen pretrained backbone resources and O(3) feature-interface wrappers."""

from .parity import (
    InversionPairedReynolds,
    O3InterfaceProjector,
    SO3FeatureBatch,
    SO3Layout,
    SO3Term,
    invert_periodic_graph,
)
from .mace import MACEBackboneAdapter, irrep_layout_from_e3nn
from .dpa4 import (
    DPA4BackboneAdapter,
    DPA4_SO3_LAYOUT,
    flatten_dpa4_latent,
)
from .resources import (
    BACKBONE_FAMILIES,
    BackboneResource,
    BackboneResourceRegistry,
    require_distribution_version,
)

__all__ = [
    "BACKBONE_FAMILIES",
    "BackboneResource",
    "BackboneResourceRegistry",
    "DPA4BackboneAdapter",
    "DPA4_SO3_LAYOUT",
    "InversionPairedReynolds",
    "MACEBackboneAdapter",
    "O3InterfaceProjector",
    "SO3FeatureBatch",
    "SO3Layout",
    "SO3Term",
    "invert_periodic_graph",
    "flatten_dpa4_latent",
    "irrep_layout_from_e3nn",
    "require_distribution_version",
]
