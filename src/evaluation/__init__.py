"""Evaluation reports and versioned acceptance fixtures."""

from .fixtures import (
    classify_candidate,
    select_point_group_fixtures,
    validate_point_group_fixture_manifest,
    write_point_group_fixture_manifest,
)
from .efficiency import profile_model_efficiency
from .pg_smoke import (
    FLOAT32_TOLERANCE,
    audit_backbone_graph_automorphism,
    point_group_smoke_schedule,
    run_point_group_smoke,
)

__all__ = [
    "classify_candidate",
    "FLOAT32_TOLERANCE",
    "audit_backbone_graph_automorphism",
    "point_group_smoke_schedule",
    "run_point_group_smoke",
    "select_point_group_fixtures",
    "validate_point_group_fixture_manifest",
    "write_point_group_fixture_manifest",
    "profile_model_efficiency",
]
