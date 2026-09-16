from __future__ import annotations

import torch
from torch import nn

from ..backbones import O3InterfaceProjector
from ..configs import ArchitectureConfig
from ..experts import PointGroupTensorModel, default_hidden_layout
from ..features import CGCNN_FEATURE_DIMENSION, CGCNN_SOURCE_LAYOUT
from ..irreps import IrrepLayout, IrrepTerm, O3FeatureBatch


GMTNET_EMBEDDING_DIMENSION = 128
GMTNET_EMBEDDED_LAYOUT = IrrepLayout(
    (IrrepTerm(GMTNET_EMBEDDING_DIMENSION, 0, "e", "gmtnet_atom_embedding"),)
)


class CGCNNFeatureTensorModel(nn.Module):
    """Full-PG tensor model with GMTNet's 92 -> 128 scalar atom embedding."""

    def __init__(
        self,
        architecture: ArchitectureConfig,
        task: str,
        expert_point_groups: tuple[str, ...],
    ) -> None:
        super().__init__()
        if architecture.branch != "B+A+PGE+R" or architecture.pg_hidden_mode != "full_pg":
            raise ValueError("CGCNN feature branch requires B+A+PGE+R/full_pg")
        hidden_layout = default_hidden_layout(architecture)
        self.atom_embedding = nn.Linear(
            CGCNN_FEATURE_DIMENSION, GMTNET_EMBEDDING_DIMENSION
        )
        self.interface = O3InterfaceProjector(GMTNET_EMBEDDED_LAYOUT, hidden_layout)
        self.downstream = PointGroupTensorModel(
            architecture,
            task,
            hidden_layout=hidden_layout,
            expert_point_groups=expert_point_groups,
        )

    def forward(self, features, graph, symmetries):
        if features.node_layout != CGCNN_SOURCE_LAYOUT:
            raise ValueError("CGCNN branch received a non-CGCNN source layout")
        embedded = O3FeatureBatch(
            self.atom_embedding(features.node_features),
            GMTNET_EMBEDDED_LAYOUT,
            features.node_batch,
        )
        return self.downstream(self.interface(embedded), graph, symmetries)
