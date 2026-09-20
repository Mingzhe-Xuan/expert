from __future__ import annotations

from dataclasses import dataclass
from torch import nn

from ..backbones import (
    DPA4BackboneAdapter,
    EquiformerV2BackboneAdapter,
    GRACEBackboneAdapter,
    MACEBackboneAdapter,
)
from ..graphs import PeriodicGraph
from ..heads import TensorPrediction
from ..irreps import IrrepLayout, O3FeatureBatch
from ..symmetry import ParentDAGSpec, SymmetryRecord


BACKBONE_ADAPTERS = {
    "mace": MACEBackboneAdapter,
    "grace": GRACEBackboneAdapter,
    "dpa4": DPA4BackboneAdapter,
    "equiformerv2": EquiformerV2BackboneAdapter,
}


@dataclass(frozen=True, slots=True)
class ModelForward:
    prediction: TensorPrediction
    backbone_features: O3FeatureBatch
    graph: PeriodicGraph


def build_backbone_adapter(
    family: str, target_layout: IrrepLayout, *, device: str = "cpu"
) -> nn.Module:
    try:
        adapter_type = BACKBONE_ADAPTERS[family]
    except KeyError as exc:
        raise ValueError(f"unsupported backbone family {family!r}") from exc
    return adapter_type(target_layout, device=device)


def periodic_graph_from_backbone(features: O3FeatureBatch, *, cutoff: float) -> PeriodicGraph:
    geometry = features.edge_geometry
    required = (
        "positions",
        "cell",
        "atomic_numbers",
        "edge_index",
        "cell_shifts",
        "edge_vectors",
        "edge_distances",
    )
    missing = [name for name in required if name not in geometry]
    if missing:
        raise ValueError(f"backbone edge geometry lacks {missing}")
    edge_vectors = geometry["edge_vectors"]
    edge_distances = edge_vectors.norm(dim=-1)
    strict = edge_distances < cutoff
    if bool(strict.all()):
        edge_index = geometry["edge_index"]
        cell_shifts = geometry["cell_shifts"]
        edge_vectors = geometry["edge_vectors"]
    else:
        edge_index = geometry["edge_index"][:, strict]
        cell_shifts = geometry["cell_shifts"][strict]
        edge_vectors = edge_vectors[strict]
        edge_distances = edge_distances[strict]
    graph = PeriodicGraph(
        positions=geometry["positions"], cell=geometry["cell"],
        atomic_numbers=geometry["atomic_numbers"].long(), node_batch=features.node_batch,
        edge_index=edge_index.long(), cell_shifts=cell_shifts.long(),
        edge_vectors=edge_vectors, edge_distances=edge_distances,
        cutoff=cutoff, boundary_convention="backbone_native_filtered_distance_lt_cutoff",
    )
    if graph.num_nodes != features.node_features.shape[0]:
        raise ValueError("backbone graph node count disagrees with tapped features")
    return graph


class BackboneTensorModel(nn.Module):
    def __init__(self, adapter: nn.Module, downstream: nn.Module, *, cutoff: float) -> None:
        super().__init__()
        if cutoff <= 0:
            raise ValueError("backbone cutoff must be positive")
        self.adapter, self.downstream, self.cutoff = adapter, downstream, float(cutoff)

    def forward_with_graph(
        self,
        graph: PeriodicGraph,
        symmetries: tuple[SymmetryRecord, ...],
        *,
        parent_dags: tuple[ParentDAGSpec | None, ...] | None = None,
        parent_residuals=None,
        point_group_numbers=None,
    ) -> ModelForward:
        features = self.adapter(graph)
        used_graph = periodic_graph_from_backbone(features, cutoff=self.cutoff)
        prediction = self.downstream(
            features,
            used_graph,
            symmetries,
            parent_dags=parent_dags,
            parent_residuals=parent_residuals,
            point_group_numbers=point_group_numbers,
        )
        return ModelForward(prediction, features, used_graph)

    def forward(
        self,
        graph: PeriodicGraph,
        symmetries: tuple[SymmetryRecord, ...],
        *,
        parent_dags: tuple[ParentDAGSpec | None, ...] | None = None,
        parent_residuals=None,
        point_group_numbers=None,
    ) -> TensorPrediction:
        return self.forward_with_graph(
            graph,
            symmetries,
            parent_dags=parent_dags,
            parent_residuals=parent_residuals,
            point_group_numbers=point_group_numbers,
        ).prediction
