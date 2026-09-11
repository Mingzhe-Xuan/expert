from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn

from src.configs import ArchitectureConfig
from src.configs import ARCHITECTURE_BRANCHES, enumerate_architecture_configs
from src.data import TensorSample, TrainingUnit
from src.experts import PointGroupTensorModel, default_hidden_layout
from src.graphs import build_periodic_graph
from src.heads import cartesian_to_irreps
from src.irreps import O3FeatureBatch
from src.models import BackboneTensorModel, periodic_graph_from_backbone
from src.training import prepare_tensor_batch, run_five_structure_smoke


def _config():
    return ArchitectureConfig("B+R", "none", "none", "full_o3", "none")


def _samples():
    unit = TrainingUnit("jarvis_tensor", "dielectric")
    lattice = torch.tensor(
        [[7.1, 0.0, 0.0], [0.4, 7.7, 0.0], [0.2, 0.3, 8.3]], dtype=torch.float64
    )
    fractional = torch.tensor(
        [[0.07, 0.12, 0.19], [0.38, 0.51, 0.73]], dtype=torch.float64
    )
    output = []
    for index in range(5):
        target = torch.diag(torch.tensor([2.0 + index, 3.0 + index, 4.0 + index]))
        output.append(
            TensorSample(
                sample_id=f"real-schema-{index}",
                unit=unit,
                lattice=lattice,
                fractional_positions=fractional,
                atomic_numbers=torch.tensor([14, 8]),
                target_cartesian=target,
                target_coefficients=cartesian_to_irreps(target, "dielectric"),
                target_unit="dimensionless",
                source={"fixture": "orchestration-only"},
            )
        )
    return tuple(output)


class AnalyticAdapter(nn.Module):
    """Orchestration seam only; never accepted as backbone evidence."""

    def __init__(self, layout):
        super().__init__()
        self.layout = layout
        self.scalar = nn.Linear(1, layout.terms[0].multiplicity)

    def forward(self, graph):
        features = graph.positions.new_zeros((graph.num_nodes, self.layout.dimension))
        features[:, : self.layout.terms[0].multiplicity] = self.scalar(
            graph.atomic_numbers.to(graph.positions.dtype).unsqueeze(-1)
        )
        geometry = {
            "positions": graph.positions,
            "cell": graph.cell,
            "atomic_numbers": graph.atomic_numbers,
            "edge_index": graph.edge_index,
            "cell_shifts": graph.cell_shifts,
            "edge_vectors": graph.edge_vectors,
            "edge_distances": graph.edge_distances,
        }
        return O3FeatureBatch(features, self.layout, graph.node_batch, edge_geometry=geometry)


def _system_builder(family, architecture, unit, point_groups, *, device):
    assert family == "mace" and unit.target == "dielectric"
    layout = default_hidden_layout(architecture)
    adapter = AnalyticAdapter(layout).to(device)
    downstream = PointGroupTensorModel(
        architecture,
        unit.target,
        hidden_layout=layout,
        expert_point_groups=point_groups,
        cutoff=6.0,
    ).to(device)
    return BackboneTensorModel(adapter, downstream, cutoff=6.0)


def test_prepare_batch_rotates_targets_and_preserves_scope() -> None:
    dielectric = prepare_tensor_batch(_samples()[:2], cutoff=6.0)
    assert dielectric.graph.num_graphs == 2
    assert dielectric.target_coefficients.shape == (2, 6)
    assert dielectric.sample_ids == ("real-schema-0", "real-schema-1")


def test_backbone_graph_view_requires_and_reuses_returned_geometry() -> None:
    graph = build_periodic_graph(
        torch.tensor([[0.0, 0.0, 0.0]]),
        4.0 * torch.eye(3),
        torch.tensor([14]),
        1.5,
    )
    layout = default_hidden_layout(_config())
    features = AnalyticAdapter(layout)(graph)
    reused = periodic_graph_from_backbone(features, cutoff=1.5)
    assert reused.edge_index.data_ptr() == features.edge_geometry["edge_index"].data_ptr()
    broken = O3FeatureBatch(
        features.node_features,
        layout,
        features.node_batch,
        edge_geometry={"positions": graph.positions},
    )
    try:
        periodic_graph_from_backbone(broken, cutoff=1.5)
    except ValueError as exc:
        assert "lacks" in str(exc)
    else:
        raise AssertionError("missing backbone geometry was accepted")


def test_five_structure_runner_updates_checkpoints_and_serializes_metrics(tmp_path) -> None:
    checkpoint = tmp_path / "smoke.pt"
    report = run_five_structure_smoke(
        backbone_family="mace",
        architecture=_config(),
        unit=TrainingUnit("jarvis_tensor", "dielectric"),
        checkpoint_path=checkpoint,
        device="cpu",
        samples=_samples(),
        system_builder=_system_builder,
    )
    assert checkpoint.is_file()
    assert report["status"] == "passed"
    assert report["sample_ids"] == {
        "train": ["real-schema-0", "real-schema-1", "real-schema-2"],
        "validation": ["real-schema-3"],
        "test": ["real-schema-4"],
    }
    assert report["active_nonbackbone_parameters"] < 5_000_000
    assert set(report["test_metrics"]) == {"trace", "traceless"}


def test_real_smoke_schedule_covers_every_unit_branch_backbone_and_mode() -> None:
    path = Path("src/configs/real_smoke_schedule.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload["runs"]
    configs = enumerate_architecture_configs()
    units = {
        ("jarvis_tensor", "dielectric"),
        ("jarvis_tensor", "elastic"),
        ("matten", "elastic"),
        ("jarvis_dfpt", "bec"),
    }
    assert len(rows) == 20
    for unit in units:
        selected = [row for row in rows if (row["dataset"], row["target"]) == unit]
        assert {configs[row["variant_index"]].branch for row in selected} == set(
            ARCHITECTURE_BRANCHES
        )
    assert {row["backbone"] for row in rows} == {
        "mace",
        "grace",
        "dpa4",
        "equiformerv2",
    }
    selected_configs = [configs[row["variant_index"]] for row in rows]
    assert {config.pg_hidden_mode for config in selected_configs} >= {
        "a1_only",
        "full_pg",
    }
    assert {config.adaptation_backend for config in selected_configs} >= {
        "full_o3",
        "o2_tp",
    }
    assert {config.o3e_backend for config in selected_configs} >= {"full_o3", "o2_tp"}
    assert {config.readout_backend for config in selected_configs} == {
        "full_o3",
        "o2_tp",
    }
    for target in ("dielectric", "elastic", "bec"):
        assert {
            row["backbone"] for row in rows if row["target"] == target
        } == {"mace", "grace", "dpa4", "equiformerv2"}
