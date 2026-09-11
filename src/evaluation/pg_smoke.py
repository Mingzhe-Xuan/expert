from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

from scipy.optimize import linear_sum_assignment
import torch

from ..backbones import BackboneResourceRegistry
from ..configs import enumerate_architecture_configs
from ..graphs import PeriodicGraph, build_periodic_graph
from ..heads import rotate_cartesian, target_representation
from ..models import build_backbone_adapter, BackboneTensorModel
from ..experts import PointGroupTensorModel, default_hidden_layout
from ..symmetry import SymmetryRecord, canonicalize_structure
from .fixtures import validate_point_group_fixture_manifest


PG_PGE_VARIANTS = (14, 17, 18, 25)
BACKBONES = ("mace", "grace", "dpa4", "equiformerv2")
TASKS = ("dielectric", "elastic", "bec")
FLOAT32_TOLERANCE = 3.0e-3


def point_group_smoke_schedule() -> tuple[dict[str, object], ...]:
    """Return 32 mandatory PGE rows followed by exhaustive 26-config rows."""

    rows = []
    for fixture_index in range(32):
        rows.append(
            {
                "fixture_index": fixture_index,
                "variant_index": PG_PGE_VARIANTS[fixture_index % len(PG_PGE_VARIANTS)],
                "backbone": BACKBONES[fixture_index % len(BACKBONES)],
                "task": TASKS[fixture_index % len(TASKS)],
                "purpose": "per_point_group_pge",
            }
        )
    for variant_index in range(26):
        rows.append(
            {
                "fixture_index": variant_index,
                "variant_index": variant_index,
                "backbone": BACKBONES[variant_index % len(BACKBONES)],
                "task": TASKS[variant_index % len(TASKS)],
                "purpose": "exhaustive_architecture",
            }
        )
    return tuple(rows)


def audit_backbone_graph_automorphism(
    graph: PeriodicGraph,
    symmetry: SymmetryRecord,
    *,
    tolerance: float = 2.0e-4,
) -> float:
    """Audit node permutations and every periodic image edge by its Cartesian vector."""

    if graph.num_graphs != 1 or symmetry.audit_permutations is None:
        raise ValueError("graph automorphism audit requires one graph and site permutations")
    def grouped_vectors(index, vectors):
        grouped: dict[tuple[int, int], list[torch.Tensor]] = {}
        for (source, target), vector in zip(
            index.T.detach().cpu(), vectors.detach().cpu()
        ):
            grouped.setdefault((int(source), int(target)), []).append(vector)
        return {key: torch.stack(values) for key, values in grouped.items()}

    reference = grouped_vectors(graph.edge_index, graph.edge_vectors)
    maximum = 0.0
    for rotation, permutation in zip(symmetry.rotations, symmetry.audit_permutations):
        mapped_index = permutation[graph.edge_index]
        rotated_vectors = graph.edge_vectors @ rotation.T
        mapped = grouped_vectors(mapped_index, rotated_vectors)
        if reference.keys() != mapped.keys() or any(
            reference[key].shape != mapped[key].shape for key in reference
        ):
            raise AssertionError("backbone periodic edge multiset is not symmetry closed")
        for key in reference:
            costs = torch.cdist(reference[key], mapped[key])
            rows, columns = linear_sum_assignment(costs.numpy())
            error = float(costs[rows, columns].max()) if len(rows) else 0.0
            maximum = max(maximum, error)
            if error >= tolerance:
                raise AssertionError("backbone periodic edge multiset is not symmetry closed")
    return maximum


def _fixture_graph(record: Mapping[str, object], cutoff: float, device: str):
    lattice = torch.tensor(record["lattice_angstrom"], dtype=torch.float32, device=device)
    fractional = torch.tensor(
        record["fractional_coordinates"], dtype=torch.float32, device=device
    )
    numbers = torch.tensor(record["atomic_numbers"], dtype=torch.long, device=device)
    canonical = canonicalize_structure(fractional @ lattice, lattice, numbers)
    graph = build_periodic_graph(
        canonical.canonical_positions,
        canonical.canonical_cell,
        canonical.atomic_numbers,
        cutoff,
    )
    return graph, canonical.symmetry


def run_point_group_smoke(
    index: int,
    *,
    fixture_path: str | Path,
    device: str = "cuda",
) -> dict[str, object]:
    schedule = point_group_smoke_schedule()
    if not 0 <= index < len(schedule):
        raise ValueError(f"smoke index must lie in [0, {len(schedule) - 1}]")
    manifest = json.loads(Path(fixture_path).read_text(encoding="utf-8"))
    validate_point_group_fixture_manifest(manifest)
    row = schedule[index]
    fixture = manifest["fixtures"][row["fixture_index"]]
    architecture = enumerate_architecture_configs()[row["variant_index"]]
    resource = BackboneResourceRegistry()[row["backbone"]]
    graph, symmetry = _fixture_graph(fixture, resource.cutoff_angstrom, device)
    hidden = default_hidden_layout(architecture)
    adapter = build_backbone_adapter(row["backbone"], hidden, device=device)
    downstream = PointGroupTensorModel(
        architecture,
        row["task"],
        hidden_layout=hidden,
        expert_point_groups=(symmetry.current_point_group,),
        cutoff=resource.cutoff_angstrom,
    ).to(device)
    system = BackboneTensorModel(
        adapter, downstream, cutoff=resource.cutoff_angstrom
    ).to(device)
    system.train()
    result = system.forward_with_graph(graph, (symmetry,))
    graph_error = audit_backbone_graph_automorphism(result.graph, symmetry)
    coefficients = result.prediction.irrep_coefficients
    loss = coefficients.square().mean()
    if not torch.isfinite(loss):
        raise AssertionError("32-PG smoke loss is non-finite")
    loss.backward()
    gradients = [
        parameter.grad
        for parameter in system.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    if not gradients or any(not torch.isfinite(value).all() for value in gradients):
        raise AssertionError("32-PG smoke gradients are missing or non-finite")
    coefficient_error = 0.0
    bec_error = None
    if row["task"] != "bec":
        representation = torch.stack(
            [
                target_representation(rotation, row["task"], dtype=coefficients.dtype)
                for rotation in symmetry.rotations
            ]
        )
        transformed = torch.einsum("gij,nj->gni", representation, coefficients)
        coefficient_error = float((transformed - coefficients).abs().max())
    else:
        canonical_bec = result.prediction.canonical_cartesian
        expected = []
        for rotation, permutation in zip(symmetry.rotations, symmetry.audit_permutations):
            transformed_bec = rotate_cartesian(canonical_bec, rotation, "bec")
            expected.append((canonical_bec[permutation] - transformed_bec).abs().max())
        bec_error = float(torch.stack(expected).max())
    maximum = max(coefficient_error, 0.0 if bec_error is None else bec_error)
    if maximum >= FLOAT32_TOLERANCE:
        raise AssertionError(f"32-PG tensor symmetry error {maximum} exceeds tolerance")
    active = downstream.active_nonbackbone_parameter_count((symmetry,))
    if active >= 5_000_000:
        raise AssertionError("32-PG active parameter budget exceeded")
    return {
        "status": "passed",
        "index": index,
        **row,
        "point_group": symmetry.current_point_group,
        "sample_id": fixture["sample_id"],
        "architecture": architecture.to_dict(),
        "loss": float(loss.detach()),
        "graph_automorphism_bound": graph_error,
        "coefficient_invariance_error": coefficient_error,
        "bec_joint_error": bec_error,
        "tolerance": FLOAT32_TOLERANCE,
        "nodes": result.graph.num_nodes,
        "edges": result.graph.num_edges,
        "active_nonbackbone_parameters": active,
    }
