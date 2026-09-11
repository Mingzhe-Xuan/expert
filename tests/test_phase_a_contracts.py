from __future__ import annotations

from collections import Counter
from dataclasses import replace
import json
from pathlib import Path

import pytest
import torch

from src.configs import ArchitectureConfig, enumerate_architecture_configs
from src.graphs import PeriodicGraph
from src.heads import TARGET_LAYOUTS, TensorPrediction
from src.irreps import ConventionMetadata, IrrepLayout, IrrepTerm, O3FeatureBatch
from src.symmetry import ParentDAGSpec, ParentEmbeddingSpec, SymmetryRecord


ROOT = Path(__file__).resolve().parents[1]
MODULES = (
    "backbones",
    "graphs",
    "symmetry",
    "irreps",
    "tensor_products",
    "experts",
    "heads",
    "data",
    "training",
    "evaluation",
    "configs",
    "cli",
)


def test_required_module_boundaries_have_readmes() -> None:
    for module in MODULES:
        directory = ROOT / "src" / module
        assert (directory / "__init__.py").is_file()
        readme = directory / "README.md"
        assert readme.is_file()
        assert "```" in readme.read_text(encoding="utf-8")


def test_architecture_matrix_matches_frozen_manifest() -> None:
    configs = enumerate_architecture_configs()
    assert len(configs) == 26
    assert Counter(config.branch for config in configs) == {
        "B+R": 2,
        "B+A+R": 4,
        "B+A+O3E+R": 8,
        "B+PGE+R": 4,
        "B+A+PGE+R": 8,
    }
    manifest = json.loads(
        (ROOT / "src" / "configs" / "architecture_variant_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["schema_version"] == 1
    assert manifest["variant_ids"] == [config.variant_id for config in configs]
    assert manifest["expected_branch_counts"] == Counter(
        config.branch for config in configs
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(
            branch="unknown",
            adaptation_backend="none",
            o3e_backend="none",
            readout_backend="full_o3",
            pg_hidden_mode="none",
        ),
        dict(
            branch="B+R",
            adaptation_backend="full_o3",
            o3e_backend="none",
            readout_backend="full_o3",
            pg_hidden_mode="none",
        ),
        dict(
            branch="B+A+O3E+R",
            adaptation_backend="full_o3",
            o3e_backend="none",
            readout_backend="full_o3",
            pg_hidden_mode="none",
        ),
        dict(
            branch="B+PGE+R",
            adaptation_backend="none",
            o3e_backend="none",
            readout_backend="full_o3",
            pg_hidden_mode="none",
        ),
        dict(
            branch="B+A+R",
            adaptation_backend="o2_tp",
            o3e_backend="none",
            readout_backend="invalid",
            pg_hidden_mode="none",
        ),
    ],
)
def test_architecture_config_rejects_invalid_combinations(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        ArchitectureConfig(**kwargs)


def test_irrep_layout_and_feature_batch_preserve_explicit_order() -> None:
    layout = IrrepLayout(
        (
            IrrepTerm(2, 0, "e", "scalar_copies"),
            IrrepTerm(1, 1, "e", "axial_vector"),
        )
    )
    assert layout.dimension == 5
    assert not layout.terms[1].natural_parity
    features = O3FeatureBatch(
        node_features=torch.zeros(3, 5),
        node_layout=layout,
        node_batch=torch.tensor([0, 0, 1]),
        edge_geometry={"edge_vectors": torch.zeros(0, 3)},
    )
    assert features.node_layout.to_spec()[1]["copy_label"] == "axial_vector"
    with pytest.raises(TypeError):
        features.edge_geometry["new"] = torch.ones(1)
    with pytest.raises(ValueError, match="width"):
        O3FeatureBatch(torch.zeros(3, 4), layout, torch.tensor([0, 0, 1]))


def test_periodic_graph_validates_image_edge_contract() -> None:
    graph = PeriodicGraph(
        positions=torch.tensor([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]]),
        cell=torch.eye(3).unsqueeze(0),
        atomic_numbers=torch.tensor([1, 1]),
        node_batch=torch.tensor([0, 0]),
        edge_index=torch.tensor([[0, 1], [1, 0]]),
        cell_shifts=torch.zeros(2, 3, dtype=torch.long),
        edge_vectors=torch.tensor([[0.5, 0.0, 0.0], [-0.5, 0.0, 0.0]]),
        edge_distances=torch.tensor([0.5, 0.5]),
        cutoff=1.0,
    )
    assert (graph.num_graphs, graph.num_nodes, graph.num_edges) == (1, 2, 2)
    with pytest.raises(ValueError, match="distances"):
        replace(graph, edge_distances=torch.tensor([0.4, 0.5]))


def test_symmetry_record_keeps_permutations_audit_only() -> None:
    record = SymmetryRecord(
        canonical_frame=torch.eye(3),
        current_point_group="m-3m",
        current_space_group=221,
        hall_number=517,
        rotations=torch.eye(3).unsqueeze(0),
        translations=torch.zeros(1, 3),
        audit_permutations=torch.tensor([[0, 1]]),
    )
    assert record.audit_permutations.tolist() == [[0, 1]]
    with pytest.raises(ValueError, match="orthogonal"):
        replace(record, canonical_frame=torch.ones(3, 3))


def _embedding() -> ParentEmbeddingSpec:
    candidate = ParentEmbeddingSpec(
        parent_hall_number=2,
        child_hall_number=1,
        parent_setting="P -1",
        child_setting="P 1",
        basis_transform=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        origin_shift=(0.0, 0.0, 0.0),
        supercell_transform=((1, 0, 0), (0, 1, 0), (0, 0, 1)),
        operations=(
            (
                ((1, 0, 0), (0, 1, 0), (0, 0, 1)),
                (0.0, 0.0, 0.0),
            ),
        ),
        parent_atomic_numbers=(6, 8),
        child_atomic_numbers=(6, 6, 8),
        atom_correspondence=(0, 0, 1),
        wyckoff_splitting=("1a->1a+1b", "1b->1c"),
        domain_variant="identity",
        convention_id="hall-standard-v1",
        version=1,
        checksum="0" * 64,
    )
    return replace(candidate, checksum=candidate.payload_checksum())


def test_parent_dag_requires_versioned_checksum_valid_embeddings() -> None:
    embedding = _embedding()
    dag = ParentDAGSpec("material-1", current_hall_number=1, embeddings=(embedding,))
    assert dag.embeddings[0].parent_hall_number == 2
    with pytest.raises(ValueError, match="checksum"):
        ParentDAGSpec(
            "material-1",
            current_hall_number=1,
            embeddings=(replace(embedding, domain_variant="changed"),),
        )


@pytest.mark.parametrize(
    ("task", "scope", "cartesian_shape"),
    [
        ("dielectric", "global", (2, 3, 3)),
        ("elastic", "global", (2, 3, 3, 3, 3)),
        ("bec", "node", (4, 3, 3)),
    ],
)
def test_tensor_prediction_enforces_target_scope_and_layout(
    task: str, scope: str, cartesian_shape: tuple[int, ...]
) -> None:
    count = cartesian_shape[0]
    prediction = TensorPrediction(
        raw_cartesian=torch.zeros(cartesian_shape),
        irrep_coefficients=torch.zeros(count, TARGET_LAYOUTS[task].dimension),
        task=task,
        scope=scope,
        node_batch=torch.zeros(count, dtype=torch.long) if scope == "node" else None,
    )
    assert prediction.scope == scope
    with pytest.raises(ValueError, match="requires"):
        replace(prediction, scope="global" if scope == "node" else "node")


def test_checkpoint_conventions_fail_closed() -> None:
    metadata = ConventionMetadata(
        schema_version=1,
        real_harmonic_convention="e3nn-real-y-lm",
        cg_convention="e3nn-wigner-3j",
        path_ordering="input1-input2-output-lexicographic",
        copy_ordering="target-contract-v1",
        subduction_checksum="a" * 64,
    )
    metadata.require_compatible({"convention_checksum": metadata.checksum})
    with pytest.raises(ValueError, match="mismatch"):
        metadata.require_compatible({"convention_checksum": "b" * 64})
