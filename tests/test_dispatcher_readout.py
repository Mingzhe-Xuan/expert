from __future__ import annotations

import inspect

import pytest
import spglib
import torch

from src.configs import ARCHITECTURE_BRANCHES, enumerate_architecture_configs
from src.data import TrainingUnit
from src.experts import PointGroupTensorModel, default_hidden_layout
from src.graphs import build_periodic_graph, collate_periodic_graphs
from src.heads import TARGET_LAYOUTS, TensorReadout, target_representation
from src.irreps import ConventionMetadata, IrrepLayout, IrrepTerm, O3FeatureBatch
from src.symmetry import ParentDAGSpec, PointGroupRegistry, SymmetryRecord
from src.training import CoefficientNormalizer, coefficient_mse, load_checkpoint, save_checkpoint
from e3nn import o3


SMALL_LAYOUT = IrrepLayout(
    (
        IrrepTerm(2, 0, "e", "scalar"),
        IrrepTerm(1, 1, "o", "polar"),
        IrrepTerm(1, 2, "e", "quadrupole"),
    )
)


def _record(symbol: str, dtype: torch.dtype = torch.float32) -> SymmetryRecord:
    group = PointGroupRegistry()[symbol]
    space_group = spglib.get_spacegroup_type(group.representative_hall_number)
    assert space_group is not None
    return SymmetryRecord(
        canonical_frame=torch.eye(3, dtype=dtype),
        current_point_group=symbol,
        current_space_group=int(space_group.number),
        hall_number=group.representative_hall_number,
        rotations=group.fractional_rotations.to(dtype=dtype),
        translations=torch.zeros(group.order, 3, dtype=dtype),
    )


def _graph(dtype: torch.dtype = torch.float32):
    first = build_periodic_graph(
        torch.tensor([[0.1, 0.2, 0.3], [0.9, 0.5, 0.7]], dtype=dtype),
        2.2 * torch.eye(3, dtype=dtype),
        torch.tensor([6, 8]),
        cutoff=1.4,
    )
    second = build_periodic_graph(
        torch.tensor([[0.2, 0.1, 0.4]], dtype=dtype),
        2.5 * torch.eye(3, dtype=dtype),
        torch.tensor([14]),
        cutoff=1.4,
    )
    return collate_periodic_graphs((first, second))


def _convention() -> ConventionMetadata:
    return ConventionMetadata(
        schema_version=1,
        real_harmonic_convention="e3nn-real-y-lm",
        cg_convention="e3nn-wigner-3j",
        path_ordering="input1-input2-output-lexicographic",
        copy_ordering="target-contract-v1",
        subduction_checksum="b" * 64,
    )


def test_three_readouts_have_correct_scope_constraints_and_no_bec_pi_g() -> None:
    torch.manual_seed(31)
    graph = _graph()
    symmetries = (_record("m-3m"), _record("2/m"))
    features = torch.randn(graph.num_nodes, SMALL_LAYOUT.dimension, requires_grad=True)
    for task in ("dielectric", "elastic", "bec"):
        head = TensorReadout(SMALL_LAYOUT, task, "full_o3", cutoff=1.4)
        prediction = head(features, graph, symmetries)
        if task == "bec":
            assert prediction.raw_cartesian.shape == (3, 3, 3)
            assert prediction.scope == "node" and prediction.symmetry_control_cartesian is None
            assert torch.allclose(prediction.asr_cartesian[:2].sum(0), torch.zeros(3, 3), atol=1e-6)
            assert torch.allclose(prediction.asr_cartesian[2:].sum(0), torch.zeros(3, 3), atol=1e-6)
        else:
            expected = (2, 3, 3) if task == "dielectric" else (2, 3, 3, 3, 3)
            assert prediction.raw_cartesian.shape == expected and prediction.scope == "global"
            for index, symbol in enumerate(("m-3m", "2/m")):
                representation = PointGroupRegistry()[symbol].representation(
                    TARGET_LAYOUTS[task], dtype=prediction.irrep_coefficients.dtype
                )
                transformed = torch.einsum(
                    "gij,j->gi", representation, prediction.irrep_coefficients[index]
                )
                assert torch.allclose(
                    transformed,
                    prediction.irrep_coefficients[index].expand_as(transformed),
                    atol=2e-5,
                    rtol=2e-5,
                )
    assert "pi_g" not in inspect.signature(TensorReadout.forward).parameters


def test_readout_is_o3_equivariant_for_proper_and_improper_rotations() -> None:
    torch.manual_seed(44)
    graph = _graph(torch.float64)
    features = torch.randn(graph.num_nodes, SMALL_LAYOUT.dimension, dtype=torch.float64)
    record = (_record("1", torch.float64), _record("1", torch.float64))
    head = TensorReadout(SMALL_LAYOUT, "bec", "full_o3", cutoff=1.4).double()
    hidden_irreps = o3.Irreps("2x0e + 1x1o + 1x2e")
    baseline = head(features, graph, record).raw_cartesian
    for rotation in (o3.rand_matrix(dtype=torch.float64), -o3.rand_matrix(dtype=torch.float64)):
        transformed_graphs = []
        for graph_index in range(graph.num_graphs):
            mask = graph.node_batch == graph_index
            transformed_graphs.append(
                build_periodic_graph(
                    graph.positions[mask] @ rotation.T,
                    graph.cell[graph_index] @ rotation.T,
                    graph.atomic_numbers[mask],
                    cutoff=1.4,
                )
            )
        transformed_graph = collate_periodic_graphs(transformed_graphs)
        transformed_features = features @ hidden_irreps.D_from_matrix(rotation).T
        actual = head(transformed_features, transformed_graph, record).raw_cartesian
        expected = torch.einsum("ai,nij,bj->nab", rotation, baseline, rotation)
        assert torch.allclose(actual, expected, atol=2e-7, rtol=2e-7)


def test_all_26_configs_construct_train_checkpoint_and_respect_budget(tmp_path) -> None:
    torch.manual_seed(52)
    configs = enumerate_architecture_configs()
    assert {config.branch for config in configs} == set(ARCHITECTURE_BRANCHES)
    assert tuple(dict.fromkeys(config.branch for config in configs)) == ARCHITECTURE_BRANCHES
    graph = _graph()
    symmetry = (_record("2/m"), _record("2/m"))
    unit = TrainingUnit("jarvis_dfpt", "bec")
    normalizer = CoefficientNormalizer.fit(
        torch.randn(graph.num_nodes, TARGET_LAYOUTS["bec"].dimension),
        TARGET_LAYOUTS["bec"],
        unit,
        split="train",
    )
    for index, config in enumerate(configs):
        layout = default_hidden_layout(config)
        model = PointGroupTensorModel(
            config,
            "bec",
            hidden_layout=layout,
            expert_point_groups=("2/m",),
            cutoff=1.4,
        )
        assert (model.adaptation is not None) == ("+A+" in config.branch)
        assert (model.o3_experts is not None) == ("+O3E+" in config.branch)
        assert (model.pg_experts is not None) == ("+PGE+" in config.branch)
        assert model.readout.backend == config.readout_backend
        assert model.active_nonbackbone_parameter_count(symmetry) < 5_000_000

        values = torch.randn(graph.num_nodes, layout.dimension, requires_grad=True)
        features = O3FeatureBatch(values, layout, graph.node_batch)
        prediction = model(features, graph, symmetry)
        target = torch.randn_like(prediction.irrep_coefficients)
        loss = coefficient_mse(
            prediction.irrep_coefficients,
            target,
            TARGET_LAYOUTS["bec"],
            normalizer,
        ).total
        loss.backward()
        assert values.grad is not None and torch.isfinite(values.grad).all()
        assert any(parameter.grad is not None for parameter in model.parameters())

        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
        path = tmp_path / f"{index:02d}.pt"
        before = {name: value.detach().clone() for name, value in model.state_dict().items()}
        save_checkpoint(
            path,
            model=model,
            optimizer=optimizer,
            architecture=config,
            unit=unit,
            convention=_convention(),
            normalizer=normalizer,
            step=1,
        )
        with torch.no_grad():
            next(model.parameters()).add_(1.0)
        loaded = load_checkpoint(
            path,
            model=model,
            optimizer=optimizer,
            expected_architecture=config,
            expected_unit=unit,
            expected_convention=_convention(),
            expected_layout=TARGET_LAYOUTS["bec"],
        )
        assert loaded.step == 1
        assert all(torch.equal(model.state_dict()[name], value) for name, value in before.items())


def test_parent_routing_requires_validated_current_hall() -> None:
    config = next(
        config for config in enumerate_architecture_configs() if config.branch == "B+PGE+R"
    )
    model = PointGroupTensorModel(
        config,
        "bec",
        hidden_layout=SMALL_LAYOUT,
        expert_point_groups=("2/m",),
        cutoff=1.4,
    )
    graph = _graph()
    symmetries = (_record("2/m"), _record("2/m"))
    features = O3FeatureBatch(
        torch.randn(graph.num_nodes, SMALL_LAYOUT.dimension), SMALL_LAYOUT, graph.node_batch
    )
    prediction = model(features, graph, symmetries)
    assert prediction.raw_cartesian.shape == (graph.num_nodes, 3, 3)
    wrong_dag = ParentDAGSpec(material_id="fixture", current_hall_number=1, embeddings=())
    with pytest.raises(ValueError, match="current Hall"):
        model(
            features,
            graph,
            symmetries,
            parent_dags=(wrong_dag, None),
        )
