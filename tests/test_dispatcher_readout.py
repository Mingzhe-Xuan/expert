from __future__ import annotations

import copy
import inspect

import pytest
import spglib
import torch

from src.configs import (
    ARCHITECTURE_BRANCHES,
    ArchitectureConfig,
    enumerate_architecture_configs,
)
from src.data import TrainingUnit
from src.experts import PointGroupTensorModel, default_hidden_layout
from src.experts.dispatcher import _extract_graph
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


def test_active_expert_counts_distinguish_expert_free_and_routed_samples() -> None:
    symmetries = (_record("1"), _record("2"))
    direct = PointGroupTensorModel(
        ArchitectureConfig("B+R", "none", "none", "full_o3", "none"),
        "dielectric",
        hidden_layout=SMALL_LAYOUT,
    )
    routed = PointGroupTensorModel(
        ArchitectureConfig("B+PGE+R", "none", "none", "full_o3", "a1_only"),
        "dielectric",
        hidden_layout=SMALL_LAYOUT,
        expert_point_groups=("1", "2"),
    )
    assert direct.active_expert_counts(symmetries) == (0, 0)
    assert routed.active_expert_counts(symmetries) == (1, 1)


def test_dispatch_batches_same_expert_once_and_matches_serial_gradients() -> None:
    torch.manual_seed(19)
    config = ArchitectureConfig("B+PGE+R", "none", "none", "full_o3", "full_pg")
    grouped = PointGroupTensorModel(
        config,
        "bec",
        hidden_layout=SMALL_LAYOUT,
        expert_point_groups=("2/m",),
        cutoff=1.4,
    )
    serial = copy.deepcopy(grouped)
    graph = _graph()
    symmetries = (_record("2/m"), _record("2/m"))
    grouped_values = torch.randn(
        graph.num_nodes, SMALL_LAYOUT.dimension, requires_grad=True
    )
    serial_values = grouped_values.detach().clone().requires_grad_(True)
    calls = []
    handle = grouped.pg_experts["pg05"].register_forward_hook(
        lambda _module, inputs, _output: calls.append(inputs[0].shape[0])
    )
    grouped_prediction = grouped(
        O3FeatureBatch(grouped_values, SMALL_LAYOUT, graph.node_batch),
        graph,
        symmetries,
    )
    handle.remove()

    serial_outputs = []
    for graph_index, symmetry in enumerate(symmetries):
        node_indices, local_graph = _extract_graph(graph, graph_index)
        serial_outputs.append(
            serial(
                O3FeatureBatch(
                    serial_values[node_indices], SMALL_LAYOUT, local_graph.node_batch
                ),
                local_graph,
                (symmetry,),
            ).raw_cartesian
        )
    serial_cartesian = torch.cat(serial_outputs)
    assert calls == [graph.num_nodes]
    assert grouped.last_dispatch_stats == {
        "expert_buckets": 1,
        "max_structures_per_expert": 2,
        "asynchronous_cuda": False,
        "cuda_streams": 0,
    }
    assert torch.allclose(
        grouped_prediction.raw_cartesian, serial_cartesian, atol=1.0e-6, rtol=1.0e-6
    )

    grouped_prediction.raw_cartesian.square().sum().backward()
    serial_cartesian.square().sum().backward()
    assert torch.allclose(grouped_values.grad, serial_values.grad, atol=2.0e-6, rtol=2.0e-6)
    grouped_gradients = {
        name: parameter.grad for name, parameter in grouped.named_parameters()
    }
    for name, parameter in serial.named_parameters():
        assert grouped_gradients[name] is not None and parameter.grad is not None
        assert torch.allclose(
            grouped_gradients[name], parameter.grad, atol=2.0e-6, rtol=2.0e-6
        )


def test_dispatch_collates_same_o3_expert_graphs_into_one_call() -> None:
    config = ArchitectureConfig(
        "B+A+O3E+R", "full_o3", "full_o3", "full_o3", "none"
    )
    model = PointGroupTensorModel(
        config,
        "bec",
        hidden_layout=SMALL_LAYOUT,
        expert_point_groups=("2/m",),
        cutoff=1.4,
    )
    graph = _graph()
    calls = []
    handle = model.o3_experts["pg05"].register_forward_hook(
        lambda _module, inputs, _output: calls.append(
            (inputs[0].shape[0], inputs[1].num_graphs)
        )
    )
    prediction = model(
        O3FeatureBatch(
            torch.randn(graph.num_nodes, SMALL_LAYOUT.dimension),
            SMALL_LAYOUT,
            graph.node_batch,
        ),
        graph,
        (_record("2/m"), _record("2/m")),
    )
    handle.remove()
    assert calls == [(graph.num_nodes, graph.num_graphs)]
    assert torch.isfinite(prediction.raw_cartesian).all()


def test_dispatch_separates_experts_and_restores_mixed_structure_order() -> None:
    config = ArchitectureConfig("B+PGE+R", "none", "none", "full_o3", "full_pg")
    grouped = PointGroupTensorModel(
        config,
        "bec",
        hidden_layout=SMALL_LAYOUT,
        expert_point_groups=("1", "2"),
        cutoff=1.4,
    )
    serial = copy.deepcopy(grouped)
    graph = _graph()
    symmetries = (_record("1"), _record("2"))
    values = torch.randn(graph.num_nodes, SMALL_LAYOUT.dimension)
    calls = {"pg01": [], "pg03": []}
    handles = [
        grouped.pg_experts[key].register_forward_hook(
            lambda _module, inputs, _output, key=key: calls[key].append(inputs[0].shape[0])
        )
        for key in calls
    ]
    actual = grouped(
        O3FeatureBatch(values, SMALL_LAYOUT, graph.node_batch), graph, symmetries
    ).raw_cartesian
    for handle in handles:
        handle.remove()

    expected = []
    for graph_index, symmetry in enumerate(symmetries):
        node_indices, local_graph = _extract_graph(graph, graph_index)
        expected.append(
            serial(
                O3FeatureBatch(values[node_indices], SMALL_LAYOUT, local_graph.node_batch),
                local_graph,
                (symmetry,),
            ).raw_cartesian
        )
    assert calls == {"pg01": [2], "pg03": [1]}
    assert grouped.last_dispatch_stats == {
        "expert_buckets": 2,
        "max_structures_per_expert": 1,
        "asynchronous_cuda": False,
        "cuda_streams": 0,
    }
    assert torch.allclose(actual, torch.cat(expected), atol=1.0e-6, rtol=1.0e-6)


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


def test_global_readout_uses_each_crystals_actual_operation_orientation() -> None:
    torch.manual_seed(313)
    graph = _graph(torch.float64)
    first = _record("mm2", torch.float64)
    frame = o3.rand_matrix(dtype=torch.float64)
    second_rotations = torch.einsum(
        "ij,gjk,lk->gil", frame, first.rotations, frame
    )
    second = SymmetryRecord(
        canonical_frame=first.canonical_frame,
        current_point_group=first.current_point_group,
        current_space_group=first.current_space_group,
        hall_number=first.hall_number,
        rotations=second_rotations,
        translations=first.translations,
    )
    features = torch.randn(
        graph.num_nodes, SMALL_LAYOUT.dimension, dtype=torch.float64, requires_grad=True
    )
    head = TensorReadout(SMALL_LAYOUT, "dielectric", "full_o3", cutoff=1.4).double()
    coefficients = head(features, graph, (first, second)).irrep_coefficients
    for index, symmetry in enumerate((first, second)):
        representations = torch.stack(
            [
                target_representation(rotation, "dielectric")
                for rotation in symmetry.rotations
            ]
        )
        transformed = torch.einsum("gij,j->gi", representations, coefficients[index])
        assert torch.allclose(
            transformed,
            coefficients[index].expand_as(transformed),
            atol=2e-8,
            rtol=2e-8,
        )
    coefficients.square().sum().backward()
    assert features.grad is not None and torch.isfinite(features.grad).all()


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
        assert tuple(term.multiplicity for term in layout.terms) == (8, 2, 2, 2, 2)
        assert layout.dimension == 56
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


def test_parent_routing_requires_validated_current_point_group() -> None:
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
    wrong_dag = ParentDAGSpec(material_id="fixture", current_point_group_number=1, embeddings=())
    with pytest.raises(ValueError, match="current point group"):
        model(
            features,
            graph,
            symmetries,
            parent_dags=(wrong_dag, None),
        )
