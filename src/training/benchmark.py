from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
import random
from pathlib import Path
from typing import Callable, Sequence

import torch
from torch import nn

from ..backbones import O3InterfaceProjector
from ..configs import ArchitectureConfig
from ..data import TensorSample, TrainingUnit
from ..evaluation import tensor_benchmark_metrics
from ..experts import PointGroupTensorModel, default_hidden_layout
from ..graphs import PeriodicGraph, collate_periodic_graphs
from ..heads import TARGET_LAYOUTS, irreps_to_cartesian
from ..irreps import IrrepLayout, IrrepTerm, O3FeatureBatch
from ..models import periodic_graph_from_backbone
from ..symmetry import SymmetryRecord
from .checkpoint import load_checkpoint, save_checkpoint
from .losses import coefficient_mse, physical_coefficient_metrics
from .normalization import CoefficientNormalizer
from .smoke import default_convention_metadata, prepare_tensor_batch


PUBLIC_TARGETS = {
    "dielectric": {"fnorm": 2.87, "ewt_25": 86.1, "ewt_10": 63.8, "ewt_5": 39.3},
    "elastic": {"fnorm": 67.38, "ewt_25": 70.6, "ewt_10": 32.2, "ewt_5": 14.4},
}
PUBLISHED_SPLIT_COUNTS = {
    "dielectric": {"train": 3770, "validation": 471, "test": 471},
    "elastic": {"train": 11376, "validation": 1422, "test": 1422},
}


def require_published_split_counts(unit: TrainingUnit, split_manifest) -> None:
    if unit.dataset != "jarvis_tensor" or unit.target not in PUBLISHED_SPLIT_COUNTS:
        raise ValueError("published split count gate only supports JARVIS tensor benchmarks")
    actual = {
        name: len(getattr(split_manifest, name))
        for name in ("train", "validation", "test")
    }
    expected = PUBLISHED_SPLIT_COUNTS[unit.target]
    if actual != expected:
        raise ValueError(
            f"{unit.target} split is not directly comparable: expected {expected}, received {actual}"
        )


@dataclass(frozen=True, slots=True)
class BenchmarkConfig:
    max_epochs: int = 200
    batch_size: int = 16
    learning_rate: float = 1.0e-3
    weight_decay: float = 1.0e-6
    patience: int = 30
    seed: int = 20260911
    normalization: str = "rms"
    gradient_clip_norm: float = 10.0

    def __post_init__(self) -> None:
        if self.max_epochs < 1 or self.batch_size < 1 or self.patience < 1:
            raise ValueError("epochs, batch size, and patience must be positive")
        if self.learning_rate <= 0 or self.weight_decay < 0 or self.gradient_clip_norm <= 0:
            raise ValueError("optimizer rates and gradient clip must be valid")
        if self.normalization not in {"rms", "variance"}:
            raise ValueError("normalization must be rms or variance")


@dataclass(frozen=True, slots=True)
class FrozenFeatureExample:
    sample_id: str
    features: torch.Tensor
    graph: PeriodicGraph
    symmetry: SymmetryRecord
    target_coefficients: torch.Tensor
    target_cartesian: torch.Tensor

    def __post_init__(self) -> None:
        if not self.sample_id or self.features.ndim != 2:
            raise ValueError("a cached feature example requires an ID and [node, feature] values")
        if self.features.shape[0] != self.graph.num_nodes:
            raise ValueError("cached features must align with graph nodes")
        if self.graph.num_graphs != 1 or self.target_coefficients.ndim != 2:
            raise ValueError("each cached benchmark example must contain exactly one graph")
        if self.target_coefficients.shape[0] != 1 or self.target_cartesian.shape[0] != 1:
            raise ValueError("global benchmark targets require one leading sample item")


@dataclass(frozen=True, slots=True)
class FrozenFeatureBatch:
    features: O3FeatureBatch
    graph: PeriodicGraph
    symmetries: tuple[SymmetryRecord, ...]
    target_coefficients: torch.Tensor
    target_cartesian: torch.Tensor
    sample_ids: tuple[str, ...]


class CachedBackboneTensorModel(nn.Module):
    """Trainable equivariant interface plus architecture over frozen source features."""

    def __init__(
        self,
        source_layout: IrrepLayout,
        architecture: ArchitectureConfig,
        task: str,
        expert_point_groups: tuple[str, ...],
    ) -> None:
        super().__init__()
        hidden_layout = default_hidden_layout(architecture)
        self.interface = O3InterfaceProjector(source_layout, hidden_layout)
        self.downstream = PointGroupTensorModel(
            architecture,
            task,
            hidden_layout=hidden_layout,
            expert_point_groups=expert_point_groups,
        )

    def forward(self, features, graph, symmetries):
        return self.downstream(self.interface(features), graph, symmetries)


def save_frozen_feature_cache(
    path: str | Path,
    *,
    backbone_family: str,
    checkpoint_sha256: str,
    unit: TrainingUnit,
    split: str,
    layout: IrrepLayout,
    examples: Sequence[FrozenFeatureExample],
    dataset_sha256: str | None = None,
) -> None:
    """Atomically persist tensor-only frozen features for hyperparameter reuse."""

    if split not in {"train", "validation", "test"} or not examples:
        raise ValueError("cache split must be train/validation/test and non-empty")
    if len(checkpoint_sha256) != 64:
        raise ValueError("cache requires the frozen checkpoint SHA-256")
    if dataset_sha256 is not None and len(dataset_sha256) != 64:
        raise ValueError("dataset_sha256 must be a SHA-256 digest")
    rows = []
    for example in examples:
        graph = example.graph
        symmetry = example.symmetry
        rows.append(
            {
                "sample_id": example.sample_id,
                "features": example.features,
                "graph": {
                    "positions": graph.positions,
                    "cell": graph.cell,
                    "atomic_numbers": graph.atomic_numbers,
                    "node_batch": graph.node_batch,
                    "edge_index": graph.edge_index,
                    "cell_shifts": graph.cell_shifts,
                    "edge_vectors": graph.edge_vectors,
                    "edge_distances": graph.edge_distances,
                    "cutoff": graph.cutoff,
                    "boundary_convention": graph.boundary_convention,
                },
                "symmetry": {
                    "canonical_frame": symmetry.canonical_frame,
                    "current_point_group": symmetry.current_point_group,
                    "current_space_group": symmetry.current_space_group,
                    "hall_number": symmetry.hall_number,
                    "rotations": symmetry.rotations,
                    "translations": symmetry.translations,
                    "audit_permutations": symmetry.audit_permutations,
                },
                "target_coefficients": example.target_coefficients,
                "target_cartesian": example.target_cartesian,
            }
        )
    payload = {
        "schema_version": 2,
        "backbone": backbone_family,
        "checkpoint_sha256": checkpoint_sha256,
        "dataset_sha256": dataset_sha256,
        "training_unit": unit.namespace,
        "split": split,
        "layout": list(layout.to_spec()),
        "component_order": layout.component_order,
        "sample_ids": [example.sample_id for example in examples],
        "examples": rows,
    }
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_frozen_feature_cache(
    path: str | Path,
    *,
    expected_backbone: str,
    expected_checkpoint_sha256: str,
    expected_unit: TrainingUnit,
    expected_split: str,
    expected_sample_ids: Sequence[str],
    expected_dataset_sha256: str | None = None,
) -> tuple[IrrepLayout, tuple[FrozenFeatureExample, ...]]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    expected = {
        "schema_version": 2,
        "backbone": expected_backbone,
        "checkpoint_sha256": expected_checkpoint_sha256,
        "training_unit": expected_unit.namespace,
        "split": expected_split,
        "sample_ids": list(expected_sample_ids),
    }
    for name, value in expected.items():
        if payload.get(name) != value:
            raise ValueError(f"frozen feature cache {name} mismatch")
    if expected_dataset_sha256 is not None and payload.get("dataset_sha256") != expected_dataset_sha256:
        raise ValueError("frozen feature cache dataset_sha256 mismatch")
    layout = IrrepLayout(
        tuple(IrrepTerm(**term) for term in payload["layout"]),
        component_order=str(payload["component_order"]),
    )
    examples = []
    for row in payload["examples"]:
        graph = PeriodicGraph(**row["graph"])
        symmetry = SymmetryRecord(**row["symmetry"])
        examples.append(
            FrozenFeatureExample(
                str(row["sample_id"]),
                row["features"],
                graph,
                symmetry,
                row["target_coefficients"],
                row["target_cartesian"],
            )
        )
    return layout, tuple(examples)


def _move_symmetry(record: SymmetryRecord, device: torch.device | str) -> SymmetryRecord:
    return SymmetryRecord(
        canonical_frame=record.canonical_frame.to(device),
        current_point_group=record.current_point_group,
        current_space_group=record.current_space_group,
        hall_number=record.hall_number,
        rotations=record.rotations.to(device),
        translations=record.translations.to(device),
        audit_permutations=(
            None if record.audit_permutations is None else record.audit_permutations.to(device)
        ),
    )


def extract_frozen_examples(
    adapter: nn.Module,
    samples: Sequence[TensorSample],
    *,
    cutoff: float,
    device: torch.device | str,
    progress: Callable[[int, int, str], None] | None = None,
) -> tuple[IrrepLayout, tuple[FrozenFeatureExample, ...]]:
    """Run a frozen backbone once and retain its pre-interface O(3) tap on CPU."""

    if not samples:
        raise ValueError("cannot extract an empty benchmark split")
    source_layout = None
    output = []
    adapter.eval()
    for index, sample in enumerate(samples, start=1):
        prepared = prepare_tensor_batch((sample,), cutoff=cutoff, device=device)
        with torch.no_grad():
            source = adapter.forward_source(prepared.graph)
        if source_layout is None:
            source_layout = source.node_layout
        elif source.node_layout != source_layout:
            raise ValueError("backbone source layout changed within one dataset")
        native_graph = periodic_graph_from_backbone(source, cutoff=cutoff).to("cpu")
        output.append(
            FrozenFeatureExample(
                sample_id=sample.sample_id,
                features=source.node_features.detach().to("cpu"),
                graph=native_graph,
                symmetry=_move_symmetry(prepared.symmetries[0], "cpu"),
                target_coefficients=prepared.target_coefficients.detach().to("cpu"),
                target_cartesian=irreps_to_cartesian(
                    prepared.target_coefficients.detach().to("cpu"), unit.target
                ),
            )
        )
        if progress is not None:
            progress(index, len(samples), sample.sample_id)
    return source_layout, tuple(output)


def collate_frozen_examples(
    examples: Sequence[FrozenFeatureExample],
    layout: IrrepLayout,
    *,
    device: torch.device | str,
) -> FrozenFeatureBatch:
    if not examples:
        raise ValueError("cannot collate an empty cached benchmark batch")
    graphs = tuple(example.graph.to(device) for example in examples)
    graph = collate_periodic_graphs(graphs)
    values = torch.cat([example.features.to(device) for example in examples])
    features = O3FeatureBatch(values, layout, graph.node_batch)
    return FrozenFeatureBatch(
        features=features,
        graph=graph,
        symmetries=tuple(_move_symmetry(example.symmetry, device) for example in examples),
        target_coefficients=torch.cat(
            [example.target_coefficients.to(device) for example in examples]
        ),
        target_cartesian=torch.cat([example.target_cartesian.to(device) for example in examples]),
        sample_ids=tuple(example.sample_id for example in examples),
    )


def _batches(
    examples: Sequence[FrozenFeatureExample],
    batch_size: int,
    *,
    shuffle_seed: int | None,
):
    indices = list(range(len(examples)))
    if shuffle_seed is not None:
        random.Random(shuffle_seed).shuffle(indices)
    for start in range(0, len(indices), batch_size):
        yield tuple(examples[index] for index in indices[start : start + batch_size])


def _evaluate(
    model: nn.Module,
    examples: Sequence[FrozenFeatureExample],
    layout: IrrepLayout,
    unit: TrainingUnit,
    normalizer: CoefficientNormalizer,
    *,
    batch_size: int,
    device: torch.device | str,
    prediction_rows: list[dict[str, object]] | None = None,
) -> tuple[float, dict[str, float | int], dict[str, dict[str, float]]]:
    model.eval()
    total_loss = 0.0
    predictions, coefficients, targets = [], [], []
    with torch.no_grad():
        for rows in _batches(examples, batch_size, shuffle_seed=None):
            batch = collate_frozen_examples(rows, layout, device=device)
            prediction = model(batch.features, batch.graph, batch.symmetries)
            loss = coefficient_mse(
                prediction.irrep_coefficients,
                batch.target_coefficients,
                TARGET_LAYOUTS[unit.target],
                normalizer,
            ).total
            total_loss += float(loss) * len(rows)
            predictions.append(prediction.raw_cartesian.cpu())
            coefficients.append(prediction.irrep_coefficients.cpu())
            targets.append(batch.target_coefficients.cpu())
            if prediction_rows is not None:
                predicted = prediction.raw_cartesian.detach().cpu()
                expected = batch.target_cartesian.detach().cpu()
                prediction_rows.extend(
                    {
                        "sample_id": sample_id,
                        "prediction": predicted[index].tolist(),
                        "target": expected[index].tolist(),
                    }
                    for index, sample_id in enumerate(batch.sample_ids)
                )
    predicted_cartesian = torch.cat(predictions)
    target_cartesian = torch.cat([example.target_cartesian for example in examples])
    predicted_coefficients = torch.cat(coefficients)
    target_coefficients = torch.cat(targets)
    return (
        total_loss / len(examples),
        tensor_benchmark_metrics(predicted_cartesian, target_cartesian, task=unit.target),
        physical_coefficient_metrics(
            predicted_coefficients, target_coefficients, TARGET_LAYOUTS[unit.target]
        ),
    )


def benchmark_target_comparison(task: str, metrics: dict[str, float | int]) -> dict[str, bool]:
    targets = PUBLIC_TARGETS[task]
    return {
        name: float(metrics[name]) < value if name == "fnorm" else float(metrics[name]) > value
        for name, value in targets.items()
    }


def train_cached_backbone_readout(
    *,
    backbone_family: str,
    unit: TrainingUnit,
    source_layout: IrrepLayout,
    train_examples: Sequence[FrozenFeatureExample],
    validation_examples: Sequence[FrozenFeatureExample],
    test_examples: Sequence[FrozenFeatureExample],
    checkpoint_path: str | Path,
    config: BenchmarkConfig = BenchmarkConfig(),
    device: torch.device | str = "cuda",
    model_builder: Callable[[IrrepLayout, str], PointGroupTensorModel] | None = None,
    architecture: ArchitectureConfig | None = None,
    expert_point_groups: tuple[str, ...] = (),
    predictions_path: str | Path | None = None,
) -> dict[str, object]:
    """Train a tensor architecture over a once-materialized frozen backbone tap."""

    if unit.target not in PUBLIC_TARGETS:
        raise ValueError("this benchmark runner supports dielectric/elastic targets")
    if not train_examples or not validation_examples or not test_examples:
        raise ValueError("all published benchmark splits must be non-empty")
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    requested_architecture = architecture
    architecture = architecture or ArchitectureConfig("B+R", "none", "none", "full_o3", "none")
    if model_builder is None:
        if requested_architecture is None:
            model = PointGroupTensorModel(
                architecture,
                unit.target,
                hidden_layout=source_layout,
                expert_point_groups=(),
            ).to(device)
        else:
            model = CachedBackboneTensorModel(
                source_layout,
                architecture,
                unit.target,
                expert_point_groups,
            ).to(device)
    else:
        model = model_builder(source_layout, unit.target).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=max(1, config.patience // 3)
    )
    normalizer = CoefficientNormalizer.fit(
        torch.cat([example.target_coefficients for example in train_examples]),
        TARGET_LAYOUTS[unit.target],
        unit,
        split="train",
        mode=config.normalization,
    )
    convention = default_convention_metadata()
    best = float("inf")
    best_epoch = 0
    stale = 0
    history = []
    for epoch in range(1, config.max_epochs + 1):
        model.train()
        total = 0.0
        for rows in _batches(train_examples, config.batch_size, shuffle_seed=config.seed + epoch):
            batch = collate_frozen_examples(rows, source_layout, device=device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(batch.features, batch.graph, batch.symmetries)
            loss = coefficient_mse(
                prediction.irrep_coefficients,
                batch.target_coefficients,
                TARGET_LAYOUTS[unit.target],
                normalizer,
            ).total
            if not torch.isfinite(loss):
                raise ValueError("benchmark training loss is non-finite")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
            optimizer.step()
            total += float(loss.detach()) * len(rows)
        validation_loss, validation_metrics, _ = _evaluate(
            model,
            validation_examples,
            source_layout,
            unit,
            normalizer,
            batch_size=config.batch_size,
            device=device,
        )
        scheduler.step(validation_loss)
        history.append(
            {
                "epoch": epoch,
                "train_loss": total / len(train_examples),
                "validation_loss": validation_loss,
                "validation_fnorm": validation_metrics["fnorm"],
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
        )
        if validation_loss < best:
            best, best_epoch, stale = validation_loss, epoch, 0
            save_checkpoint(
                checkpoint_path,
                model=model,
                optimizer=optimizer,
                architecture=architecture,
                unit=unit,
                convention=convention,
                normalizer=normalizer,
                step=epoch,
            )
        else:
            stale += 1
            if stale >= config.patience:
                break
    loaded = load_checkpoint(
        checkpoint_path,
        model=model,
        optimizer=None,
        expected_architecture=architecture,
        expected_unit=unit,
        expected_convention=convention,
        expected_layout=TARGET_LAYOUTS[unit.target],
        map_location=device,
    )
    prediction_rows: list[dict[str, object]] = []
    test_loss, test_metrics, irrep_metrics = _evaluate(
        model,
        test_examples,
        source_layout,
        unit,
        loaded.normalizer,
        batch_size=config.batch_size,
        device=device,
        prediction_rows=prediction_rows,
    )
    if predictions_path is not None:
        target = Path(predictions_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f".{target.name}.tmp")
        try:
            with temporary.open("w", encoding="utf-8", newline="\n") as stream:
                for row in prediction_rows:
                    stream.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
            os.replace(temporary, target)
        finally:
            if temporary.exists():
                temporary.unlink()
    report = {
        "schema_version": 1,
        "status": "passed",
        "backbone": backbone_family,
        "training_unit": unit.namespace,
        "architecture": architecture.to_dict(),
        "source_layout": list(source_layout.to_spec()),
        "config": asdict(config),
        "split_counts": {
            "train": len(train_examples),
            "validation": len(validation_examples),
            "test": len(test_examples),
        },
        "best_epoch": best_epoch,
        "best_validation_loss": best,
        "test_loss": test_loss,
        "test_metrics": test_metrics,
        "test_irrep_metrics": irrep_metrics,
        "expert_point_groups": list(expert_point_groups),
        "routing": "current_point_group_only",
        "predictions": None if predictions_path is None else str(predictions_path),
        "history": history,
        "checkpoint": str(checkpoint_path),
    }
    if unit.dataset == "jarvis_tensor":
        report["public_targets"] = PUBLIC_TARGETS[unit.target]
        report["exceeds_public_target"] = benchmark_target_comparison(unit.target, test_metrics)
    return report
