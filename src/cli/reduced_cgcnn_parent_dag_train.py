from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import time

from ..data import TrainingUnit, load_training_dataset
from ..features import CGCNN_SOURCE_LAYOUT, cgcnn_feature_metadata, cgcnn_feature_sha256
from ..models import CGCNNFeatureTensorModel
from ..symmetry import (
    DEFAULT_PARENT_SYMPRECS,
    discover_material_parent_routing,
    load_parent_routing_cache,
    parent_detection_config_sha256,
    parent_routing_coverage,
    save_parent_routing_cache,
)
from ..training import (
    BenchmarkConfig,
    load_frozen_feature_cache,
    save_frozen_feature_cache,
    train_cached_backbone_readout,
    write_smoke_report,
)
from .reduced_cgcnn_full_pg_train import (
    ARCHITECTURE,
    BACKBONE_NAME,
    GRAPH_CUTOFF_ANGSTROM,
    _materialize_examples,
)
from .reporting import execution_metadata, write_single_case_junit
from .reduced_protocol import (
    REDUCED_POINT_GROUPS,
    canonical_expert_point_groups,
    point_group_stratified_smoke_ids,
)


MODEL_NAME = "CGCNN B+A+PGE+R full_pg parent-DAG"


def _parent_enriched_examples(
    examples, cache_path: Path, *, sample_ids, dataset_sha256: str
):
    if cache_path.is_file():
        routings = load_parent_routing_cache(
            cache_path, sample_ids=sample_ids, dataset_sha256=dataset_sha256
        )
    else:
        values = []
        for index, example in enumerate(examples, start=1):
            values.append(
                discover_material_parent_routing(
                    example.sample_id,
                    example.graph.positions,
                    example.graph.cell[0],
                    example.graph.atomic_numbers,
                    example.symmetry,
                )
            )
            if index == 1 or index == len(examples) or index % 25 == 0:
                print(
                    json.dumps(
                        {
                            "event": "material_parent_dag_detection",
                            "current": index,
                            "total": len(examples),
                            "sample_id": example.sample_id,
                            "parent_count": len(values[-1].dag.embeddings),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
        routings = tuple(values)
        save_parent_routing_cache(
            cache_path,
            routings,
            sample_ids=sample_ids,
            dataset_sha256=dataset_sha256,
        )
    enriched = tuple(
        replace(
            example,
            parent_dag=routing.dag,
            parent_residuals=dict(routing.residuals),
        )
        for example, routing in zip(examples, routings)
    )
    return enriched, parent_routing_coverage(routings)


def run(arguments: argparse.Namespace) -> dict[str, object]:
    unit = TrainingUnit("curated_reduced_total", "dielectric")
    dataset = load_training_dataset(unit, manifest_path=arguments.manifest)
    expected = {"train": 5001, "validation": 637, "test": 677}
    full_ids = {
        name: tuple(getattr(dataset.split_manifest, name))
        for name in ("train", "validation", "test")
    }
    if {name: len(ids) for name, ids in full_ids.items()} != expected:
        raise ValueError("reduced dielectric_total split drift")
    selected_ids = point_group_stratified_smoke_ids(dataset) if arguments.smoke else full_ids
    dataset_sha256 = str(dataset[0].source["manifest_sha256"])
    feature_sha256 = cgcnn_feature_sha256()
    scope = "smoke" if arguments.smoke else "full"
    splits, coverage = {}, {}
    for split in ("train", "validation", "test"):
        feature_cache = (
            arguments.cache_root
            / "cgcnn-full-pg"
            / unit.namespace
            / scope
            / f"{split}.pt"
        )
        if feature_cache.is_file():
            layout, examples = load_frozen_feature_cache(
                feature_cache,
                expected_backbone=BACKBONE_NAME,
                expected_checkpoint_sha256=feature_sha256,
                expected_unit=unit,
                expected_split=split,
                expected_sample_ids=selected_ids[split],
                expected_dataset_sha256=dataset_sha256,
            )
            if layout != CGCNN_SOURCE_LAYOUT:
                raise ValueError("cached CGCNN source layout mismatch")
        else:
            examples = _materialize_examples(dataset, selected_ids[split], progress=True)
            save_frozen_feature_cache(
                feature_cache,
                backbone_family=BACKBONE_NAME,
                checkpoint_sha256=feature_sha256,
                unit=unit,
                split=split,
                layout=CGCNN_SOURCE_LAYOUT,
                examples=examples,
                dataset_sha256=dataset_sha256,
            )
        routing_cache = (
            arguments.cache_root
            / "cgcnn-parent-dag"
            / unit.namespace
            / scope
            / f"{split}.pt"
        )
        splits[split], coverage[split] = _parent_enriched_examples(
            examples,
            routing_cache,
            sample_ids=selected_ids[split],
            dataset_sha256=dataset_sha256,
        )

    expert_point_groups = canonical_expert_point_groups(
        splits["train"], splits["validation"], splits["test"]
    )
    common = {
        "schema_version": 1,
        "status": "passed",
        "model": MODEL_NAME,
        "routing": "material_parent_dag",
        "training_unit": unit.namespace,
        "dataset_sha256": dataset_sha256,
        "feature_embedding": cgcnn_feature_metadata(),
        "parent_detection": {
            "symprecs_angstrom": list(DEFAULT_PARENT_SYMPRECS),
            "config_sha256": parent_detection_config_sha256(),
            "coverage_by_split": coverage,
        },
        "expert_point_groups": list(expert_point_groups),
        "split_counts": {name: len(rows) for name, rows in splits.items()},
        "execution": execution_metadata(),
    }
    if arguments.prepare_only:
        return {**common, "mode": "prepare_only"}

    report = train_cached_backbone_readout(
        backbone_family=BACKBONE_NAME,
        unit=unit,
        source_layout=CGCNN_SOURCE_LAYOUT,
        train_examples=splits["train"],
        validation_examples=splits["validation"],
        test_examples=splits["test"],
        checkpoint_path=arguments.checkpoint,
        predictions_path=arguments.predictions,
        architecture=ARCHITECTURE,
        expert_point_groups=expert_point_groups,
        model_builder=lambda _layout, task: CGCNNFeatureTensorModel(
            ARCHITECTURE, task, expert_point_groups
        ),
        config=BenchmarkConfig(
            max_epochs=arguments.epochs,
            batch_size=arguments.batch_size,
            learning_rate=arguments.learning_rate,
            end_learning_rate=arguments.end_learning_rate,
            weight_decay=arguments.weight_decay,
            patience=arguments.epochs,
            seed=arguments.seed,
            training_protocol="gmtnet",
        ),
        device=arguments.device,
    )
    report.update(common)
    report["graph_cutoff_angstrom"] = GRAPH_CUTOFF_ANGSTROM
    report["source_retained_point_groups"] = list(REDUCED_POINT_GROUPS)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train CGCNN full-PG with material-specific parent-DAG routing"
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/manifests/curated_tensors_reduced_gt_5pct.json"),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--junit", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--end-learning-rate", type=float, default=1.0e-5)
    parser.add_argument("--weight-decay", type=float, default=1.0e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    arguments = parser.parse_args()
    started = time.perf_counter()
    error = None
    try:
        report = run(arguments)
    except Exception as caught:
        error = caught
        report = {
            "schema_version": 1,
            "status": "failed",
            "error_type": type(caught).__name__,
            "error": str(caught),
            "execution": execution_metadata(),
        }
    write_smoke_report(arguments.summary, report)
    write_single_case_junit(
        arguments.junit,
        suite_name="reduced_cgcnn_parent_dag",
        seconds=time.perf_counter() - started,
        error=error,
    )
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    if error is not None:
        raise error


if __name__ == "__main__":
    main()
