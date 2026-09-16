from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from ..configs import ArchitectureConfig
from ..data import TrainingUnit, load_training_dataset
from ..features import (
    CGCNN_SOURCE_LAYOUT,
    cgcnn_feature_metadata,
    cgcnn_feature_sha256,
    cgcnn_node_features,
)
from ..heads import irreps_to_cartesian
from ..models import CGCNNFeatureTensorModel
from ..training import (
    BenchmarkConfig,
    FrozenFeatureExample,
    load_frozen_feature_cache,
    prepare_tensor_batch,
    save_frozen_feature_cache,
    train_cached_backbone_readout,
    write_smoke_report,
)
from .reporting import execution_metadata, write_single_case_junit
from .reduced_protocol import (
    REDUCED_POINT_GROUPS,
    canonical_expert_point_groups,
    point_group_stratified_smoke_ids,
)


ARCHITECTURE = ArchitectureConfig(
    "B+A+PGE+R", "full_o3", "none", "full_o3", "full_pg"
)
BACKBONE_NAME = "gmtnet_cgcnn_features"
GRAPH_CUTOFF_ANGSTROM = 4.0


def _materialize_examples(dataset, sample_ids, *, progress):
    examples = []
    for index, sample_id in enumerate(sample_ids, start=1):
        sample = dataset.by_id(sample_id)
        prepared = prepare_tensor_batch(
            (sample,), cutoff=GRAPH_CUTOFF_ANGSTROM, device="cpu"
        )
        examples.append(
            FrozenFeatureExample(
                sample_id=sample.sample_id,
                features=cgcnn_node_features(prepared.graph.atomic_numbers),
                graph=prepared.graph,
                symmetry=prepared.symmetries[0],
                target_coefficients=prepared.target_coefficients,
                target_cartesian=irreps_to_cartesian(
                    prepared.target_coefficients, sample.unit.target
                ),
            )
        )
        if progress and (index == 1 or index == len(sample_ids) or index % 25 == 0):
            print(
                json.dumps(
                    {
                        "event": "cgcnn_feature_materialization",
                        "current": index,
                        "total": len(sample_ids),
                        "sample_id": sample_id,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    return tuple(examples)


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
    splits = {}
    for split in ("train", "validation", "test"):
        cache = arguments.cache_root / "cgcnn-full-pg" / unit.namespace / scope / f"{split}.pt"
        if cache.is_file():
            layout, examples = load_frozen_feature_cache(
                cache,
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
            examples = _materialize_examples(
                dataset, selected_ids[split], progress=True
            )
            save_frozen_feature_cache(
                cache,
                backbone_family=BACKBONE_NAME,
                checkpoint_sha256=feature_sha256,
                unit=unit,
                split=split,
                layout=CGCNN_SOURCE_LAYOUT,
                examples=examples,
                dataset_sha256=dataset_sha256,
            )
        splits[split] = examples
    if arguments.prepare_only:
        return {
            "schema_version": 1,
            "status": "passed",
            "mode": "prepare_only",
            "model": "CGCNN B+A+PGE+R full_pg",
            "training_unit": unit.namespace,
            "dataset_sha256": dataset_sha256,
            "feature_embedding": cgcnn_feature_metadata(),
            "split_counts": {name: len(rows) for name, rows in splits.items()},
            "execution": execution_metadata(),
        }
    expert_point_groups = canonical_expert_point_groups(
        splits["train"], splits["validation"], splits["test"]
    )
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
    report["model"] = "CGCNN B+A+PGE+R full_pg"
    report["dataset_sha256"] = dataset_sha256
    report["feature_embedding"] = cgcnn_feature_metadata()
    report["graph_cutoff_angstrom"] = GRAPH_CUTOFF_ANGSTROM
    report["source_retained_point_groups"] = list(REDUCED_POINT_GROUPS)
    report["execution"] = execution_metadata()
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train CGCNN-feature B+A+PGE+R/full_pg with GMTNet-aligned optimization"
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
        suite_name="reduced_cgcnn_full_pg",
        seconds=time.perf_counter() - started,
        error=error,
    )
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    if error is not None:
        raise error


if __name__ == "__main__":
    main()
