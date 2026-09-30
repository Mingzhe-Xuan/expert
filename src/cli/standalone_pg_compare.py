"""Compare standalone PG execution paths on real frozen DPA training examples."""
import argparse
from pathlib import Path

from ..backbones import BackboneResourceRegistry
from ..data import TrainingUnit, load_training_dataset
from ..profiling.standalone_pg import compare_standalone_pg
from ..symmetry import PointGroupAncestorDAG
from .global_experts_compare import select_profile_splits
from .reduced_protocol import point_group_stratified_smoke_ids, canonical_expert_point_groups
from .reduced_dpa4_gmtnet_train import _load_dpa4_feature_splits
from .reduced_dpa4_relative_pg_train import ARCHITECTURE
from .reduced_cgcnn_parent_dag_train import _parent_enriched_examples, _shared_edge_ids


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/manifests/curated_tensors_reduced_gt_5pct.json"))
    parser.add_argument("--dpa-cache-root", type=Path, default=Path("results/reduced-benchmark/cache"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-samples", type=int, default=7)
    parser.add_argument("--feature-shards", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("use a fresh comparison directory")
    unit = TrainingUnit("curated_reduced_total", "dielectric")
    dataset = load_training_dataset(unit, manifest_path=args.manifest)
    selected = select_profile_splits(dataset, point_group_stratified_smoke_ids(dataset), args.train_samples)
    dataset_hash = str(dataset[0].source["manifest_sha256"])
    resource = BackboneResourceRegistry()["dpa4"]
    layout, splits = _load_dpa4_feature_splits(cache_root=args.dpa_cache_root, unit=unit,
        full_ids={n: tuple(getattr(dataset.split_manifest, n)) for n in ("train", "validation", "test")},
        selected_ids=selected, shard_count=args.feature_shards,
        checkpoint_sha256=resource.sha256, dataset_sha256=dataset_hash)
    classes = PointGroupAncestorDAG.from_path()
    # Unique per-run native-DPA routing; never reuse the GMTNet k-neighbour graph cache.
    rows, coverage = _parent_enriched_examples(splits["train"],
        args.output_dir.parent / (args.output_dir.name + "-native-dpa-routing.pt"),
        sample_ids=selected["train"], dataset_sha256=dataset_hash, class_dag=classes)
    compare_standalone_pg(layout, ARCHITECTURE, rows,
        expert_point_groups=canonical_expert_point_groups(rows), edge_ids=_shared_edge_ids({"train": rows}),
        output_dir=args.output_dir, device=args.device,
        provenance={"dataset_sha256": dataset_hash, "checkpoint_sha256": resource.sha256,
                    "feature_shards": args.feature_shards, "routing_coverage": coverage,
                    "input": "native frozen DPA4 equivariant source, canonical frame"})


if __name__ == "__main__":
    main()
