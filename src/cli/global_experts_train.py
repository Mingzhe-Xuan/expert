"""Train the additive algorithm.md GMTNet + hierarchical Full-PG model."""

import argparse
from dataclasses import fields
import json
from pathlib import Path
import random
from types import SimpleNamespace

import numpy as np
import torch

from ..baselines.gmtnet.runner import (
    GMTNET_OFFICIAL_COMMIT,
    _prepare_cache,
    _replace_atom_embedding,
    _attach_dpa4_node_embeddings,
)
from ..backbones import BackboneResourceRegistry
from ..data import TrainingUnit, load_training_dataset
from ..models.global_experts import GlobalExpertsConfig, GlobalExpertsModel
from ..symmetry import PointGroupAncestorDAG
from ..training.global_experts import GlobalExpertsTrainConfig, train_global_experts
from ..training.global_experts.data import (
    prepare_routing,
    file_sha256,
    attach_invariant_inputs,
)
from ..training.global_experts.official import load_official_modules
from .reduced_protocol import point_group_stratified_smoke_ids
from .reduced_dpa4_gmtnet_train import _load_dpa4_feature_splits


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--official-root", type=Path, required=True)
    result.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/manifests/curated_tensors_reduced_gt_5pct.json"),
    )
    result.add_argument("--graph-cache", type=Path, required=True)
    result.add_argument("--routing-cache", type=Path, required=True)
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--device", default="cuda")
    result.add_argument("--symprec", type=float, default=1e-5)
    result.add_argument("--smoke", action="store_true")
    result.add_argument("--prepare-only", action="store_true")
    result.add_argument("--global-checkpoint", type=Path)
    result.add_argument("--global-checkpoint-sha256")
    result.add_argument("--node-features", type=Path)
    result.add_argument("--node-feature-sha256")
    result.add_argument("--input-features", choices=("dpa4", "cgcnn"), default="dpa4")
    result.add_argument(
        "--dpa-cache-root", type=Path, default=Path("results/reduced-benchmark/cache")
    )
    result.add_argument("--feature-shards", type=int, default=64)
    for config_type in (GlobalExpertsConfig, GlobalExpertsTrainConfig):
        for field in fields(config_type):
            option = "--" + field.name.replace("_", "-")
            if isinstance(field.default, bool):
                result.add_argument(
                    option, action=argparse.BooleanOptionalAction, default=field.default
                )
            else:
                result.add_argument(
                    option,
                    type=float if field.default is None else type(field.default),
                    default=field.default,
                )
    return result


def run(arguments, *, executor=train_global_experts, split_selector=None):
    if bool(arguments.node_features) != bool(arguments.node_feature_sha256):
        raise ValueError("--node-features and --node-feature-sha256 are required together")
    if arguments.input_features == "cgcnn" and arguments.node_features:
        raise ValueError("--input-features cgcnn conflicts with --node-features")
    model_config = GlobalExpertsConfig(
        **{f.name: getattr(arguments, f.name) for f in fields(GlobalExpertsConfig)}
    )
    training_config = GlobalExpertsTrainConfig(
        **{f.name: getattr(arguments, f.name) for f in fields(GlobalExpertsTrainConfig)}
    )
    dataset = load_training_dataset(
        TrainingUnit("curated_reduced_total", "dielectric"),
        manifest_path=arguments.manifest,
    )
    selected = (
        point_group_stratified_smoke_ids(dataset)
        if arguments.smoke
        else {
            key: tuple(getattr(dataset.split_manifest, key))
            for key in ("train", "validation", "test")
        }
    )
    if split_selector is not None:
        selected = split_selector(dataset, selected)
    dataset_hash = str(dataset[0].source["manifest_sha256"])
    official, _, _ = load_official_modules(arguments.official_root)
    splits = _prepare_cache(
        dataset, arguments.official_root, arguments.graph_cache, dataset_hash, selected
    )
    for name, rows in splits.items():
        if [r["sample_id"] for r in rows] != list(selected[name]):
            raise ValueError("cached graph sample order mismatch")
    class_dag = PointGroupAncestorDAG.from_path()
    splits, provenance = prepare_routing(
        splits,
        dataset,
        arguments.routing_cache,
        dataset_sha256=dataset_hash,
        graph_cache_sha256=file_sha256(arguments.graph_cache),
        class_dag=class_dag,
        symprec=arguments.symprec,
    )
    provenance["input_embedding"] = {"kind": "jarvis_cgcnn", "input_dimension": 92}
    if arguments.node_features:
        splits, provenance["input_embedding"] = attach_invariant_inputs(
            splits,
            arguments.node_features,
            expected_sha256=arguments.node_feature_sha256,
        )
    elif arguments.node_feature_sha256:
        raise ValueError("--node-feature-sha256 requires --node-features")
    elif arguments.input_features == "dpa4":
        resource = BackboneResourceRegistry()["dpa4"]
        if not resource.sha256:
            raise ValueError("DPA4 resource lacks a frozen checkpoint SHA-256")
        layout, feature_splits = _load_dpa4_feature_splits(
            cache_root=arguments.dpa_cache_root,
            unit=dataset.unit,
            full_ids={
                name: tuple(getattr(dataset.split_manifest, name))
                for name in ("train", "validation", "test")
            },
            selected_ids=selected,
            shard_count=arguments.feature_shards,
            checkpoint_sha256=resource.sha256,
            dataset_sha256=dataset_hash,
        )
        splits = _attach_dpa4_node_embeddings(splits, feature_splits, layout)
        provenance["input_embedding"] = {
            "kind": "frozen_dpa4_o3_invariant_per_copy",
            "input_dimension": sum(term.multiplicity for term in layout.terms),
            "source_dimension": layout.dimension,
            "checkpoint_sha256": resource.sha256,
            "feature_shards": arguments.feature_shards,
        }
    if arguments.prepare_only:
        return {
            "status": "prepared",
            "provenance": provenance,
            "split_counts": {name: len(rows) for name, rows in splits.items()},
        }
    random.seed(training_config.seed)
    np.random.seed(training_config.seed)
    torch.manual_seed(training_config.seed)
    baseline = official.GMTNet(
        SimpleNamespace(target="dielectric", use_mask=True, reduce_cell=False)
    )
    width = provenance["input_embedding"]["input_dimension"]
    if width != 92:
        _replace_atom_embedding(baseline, width)
    # Construct the backbone's optional attention before loading a matching checkpoint.
    from ..baselines.gmtnet import configure_equivariant_attention

    configure_equivariant_attention(
        baseline, use_equiv_attn=model_config.use_equiv_attn
    )
    if arguments.global_checkpoint:
        digest = file_sha256(arguments.global_checkpoint)
        if digest != arguments.global_checkpoint_sha256:
            raise ValueError("global checkpoint hash mismatch")
        saved = torch.load(
            arguments.global_checkpoint, map_location="cpu", weights_only=True
        )
        if saved.get("official_commit") != GMTNET_OFFICIAL_COMMIT:
            raise ValueError("global checkpoint official source mismatch")
        missing, unexpected = baseline.load_state_dict(
            saved["model_state"], strict=False
        )
        if unexpected or any(".attn_mlp." not in name for name in missing):
            raise ValueError("incompatible global checkpoint parameters")
        provenance["global_initialization_sha256"] = digest
    elif arguments.global_checkpoint_sha256:
        raise ValueError("--global-checkpoint-sha256 requires --global-checkpoint")
    rows = [row for split in splits.values() for row in split]
    active = sorted(
        {
            number
            for row in rows
            for number in class_dag.ancestors(
                row["routing"].dag.current_point_group_number
            )
        }
    )
    edge_ids = sorted(
        {e.edge_id for row in rows for e in row["routing"].dag.embeddings}
    )
    model = GlobalExpertsModel(
        baseline,
        official.equality_adjustment,
        expert_numbers=active,
        edge_ids=edge_ids,
        config=model_config,
        class_dag=class_dag,
    )
    return executor(
        model,
        splits,
        output_dir=arguments.output_dir,
        provenance=provenance,
        config=training_config,
        device=arguments.device,
    )


def main():
    arguments = parser().parse_args()
    result = run(arguments)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
