"""One paired-grid training run. Execute via Slurm, one task per Python process."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import time
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from ...baselines.gmtnet.runner import (
    GMTNET_OFFICIAL_COMMIT, GMTNetConfig, _batches, _collate, _learning_rate_after_step,
    _predict, _replace_atom_embedding, _write_predictions,
)
from ...cli.reporting import execution_metadata
from ...evaluation import tensor_benchmark_metrics
from ...features import cgcnn_node_features
from .data import dataset_from_records, standard_voigt
from .prepare import official_modules, prepare_graph, save_tensor, synchronize
from .protocol import grid, load_records, nested_ids, run_name, sha256, write_json


def metrics(prediction, target, task):
    if task == "elastic":
        prediction, target = standard_voigt(prediction), standard_voigt(target)
    result = tensor_benchmark_metrics(prediction.double(), target.double())
    result["mae"] = float((prediction.double()-target.double()).abs().mean())
    return result


def attach(row, feature):
    expected = cgcnn_node_features(feature["atomic_numbers"]).to(row["graph"]["x"])
    if expected.shape != row["graph"]["x"].shape or not torch.allclose(
        expected, row["graph"]["x"], atol=1e-6, rtol=1e-6
    ):
        raise ValueError("graph and pretrained feature atom order mismatch")
    embedding = feature["embedding"]
    if not torch.isfinite(embedding).all() or embedding.shape[0] != expected.shape[0]:
        raise ValueError("invalid pretrained embeddings")
    return {**row, "graph": {**row["graph"], "x": embedding.to(expected)}}


def load_cache(path, provenance, *, smoke, inference=False):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload["provenance"] != provenance or payload["smoke"] != smoke or payload["inference"] != inference:
        raise ValueError(f"cache provenance mismatch: {path}")
    return payload


def run(args):
    spec = grid()[args.index]
    task = spec["task"]
    out = args.output / run_name(spec)
    if (out / "summary.json").exists():
        raise FileExistsError(f"completed run exists: {out}")
    out.mkdir(parents=True, exist_ok=True)
    rows, provenance = load_records(task, group_scope=args.group_scope)
    ids = nested_ids(rows, spec["fraction"], smoke=args.smoke)
    prep = args.cache / task
    graphs = load_cache(prep / "graphs.pt", provenance, smoke=args.smoke)
    splits = {name: [graphs["by_id"][sid] for sid in values] for name, values in ids.items()}
    features = None
    if spec["model"] == "pretrain":
        features = load_cache(prep / "features.pt", provenance, smoke=args.smoke)
        from ...backbones import BackboneResourceRegistry
        if features["checkpoint_sha256"] != BackboneResourceRegistry()["dpa4"].sha256:
            raise ValueError("pretrained checkpoint fingerprint mismatch")
        splits = {name: [attach(row, features["by_id"][row["sample_id"]]) for row in values]
                  for name, values in splits.items()}
    model_module, official_graphs, official_data = official_modules(args.official_root, task)
    from torch_geometric.data import Batch
    data_type = official_graphs.Data
    config = GMTNetConfig(epochs=args.epochs, seed=spec["seed"], batch_size=64,
                         minimum_checkpoint_epoch_exclusive=(0 if args.smoke else args.epochs // 2))
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    factory = model_module.GMTNet if task == "dielectric" else model_module.ComformerEquivariant
    # Match the pinned official per-task defaults; identical settings across the pair.
    model = factory(SimpleNamespace(target=task, use_mask=(task == "dielectric"), reduce_cell=False)).to(args.device)
    if features is not None:
        _replace_atom_embedding(model, splits["train"][0]["graph"]["x"].shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate,
                                  weight_decay=config.weight_decay)
    batch_size = min(config.batch_size, len(splits["train"]))
    steps_per_epoch = len(splits["train"]) // batch_size
    criterion = nn.HuberLoss()
    best = float("inf")
    history = []
    step = 0
    synchronize(args.device)
    train_start = time.perf_counter()
    for epoch in range(1, config.epochs+1):
        epoch_start = time.perf_counter()
        model.train()
        loss_sum, fnorm_sum, seen = 0.0, 0.0, 0
        for batch_rows in _batches(splits["train"], batch_size, seed=config.seed+epoch, drop_last=True):
            graph, mask, equality, labels = _collate(batch_rows, data_type, Batch, args.device)
            optimizer.zero_grad(set_to_none=True)
            output = model(graph, mask, equality)
            loss = criterion(output, labels)
            if not torch.isfinite(loss):
                raise ValueError("non-finite training loss")
            loss.backward()
            optimizer.step()
            step += 1
            lr = _learning_rate_after_step(config, step=step, steps_per_epoch=steps_per_epoch)
            for group in optimizer.param_groups:
                group["lr"] = lr
            n = len(batch_rows)
            loss_sum += float(loss.detach()) * n
            fnorm_sum += float(torch.linalg.vector_norm((output.detach()-labels).flatten(1), dim=1).sum())
            seen += n
        synchronize(args.device)
        optimization_seconds = time.perf_counter()-epoch_start
        validation_start = time.perf_counter()
        pred = _predict(model, splits["validation"], config.batch_size, data_type, Batch, args.device)
        target = torch.stack([row["target"] for row in splits["validation"]])
        validation = metrics(pred, target, task)
        synchronize(args.device)
        validation_seconds = time.perf_counter()-validation_start
        row = {"epoch": epoch, "training_loss": loss_sum/seen, "training_fnorm_online": fnorm_sum/seen,
               "validation_loss": float(criterion(pred, target)), "validation_fnorm": validation["fnorm"],
               "validation_mae": validation["mae"], "learning_rate": lr,
               "optimization_seconds": optimization_seconds, "validation_seconds": validation_seconds}
        # Validation Fnorm is the prespecified primary selector for both tasks/models.
        if epoch > config.minimum_checkpoint_epoch_exclusive and validation["fnorm"] < best:
            best, best_epoch = validation["fnorm"], epoch
            save_tensor(out / "best.pt", {"state_dict": model.state_dict(), "spec": spec,
                                         "provenance": provenance, "epoch": epoch})
        synchronize(args.device)
        row["epoch_seconds"] = time.perf_counter()-epoch_start
        row["elapsed_training_seconds"] = time.perf_counter()-train_start
        history.append(row)
        write_json(out / "history.json", history)
        print(json.dumps(row), flush=True)
    synchronize(args.device)
    training_seconds = time.perf_counter()-train_start
    saved = torch.load(out / "best.pt", map_location=args.device, weights_only=True)
    model.load_state_dict(saved["state_dict"], strict=True)
    test_pred = _predict(model, splits["test"], config.batch_size, data_type, Batch, args.device)
    test_target = torch.stack([row["target"] for row in splits["test"]])
    test_metrics = metrics(test_pred, test_target, task)
    _write_predictions(out / "predictions.jsonl", splits["test"], test_pred)
    inference = benchmark_inference(args, task, provenance, rows, model, data_type, Batch,
                                    official_data, official_graphs, features is not None)
    # Exact per-record measured extraction costs, plus fixed initialization and full
    # cache serialization overhead. No linear extrapolation from sample counts.
    preparation_ids = ids["train"] + ids["validation"]
    graph_meta = json.loads((prep / "graphs.json").read_text())
    graph_seconds = sum(graphs["seconds_by_id"][sid] for sid in preparation_ids) + graphs["initialization_seconds"] + graph_meta["save_seconds"]
    feature_seconds = 0.0
    if features is not None:
        feature_meta = json.loads((prep / "features.json").read_text())
        feature_seconds = sum(features["seconds_by_id"][sid] for sid in preparation_ids) + features["initialization_seconds"] + feature_meta["save_seconds"]
    report = {"status": "passed", "spec": spec, "provenance": provenance, "split_ids": ids,
              "epochs": args.epochs, "smoke": args.smoke, "best_epoch": best_epoch,
              "best_validation_fnorm": best, "selection": "validation Fnorm; second half of training",
              "official_commit": GMTNET_OFFICIAL_COMMIT, "use_equivariant_attention": False,
              "history": history, "test_metrics": test_metrics, "execution": execution_metadata(),
              "checkpoint_sha256": sha256(out / "best.pt"), "prediction_sha256": sha256(out / "predictions.jsonl"),
              "timing": {"feature_preparation_seconds": feature_seconds,
                         "graph_preparation_seconds": graph_seconds, "training_seconds": training_seconds,
                         "total_seconds": feature_seconds+graph_seconds+training_seconds,
                         "preparation_accounting": "selected train+validation record costs + initialization + full serialization overhead; reused per seed, not extra work",
                         "inference": inference}}
    write_json(out / "summary.json", report)


def benchmark_inference(args, task, provenance, rows, model, data_type, batch_type, data, graphs, pretrain):
    """Batch-one warm resident-model latency from uncached structures, excluding I/O."""
    dataset = dataset_from_records(rows, provenance)
    selected = list(dataset.split_manifest.test[:(8 if args.smoke else 32)])
    fresh = None
    if pretrain:
        fresh = load_cache(args.cache / task / "inference_features.pt", provenance,
                           smoke=args.smoke, inference=True)
        selected = [sid for sid in selected if sid in fresh["by_id"]]
    # Warm downstream kernels once; graph construction remains inside timed trials.
    graph_row = prepare_graph(dataset.by_id(selected[0]), data, graphs)
    if pretrain:
        graph_row = attach(graph_row, fresh["by_id"][selected[0]])
    _predict(model, [graph_row], 1, data_type, batch_type, args.device)
    trials = []
    for sid in selected:
        synchronize(args.device)
        start = time.perf_counter()
        row = prepare_graph(dataset.by_id(sid), data, graphs)
        graph_time = time.perf_counter()-start
        if pretrain:
            row = attach(row, fresh["by_id"][sid])
        synchronize(args.device)
        start = time.perf_counter()
        pred = _predict(model, [row], 1, data_type, batch_type, args.device)
        synchronize(args.device)
        downstream_time = time.perf_counter()-start
        feature_time = fresh["seconds_by_id"][sid] if pretrain else 0.0
        trials.append({"sample_id": sid, "feature_seconds": feature_time, "graph_seconds": graph_time,
                       "downstream_seconds": downstream_time,
                       "total_seconds": feature_time+graph_time+downstream_time})
        if not torch.isfinite(pred).all():
            raise ValueError("non-finite uncached inference")
    return {"batch_size": 1, "samples": trials,
            "mean_seconds_per_structure": float(np.mean([r["total_seconds"] for r in trials])),
            "feature_initialization_seconds": fresh["initialization_seconds"] if pretrain else 0.0,
            "definition": "sum of separately measured fresh feature extraction, graph/mask construction and downstream prediction; resident models; no cache-hit timing"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=int, choices=range(48), required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--official-root", type=Path, required=True)
    parser.add_argument("--group-scope", choices=("all", "seven"), default="all")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--smoke", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
