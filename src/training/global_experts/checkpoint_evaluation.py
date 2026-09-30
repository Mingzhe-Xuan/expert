"""Reuse retained global-experts checkpoints without constructing an optimizer."""

from dataclasses import asdict
import json
import math
from pathlib import Path
import time

import torch

from ...evaluation import tensor_benchmark_metrics
from .data import file_sha256
from .runner import load_checkpoint, normalized_training_config, predict


def select_retained(source, source_dir, minimum_epoch=100):
    history = source["history"]
    if minimum_epoch < 0 or [r["epoch"] for r in history] != list(
        range(1, source["training_config"]["epochs"] + 1)
    ):
        raise ValueError("invalid epoch boundary or noncontiguous history")
    if any(not math.isfinite(r["validation_mae"]) for r in history):
        raise ValueError("nonfinite validation metric")
    eligible = [r for r in history if r["epoch"] > minimum_epoch]
    candidates = []
    for row in eligible:
        path = Path(source_dir) / f"epoch-{row['epoch']:04d}.pt"
        if path.is_file():
            candidates.append((row, path))
        if row["epoch"] == source["best_epoch"]:
            best = Path(source_dir) / "best.pt"
            if best.is_file():
                candidates.append((row, best))
    if not candidates:
        raise ValueError("no retained checkpoint satisfies exclusive epoch boundary")
    selected, path = min(candidates, key=lambda pair: (
        pair[0]["validation_mae"], pair[0]["epoch"], str(pair[1])))
    return selected, path, min(eligible, key=lambda r: (r["validation_mae"], r["epoch"]))


def evaluate_checkpoint(model, splits, *, output_dir, provenance, config, device,
                        source_summary, source_sha256, minimum_epoch=100):
    started = time.monotonic()
    source_summary = Path(source_summary)
    if file_sha256(source_summary) != source_sha256:
        raise ValueError("source summary SHA mismatch")
    source = json.loads(source_summary.read_text(encoding="utf-8"))
    # Summaries encode tuples as JSON lists; checkpoints retain their Python types.
    if (source.get("status") != "passed"
            or source["provenance"] != json.loads(json.dumps(provenance))
            or source["model_metadata"] != json.loads(json.dumps(model.metadata()))
            or normalized_training_config(source["training_config"]) != asdict(config)
            or source["split_counts"] != {k: len(v) for k, v in splits.items()}):
        raise ValueError("source identity mismatch")
    selected, checkpoint, historical = select_retained(
        source, source_summary.parent, minimum_epoch)
    output_dir = Path(output_dir)
    if output_dir.resolve() == source_summary.parent.resolve() or (
        output_dir.exists() and any(output_dir.iterdir())
    ):
        raise FileExistsError("use a new empty evaluation directory")
    checkpoint_sha = file_sha256(checkpoint)
    model.to(device)
    payload = load_checkpoint(model, checkpoint, provenance=provenance, device=device)
    if (payload["epoch"] != selected["epoch"]
            or normalized_training_config(payload["training_config"])
            != normalized_training_config(source["training_config"])
            or payload["step"] != math.ceil(len(splits["train"]) / config.batch_size)
            * selected["epoch"]):
        raise ValueError("checkpoint epoch/config/step mismatch")
    rows = splits["test"]
    ids = [row["sample_id"] for split in splits.values() for row in split]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate or overlapping sample IDs")
    target = torch.stack([r["target"] for r in rows])
    if not torch.isfinite(target).all() or any(
        not r.get("target_mask", torch.ones(1, dtype=torch.bool)).all() for r in rows
    ):
        raise ValueError("test targets must be complete and finite")
    prediction = predict(model, rows, config.batch_size, device)
    if prediction.shape != target.shape or not torch.isfinite(prediction).all():
        raise ValueError("invalid test predictions")
    metrics = tensor_benchmark_metrics(prediction, target, task="dielectric")
    if file_sha256(checkpoint) != checkpoint_sha:
        raise ValueError("source checkpoint changed during evaluation")
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "predictions.jsonl").open("w", encoding="utf-8") as stream:
        for row, pred in zip(rows, prediction):
            stream.write(json.dumps({"sample_id": row["sample_id"],
                "prediction": pred.tolist(), "target": row["target"].tolist()}) + "\n")
    report = {**source, "evaluation_only": True, "optimizer_steps": 0,
        "source_summary_sha256": source_sha256,
        "source_checkpoint": str(checkpoint), "source_checkpoint_sha256": checkpoint_sha,
        "minimum_checkpoint_epoch_exclusive": minimum_epoch,
        "selection": "minimum validation MAE among retained eligible checkpoints",
        "historical_eligible_best_epoch": historical["epoch"],
        "historical_eligible_best_validation_mae": historical["validation_mae"],
        "best_epoch": selected["epoch"],
        "best_validation_mae": selected["validation_mae"], "test_metrics": metrics,
        "evaluation_seconds": time.monotonic() - started}
    (output_dir / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
