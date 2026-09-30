"""Independent supervised trainer; legacy runners are deliberately unchanged."""

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path

import torch
from torch import nn
from torch_geometric.data import Batch, Data

from ...baselines.gmtnet.runner import _batches, _validation_history_metrics
from ...evaluation import tensor_benchmark_metrics


@dataclass(frozen=True)
class GlobalExpertsTrainConfig:
    epochs: int = 200
    batch_size: int = 64
    learning_rate: float = 1e-3
    end_learning_rate: float = 1e-5
    weight_decay: float = 1e-5
    seed: int = 42
    huber_delta: float = 1.0
    checkpoint_interval: int = 20
    gradient_clip: float | None = None
    decay_epochs: int = 0
    minimum_checkpoint_epoch_exclusive: int = 100

    def __post_init__(self):
        if self.epochs < 1 or self.batch_size < 1 or self.checkpoint_interval < 0:
            raise ValueError("invalid training counts")
        if not 0 <= self.decay_epochs <= self.epochs:
            raise ValueError("decay_epochs must be zero or within the training horizon")
        if not 0 <= self.minimum_checkpoint_epoch_exclusive < self.epochs:
            raise ValueError("checkpoint boundary must leave an eligible epoch")
        if (
            not 0 < self.end_learning_rate <= self.learning_rate
            or self.huber_delta <= 0
            or self.weight_decay < 0
        ):
            raise ValueError("invalid optimization parameters")
        if not all(
            math.isfinite(v)
            for v in (
                self.learning_rate,
                self.end_learning_rate,
                self.weight_decay,
                self.huber_delta,
            )
        ):
            raise ValueError("optimization parameters must be finite")
        if self.gradient_clip is not None and (
            not math.isfinite(self.gradient_clip) or self.gradient_clip <= 0
        ):
            raise ValueError("gradient clip must be finite and positive")


def learning_rate_after_step(config, step, steps_per_epoch):
    """LR for the next update; step zero is the optimizer's initial learning rate."""
    decay_steps = steps_per_epoch * (config.decay_epochs or config.epochs)
    if step < 0 or steps_per_epoch < 1:
        raise ValueError("invalid step count")
    return config.end_learning_rate + (
        config.learning_rate - config.end_learning_rate
    ) * (1 - min(step / decay_steps, 1.0))


def normalized_training_config(saved):
    """Pre-post100 checkpoints selected across all epochs and decayed throughout."""
    return asdict(GlobalExpertsTrainConfig(**{
        "decay_epochs": 0, "minimum_checkpoint_epoch_exclusive": 0, **saved,
    }))


def collate(rows, device):
    graphs = [Data(**row["graph"], sample_id=row["sample_id"]) for row in rows]
    data = Batch.from_data_list(graphs).to(device)
    masks = torch.stack([row["feature_mask"] for row in rows]).to(device)
    equality = torch.stack([row["equality"] for row in rows]).to(device)
    targets = torch.stack([row["target"] for row in rows]).to(device)
    valid = torch.stack(
        [
            row.get("target_mask", torch.ones_like(row["target"], dtype=torch.bool))
            for row in rows
        ]
    ).to(device)
    return data, masks, equality, tuple(row["routing"] for row in rows), targets, valid


def masked_huber(prediction, target, valid, *, delta=1.0):
    if prediction.shape != target.shape or valid.shape != target.shape:
        raise ValueError("prediction/target/observation mask shapes differ")
    if valid.dtype != torch.bool or not valid.any():
        raise ValueError("observation mask must contain a valid entry")
    selected_prediction, selected_target = prediction[valid], target[valid]
    if (
        not torch.isfinite(selected_prediction).all()
        or not torch.isfinite(selected_target).all()
    ):
        raise ValueError("observed predictions/targets must be finite")
    return nn.functional.huber_loss(selected_prediction, selected_target, delta=delta)


def predict(model, rows, batch_size, device):
    model.eval()
    outputs = []
    for batch in _batches(rows, batch_size, seed=None, drop_last=False):
        data, mask, equality, routing, _, _ = collate(batch, device)
        # The official response readout differentiates its probe even at inference.
        with torch.enable_grad():
            output = model(data, mask, equality, routing)
        outputs.append((0.5 * (output + output.transpose(-1, -2))).detach().cpu())
    return torch.cat(outputs)


def load_checkpoint(model, path, *, provenance, device="cpu"):
    payload = torch.load(path, map_location=device, weights_only=True)
    if (
        payload.get("schema_version") != 1
        or payload.get("model_metadata") != model.metadata()
        or payload.get("provenance") != provenance
    ):
        raise ValueError("global-experts checkpoint identity mismatch")
    model.load_state_dict(payload["model_state"], strict=True)
    return payload


def train_global_experts(
    model,
    splits,
    *,
    output_dir,
    provenance,
    config=GlobalExpertsTrainConfig(),
    device="cpu",
):
    if set(splits) != {"train", "validation", "test"} or any(
        not rows for rows in splits.values()
    ):
        raise ValueError("three nonempty splits required")
    ids = [row["sample_id"] for rows in splits.values() for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate or overlapping sample IDs")
    for split in ("validation", "test"):
        for row in splits[split]:
            if (
                not torch.isfinite(row["target"]).all()
                or not row.get("target_mask", torch.ones(1, dtype=torch.bool)).all()
            ):
                raise ValueError(
                    "tensor benchmark evaluation requires complete targets"
                )
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("use a new empty experiment directory")
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(config.seed)
    model.to(device)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    steps_per_epoch = math.ceil(len(splits["train"]) / config.batch_size)
    history, best_mae, best_epoch, step = [], float("inf"), 0, 0
    best = output_dir / "best.pt"

    def save(path, epoch):
        torch.save(
            {
                "schema_version": 1,
                "model_metadata": model.metadata(),
                "provenance": provenance,
                "training_config": asdict(config),
                "epoch": epoch,
                "step": step,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
            },
            path,
        )

    for epoch in range(1, config.epochs + 1):
        model.train()
        total, entries = 0.0, 0
        for rows in _batches(
            splits["train"],
            config.batch_size,
            seed=config.seed + epoch,
            drop_last=False,
        ):
            data, mask, equality, routing, target, valid = collate(rows, device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(data, mask, equality, routing)
            loss = masked_huber(prediction, target, valid, delta=config.huber_delta)
            loss.backward()
            gradients = [p.grad for p in model.parameters() if p.grad is not None]
            if not gradients or any(not torch.isfinite(g).all() for g in gradients):
                raise ValueError("nonfinite or missing model gradients")
            if config.gradient_clip is not None:
                nn.utils.clip_grad_norm_(
                    model.parameters(), config.gradient_clip, error_if_nonfinite=True
                )
            optimizer.step()
            count = int(valid.sum())
            total += float(loss.detach()) * count
            entries += count
            step += 1
            for group in optimizer.param_groups:
                group["lr"] = learning_rate_after_step(config, step, steps_per_epoch)
        prediction = predict(model, splits["validation"], config.batch_size, device)
        target = torch.stack([r["target"] for r in splits["validation"]])
        metrics = _validation_history_metrics(prediction, target)
        history.append(
            {
                "epoch": epoch,
                "training_loss": total / entries,
                **metrics,
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
        )
        if (epoch > config.minimum_checkpoint_epoch_exclusive
                and metrics["validation_mae"] < best_mae):
            best_mae, best_epoch = metrics["validation_mae"], epoch
            save(best, epoch)
        if config.checkpoint_interval and epoch % config.checkpoint_interval == 0:
            save(output_dir / f"epoch-{epoch:04d}.pt", epoch)
        print(json.dumps(history[-1]), flush=True)
    load_checkpoint(model, best, provenance=provenance, device=device)
    prediction = predict(model, splits["test"], config.batch_size, device)
    target = torch.stack([r["target"] for r in splits["test"]])
    metrics = tensor_benchmark_metrics(prediction, target, task="dielectric")
    with (output_dir / "predictions.jsonl").open("w", encoding="utf-8") as file:
        for row, pred in zip(splits["test"], prediction):
            file.write(
                json.dumps(
                    {
                        "sample_id": row["sample_id"],
                        "prediction": pred.tolist(),
                        "target": row["target"].tolist(),
                    }
                )
                + "\n"
            )
    summary = {
        "schema_version": 1,
        "status": "passed",
        "model": "GMTNet global + Full-PG experts",
        "model_metadata": model.metadata(),
        "provenance": provenance,
        "training_config": asdict(config),
        "history": history,
        "best_epoch": best_epoch,
        "best_validation_mae": best_mae,
        "test_metrics": metrics,
        "split_counts": {key: len(rows) for key, rows in splits.items()},
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)
    return summary
