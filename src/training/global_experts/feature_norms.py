"""Read-only crystal-level norm diagnostics at the actual feature fusion point."""

import json
import math
from pathlib import Path

import torch

from .data import file_sha256
from .runner import collate, load_checkpoint, normalized_training_config


def norm_rows(global_features, auxiliary, scale, mask=None):
    g, z = global_features.detach().double(), auxiliary.detach().double()
    if g.ndim != 2 or z.shape != g.shape or not math.isfinite(scale):
        raise ValueError("invalid feature shapes or scale")
    if mask is not None:
        mask = mask.detach().double()
        if mask.shape != (len(g), g.shape[1], g.shape[1]):
            raise ValueError("invalid mask shape")
        g = torch.bmm(mask, g.unsqueeze(-1)).squeeze(-1)
        z = torch.bmm(mask, z.unsqueeze(-1)).squeeze(-1)
    if not torch.isfinite(g).all() or not torch.isfinite(z).all():
        raise ValueError("nonfinite features")
    gn, zn, fn = (v.norm(dim=-1).cpu().tolist() for v in (g, z, g + scale*z))
    return [{"global_norm": a, "pg_norm": b, "scaled_pg_norm": abs(scale)*b,
             "fused_norm": c, "scaled_pg_over_global": abs(scale)*b/a if a else None,
             "pg_over_global": b/a if a else None} for a, b, c in zip(gn, zn, fn)]


def summarize(rows):
    output = {}
    for key in rows[0]:
        values = torch.tensor([r[key] for r in rows if r[key] is not None], dtype=torch.float64)
        output[key] = {"count": len(values), "undefined": len(rows)-len(values)}
        if len(values):
            output[key].update(mean=float(values.mean()), min=float(values.min()),
                p05=float(values.quantile(.05)), median=float(values.quantile(.5)),
                p95=float(values.quantile(.95)), max=float(values.max()))
    return output


def diagnose(model, splits, *, output_dir, provenance, config, device,
             source_summary, source_sha256, checkpoint_sha256, predictions_sha256):
    source_summary = Path(source_summary)
    if file_sha256(source_summary) != source_sha256:
        raise ValueError("source summary SHA mismatch")
    source = json.loads(source_summary.read_text(encoding="utf-8"))
    checkpoint = source_summary.parent / "best.pt"
    accepted = source_summary.parent / "predictions.jsonl"
    if file_sha256(checkpoint) != checkpoint_sha256 or file_sha256(accepted) != predictions_sha256:
        raise ValueError("checkpoint/predictions SHA mismatch")
    if (source["status"] != "passed" or source["provenance"] != json.loads(json.dumps(provenance))
            or source["model_metadata"] != json.loads(json.dumps(model.metadata()))):
        raise ValueError("source metadata mismatch")
    target = Path(output_dir)
    if target.exists() and any(target.iterdir()):
        raise FileExistsError("use a new empty output directory")
    model.to(device).eval()
    payload = load_checkpoint(model, checkpoint, provenance=provenance, device=device)
    if (payload["epoch"] != source["best_epoch"]
            or normalized_training_config(payload["training_config"])
            != normalized_training_config(source["training_config"])):
        raise ValueError("checkpoint selection/config mismatch")
    reference = [json.loads(line) for line in accepted.read_text(encoding="utf-8").splitlines()]
    rows = splits["test"]
    ids = [row["sample_id"] for row in rows]
    if (ids != [r["sample_id"] for r in reference] or len(set(ids)) != len(ids)
            or ids != source["provenance"]["split_ids"]["test"]):
        raise ValueError("test IDs mismatch")
    scale = float(model.branch_logit.detach().sigmoid())
    records, predictions = [], []
    for start in range(0, len(rows), config.batch_size):
        batch = rows[start:start+config.batch_size]
        data, mask, equality, routing, _, _ = collate(batch, device)
        with torch.enable_grad():
            prediction, diagnostics = model(data, mask, equality, routing, return_diagnostics=True)
        predictions.append((.5*(prediction + prediction.transpose(-1,-2))).detach().cpu())
        g, z = diagnostics["global_features"], diagnostics["auxiliary"]
        before = norm_rows(g, z, scale)
        after = norm_rows(g, z, scale, mask if model.global_model.mask else None)
        records.extend({"sample_id": row["sample_id"], "before_mask": b, "after_mask": a}
                       for row, b, a in zip(batch, before, after))
        del prediction, diagnostics, g, z
    predictions = torch.cat(predictions)
    expected = torch.tensor([r["prediction"] for r in reference], dtype=predictions.dtype)
    torch.testing.assert_close(predictions, expected, atol=2e-5, rtol=2e-4)
    if file_sha256(checkpoint) != checkpoint_sha256:
        raise ValueError("checkpoint changed")
    result = {"status": "passed", "checkpoint_epoch": payload["epoch"],
        "checkpoint_sha256": checkpoint_sha256, "source_summary_sha256": source_sha256,
        "sample_count": len(rows), "optimizer_steps": 0, "lambda": scale,
        "branch_logit": float(model.branch_logit.detach()),
        "initial_lambda": 1/(1+math.exp(-model.config.branch_initial_logit)),
        "definition": "per-crystal L2 in shared global carrier, after PG projection/routing/pooling",
        "official_mask_enabled": bool(model.global_model.mask),
        "max_prediction_difference": float((predictions-expected).abs().max()),
        **{stage: summarize([r[stage] for r in records]) for stage in ("before_mask", "after_mask")}}
    target.mkdir(parents=True, exist_ok=True)
    (target/"summary.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    (target/"norms.jsonl").write_text("".join(json.dumps(r, allow_nan=False)+"\n" for r in records), encoding="utf-8")
    return result
