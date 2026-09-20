from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr

from .benchmark import tensor_benchmark_metrics


MODEL_LABELS = ("current-pg", "GMTNet")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if not rows:
        raise ValueError(f"{path} is empty")
    return rows


def load_dataset_index(path: Path) -> dict[str, Any]:
    rows = _jsonl(path)
    by_id: dict[str, dict[str, Any]] = {}
    counts: dict[int, Counter[str]] = {}
    point_groups: dict[int, set[str]] = {}
    for row in rows:
        record_id = str(row["record_id"])
        if record_id in by_id:
            raise ValueError(f"duplicate dataset record_id: {record_id}")
        split = str(row["split"])
        if split not in {"train", "validation", "test"}:
            raise ValueError(f"invalid split for {record_id}: {split}")
        space_group = int(row["space_group"])
        if not 1 <= space_group <= 230:
            raise ValueError(f"invalid space group for {record_id}: {space_group}")
        point_group = str(row["point_group"])
        by_id[record_id] = {
            "split": split,
            "space_group": space_group,
            "point_group": point_group,
        }
        counts.setdefault(space_group, Counter())[split] += 1
        point_groups.setdefault(space_group, set()).add(point_group)
    inconsistent = {key: value for key, value in point_groups.items() if len(value) != 1}
    if inconsistent:
        raise ValueError(f"space groups map to multiple point groups: {inconsistent}")
    return {
        "by_id": by_id,
        "counts": counts,
        "point_groups": {key: next(iter(value)) for key, value in point_groups.items()},
        "row_count": len(rows),
    }


def load_predictions(path: Path) -> dict[str, Any]:
    rows = _jsonl(path)
    ids = [str(row["sample_id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{path} contains duplicate sample IDs")
    prediction = torch.tensor([row["prediction"] for row in rows], dtype=torch.float64)
    target = torch.tensor([row["target"] for row in rows], dtype=torch.float64)
    if prediction.ndim != 3 or tuple(prediction.shape[1:]) != (3, 3):
        raise ValueError(f"{path} predictions must have shape [N,3,3]")
    if target.shape != prediction.shape:
        raise ValueError(f"{path} targets must match prediction shape")
    if not torch.isfinite(prediction).all() or not torch.isfinite(target).all():
        raise ValueError(f"{path} contains non-finite tensors")
    return {"ids": ids, "prediction": prediction, "target": target}


def _relationship(x: Sequence[float], y: Sequence[float]) -> dict[str, float | int | None]:
    if len(x) != len(y):
        raise ValueError("relationship inputs must have equal length")
    if len(x) < 3 or len(set(x)) < 2 or len(set(y)) < 2:
        return {
            "group_count": len(x),
            "spearman_rho": None,
            "spearman_p": None,
            "pearson_r": None,
            "pearson_p": None,
        }
    spearman = spearmanr(x, y)
    pearson = pearsonr(x, y)
    return {
        "group_count": len(x),
        "spearman_rho": float(spearman.statistic),
        "spearman_p": float(spearman.pvalue),
        "pearson_r": float(pearson.statistic),
        "pearson_p": float(pearson.pvalue),
    }


def analyze_space_groups(
    dataset_path: Path,
    model_paths: Mapping[str, Path],
    *,
    cohort_thresholds: Sequence[int] = (5, 10),
    relative_epsilon: float = 1.0e-5,
) -> dict[str, Any]:
    if tuple(model_paths) != MODEL_LABELS:
        raise ValueError(f"models must be ordered exactly as {MODEL_LABELS}")
    thresholds = tuple(int(value) for value in cohort_thresholds)
    if not thresholds or any(value < 1 for value in thresholds) or len(set(thresholds)) != len(thresholds):
        raise ValueError("cohort thresholds must be unique positive integers")

    dataset = load_dataset_index(dataset_path)
    predictions = {label: load_predictions(path) for label, path in model_paths.items()}
    reference = predictions[MODEL_LABELS[0]]
    reference_ids = reference["ids"]
    expected_test_ids = {
        record_id for record_id, row in dataset["by_id"].items() if row["split"] == "test"
    }
    if len(reference_ids) != len(expected_test_ids) or set(reference_ids) != expected_test_ids:
        missing = sorted(expected_test_ids.difference(reference_ids))[:5]
        extra = sorted(set(reference_ids).difference(expected_test_ids))[:5]
        raise ValueError(f"prediction IDs do not equal frozen test IDs; missing={missing}, extra={extra}")
    for label, artifact in predictions.items():
        if artifact["ids"] != reference_ids:
            raise ValueError(f"{label} prediction IDs or order differ")
        if not torch.allclose(
            torch.linalg.eigvalsh(artifact["target"]),
            torch.linalg.eigvalsh(reference["target"]),
            atol=2.0e-4,
            rtol=2.0e-5,
        ):
            raise ValueError(f"{label} targets are not frame-equivalent")

    indices_by_group: dict[int, list[int]] = {}
    for index, record_id in enumerate(reference_ids):
        space_group = dataset["by_id"][record_id]["space_group"]
        indices_by_group.setdefault(space_group, []).append(index)

    groups: list[dict[str, Any]] = []
    for space_group in sorted(indices_by_group):
        indices = torch.tensor(indices_by_group[space_group], dtype=torch.long)
        counts = dataset["counts"][space_group]
        reference_target = reference["target"].index_select(0, indices)
        row: dict[str, Any] = {
            "space_group": space_group,
            "point_group": dataset["point_groups"][space_group],
            "train_count": int(counts["train"]),
            "validation_count": int(counts["validation"]),
            "test_count": int(counts["test"]),
            "mean_target_fnorm": float(
                torch.linalg.vector_norm(reference_target.reshape(len(indices), -1), dim=1).mean()
            ),
            "models": {},
        }
        for label, artifact in predictions.items():
            prediction = artifact["prediction"].index_select(0, indices)
            target = artifact["target"].index_select(0, indices)
            metrics = tensor_benchmark_metrics(prediction, target, task="dielectric")
            distance = torch.linalg.vector_norm((prediction - target).reshape(len(indices), -1), dim=1)
            target_norm = torch.linalg.vector_norm(target.reshape(len(indices), -1), dim=1)
            metrics["mean_relative_fnorm"] = float((distance / (target_norm + relative_epsilon)).mean())
            row["models"][label] = metrics
        row["delta_current_pg_minus_gmtnet"] = {
            metric: float(row["models"]["current-pg"][metric] - row["models"]["GMTNet"][metric])
            for metric in ("rmse", "fnorm", "mean_relative_fnorm", "ewt_25", "ewt_10", "ewt_5")
        }
        groups.append(row)

    relationships: dict[str, Any] = {}
    for minimum in thresholds:
        cohort = [row for row in groups if row["test_count"] >= minimum and row["train_count"] > 0]
        x = [math.log10(row["train_count"]) for row in cohort]
        cohort_report: dict[str, Any] = {
            "minimum_test_count": minimum,
            "space_groups": [row["space_group"] for row in cohort],
            "models": {},
            "target_scale": _relationship(x, [row["mean_target_fnorm"] for row in cohort]),
        }
        for label in MODEL_LABELS:
            cohort_report["models"][label] = {
                metric: _relationship(x, [row["models"][label][metric] for row in cohort])
                for metric in ("rmse", "fnorm", "mean_relative_fnorm")
            }
        relationships[f"test_count_gte_{minimum}"] = cohort_report

    return {
        "schema_version": 1,
        "grouping": "source_space_group_number_from_curated_dataset",
        "relationship_x": "log10_train_count",
        "relative_epsilon": relative_epsilon,
        "dataset": {
            "path": dataset_path.as_posix(),
            "sha256": file_sha256(dataset_path),
            "row_count": dataset["row_count"],
            "split_counts": dict(
                sorted(Counter(row["split"] for row in dataset["by_id"].values()).items())
            ),
        },
        "prediction_sha256": {
            label: file_sha256(path) for label, path in model_paths.items()
        },
        "test_id_count": len(reference_ids),
        "space_group_count": len(groups),
        "cohort_thresholds": list(thresholds),
        "groups": groups,
        "relationships": relationships,
    }


def write_csv(report: Mapping[str, Any], path: Path) -> None:
    fields = [
        "space_group", "point_group", "train_count", "validation_count", "test_count",
        "mean_target_fnorm",
    ]
    metric_names = ("rmse", "fnorm", "mean_relative_fnorm", "ewt_25", "ewt_10", "ewt_5")
    fields += [f"{label}_{metric}" for label in MODEL_LABELS for metric in metric_names]
    fields += [f"delta_{metric}" for metric in metric_names]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for group in report["groups"]:
            row = {key: group[key] for key in fields[:6]}
            for label in MODEL_LABELS:
                for metric in metric_names:
                    row[f"{label}_{metric}"] = group["models"][label][metric]
            for metric in metric_names:
                row[f"delta_{metric}"] = group["delta_current_pg_minus_gmtnet"][metric]
            writer.writerow(row)


def write_markdown(report: Mapping[str, Any], path: Path) -> None:
    lines = [
        "# Reduced dielectric-total errors by source space group",
        "",
        f"Test structures: {report['test_id_count']}; observed test space groups: {report['space_group_count']}.",
        "All metrics use the frozen split. Correlations are unweighted across eligible space groups.",
        "",
        "## Training-count relationship",
        "",
        "| Cohort | Model | Metric | Groups | Spearman rho | p | Pearson r | p |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for cohort_name, cohort in report["relationships"].items():
        for label in MODEL_LABELS:
            for metric, relation in cohort["models"][label].items():
                def fmt(value: float | None) -> str:
                    return "NA" if value is None else f"{value:.4f}"
                lines.append(
                    f"| {cohort_name} | {label} | {metric} | {relation['group_count']} | "
                    f"{fmt(relation['spearman_rho'])} | {fmt(relation['spearman_p'])} | "
                    f"{fmt(relation['pearson_r'])} | {fmt(relation['pearson_p'])} |"
                )
    lines += [
        "",
        "## Per-space-group metrics",
        "",
        "| SG | PG | Train | Val | Test | current-pg RMSE | GMTNet RMSE | current-pg Fnorm | GMTNet Fnorm | current-pg EwT25 | GMTNet EwT25 |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for group in report["groups"]:
        current = group["models"]["current-pg"]
        gmt = group["models"]["GMTNet"]
        lines.append(
            f"| {group['space_group']} | {group['point_group']} | {group['train_count']} | "
            f"{group['validation_count']} | {group['test_count']} | {current['rmse']:.4f} | "
            f"{gmt['rmse']:.4f} | {current['fnorm']:.4f} | {gmt['fnorm']:.4f} | "
            f"{current['ewt_25']:.2f}% | {gmt['ewt_25']:.2f}% |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def render_relationship_plot(report: Mapping[str, Any], path: Path, *, minimum_test_count: int = 5) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    groups = [row for row in report["groups"] if row["test_count"] >= minimum_test_count]
    if not groups:
        raise ValueError("no space groups satisfy the plotting threshold")
    matplotlib.rcParams["svg.hashsalt"] = "space-group-error-v1"
    colors = {"current-pg": "#0072B2", "GMTNet": "#D55E00"}
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.8), constrained_layout=True)
    for axis, metric, ylabel in (
        (axes[0], "fnorm", "Test Fnorm"),
        (axes[1], "mean_relative_fnorm", "Mean relative Fnorm"),
    ):
        for label in MODEL_LABELS:
            axis.scatter(
                [row["train_count"] for row in groups],
                [row["models"][label][metric] for row in groups],
                s=[24 + 4 * math.sqrt(row["test_count"]) for row in groups],
                color=colors[label], alpha=0.75, edgecolors="white", linewidths=0.5, label=label,
            )
        axis.set_xscale("log")
        axis.set_xlabel("Training structures in source space group (log scale)")
        axis.set_ylabel(ylabel)
        axis.grid(True, alpha=0.25)
        axis.legend(frameon=False)
    fig.suptitle(
        f"Reduced dielectric-total: error vs training coverage (test count >= {minimum_test_count})"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, metadata={"Date": None})
    plt.close(fig)
