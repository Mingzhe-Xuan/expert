from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import torch

from ..evaluation import tensor_benchmark_metrics


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(path: Path):
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    ids = [str(row["sample_id"]) for row in rows]
    if len(ids) != 677 or len(ids) != len(set(ids)):
        raise ValueError(f"{path.name} must contain exactly 677 unique test IDs")
    prediction = torch.tensor([row["prediction"] for row in rows], dtype=torch.float64)
    target = torch.tensor([row["target"] for row in rows], dtype=torch.float64)
    if prediction.shape != (677, 3, 3) or target.shape != prediction.shape:
        raise ValueError(f"{path.name} tensors must have shape [677,3,3]")
    return ids, prediction, target


def _write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(text, encoding="utf-8", newline="\n")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate and compare reduced DPA4/CGCNN-full-PG/GMTNet predictions"
    )
    parser.add_argument("--dpa4", type=Path, required=True)
    parser.add_argument("--gmtnet", type=Path, required=True)
    parser.add_argument("--cgcnn-full-pg", type=Path)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--table", type=Path, required=True)
    arguments = parser.parse_args()
    dpa_ids, dpa_prediction, dpa_target = _load(arguments.dpa4)
    gmt_ids, gmt_prediction, gmt_target = _load(arguments.gmtnet)
    if dpa_ids != gmt_ids:
        raise ValueError("DPA4 and GMTNet test IDs or order differ")
    # DPA4 predicts in its deterministic canonical frame; GMTNet predicts in the
    # source frame. Eigenvalues prove both rows refer to the same symmetric target.
    if not torch.allclose(
        torch.linalg.eigvalsh(dpa_target),
        torch.linalg.eigvalsh(gmt_target),
        atol=2.0e-4,
        rtol=2.0e-5,
    ):
        raise ValueError("model prediction files do not contain frame-equivalent targets")
    metrics = {
        "DPA4 B+A+PGE+R full_pg": tensor_benchmark_metrics(
            dpa_prediction, dpa_target, task="dielectric"
        ),
        "GMTNet": tensor_benchmark_metrics(gmt_prediction, gmt_target, task="dielectric"),
    }
    prediction_sha256 = {
        "dpa4": _sha256(arguments.dpa4),
        "gmtnet": _sha256(arguments.gmtnet),
    }
    if arguments.cgcnn_full_pg is not None:
        cgcnn_ids, cgcnn_prediction, cgcnn_target = _load(arguments.cgcnn_full_pg)
        if cgcnn_ids != dpa_ids:
            raise ValueError("CGCNN full-PG and reference test IDs or order differ")
        if not torch.allclose(
            torch.linalg.eigvalsh(cgcnn_target),
            torch.linalg.eigvalsh(dpa_target),
            atol=2.0e-4,
            rtol=2.0e-5,
        ):
            raise ValueError("CGCNN full-PG targets are not frame-equivalent")
        metrics["CGCNN B+A+PGE+R full_pg"] = tensor_benchmark_metrics(
            cgcnn_prediction, cgcnn_target, task="dielectric"
        )
        prediction_sha256["cgcnn_full_pg"] = _sha256(arguments.cgcnn_full_pg)
    report = {
        "schema_version": 1,
        "status": "passed",
        "test_ids": dpa_ids,
        "test_count": len(dpa_ids),
        "target_equivalence": "symmetric_tensor_eigenvalues_atol_2e-4_rtol_2e-5",
        "prediction_sha256": prediction_sha256,
        "metrics": metrics,
    }
    _write_atomic(arguments.summary, json.dumps(report, indent=2, sort_keys=True) + "\n")
    lines = [
        "| Model | RMSE ↓ | Fnorm ↓ | EwT 25% ↑ | EwT 10% ↑ | EwT 5% ↑ |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for model, row in metrics.items():
        lines.append(
            f"| {model} | {row['rmse']:.6f} | {row['fnorm']:.6f} | "
            f"{row['ewt_25']:.2f}% | {row['ewt_10']:.2f}% | {row['ewt_5']:.2f}% |"
        )
    _write_atomic(arguments.table, "\n".join(lines) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
