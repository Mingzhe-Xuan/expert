from __future__ import annotations

from dataclasses import asdict, dataclass
import importlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
from types import SimpleNamespace
from types import ModuleType
from typing import Sequence

import numpy as np
import torch
from torch import nn

from ...data import IndependentTensorDataset, TensorSample
from ...evaluation import tensor_benchmark_metrics


GMTNET_OFFICIAL_COMMIT = "7a606a459ee48a320ed38450e391811fb43d5e19"


@dataclass(frozen=True, slots=True)
class GMTNetConfig:
    epochs: int = 200
    batch_size: int = 64
    learning_rate: float = 1.0e-3
    end_learning_rate: float = 1.0e-5
    weight_decay: float = 1.0e-5
    seed: int = 42

    def __post_init__(self) -> None:
        if self.epochs < 1 or self.batch_size < 1:
            raise ValueError("epochs and batch size must be positive")
        if not 0 < self.end_learning_rate <= self.learning_rate:
            raise ValueError("learning-rate range is invalid")
        if self.weight_decay < 0:
            raise ValueError("weight decay must be non-negative")


def _load_official_modules(official_root: Path):
    root = official_root.resolve()
    if not (root / "gmtnet.py").is_file() or not (root / "graphs.py").is_file():
        raise FileNotFoundError("official GMTNet checkout lacks gmtnet.py or graphs.py")
    commit = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if commit != GMTNET_OFFICIAL_COMMIT:
        raise ValueError(f"GMTNet checkout commit mismatch: {commit}")
    try:
        importlib.import_module("torch_scatter")
    except ImportError:
        from torch_geometric.utils import scatter as pyg_scatter

        compatibility = ModuleType("torch_scatter")
        compatibility.scatter = pyg_scatter
        sys.modules["torch_scatter"] = compatibility
    sys.path.insert(0, str(root))
    try:
        return (
            importlib.import_module("gmtnet"),
            importlib.import_module("graphs"),
            importlib.import_module("data"),
        )
    finally:
        sys.path.pop(0)


def _plain_graph(graph) -> dict[str, torch.Tensor]:
    return {"x": graph.x.cpu(), "edge_index": graph.edge_index.cpu(), "edge_attr": graph.edge_attr.cpu()}


def _prepare_row(sample: TensorSample, official_data, official_graphs, adaptor) -> dict[str, object]:
    from pymatgen.core import Structure

    structure = Structure(
        lattice=sample.lattice.cpu().numpy(),
        species=sample.atomic_numbers.cpu().tolist(),
        coords=sample.fractional_positions.cpu().numpy(),
    )
    symmetry = official_data.get_symmetry_dataset(structure, symprec=1.0e-5)
    rotations = official_data.rm_duplicates(np.asarray(symmetry["rotations"]))
    lattice_columns = structure.lattice.matrix.T
    cartesian_rotations = np.matmul(
        lattice_columns, np.matmul(rotations, np.linalg.inv(lattice_columns))
    )
    if not official_data.is_group(cartesian_rotations):
        raise ValueError(f"official GMTNet symmetry operations are not a group for {sample.sample_id}")
    d_matrices = official_data.irreps_output.D_from_matrix(
        torch.as_tensor(cartesian_rotations, dtype=torch.float32)
    )
    marker = torch.arange(32, dtype=torch.float32) + 10.0
    marker[8:] *= 100.0
    summed = d_matrices.sum(dim=0)
    feature_marker = summed @ marker
    ideal_matrix = official_data.converter.to_cartesian(
        feature_marker[[0, 2, 3, 4, 8, 9, 10, 11, 12]]
    )
    feature_mask = summed / d_matrices.shape[0]
    feature_mask = feature_mask * (feature_mask > 1.0e-5)
    equality = official_data.find_almost_equal_entries(ideal_matrix)
    atoms = adaptor.get_atoms(structure)
    graph = official_graphs.atoms2graphs(
        atoms,
        cutoff=4.0,
        max_neighbors=16,
        reduce=False,
        equivalent_atoms=symmetry["equivalent_atoms"],
        use_canonize=True,
    )
    return {
        "sample_id": sample.sample_id,
        "graph": _plain_graph(graph),
        "feature_mask": feature_mask.cpu(),
        "equality": equality.cpu(),
        "target": sample.target_cartesian.to(torch.float32).cpu(),
    }


def _prepare_cache(
    dataset: IndependentTensorDataset,
    official_root: Path,
    cache_path: Path,
    dataset_sha256: str,
) -> dict[str, list[dict[str, object]]]:
    if cache_path.is_file():
        payload = torch.load(cache_path, map_location="cpu", weights_only=True)
        if payload.get("schema_version") != 1 or payload.get("official_commit") != GMTNET_OFFICIAL_COMMIT:
            raise ValueError("GMTNet graph cache implementation metadata mismatch")
        if payload.get("dataset_sha256") != dataset_sha256:
            raise ValueError("GMTNet graph cache dataset SHA-256 mismatch")
        expected_ids = {
            name: list(getattr(dataset.split_manifest, name))
            for name in ("train", "validation", "test")
        }
        if payload.get("split_ids") != expected_ids:
            raise ValueError("GMTNet graph cache split IDs mismatch")
        return payload["splits"]

    _, official_graphs, official_data = _load_official_modules(official_root)
    from pymatgen.io.jarvis import JarvisAtomsAdaptor

    adaptor = JarvisAtomsAdaptor()
    splits = {}
    for split in ("train", "validation", "test"):
        ids = getattr(dataset.split_manifest, split)
        rows = []
        for index, sample_id in enumerate(ids, start=1):
            rows.append(_prepare_row(dataset.by_id(sample_id), official_data, official_graphs, adaptor))
            if index == 1 or index == len(ids) or index % 25 == 0:
                print(json.dumps({"event": "gmtnet_preprocess", "split": split,
                                  "current": index, "total": len(ids), "sample_id": sample_id}), flush=True)
        splits[split] = rows
    payload = {
        "schema_version": 1,
        "official_commit": GMTNET_OFFICIAL_COMMIT,
        "dataset_sha256": dataset_sha256,
        "split_ids": {name: list(getattr(dataset.split_manifest, name)) for name in splits},
        "splits": splits,
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_name(f".{cache_path.name}.tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, cache_path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return splits


def _batches(rows: Sequence[dict[str, object]], batch_size: int, *, seed: int | None, drop_last: bool):
    indices = list(range(len(rows)))
    if seed is not None:
        random.Random(seed).shuffle(indices)
    stop = len(indices) - len(indices) % batch_size if drop_last else len(indices)
    for start in range(0, stop, batch_size):
        yield [rows[index] for index in indices[start : start + batch_size]]


def _collate(rows, data_type, batch_type, device):
    graphs = [data_type(**row["graph"]) for row in rows]
    return (
        batch_type.from_data_list(graphs).to(device),
        torch.stack([row["feature_mask"] for row in rows]).to(device),
        torch.stack([row["equality"] for row in rows]).to(device),
        torch.stack([row["target"] for row in rows]).to(device),
    )


def _predict(model, rows, batch_size, data_type, batch_type, device):
    model.eval()
    predictions = []
    for batch_rows in _batches(rows, batch_size, seed=None, drop_last=False):
        graph, mask, equality, _ = _collate(batch_rows, data_type, batch_type, device)
        with torch.enable_grad():
            output = model(graph, mask, equality)
        predictions.append((0.5 * (output + output.transpose(-1, -2))).detach().cpu())
    return torch.cat(predictions)


def run_gmtnet_benchmark(
    dataset: IndependentTensorDataset,
    *,
    official_root: str | Path,
    cache_path: str | Path,
    checkpoint_path: str | Path,
    predictions_path: str | Path,
    config: GMTNetConfig = GMTNetConfig(),
    device: str | torch.device = "cuda",
) -> dict[str, object]:
    if dataset.unit.namespace != "curated_reduced_total__dielectric":
        raise ValueError("GMTNet reduced benchmark received the wrong training unit")
    dataset_sha256 = str(dataset[0].source["manifest_sha256"])
    root = Path(official_root)
    splits = _prepare_cache(dataset, root, Path(cache_path), dataset_sha256)
    official_model, official_graphs, _ = _load_official_modules(root)
    data_type = official_graphs.Data
    batch_type = importlib.import_module("torch_geometric.data.batch").Batch
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    model = official_model.GMTNet(
        SimpleNamespace(target="dielectric", use_mask=True, reduce_cell=False)
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    steps_per_epoch = len(splits["train"]) // config.batch_size
    total_steps = steps_per_epoch * config.epochs
    criterion = nn.HuberLoss()
    best_mae = float("inf")
    best_epoch = 0
    history = []
    checkpoint = Path(checkpoint_path)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    step = 0
    for epoch in range(1, config.epochs + 1):
        model.train()
        total_loss = 0.0
        seen = 0
        for batch_rows in _batches(
            splits["train"], config.batch_size, seed=config.seed + epoch, drop_last=True
        ):
            graph, mask, equality, labels = _collate(batch_rows, data_type, batch_type, device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(graph, mask, equality)
            loss = criterion(outputs, labels)
            if not torch.isfinite(loss):
                raise ValueError("GMTNet training loss is non-finite")
            loss.backward()
            optimizer.step()
            step += 1
            fraction = min(step, total_steps) / total_steps
            lr = (config.learning_rate - config.end_learning_rate) * (1.0 - fraction) + config.end_learning_rate
            for group in optimizer.param_groups:
                group["lr"] = lr
            total_loss += float(loss.detach()) * len(batch_rows)
            seen += len(batch_rows)
        validation_prediction = _predict(
            model, splits["validation"], config.batch_size, data_type, batch_type, device
        )
        validation_target = torch.stack([row["target"] for row in splits["validation"]])
        validation_mae = float((validation_prediction - validation_target).abs().mean())
        history.append({"epoch": epoch, "training_loss": total_loss / seen,
                        "validation_mae": validation_mae, "learning_rate": optimizer.param_groups[0]["lr"]})
        if validation_mae < best_mae:
            best_mae, best_epoch = validation_mae, epoch
            torch.save({"schema_version": 1, "official_commit": GMTNET_OFFICIAL_COMMIT,
                        "dataset_sha256": dataset_sha256, "config": asdict(config),
                        "epoch": epoch, "model_state": model.state_dict()}, checkpoint)
        print(json.dumps(history[-1], sort_keys=True), flush=True)
    saved = torch.load(checkpoint, map_location=device, weights_only=True)
    if saved.get("official_commit") != GMTNET_OFFICIAL_COMMIT or saved.get("dataset_sha256") != dataset_sha256:
        raise ValueError("GMTNet checkpoint provenance mismatch")
    model.load_state_dict(saved["model_state"], strict=True)
    test_prediction = _predict(model, splits["test"], config.batch_size, data_type, batch_type, device)
    test_target = torch.stack([row["target"] for row in splits["test"]])
    metrics = tensor_benchmark_metrics(test_prediction, test_target, task="dielectric")
    prediction_file = Path(predictions_path)
    prediction_file.parent.mkdir(parents=True, exist_ok=True)
    temporary = prediction_file.with_name(f".{prediction_file.name}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            for row, prediction in zip(splits["test"], test_prediction):
                stream.write(json.dumps({"sample_id": row["sample_id"],
                                         "prediction": prediction.tolist(),
                                         "target": row["target"].tolist()},
                                        sort_keys=True, separators=(",", ":")) + "\n")
        os.replace(temporary, prediction_file)
    finally:
        if temporary.exists():
            temporary.unlink()
    return {
        "schema_version": 1,
        "status": "passed",
        "model": "GMTNet",
        "official_commit": GMTNET_OFFICIAL_COMMIT,
        "dataset_sha256": dataset_sha256,
        "training_unit": dataset.unit.namespace,
        "split_counts": {name: len(rows) for name, rows in splits.items()},
        "config": asdict(config),
        "protocol_repairs": ["no_wandb", "explicit_paths", "complete_validation_and_test_batches",
                             "best_validation_mae_from_epoch_1",
                             "pyg_scatter_compatibility_if_torch_scatter_unavailable"],
        "best_epoch": best_epoch,
        "best_validation_mae": best_mae,
        "test_metrics": metrics,
        "checkpoint": str(checkpoint),
        "predictions": str(prediction_file),
        "history": history,
    }
