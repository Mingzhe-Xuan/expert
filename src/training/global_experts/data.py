"""Identity-bound routing cache computed on the exact official GMTNet graph."""

import hashlib
import math
from pathlib import Path

import torch

from ...features import cgcnn_node_features
from ...models.global_experts.frames import standardize_o3
from ...models.global_experts.model import CrystalRouting
from ...symmetry import PointGroupAncestorDAG, build_point_group_parent_dag
from ...symmetry.parent_detection import route_material_on_point_group_dag
from ...baselines.gmtnet.runner import GMTNET_OFFICIAL_COMMIT


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def row_digest(row):
    digest = hashlib.sha256(str(row["sample_id"]).encode())
    for key in ("x", "edge_index", "edge_attr"):
        value = row["graph"][key].detach().cpu().contiguous()
        digest.update(str((key, value.dtype, tuple(value.shape))).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def prepare_routing(
    splits,
    samples,
    path,
    *,
    dataset_sha256,
    graph_cache_sha256,
    class_dag=None,
    symprec=1e-5
):
    """Return enriched rows and cache provenance; never reuse a stale cache."""
    if not math.isfinite(symprec) or symprec <= 0:
        raise ValueError("symprec must be finite and positive")
    class_dag = class_dag or PointGroupAncestorDAG.from_path()
    sample_by_id = {sample.sample_id: sample for sample in samples}
    identity = {
        "schema_version": 1,
        "dataset_sha256": dataset_sha256,
        "graph_cache_sha256": graph_cache_sha256,
        "dag_sha256": class_dag.asset_sha256,
        "registry_sha256": class_dag.asset_sha256,
        "official_commit": GMTNET_OFFICIAL_COMMIT,
        "frame_convention": "lattice-polar-then-spglib-o3-v1",
        "symprec": symprec,
        "residual_convention": "minimum-oriented-parent-minus-child-relative-edge-rms-v1",
        "split_ids": {
            name: [r["sample_id"] for r in rows] for name, rows in splits.items()
        },
        "graph_rows": {
            r["sample_id"]: row_digest(r) for rows in splits.values() for r in rows
        },
    }
    flat_ids = [key for ids in identity["split_ids"].values() for key in ids]
    if len(set(flat_ids)) != len(flat_ids) or not set(flat_ids) <= sample_by_id.keys():
        raise ValueError("split IDs must be unique, disjoint, and present in dataset")
    for rows in splits.values():
        for row in rows:
            sample = sample_by_id[row["sample_id"]]
            expected = cgcnn_node_features(sample.atomic_numbers).to(row["graph"]["x"])
            if expected.shape != row["graph"]["x"].shape or not torch.allclose(
                expected, row["graph"]["x"], atol=1e-6, rtol=1e-6
            ):
                raise ValueError(
                    "official graph node species/order or features mismatch"
                )
    path = Path(path)
    if path.is_file():
        saved = torch.load(path, map_location="cpu", weights_only=True)
        if saved.get("identity") != identity:
            raise ValueError("routing cache identity mismatch")
        records = saved["records"]
        if set(records) != set(flat_ids):
            raise ValueError("routing cache sample set mismatch")
    else:
        records = {}
        for rows in splits.values():
            for row in rows:
                sample = sample_by_id[row["sample_id"]]
                canonical, frame = standardize_o3(
                    sample.cartesian_positions.double(),
                    sample.lattice.double(),
                    sample.atomic_numbers,
                    symprec=symprec,
                )
                number = class_dag.number(canonical.symmetry.current_point_group)
                dag = build_point_group_parent_dag(sample.sample_id, number, class_dag)
                routing = route_material_on_point_group_dag(
                    sample.sample_id,
                    row["graph"]["edge_attr"].double() @ frame.T,
                    row["graph"]["edge_index"],
                    sample.atomic_numbers,
                    canonical.symmetry,
                    dag,
                    class_dag,
                )
                records[sample.sample_id] = {
                    "point_group_number": number,
                    "frame": frame.cpu(),
                    "residuals": dict(routing.residuals),
                }
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"identity": identity, "records": records}, path)
    output = {}
    for name, rows in splits.items():
        output[name] = []
        for row in rows:
            record = records[row["sample_id"]]
            dag = build_point_group_parent_dag(
                row["sample_id"], record["point_group_number"], class_dag
            )
            if set(record["residuals"]) != {e.checksum for e in dag.embeddings}:
                raise ValueError("cached residual keys mismatch")
            if any(
                not math.isfinite(float(v)) or float(v) < 0
                for v in record["residuals"].values()
            ):
                raise ValueError("invalid cached residual")
            frame = record["frame"]
            if frame.shape != (3, 3) or not torch.allclose(
                frame @ frame.T, torch.eye(3).to(frame), atol=1e-7, rtol=1e-7
            ):
                raise ValueError("invalid cached standard frame")
            output[name].append(
                {
                    **row,
                    "routing": CrystalRouting(
                        row["sample_id"], dag, record["residuals"], frame
                    ),
                }
            )
    return output, {**identity, "routing_cache_sha256": file_sha256(path)}


def attach_invariant_inputs(splits, archive_path, *, expected_sha256):
    """Optional frozen invariant node embeddings, retaining exact graph/node order.

    Archive: {schema_version: 1, features: {id: Tensor[N,C]},
              graph_rows: {id: original row_digest}, kind: str}.
    Caller explicitly supplies the archive SHA-256; graph digests bind each node order.
    """
    if not expected_sha256 or file_sha256(archive_path) != expected_sha256:
        raise ValueError("invariant node feature archive hash mismatch")
    archive = torch.load(archive_path, map_location="cpu", weights_only=True)
    ids = {r["sample_id"] for rows in splits.values() for r in rows}
    if archive.get("schema_version") != 1 or not ids <= archive["features"].keys():
        raise ValueError("invalid invariant node feature archive")
    width = None
    result = {}
    for split, rows in splits.items():
        result[split] = []
        for row in rows:
            key = row["sample_id"]
            if archive["graph_rows"].get(key) != row_digest(row):
                raise ValueError("invariant node feature graph/order mismatch")
            features = archive["features"][key]
            if (
                features.ndim != 2
                or len(features) != len(row["graph"]["x"])
                or not torch.isfinite(features).all()
            ):
                raise ValueError("invalid invariant node feature shape/values")
            if width is not None and features.shape[1] != width:
                raise ValueError("invariant node feature width drift")
            width = features.shape[1]
            result[split].append(
                {**row, "graph": {**row["graph"], "x": features.detach().float()}}
            )
    if not width:
        raise ValueError("invariant node features must have positive width")
    return result, {
        "kind": archive.get("kind", "frozen_invariant"),
        "sha256": expected_sha256,
        "input_dimension": width,
    }
