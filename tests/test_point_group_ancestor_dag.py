from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.symmetry import (
    POINT_GROUP_DAG_CONVENTION,
    PointGroupAncestorDAG,
    load_point_group_number_cache,
    save_point_group_number_cache,
)


def test_offline_point_group_dag_has_complete_deterministic_ancestor_closures() -> None:
    dag = PointGroupAncestorDAG.from_path()
    assert len(dag.symbols_by_number) == 32
    assert dag.edge_count == 80
    assert dag.metadata()["convention_id"] == POINT_GROUP_DAG_CONVENTION
    expected = {
        "2/m": ("2/m", "mmm", "4/m", "4/mmm", "-3m", "6/m", "6/mmm", "m-3", "m-3m"),
        "mm2": ("mm2", "mmm", "4mm", "-42m", "4/mmm", "6mm", "-6m2", "6/mmm", "m-3", "-43m", "m-3m"),
        "mmm": ("mmm", "4/mmm", "6/mmm", "m-3", "m-3m"),
        "4/mmm": ("4/mmm", "m-3m"),
        "-3m": ("-3m", "6/mmm", "m-3m"),
        "-43m": ("-43m", "m-3m"),
        "m-3m": ("m-3m",),
    }
    for current, symbols in expected.items():
        active = dag.ancestors(dag.number(current))
        assert active == tuple(sorted(set(active)))
        assert dag.symbols(active) == symbols


def test_offline_point_group_dag_rejects_edge_metadata_drift(tmp_path) -> None:
    source = PointGroupAncestorDAG.from_path()
    payload_path = tmp_path / "subgroup-chain.json"
    original = json.loads(
        Path("assets/docs/subgroup_chain.json").read_text(encoding="utf-8")
    )
    original["metadata"]["class_cover_edge_count"] = source.edge_count - 1
    payload_path.write_text(json.dumps(original), encoding="utf-8")
    with pytest.raises(ValueError, match="edge count"):
        PointGroupAncestorDAG.from_path(payload_path)


def test_point_group_number_cache_is_ordered_and_hash_scoped(tmp_path) -> None:
    dag = PointGroupAncestorDAG.from_path()
    path = tmp_path / "point-groups.json"
    digest = "a" * 64
    save_point_group_number_cache(
        path,
        sample_ids=("a", "b"),
        point_group_numbers=(dag.number("2/m"), dag.number("m-3m")),
        dataset_sha256=digest,
        dag=dag,
    )
    assert load_point_group_number_cache(
        path, sample_ids=("a", "b"), dataset_sha256=digest, dag=dag
    ) == (5, 32)
    with pytest.raises(ValueError, match="sample_ids mismatch"):
        load_point_group_number_cache(
            path, sample_ids=("b", "a"), dataset_sha256=digest, dag=dag
        )
    with pytest.raises(ValueError, match="dataset_sha256 mismatch"):
        load_point_group_number_cache(
            path, sample_ids=("a", "b"), dataset_sha256="b" * 64, dag=dag
        )
