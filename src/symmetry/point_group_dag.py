from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Sequence

from .registry import DEFAULT_REGISTRY_PATH, canonical_point_group_symbol


POINT_GROUP_DAG_CONVENTION = "crystallographic-point-group-all-ancestors-v1"


@dataclass(frozen=True, slots=True)
class PointGroupAncestorDAG:
    """Offline point-group class DAG with deterministic transitive parent lookup."""

    symbols_by_number: Mapping[int, str]
    parents_by_child: Mapping[int, tuple[int, ...]]
    asset_sha256: str
    edge_count: int

    @classmethod
    def from_path(cls, path: str | Path = DEFAULT_REGISTRY_PATH) -> "PointGroupAncestorDAG":
        source = Path(path)
        encoded = source.read_bytes()
        payload = json.loads(encoded.decode("utf-8"))
        if payload.get("schema_version") != 1:
            raise ValueError("unsupported point-group DAG schema")
        metadata = payload.get("metadata", {})
        if metadata.get("point_group_count") != 32:
            raise ValueError("point-group DAG must contain exactly 32 classes")
        if metadata.get("edge_direction") != "parent_to_child":
            raise ValueError("point-group DAG edges must be parent_to_child")
        raw_groups = payload.get("point_groups")
        if not isinstance(raw_groups, dict) or set(raw_groups) != {
            str(number) for number in range(1, 33)
        }:
            raise ValueError("point-group DAG keys must be exactly 1..32")

        symbols_by_number = {
            number: canonical_point_group_symbol(str(raw_groups[str(number)]["hm_symbol"]))
            for number in range(1, 33)
        }
        if len(set(symbols_by_number.values())) != 32:
            raise ValueError("point-group DAG symbols must be unique")
        number_by_symbol = {symbol: number for number, symbol in symbols_by_number.items()}
        parents: dict[int, set[int]] = {number: set() for number in range(1, 33)}
        edges: set[tuple[int, int]] = set()
        for parent_number in range(1, 33):
            raw = raw_groups[str(parent_number)]
            if int(raw.get("number", -1)) != parent_number:
                raise ValueError("point-group DAG embedded number mismatch")
            children = raw.get("maximal_subgroup_class_counts")
            if not isinstance(children, dict):
                raise ValueError("point-group DAG maximal subgroup map must be an object")
            for child_symbol in children:
                canonical_child = canonical_point_group_symbol(str(child_symbol))
                if canonical_child not in number_by_symbol:
                    raise ValueError("point-group DAG edge references an unknown child")
                child_number = number_by_symbol[canonical_child]
                if child_number == parent_number:
                    raise ValueError("point-group DAG cannot contain self edges")
                edges.add((parent_number, child_number))
                parents[child_number].add(parent_number)
        declared_edges = int(metadata.get("class_cover_edge_count", -1))
        if len(edges) != declared_edges:
            raise ValueError("point-group DAG cover edge count mismatch")

        visiting: set[int] = set()
        visited: set[int] = set()

        def visit(number: int) -> None:
            if number in visiting:
                raise ValueError("point-group parent relation must be acyclic")
            if number in visited:
                return
            visiting.add(number)
            for parent in parents[number]:
                visit(parent)
            visiting.remove(number)
            visited.add(number)

        for number in range(1, 33):
            visit(number)
        return cls(
            MappingProxyType(symbols_by_number),
            MappingProxyType(
                {number: tuple(sorted(values)) for number, values in parents.items()}
            ),
            hashlib.sha256(encoded).hexdigest(),
            len(edges),
        )

    def number(self, symbol: str) -> int:
        canonical = canonical_point_group_symbol(symbol)
        for number, candidate in self.symbols_by_number.items():
            if candidate == canonical:
                return number
        raise KeyError(symbol)

    def ancestors(self, current_number: int, *, include_current: bool = True) -> tuple[int, ...]:
        if current_number not in self.symbols_by_number:
            raise ValueError("point-group number must be in [1, 32]")
        reached: set[int] = {current_number} if include_current else set()
        frontier = list(self.parents_by_child[current_number])
        while frontier:
            parent = frontier.pop()
            if parent in reached:
                continue
            reached.add(parent)
            frontier.extend(self.parents_by_child[parent])
        return tuple(sorted(reached))

    def symbols(self, numbers: Sequence[int]) -> tuple[str, ...]:
        return tuple(self.symbols_by_number[number] for number in numbers)

    def metadata(self) -> dict[str, object]:
        return {
            "convention_id": POINT_GROUP_DAG_CONVENTION,
            "asset_sha256": self.asset_sha256,
            "point_group_count": len(self.symbols_by_number),
            "cover_edge_count": self.edge_count,
            "activation": "current_plus_all_transitive_parents",
        }


def save_point_group_number_cache(
    path: str | Path,
    *,
    sample_ids: Sequence[str],
    point_group_numbers: Sequence[int],
    dataset_sha256: str,
    dag: PointGroupAncestorDAG,
) -> None:
    if not sample_ids or len(sample_ids) != len(point_group_numbers):
        raise ValueError("point-group number cache rows must align with non-empty sample IDs")
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("point-group number cache sample IDs must be unique")
    for number in point_group_numbers:
        dag.ancestors(int(number))
    if len(dataset_sha256) != 64:
        raise ValueError("point-group number cache requires a dataset SHA-256")
    payload = {
        "schema_version": 1,
        "convention_id": POINT_GROUP_DAG_CONVENTION,
        "dataset_sha256": dataset_sha256,
        "dag_asset_sha256": dag.asset_sha256,
        "sample_ids": list(sample_ids),
        "point_group_numbers": [int(number) for number in point_group_numbers],
    }
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_point_group_number_cache(
    path: str | Path,
    *,
    sample_ids: Sequence[str],
    dataset_sha256: str,
    dag: PointGroupAncestorDAG,
) -> tuple[int, ...]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    expected = {
        "schema_version": 1,
        "convention_id": POINT_GROUP_DAG_CONVENTION,
        "dataset_sha256": dataset_sha256,
        "dag_asset_sha256": dag.asset_sha256,
        "sample_ids": list(sample_ids),
    }
    for name, value in expected.items():
        if payload.get(name) != value:
            raise ValueError(f"point-group number cache {name} mismatch")
    raw_numbers = payload.get("point_group_numbers")
    if not isinstance(raw_numbers, list) or len(raw_numbers) != len(sample_ids):
        raise ValueError("point-group number cache values do not align with sample IDs")
    numbers = tuple(int(number) for number in raw_numbers)
    for number in numbers:
        dag.ancestors(number)
    return numbers

