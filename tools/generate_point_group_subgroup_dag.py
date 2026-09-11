"""Generate the subgroup lattice of the 32 crystallographic point groups.

The concrete matrix groups come from spglib's Hall-setting database.  For each
point-group class, all subgroups are enumerated from the multiplication table,
classified again with spglib, and reduced to maximal-subgroup cover relations.
The generated JSON retains operation indices, so class-level edges can later be
expanded into oriented embedding records instead of being treated as symbols.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict, deque
from pathlib import Path

import numpy as np
import spglib


HM_SYMBOLS = {
    1: "1",
    2: "-1",
    3: "2",
    4: "m",
    5: "2/m",
    6: "222",
    7: "mm2",
    8: "mmm",
    9: "4",
    10: "-4",
    11: "4/m",
    12: "422",
    13: "4mm",
    14: "-42m",
    15: "4/mmm",
    16: "3",
    17: "-3",
    18: "32",
    19: "3m",
    20: "-3m",
    21: "6",
    22: "-6",
    23: "6/m",
    24: "622",
    25: "6mm",
    26: "-6m2",
    27: "6/mmm",
    28: "23",
    29: "m-3",
    30: "432",
    31: "-43m",
    32: "m-3m",
}


def matrix_key(matrix: np.ndarray) -> tuple[int, ...]:
    return tuple(int(x) for x in np.asarray(matrix, dtype=int).reshape(-1))


def unique_rotations(hall_number: int) -> list[np.ndarray]:
    data = spglib.get_symmetry_from_database(hall_number)
    unique = {matrix_key(rotation): np.asarray(rotation, dtype=int) for rotation in data["rotations"]}
    return [unique[key] for key in sorted(unique)]


def representative_groups() -> dict[int, dict]:
    representatives: dict[int, dict] = {}
    for hall_number in range(1, 531):
        space_group_type = spglib.get_spacegroup_type(hall_number)
        rotations = unique_rotations(hall_number)
        symbol, point_group_number, _ = spglib.get_pointgroup(np.asarray(rotations, dtype=int))
        if point_group_number in representatives:
            continue
        representatives[point_group_number] = {
            "hall_number": hall_number,
            "space_group_number": int(space_group_type.number),
            "space_group_symbol": space_group_type.international_short,
            "detected_symbol": symbol.strip(),
            "schoenflies": space_group_type.pointgroup_schoenflies,
            "rotations": rotations,
        }
    if set(representatives) != set(HM_SYMBOLS):
        missing = sorted(set(HM_SYMBOLS) - set(representatives))
        raise RuntimeError(f"Failed to find representative Hall settings for point groups: {missing}")
    return representatives


def multiplication_table(rotations: list[np.ndarray]) -> tuple[list[list[int]], int]:
    index = {matrix_key(rotation): idx for idx, rotation in enumerate(rotations)}
    table: list[list[int]] = []
    for left in rotations:
        row = []
        for right in rotations:
            product = matrix_key(left @ right)
            if product not in index:
                raise RuntimeError("Representative rotations are not closed")
            row.append(index[product])
        table.append(row)
    identity = index[matrix_key(np.eye(3, dtype=int))]
    return table, identity


def generated_subgroup(generators: frozenset[int], table: list[list[int]], identity: int) -> frozenset[int]:
    subgroup = set(generators)
    subgroup.add(identity)
    changed = True
    while changed:
        changed = False
        elements = tuple(subgroup)
        for left in elements:
            for right in elements:
                product = table[left][right]
                if product not in subgroup:
                    subgroup.add(product)
                    changed = True
    return frozenset(subgroup)


def enumerate_subgroups(table: list[list[int]], identity: int) -> list[frozenset[int]]:
    group_order = len(table)
    trivial = frozenset({identity})
    seen = {trivial}
    queue = deque([trivial])
    while queue:
        subgroup = queue.popleft()
        for element in range(group_order):
            if element in subgroup:
                continue
            candidate = generated_subgroup(subgroup | {element}, table, identity)
            if candidate not in seen:
                seen.add(candidate)
                queue.append(candidate)
    return sorted(seen, key=lambda subgroup: (len(subgroup), tuple(sorted(subgroup))))


def classify_subgroup(subgroup: frozenset[int], rotations: list[np.ndarray]) -> int:
    subgroup_rotations = np.asarray([rotations[idx] for idx in sorted(subgroup)], dtype=int)
    _, point_group_number, _ = spglib.get_pointgroup(subgroup_rotations)
    return int(point_group_number)


def maximal_children(
    parent: frozenset[int], subgroups: list[frozenset[int]]
) -> list[frozenset[int]]:
    proper = [candidate for candidate in subgroups if candidate < parent]
    children = []
    for candidate in proper:
        if not any(candidate < intermediate < parent for intermediate in proper):
            children.append(candidate)
    return sorted(children, key=lambda subgroup: (-len(subgroup), tuple(sorted(subgroup))))


def maximal_chains(
    full_group: frozenset[int],
    subgroups: list[frozenset[int]],
    point_group_numbers: dict[frozenset[int], int],
) -> Counter[tuple[int, ...]]:
    children = {subgroup: maximal_children(subgroup, subgroups) for subgroup in subgroups}
    chains: Counter[tuple[int, ...]] = Counter()

    def visit(subgroup: frozenset[int], path: tuple[int, ...]) -> None:
        next_path = path + (point_group_numbers[subgroup],)
        if len(subgroup) == 1:
            chains[next_path] += 1
            return
        for child in children[subgroup]:
            visit(child, next_path)

    visit(full_group, ())
    return chains


def format_class_counts(
    counts: Counter[int], parent_order: int, orders: dict[int, int]
) -> str:
    if not counts:
        return "—"
    parts = []
    for point_group_number in sorted(counts, key=lambda number: (-orders[number], number)):
        count = counts[point_group_number]
        index = parent_order // orders[point_group_number]
        suffix = f" ×{count}" if count > 1 else ""
        parts.append(f"`{HM_SYMBOLS[point_group_number]}`{suffix} (i={index})")
    return "; ".join(parts)


def build_lattice() -> dict:
    representatives = representative_groups()
    orders = {number: len(record["rotations"]) for number, record in representatives.items()}
    point_groups = {}
    cover_edges = defaultdict(int)

    for number in sorted(representatives):
        record = representatives[number]
        rotations = record["rotations"]
        table, identity = multiplication_table(rotations)
        subgroups = enumerate_subgroups(table, identity)
        full_group = frozenset(range(len(rotations)))
        classifications = {subgroup: classify_subgroup(subgroup, rotations) for subgroup in subgroups}
        maximal = maximal_children(full_group, subgroups)

        all_counts = Counter(
            classifications[subgroup] for subgroup in subgroups if subgroup != full_group
        )
        maximal_counts = Counter(classifications[subgroup] for subgroup in maximal)
        for child_number, count in maximal_counts.items():
            cover_edges[(number, child_number)] += count

        chains = maximal_chains(full_group, subgroups, classifications)
        point_groups[str(number)] = {
            "number": number,
            "hm_symbol": HM_SYMBOLS[number],
            "schoenflies": record["schoenflies"],
            "order": len(rotations),
            "representative_hall_number": record["hall_number"],
            "representative_space_group_number": record["space_group_number"],
            "representative_space_group_symbol": record["space_group_symbol"],
            "operation_matrices": [rotation.tolist() for rotation in rotations],
            "subgroup_instances": [
                {
                    "subgroup_id": f"pg{number:02d}_subgroup_{idx:03d}",
                    "point_group_number": classifications[subgroup],
                    "hm_symbol": HM_SYMBOLS[classifications[subgroup]],
                    "order": len(subgroup),
                    "operation_indices": sorted(subgroup),
                    "is_maximal": subgroup in maximal,
                }
                for idx, subgroup in enumerate(subgroups)
                if subgroup != full_group
            ],
            "all_subgroup_class_counts": {
                HM_SYMBOLS[key]: value for key, value in sorted(all_counts.items())
            },
            "maximal_subgroup_class_counts": {
                HM_SYMBOLS[key]: value for key, value in sorted(maximal_counts.items())
            },
            "maximal_chain_class_sequences": [
                {
                    "symbols": [HM_SYMBOLS[item] for item in sequence],
                    "embedding_chain_count": count,
                }
                for sequence, count in sorted(
                    chains.items(),
                    key=lambda item: (len(item[0]), tuple(HM_SYMBOLS[number] for number in item[0])),
                )
            ],
        }

    return {
        "metadata": {
            "generator": "tools/generate_point_group_subgroup_dag.py",
            "spglib_version": spglib.__version__,
            "relation": "all concrete matrix subgroups; maximal edges are inclusion covers",
            "edge_direction": "parent_to_child",
            "classification": "spglib.get_pointgroup",
        },
        "point_groups": point_groups,
        "class_cover_edges": [
            {
                "parent_number": parent,
                "parent_symbol": HM_SYMBOLS[parent],
                "child_number": child,
                "child_symbol": HM_SYMBOLS[child],
                "point_group_index": orders[parent] // orders[child],
                "embedding_count_in_representative_parent": count,
            }
            for (parent, child), count in sorted(
                cover_edges.items(), key=lambda item: (-orders[item[0][0]], item[0][0], -orders[item[0][1]], item[0][1])
            )
        ],
    }


def render_markdown(data: dict) -> str:
    point_groups = data["point_groups"]
    orders = {int(number): record["order"] for number, record in point_groups.items()}
    lines = [
        "# Subgroup lattice of the 32 crystallographic point groups",
        "",
        "This file is generated by `tools/generate_point_group_subgroup_dag.py` from concrete rotation matrices in the spglib Hall-setting database. Subgroups are enumerated as matrix subsets, classified with `spglib.get_pointgroup`, and reduced by set inclusion. An edge therefore denotes a maximal-subgroup cover in at least one concrete orientation, not merely an abstract-group isomorphism.",
        "",
        f"Generator library: spglib `{data['metadata']['spglib_version']}`. Edge direction: parent → child.",
        "",
        "## Complete maximal-subgroup DAG",
        "",
        "Following the maximal-subgroup column recursively gives every subgroup chain. Multiplicity `×n` counts distinct oriented subgroup instances inside the representative parent matrix group; `i` is the point-group index.",
        "",
        "| No. | Parent (HM; Schoenflies) | Order | Immediate maximal subgroups |",
        "|---:|---|---:|---|",
    ]
    for number in range(1, 33):
        record = point_groups[str(number)]
        counts = Counter(
            {
                next(key for key, symbol in HM_SYMBOLS.items() if symbol == child_symbol): count
                for child_symbol, count in record["maximal_subgroup_class_counts"].items()
            }
        )
        lines.append(
            f"| {number} | `{record['hm_symbol']}`; {record['schoenflies']} | {record['order']} | "
            f"{format_class_counts(counts, record['order'], orders)} |"
        )

    lines.extend(
        [
            "",
            "## All subgroup classes of every parent",
            "",
            "This table includes non-maximal descendants as well. Counts again refer to concrete oriented subgroup instances in the representative parent.",
            "",
            "| Parent | All proper subgroup classes |",
            "|---|---|",
        ]
    )
    for number in range(1, 33):
        record = point_groups[str(number)]
        counts = Counter(
            {
                next(key for key, symbol in HM_SYMBOLS.items() if symbol == child_symbol): count
                for child_symbol, count in record["all_subgroup_class_counts"].items()
            }
        )
        lines.append(
            f"| `{record['hm_symbol']}` | {format_class_counts(counts, record['order'], orders)} |"
        )

    lines.extend(["", "## All maximal class-chain sequences", ""])
    for number in range(1, 33):
        record = point_groups[str(number)]
        lines.append(f"### `{record['hm_symbol']}`")
        lines.append("")
        for chain in record["maximal_chain_class_sequences"]:
            sequence = " → ".join(f"`{symbol}`" for symbol in chain["symbols"])
            count = chain["embedding_chain_count"]
            suffix = f" (embedding chains: {count})" if count > 1 else ""
            lines.append(f"- {sequence}{suffix}")
        lines.append("")

    lines.extend(
        [
            "## Interpretation boundary",
            "",
            "This is the complete point-group subgroup lattice. It is only the candidate class/orientation layer of a material parent DAG. A physical parent edge must additionally be validated at space-group level with Hall setting, basis/origin transform, translation subgroup, common cell, species-preserving atom correspondence, Wyckoff splitting, and a frozen domain variant.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    args = parser.parse_args()

    data = build_lattice()
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    args.markdown.write_text(render_markdown(data), encoding="utf-8")


if __name__ == "__main__":
    main()
