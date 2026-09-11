import json
from collections import Counter
from pathlib import Path


ASSET_PATH = (
    Path(__file__).resolve().parents[2] / "docs" / "subgroup_chain.json"
)


def _matrix_key(matrix):
    return tuple(value for row in matrix for value in row)


def _matmul(left, right):
    return [
        [sum(left[i][k] * right[k][j] for k in range(3)) for j in range(3)]
        for i in range(3)
    ]


def test_subgroup_chain_asset_is_complete_and_consistent():
    data = json.loads(ASSET_PATH.read_text(encoding="utf-8"))
    point_groups = data["point_groups"]
    edges = data["class_cover_edges"]
    metadata = data["metadata"]
    policy = data["proposal_runtime_policy"]

    assert data["schema_version"] == 1
    assert set(point_groups) == {str(number) for number in range(1, 33)}
    assert metadata["point_group_count"] == 32
    assert metadata["class_cover_edge_count"] == len(edges) == 80
    assert metadata["oriented_subgroup_instance_count"] == sum(
        len(record["subgroup_instances"]) for record in point_groups.values()
    ) == 433
    assert metadata["maximal_chain_class_sequence_count"] == sum(
        len(record["maximal_chain_class_sequences"])
        for record in point_groups.values()
    ) == 222
    assert policy["class_skeleton_roots"] == ["m-3m", "6/mmm"]
    assert policy["max_supergroup_index"] == 4
    assert policy["max_parent_depth"] == 2
    assert policy["hard_top_k"] is False
    assert policy["requires_hall_level_parent_embedding_registry"] is True

    cover_pairs = {
        (edge["parent_symbol"], edge["child_symbol"]) for edge in edges
    }
    assert len(cover_pairs) == len(edges)

    for record in point_groups.values():
        matrices = record["operation_matrices"]
        matrix_to_index = {
            _matrix_key(matrix): index for index, matrix in enumerate(matrices)
        }
        assert len(matrix_to_index) == record["order"] == len(matrices)

        subgroup_ids = set()
        maximal_counts = Counter()
        for instance in record["subgroup_instances"]:
            assert instance["subgroup_id"] not in subgroup_ids
            subgroup_ids.add(instance["subgroup_id"])
            indices = set(instance["operation_indices"])
            assert len(indices) == instance["order"]
            assert indices <= set(range(record["order"]))
            if instance["is_maximal"]:
                maximal_counts[instance["hm_symbol"]] += 1
            for left in indices:
                for right in indices:
                    product = matrix_to_index[
                        _matrix_key(_matmul(matrices[left], matrices[right]))
                    ]
                    assert product in indices

        assert dict(maximal_counts) == record["maximal_subgroup_class_counts"]
        for chain in record["maximal_chain_class_sequences"]:
            symbols = chain["symbols"]
            assert symbols[0] == record["hm_symbol"]
            assert symbols[-1] == "1"
            assert chain["embedding_chain_count"] >= 1
            assert all(pair in cover_pairs for pair in zip(symbols, symbols[1:]))
