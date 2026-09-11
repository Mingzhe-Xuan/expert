import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "docs" / "ref" / "crystallographic_point_group_subgroups.json"


def load_artifact():
    return json.loads(ARTIFACT.read_text(encoding="utf-8"))


def test_all_32_crystallographic_point_groups_are_present():
    data = load_artifact()
    assert set(data["point_groups"]) == {str(number) for number in range(1, 33)}
    assert data["point_groups"]["1"]["hm_symbol"] == "1"
    assert data["point_groups"]["32"]["hm_symbol"] == "m-3m"
    assert data["point_groups"]["32"]["order"] == 48


def test_every_subgroup_instance_obeys_lagrange_and_chain_ordering():
    data = load_artifact()
    symbol_to_order = {
        record["hm_symbol"]: record["order"] for record in data["point_groups"].values()
    }
    for record in data["point_groups"].values():
        parent_order = record["order"]
        for subgroup in record["subgroup_instances"]:
            assert subgroup["order"] < parent_order
            assert parent_order % subgroup["order"] == 0
            assert len(subgroup["operation_indices"]) == subgroup["order"]
        for chain in record["maximal_chain_class_sequences"]:
            symbols = chain["symbols"]
            orders = [symbol_to_order[symbol] for symbol in symbols]
            assert symbols[0] == record["hm_symbol"]
            assert symbols[-1] == "1"
            assert all(left > right for left, right in zip(orders, orders[1:]))


def test_key_cross_crystal_system_cover_relations_are_present():
    data = load_artifact()
    edges = {
        (edge["parent_symbol"], edge["child_symbol"]): edge
        for edge in data["class_cover_edges"]
    }
    assert edges[("m-3m", "-3m")]["embedding_count_in_representative_parent"] == 4
    assert edges[("6/mmm", "-3m")]["embedding_count_in_representative_parent"] == 2
    assert edges[("m-3m", "4/mmm")]["embedding_count_in_representative_parent"] == 3
    assert edges[("6/mmm", "mmm")]["embedding_count_in_representative_parent"] == 3


def test_declared_maximal_counts_match_maximal_instances():
    data = load_artifact()
    for record in data["point_groups"].values():
        observed = {}
        for subgroup in record["subgroup_instances"]:
            if subgroup["is_maximal"]:
                symbol = subgroup["hm_symbol"]
                observed[symbol] = observed.get(symbol, 0) + 1
        assert observed == record["maximal_subgroup_class_counts"]
