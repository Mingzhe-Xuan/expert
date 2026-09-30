"""Train-split structural active-parameter accounting and dense capacity search."""

from dataclasses import replace
from statistics import mean
from e3nn import o3


def trainable_count(modules):
    unique = {id(p): p for module in modules for p in module.parameters() if p.requires_grad}
    return sum(p.numel() for p in unique.values())


def pg_active_budget(model, train_rows):
    if model.config.auxiliary_type != "pg" or not train_rows:
        raise ValueError("nonempty training rows and PG reference model required")
    shared = trainable_count([model.input_map, model.output_map, model.adapter])
    shared += model.branch_logit.numel() if model.branch_logit.requires_grad else 0
    active_counts = []
    for row in train_rows:
        dag = row["routing"].dag
        groups = {dag.current_point_group_number}
        edges = set()
        for path in dag.current_to_root_embedding_paths():
            for edge in path:
                groups.update((edge.parent_point_group_number, edge.child_point_group_number))
                edges.add(edge.edge_id)
        if not edges.issubset(set(model.router.edge_ids)):
            raise ValueError("unregistered training router edge")
        count = trainable_count([model.experts[str(pg)] for pg in groups])
        count += len(edges) if model.router.log_scales.requires_grad else 0
        active_counts.append(count)
    return {"training_crystals": len(train_rows), "shared_parameters": shared,
            "mean_active_experts_router": mean(active_counts),
            "mean_active_auxiliary": shared + mean(active_counts),
            "min_active_experts_router": min(active_counts),
            "max_active_experts_router": max(active_counts),
            "stored_auxiliary_parameters": shared + trainable_count([model.experts, model.router]),
            "definition": "train-only crystal mean; unique structural PG/edge activity; no alpha weighting"}


def linear_count(left, right):
    return sum(a * b for a, i in left for b, j in right if i == j)


def dense_parameter_count(external, hidden, depth, radial_width, gate_width, edge_lmax):
    external, hidden = o3.Irreps(external), o3.Irreps(hidden)
    edge = o3.Irreps.spherical_harmonics(edge_lmax)
    paths = sum(a * b * c for a, i in hidden for b, j in edge for c, k in hidden if k in i * j)
    copies = hidden.num_irreps
    invariants = copies + sum(a for a, i in hidden if i.l == 0 and i.p == 1)
    gate = (invariants + 1) * gate_width + (gate_width + 1) * copies
    block = paths + 3 * radial_width + 1 + gate + linear_count(hidden, hidden) + 1
    projections = 0 if external == hidden else 2 * linear_count(external, hidden)
    return depth * block + projections


def match_dense_config(config, budget):
    """Choose meaningful depth/multiplicities/gate width; no padding parameters."""
    target = budget["mean_active_experts_router"]
    external = o3.Irreps(config.expert_irreps)
    candidates = []
    for multiplier in range(1, 17):
        hidden = o3.Irreps([(mul * multiplier, ir) for mul, ir in external])
        for depth in range(2, 9):
            base = dense_parameter_count(external, hidden, depth,
                                         config.dense_radial_width, 8, config.adapter_edge_lmax)
            slope = depth * (2 * hidden.num_irreps + 1 +
                             sum(mul for mul, ir in hidden if ir.l == 0 and ir.p == 1))
            ideal = (target - base) / slope + 8
            for gate in {max(8, min(256, int(ideal))), max(8, min(256, int(ideal) + 1))}:
                count = base + slope * (gate - 8)
                candidates.append((abs(count - target), multiplier, depth, gate, count, str(hidden)))
    error, _, depth, gate, count, hidden = min(candidates)
    result = replace(config, auxiliary_type="dense", dense_depth=depth,
                     dense_gate_width=gate, dense_hidden_irreps=hidden)
    return result, {**budget, "dense_experts_replacement_parameters": count,
                    "dense_active_auxiliary": budget["shared_parameters"] + count,
                    "replacement_relative_error": error / max(target, 1),
                    "auxiliary_relative_error": error / max(budget["mean_active_auxiliary"], 1),
                    "selected_config": result.metadata()}
