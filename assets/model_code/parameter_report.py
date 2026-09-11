from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pg_tensor_model import ModelConfig, PointGroupTensorModel


ACTIVE_PARAMETER_BUDGET = 5_000_000


def count(module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def build_report(model: PointGroupTensorModel) -> dict:
    group_rows = []
    all_group_parameter_ids = {
        id(parameter)
        for branch in model.branches.values()
        for parameter in branch.parameters()
    }
    final_ids = {
        task: {id(parameter) for parameter in module.parameters()}
        for task, module in model.final_readouts.items()
    }
    base_shared = sum(
        parameter.numel()
        for parameter in model.parameters()
        if id(parameter) not in all_group_parameter_ids
        and all(id(parameter) not in ids for ids in final_ids.values())
    )
    final_counts = {task: count(module) for task, module in model.final_readouts.items()}

    per_group = {}
    for symbol in model.dag.symbols:
        branch = model.branches[model.symbol_to_key[symbol]]
        block_counts = [count(block) for block in branch.blocks]
        head_counts = {task: count(head) for task, head in branch.readouts.items()}
        record = model.dag.records[symbol]
        row = {
            "point_group_number": record.number,
            "hm_symbol": symbol,
            "schoenflies": record.schoenflies,
            "group_order": record.order,
            "a1_multiplicity_by_l": list(branch.a1_multiplicities),
            "a1_node_width": branch.node_width,
            "a1_edge_width": branch.edge_width,
            "pg_block_1_parameters": block_counts[0] if block_counts else 0,
            "pg_block_2_parameters": block_counts[1] if block_counts else 0,
            "dielectric_head_parameters": head_counts["dielectric"],
            "elastic_head_parameters": head_counts["elastic"],
            "expert_total_parameters": count(branch),
        }
        per_group[symbol] = (row, block_counts, head_counts)
        group_rows.append(row)

    for symbol in model.dag.symbols:
        row, _, _ = per_group[symbol]
        active = model.dag.ancestors(symbol)
        paths = model.dag.root_to_current_paths(symbol)
        active_edges = {(p, c) for path in paths for p, c in zip(path[:-1], path[1:])}
        # Only gate sigmas on traversed edges participate in this forward.
        active_gate_params = len(active_edges)
        inactive_gate_params = count(model.hierarchical_gate) - active_gate_params
        shared_for_sample = base_shared - inactive_gate_params
        row["active_expert_count"] = len(active)
        row["active_experts"] = active
        for task in ("dielectric", "elastic"):
            active_group = 0
            for active_symbol in active:
                _, blocks, heads = per_group[active_symbol]
                active_group += sum(blocks) + heads[task]
            row[f"active_parameters_{task}"] = (
                shared_for_sample + final_counts[task] + active_group
            )
            row[f"under_5m_{task}"] = (
                row[f"active_parameters_{task}"] < ACTIVE_PARAMETER_BUDGET
            )

    total = count(model)
    instantiated_by_task = {}
    for task in ("dielectric", "elastic"):
        instantiated_by_task[task] = base_shared + final_counts[task] + sum(
            sum(blocks) + heads[task] for _, blocks, heads in per_group.values()
        )
    return {
        "configuration": {
            "hidden_irreps": str(model.hidden_irreps),
            "lmax": model.config.lmax,
            "o2_mmax": model.config.o2_mmax,
            "radial_basis": model.config.radial_basis,
            "radial_channels": model.config.radial_channels,
            "route_width": model.config.route_width,
            "pg_layers": model.config.pg_layers,
            "pg_parameters_shared_across_depth": False,
            "backbone_included": False,
        },
        "totals": {
            "all_instantiated_parameters": total,
            "instantiated_parameters_single_task": instantiated_by_task,
            "base_shared_parameters_both_task_gates_included": base_shared,
            "final_readout_parameters": final_counts,
            "uniform_over_32_groups_mean_active_experts": sum(
                row["active_expert_count"] for row in group_rows
            )
            / len(group_rows),
            "uniform_over_32_groups_mean_active_parameters_dielectric": sum(
                row["active_parameters_dielectric"] for row in group_rows
            )
            / len(group_rows),
            "uniform_over_32_groups_mean_active_parameters_elastic": sum(
                row["active_parameters_elastic"] for row in group_rows
            )
            / len(group_rows),
            "maximum_active_experts": max(row["active_expert_count"] for row in group_rows),
            "maximum_active_parameters_dielectric": max(
                row["active_parameters_dielectric"] for row in group_rows
            ),
            "maximum_active_parameters_elastic": max(
                row["active_parameters_elastic"] for row in group_rows
            ),
            "active_parameter_budget_strict_upper_bound": ACTIVE_PARAMETER_BUDGET,
            "point_groups_at_or_over_5m_dielectric": [
                row["hm_symbol"] for row in group_rows if not row["under_5m_dielectric"]
            ],
            "point_groups_at_or_over_5m_elastic": [
                row["hm_symbol"] for row in group_rows if not row["under_5m_elastic"]
            ],
        },
        "point_groups": group_rows,
        "notes": [
            "Active experts are the current point group plus every abstract class-DAG ancestor.",
            "The reported mean is an unweighted mean over 32 point-group classes, not a dataset estimate.",
            "C1 bypasses both PG blocks but still has task-specific constrained heads.",
            "Buffers such as Reynolds bases and fixed CG/Wigner matrices are not trainable parameters.",
        ],
    }


def write_markdown(report: dict, path: Path) -> None:
    totals = report["totals"]
    lines = [
        "# Parameter report (backbone excluded)",
        "",
        f"- Total instantiated parameters: `{totals['all_instantiated_parameters']:,}`",
        f"- Dielectric-only instantiated parameters: `{totals['instantiated_parameters_single_task']['dielectric']:,}`",
        f"- Elastic-only instantiated parameters: `{totals['instantiated_parameters_single_task']['elastic']:,}`",
        f"- Shared/base parameters: `{totals['base_shared_parameters_both_task_gates_included']:,}`",
        f"- Unweighted mean active experts over 32 PG classes: `{totals['uniform_over_32_groups_mean_active_experts']:.2f}`",
        f"- Mean active parameters, dielectric: `{totals['uniform_over_32_groups_mean_active_parameters_dielectric']:,.0f}`",
        f"- Mean active parameters, elastic: `{totals['uniform_over_32_groups_mean_active_parameters_elastic']:,.0f}`",
        f"- Maximum active parameters, dielectric: `{totals['maximum_active_parameters_dielectric']:,}`",
        f"- Maximum active parameters, elastic: `{totals['maximum_active_parameters_elastic']:,}`",
        f"- Strict active-parameter budget: `<{totals['active_parameter_budget_strict_upper_bound']:,}`",
        f"- PG classes at or over 5M: dielectric `{totals['point_groups_at_or_over_5m_dielectric']}`, elastic `{totals['point_groups_at_or_over_5m_elastic']}`",
        "",
        "The mean is not dataset-weighted. Each row uses current PG plus all abstract class-DAG ancestors.",
        "",
        "| # | PG | A1 by l=0..4 | node | edge | block 1 | block 2 | dielectric head | elastic head | expert total | active experts | active dielectric | active elastic |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["point_groups"]:
        lines.append(
            "| {point_group_number} | {hm_symbol} | {a1} | {a1_node_width:,} | "
            "{a1_edge_width:,} | {pg_block_1_parameters:,} | {pg_block_2_parameters:,} | "
            "{dielectric_head_parameters:,} | {elastic_head_parameters:,} | "
            "{expert_total_parameters:,} | {active_expert_count} | "
            "{active_parameters_dielectric:,} | {active_parameters_elastic:,} |".format(
                a1=",".join(map(str, row["a1_multiplicity_by_l"])), **row
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports")
    args = parser.parse_args()
    model = PointGroupTensorModel(ModelConfig())
    report = build_report(model)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "parameters_by_point_group.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    with (args.output_dir / "parameters_by_point_group.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        rows = report["point_groups"]
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    write_markdown(report, args.output_dir / "parameters_by_point_group.md")
    print(json.dumps(report["totals"], indent=2))


if __name__ == "__main__":
    main()
