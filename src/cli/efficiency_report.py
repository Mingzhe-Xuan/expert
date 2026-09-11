from __future__ import annotations

import argparse
from glob import glob
import json
import math
from pathlib import Path
from typing import Iterable, Mapping


EXPECTED_ROWS = {"point_group": 58, "real_subset": 20}
FLOPS_SCOPE = "torch_dispatched_active_downstream_forward"
REQUIRED_EFFICIENCY_FIELDS = {
    "schema_version",
    "architecture",
    "task",
    "point_groups",
    "pg_hidden_mode",
    "adaptation_backend",
    "o3e_backend",
    "readout_backend",
    "total_parameters",
    "torch_registered_parameters",
    "external_frozen_parameters",
    "trainable_parameters",
    "active_nonbackbone_parameters",
    "active_expert_count_per_sample",
    "active_expert_count_mean",
    "active_expert_count_max",
    "active_downstream_flops",
    "flops_scope",
    "end_to_end_forward_latency_ms",
    "latency_repetitions",
    "peak_cuda_allocated_bytes",
    "device",
}


def _finite_nonnegative(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{field} must be finite and non-negative")
    return result


def _validate_efficiency(value: object, source: Path) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{source}: missing efficiency object")
    missing = sorted(REQUIRED_EFFICIENCY_FIELDS - set(value))
    if missing:
        raise ValueError(f"{source}: missing efficiency fields: {', '.join(missing)}")
    if value["schema_version"] != 1:
        raise ValueError(f"{source}: unsupported efficiency schema_version")
    strings = (
        "architecture",
        "task",
        "pg_hidden_mode",
        "adaptation_backend",
        "o3e_backend",
        "readout_backend",
        "device",
    )
    if any(not isinstance(value[field], str) or not value[field] for field in strings):
        raise ValueError(f"{source}: efficiency identity fields must be non-empty strings")
    groups = value["point_groups"]
    counts = value["active_expert_count_per_sample"]
    if (
        not isinstance(groups, list)
        or not groups
        or any(not isinstance(group, str) or not group for group in groups)
    ):
        raise ValueError(f"{source}: point_groups must be a non-empty string list")
    if (
        not isinstance(counts, list)
        or len(counts) != len(groups)
        or any(isinstance(count, bool) or not isinstance(count, int) or count < 0 for count in counts)
    ):
        raise ValueError(f"{source}: active expert counts must align with point_groups")
    integer_fields = (
        "total_parameters",
        "torch_registered_parameters",
        "external_frozen_parameters",
        "trainable_parameters",
        "active_nonbackbone_parameters",
        "active_expert_count_max",
        "active_downstream_flops",
        "latency_repetitions",
    )
    if any(
        isinstance(value[field], bool) or not isinstance(value[field], int) or value[field] < 0
        for field in integer_fields
    ):
        raise ValueError(f"{source}: efficiency counts must be non-negative integers")
    _finite_nonnegative(value["active_expert_count_mean"], "active_expert_count_mean")
    _finite_nonnegative(
        value["end_to_end_forward_latency_ms"], "end_to_end_forward_latency_ms"
    )
    peak = value["peak_cuda_allocated_bytes"]
    if peak is not None and (isinstance(peak, bool) or not isinstance(peak, int) or peak < 0):
        raise ValueError(f"{source}: peak_cuda_allocated_bytes must be null or non-negative")
    if value["flops_scope"] != FLOPS_SCOPE:
        raise ValueError(f"{source}: unsupported FLOPs scope {value['flops_scope']!r}")
    if value["latency_repetitions"] != 1:
        raise ValueError(f"{source}: latency must describe one real end-to-end forward")
    if value["total_parameters"] != (
        value["torch_registered_parameters"] + value["external_frozen_parameters"]
    ):
        raise ValueError(f"{source}: total parameter count is inconsistent")
    if value["trainable_parameters"] > value["torch_registered_parameters"]:
        raise ValueError(f"{source}: trainable parameter count is inconsistent")
    if value["active_nonbackbone_parameters"] >= 5_000_000:
        raise ValueError(f"{source}: active non-backbone parameter budget exceeded")
    if value["active_downstream_flops"] <= 0:
        raise ValueError(f"{source}: active downstream FLOPs must be positive")
    if not str(value["device"]).startswith("cuda") or peak is None:
        raise ValueError(f"{source}: final efficiency evidence must come from CUDA")
    if value["active_expert_count_max"] != max(counts):
        raise ValueError(f"{source}: active expert maximum is inconsistent")
    if not math.isclose(
        float(value["active_expert_count_mean"]), sum(counts) / len(counts)
    ):
        raise ValueError(f"{source}: active expert mean is inconsistent")
    return dict(value)


def _load_record(path: Path, suite: str) -> dict[str, object]:
    summary = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(summary, dict) or summary.get("status") != "passed":
        raise ValueError(f"{path}: smoke summary is not passed")
    index = summary.get("index")
    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        raise ValueError(f"{path}: smoke index must be a non-negative integer")
    efficiency = _validate_efficiency(summary.get("efficiency"), path)
    architecture = summary.get("architecture")
    if (
        not isinstance(architecture, dict)
        or architecture.get("variant_id") != efficiency["architecture"]
    ):
        raise ValueError(f"{path}: outer and efficiency architecture identities disagree")
    outer_task = summary.get("task")
    if suite == "real_subset":
        unit = summary.get("training_unit")
        if not isinstance(unit, str) or not unit.endswith(f"::{efficiency['task']}"):
            raise ValueError(f"{path}: training unit and efficiency task disagree")
    elif outer_task != efficiency["task"]:
        raise ValueError(f"{path}: outer and efficiency task identities disagree")
    return {
        "suite": suite,
        "index": index,
        "source": path.as_posix(),
        "backbone": summary.get("backbone"),
        "training_unit": summary.get("training_unit"),
        "purpose": summary.get("purpose"),
        **efficiency,
    }


def aggregate_efficiency_reports(
    paths_by_suite: Mapping[str, Iterable[Path]],
    *,
    expected_rows: Mapping[str, int] = EXPECTED_ROWS,
) -> dict[str, object]:
    """Validate complete smoke matrices and return one stable machine report."""

    if set(paths_by_suite) != set(expected_rows):
        raise ValueError("report suites must exactly match expected suites")
    records: list[dict[str, object]] = []
    identities: set[tuple[str, int]] = set()
    for suite in sorted(expected_rows):
        paths = sorted(
            (Path(path) for path in paths_by_suite[suite]),
            key=lambda path: path.as_posix(),
        )
        suite_records = [_load_record(path, suite) for path in paths]
        indices = {int(record["index"]) for record in suite_records}
        expected_indices = set(range(expected_rows[suite]))
        if len(suite_records) != expected_rows[suite] or indices != expected_indices:
            raise ValueError(
                f"{suite}: expected indices 0..{expected_rows[suite] - 1} exactly once"
            )
        for record in suite_records:
            identity = (suite, int(record["index"]))
            if identity in identities:
                raise ValueError(f"duplicate smoke identity {identity}")
            identities.add(identity)
            records.append(record)
    records.sort(key=lambda record: (str(record["suite"]), int(record["index"])))
    return {
        "schema_version": 1,
        "status": "passed",
        "flops_scope": FLOPS_SCOPE,
        "latency_scope": "one_real_end_to_end_forward",
        "row_count": len(records),
        "suite_counts": {suite: expected_rows[suite] for suite in sorted(expected_rows)},
        "records": records,
    }


def render_efficiency_markdown(report: Mapping[str, object]) -> str:
    records = report["records"]
    lines = [
        "# Runtime efficiency report",
        "",
        f"Status: **{report['status']}**; rows: **{report['row_count']}**.",
        "",
        f"FLOPs scope: `{report['flops_scope']}`. Latency scope: `{report['latency_scope']}`. ",
        "FLOPs cover only the active Torch downstream forward on cached backbone features; ",
        "latency is one real synchronized end-to-end forward.",
        "",
        "| Suite | Row | Backbone | Training unit | Architecture | Task | PG | PG mode | A backend | O3E backend | Readout | Total params | Trainable | Active non-B | Experts mean/max | Active downstream FLOPs | E2E ms | CUDA peak MiB | Device |",
        "|---|---:|---|---|---|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for record in records:
        peak = record["peak_cuda_allocated_bytes"]
        peak_mib = "n/a" if peak is None else f"{int(peak) / (1024 ** 2):.2f}"
        values = (
            record["suite"],
            record["index"],
            record.get("backbone") or "n/a",
            record.get("training_unit") or "n/a",
            record["architecture"],
            record["task"],
            ", ".join(record["point_groups"]),
            record["pg_hidden_mode"],
            record["adaptation_backend"],
            record["o3e_backend"],
            record["readout_backend"],
            record["total_parameters"],
            record["trainable_parameters"],
            record["active_nonbackbone_parameters"],
            f"{float(record['active_expert_count_mean']):.2f}/{record['active_expert_count_max']}",
            record["active_downstream_flops"],
            f"{float(record['end_to_end_forward_latency_ms']):.3f}",
            peak_mib,
            record["device"],
        )
        escaped = [str(value).replace("|", "\\|").replace("\n", " ") for value in values]
        lines.append("| " + " | ".join(escaped) + " |")
    return "\n".join(lines) + "\n"


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate complete runtime-efficiency evidence")
    parser.add_argument("--point-group-glob", required=True)
    parser.add_argument("--real-subset-glob", required=True)
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path, required=True)
    arguments = parser.parse_args()
    report = aggregate_efficiency_reports(
        {
            "point_group": [Path(path) for path in glob(arguments.point_group_glob)],
            "real_subset": [Path(path) for path in glob(arguments.real_subset_glob)],
        }
    )
    json_content = json.dumps(report, indent=2, sort_keys=True) + "\n"
    markdown_content = render_efficiency_markdown(report)
    _atomic_write(arguments.json_output, json_content)
    _atomic_write(arguments.markdown_output, markdown_content)
    print(json_content, end="")


if __name__ == "__main__":
    main()
