"""Render point-group frequency counts from a tensor-curation report as SVG."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence
from xml.sax.saxutils import escape


POINT_GROUP_SYSTEMS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Triclinic", ("1", "-1")),
    ("Monoclinic", ("2", "m", "2/m")),
    ("Orthorhombic", ("222", "mm2", "mmm")),
    ("Tetragonal", ("4", "-4", "4/m", "422", "4mm", "-42m", "4/mmm")),
    ("Trigonal", ("3", "-3", "32", "3m", "-3m")),
    ("Hexagonal", ("6", "-6", "6/m", "622", "6mm", "-6m2", "6/mmm")),
    ("Cubic", ("23", "m-3", "432", "-43m", "m-3m")),
)
POINT_GROUPS: tuple[str, ...] = tuple(
    point_group for _, groups in POINT_GROUP_SYSTEMS for point_group in groups
)

SUBTYPE_STYLES: tuple[tuple[str, str, str], ...] = (
    ("dielectric_electronic", "Dielectric — electronic", "#277DA1"),
    ("dielectric_ionic", "Dielectric — ionic", "#43AA8B"),
    ("dielectric_total", "Dielectric — total", "#F8961E"),
    ("elastic_stiffness", "Elastic — stiffness", "#8E5EA2"),
)


def _validated_counts(report: Mapping[str, Any], subtype: str) -> list[int]:
    try:
        section = report["subtypes"][subtype]
        raw_counts = section["point_group_counts"]["recommended"]
        expected_total = section["recommended_records"]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"missing recommended point-group data for {subtype}") from exc

    if not isinstance(raw_counts, Mapping):
        raise ValueError(f"point-group counts for {subtype} must be an object")
    observed = set(raw_counts)
    expected = set(POINT_GROUPS)
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise ValueError(f"invalid point groups for {subtype}: missing={missing}, extra={extra}")

    counts: list[int] = []
    for point_group in POINT_GROUPS:
        value = raw_counts[point_group]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"count for {subtype}/{point_group} must be a non-negative integer")
        counts.append(value)
    if isinstance(expected_total, bool) or not isinstance(expected_total, int):
        raise ValueError(f"recommended_records for {subtype} must be an integer")
    if sum(counts) != expected_total:
        raise ValueError(
            f"point-group counts for {subtype} sum to {sum(counts)}, expected {expected_total}"
        )
    return counts


def _nice_axis_max(value: int) -> int:
    if value <= 0:
        return 4
    rough_step = value / 4.0
    magnitude = 10 ** math.floor(math.log10(rough_step))
    normalized = rough_step / magnitude
    step_factor = 1 if normalized <= 1 else 2 if normalized <= 2 else 5 if normalized <= 5 else 10
    step = step_factor * magnitude
    return int(math.ceil(value / step) * step)


def _text(
    x: float,
    y: float,
    value: str,
    *,
    size: int = 18,
    anchor: str = "start",
    weight: int = 400,
    fill: str = "#243447",
    transform: str | None = None,
) -> str:
    transform_attr = f' transform="{escape(transform)}"' if transform else ""
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" text-anchor="{anchor}" '
        f'font-weight="{weight}" fill="{fill}"{transform_attr}>{escape(value)}</text>'
    )


def render_point_group_frequency_svg(report: Mapping[str, Any]) -> str:
    """Return a deterministic four-panel frequency plot as an SVG string."""

    width, height = 1800, 1320
    left, right = 112.0, 38.0
    plot_width = width - left - right
    panel_top, panel_height, panel_gap = 112.0, 220.0, 42.0
    slot_width = plot_width / len(POINT_GROUPS)
    bar_width = slot_width * 0.72
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}" role="img" '
            'aria-labelledby="title description">'
        ),
        '<title id="title">Recommended dataset frequency by crystallographic point group</title>',
        (
            '<desc id="description">Four bar-chart panels show record counts for electronic, '
            "ionic, and total dielectric tensors and elastic stiffness tensors across all 32 "
            "point groups.</desc>"
        ),
        '<rect width="100%" height="100%" fill="#FFFFFF"/>',
        _text(
            width / 2,
            43,
            "Recommended dataset frequency by crystallographic point group",
            size=29,
            anchor="middle",
            weight=700,
        ),
        _text(
            width / 2,
            76,
            "Independent y-axes preserve within-property imbalance; bars show record counts",
            size=17,
            anchor="middle",
            fill="#5B6770",
        ),
    ]

    for panel_index, (subtype, label, color) in enumerate(SUBTYPE_STYLES):
        counts = _validated_counts(report, subtype)
        total = sum(counts)
        axis_max = _nice_axis_max(max(counts))
        top = panel_top + panel_index * (panel_height + panel_gap)
        bottom = top + panel_height

        lines.append(f'<g id="{subtype}">')
        for tick_index in range(5):
            tick_value = axis_max * tick_index // 4
            y = bottom - panel_height * tick_index / 4
            lines.append(
                f'<line x1="{left:.1f}" y1="{y:.1f}" x2="{left + plot_width:.1f}" y2="{y:.1f}" '
                'stroke="#DCE3E8" stroke-width="1"/>'
            )
            lines.append(
                _text(
                    left - 12,
                    y + 6,
                    f"{tick_value:,}",
                    size=14,
                    anchor="end",
                    fill="#607080",
                )
            )

        offset = 0
        for _, groups in POINT_GROUP_SYSTEMS[:-1]:
            offset += len(groups)
            x = left + offset * slot_width
            lines.append(
                f'<line x1="{x:.1f}" y1="{top:.1f}" x2="{x:.1f}" y2="{bottom:.1f}" '
                'stroke="#AAB7C1" stroke-width="1" stroke-dasharray="4 5"/>'
            )

        for index, count in enumerate(counts):
            x = left + index * slot_width + (slot_width - bar_width) / 2
            bar_height = panel_height * count / axis_max
            y = bottom - bar_height
            lines.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width:.1f}" height="{bar_height:.1f}" '
                f'rx="2" fill="{color}"><title>{escape(POINT_GROUPS[index])}: {count:,} records '
                f'({count / total:.1%})</title></rect>'
            )

        peak_index = max(range(len(counts)), key=counts.__getitem__)
        peak_count = counts[peak_index]
        lines.append(_text(left, top - 13, label, size=19, weight=700, fill=color))
        lines.append(
            _text(
                left + plot_width,
                top - 13,
                f"max: {POINT_GROUPS[peak_index]} — {peak_count:,} ({peak_count / total:.1%})",
                size=15,
                anchor="end",
                weight=600,
                fill="#44515C",
            )
        )
        axis_center = top + panel_height / 2
        lines.append(
            _text(
                28,
                axis_center,
                "Frequency (records)",
                size=14,
                anchor="middle",
                fill="#607080",
                transform=f"rotate(-90 28 {axis_center:.1f})",
            )
        )
        lines.append("</g>")

    bottom = panel_top + 3 * (panel_height + panel_gap) + panel_height
    for index, point_group in enumerate(POINT_GROUPS):
        x = left + (index + 0.5) * slot_width
        tick_y = bottom + 29
        lines.append(
            _text(
                x,
                tick_y,
                point_group,
                size=14,
                anchor="end",
                transform=f"rotate(-55 {x:.1f} {tick_y:.1f})",
            )
        )

    offset = 0
    for system_name, groups in POINT_GROUP_SYSTEMS:
        center = left + (offset + len(groups) / 2) * slot_width
        lines.append(
            _text(
                center,
                bottom + 112,
                system_name,
                size=14,
                anchor="middle",
                weight=600,
                fill="#53616C",
            )
        )
        offset += len(groups)
    lines.append(
        _text(
            left + plot_width / 2,
            height - 27,
            "Crystallographic point group (Hermann–Mauguin notation)",
            size=17,
            anchor="middle",
            weight=600,
        )
    )
    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def render_report(input_path: Path, output_path: Path) -> None:
    report = json.loads(input_path.read_text(encoding="utf-8"))
    svg = render_point_group_frequency_svg(report)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(svg, encoding="utf-8", newline="\n")


def _parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=root / "docs" / "analysis" / "curated_tensor_datasets.json",
        help="Machine-readable curation report.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "docs" / "analysis" / "curated_tensor_point_group_frequency.svg",
        help="Output SVG path.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    render_report(args.input, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
