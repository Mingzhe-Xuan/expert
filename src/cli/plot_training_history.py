from __future__ import annotations

import argparse
from pathlib import Path

from ..evaluation.training_history import (
    load_current_group_history,
    load_gmtnet_history,
    load_parent_dag_history,
    render_current_group_history,
    render_routing_comparison_history,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot a validated CGCNN current-group-only training summary"
    )
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--gmtnet-summary", type=Path)
    parser.add_argument("--gmtnet-expected-sha256")
    parser.add_argument("--parent-dag-summary", type=Path)
    parser.add_argument("--parent-dag-expected-sha256")
    parser.add_argument("--svg", type=Path, required=True)
    parser.add_argument("--png", type=Path, required=True)
    arguments = parser.parse_args()
    summary = load_current_group_history(
        arguments.summary, expected_sha256=arguments.expected_sha256
    )
    if (arguments.gmtnet_summary is None) != (arguments.gmtnet_expected_sha256 is None):
        parser.error("GMTNet summary and expected SHA-256 must be supplied together")
    if (arguments.parent_dag_summary is None) != (
        arguments.parent_dag_expected_sha256 is None
    ):
        parser.error("parent-DAG summary and expected SHA-256 must be supplied together")
    if arguments.gmtnet_summary is not None and arguments.parent_dag_summary is not None:
        parser.error("GMTNet and parent-DAG overlays are separate figures")
    gmtnet = (
        None
        if arguments.gmtnet_summary is None
        else load_gmtnet_history(
            arguments.gmtnet_summary,
            expected_sha256=arguments.gmtnet_expected_sha256,
        )
    )
    if arguments.parent_dag_summary is not None:
        parent = load_parent_dag_history(
            arguments.parent_dag_summary,
            expected_sha256=arguments.parent_dag_expected_sha256,
        )
        render_routing_comparison_history(
            summary, parent, svg_path=arguments.svg, png_path=arguments.png
        )
    else:
        render_current_group_history(
            summary, gmtnet_summary=gmtnet, svg_path=arguments.svg, png_path=arguments.png
        )


if __name__ == "__main__":
    main()
