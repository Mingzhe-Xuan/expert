from __future__ import annotations

import argparse
from pathlib import Path

from ..evaluation.training_history import (
    load_experiment_history,
    render_all_experiment_histories,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot all accepted experiment histories in one figure")
    parser.add_argument(
        "--summary",
        nargs=3,
        action="append",
        metavar=("LABEL", "PATH", "SHA256"),
        required=True,
    )
    parser.add_argument("--svg", type=Path, required=True)
    parser.add_argument("--png", type=Path, required=True)
    arguments = parser.parse_args()
    summaries = [
        (
            label,
            load_experiment_history(Path(path), expected_sha256=sha256),
        )
        for label, path, sha256 in arguments.summary
    ]
    render_all_experiment_histories(summaries, svg_path=arguments.svg, png_path=arguments.png)


if __name__ == "__main__":
    main()
