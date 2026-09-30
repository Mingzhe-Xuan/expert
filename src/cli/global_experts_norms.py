"""Measure global versus lambda-scaled PG feature norms without updating weights."""

from functools import partial
import json
from pathlib import Path

from .global_experts_train import parser, run
from ..training.global_experts.data import file_sha256
from ..training.global_experts.runner import normalized_training_config
from ..training.global_experts.feature_norms import diagnose


def main():
    cli = parser()
    cli.add_argument("--source-summary", type=Path, required=True)
    cli.add_argument("--source-summary-sha256", required=True)
    cli.add_argument("--checkpoint-sha256", required=True)
    cli.add_argument("--predictions-sha256", required=True)
    args = cli.parse_args()
    if args.smoke or args.prepare_only or args.global_checkpoint:
        cli.error("norm diagnostics require full-data best checkpoint")
    if file_sha256(args.source_summary) != args.source_summary_sha256:
        cli.error("summary SHA mismatch")
    source = json.loads(args.source_summary.read_text(encoding="utf-8"))
    for key, value in normalized_training_config(source["training_config"]).items():
        setattr(args, key, value)
    report = run(args, executor=partial(diagnose, source_summary=args.source_summary,
        source_sha256=args.source_summary_sha256, checkpoint_sha256=args.checkpoint_sha256,
        predictions_sha256=args.predictions_sha256))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
