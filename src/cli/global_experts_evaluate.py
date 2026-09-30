"""Evaluate an existing full-model checkpoint; no training or optimizer steps."""

from functools import partial
import json
from pathlib import Path

from .global_experts_train import parser, run
from ..training.global_experts.checkpoint_evaluation import evaluate_checkpoint


def main():
    cli = parser()
    cli.description = __doc__
    cli.add_argument("--source-summary", type=Path, required=True)
    cli.add_argument("--source-summary-sha256", required=True)
    cli.add_argument("--minimum-checkpoint-epoch-exclusive", type=int, default=100)
    args = cli.parse_args()
    if args.prepare_only or args.smoke or args.global_checkpoint:
        cli.error("checkpoint evaluation requires full data and no initialization override")
    report = run(args, executor=partial(evaluate_checkpoint,
        source_summary=args.source_summary, source_sha256=args.source_summary_sha256,
        minimum_epoch=args.minimum_checkpoint_epoch_exclusive))
    print(json.dumps({k: report[k] for k in (
        "status", "best_epoch", "test_metrics", "evaluation_seconds")}), flush=True)


if __name__ == "__main__":
    main()
