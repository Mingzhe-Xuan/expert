"""Evaluate an existing full-model checkpoint; no training or optimizer steps."""

from functools import partial
import json
from pathlib import Path

from .global_experts_train import parser, run
from ..training.global_experts.checkpoint_evaluation import evaluate_checkpoint
from ..training.global_experts.data import file_sha256
from ..training.global_experts.runner import normalized_training_config


def main():
    cli = parser()
    cli.description = __doc__
    cli.add_argument("--source-summary", type=Path, required=True)
    cli.add_argument("--source-summary-sha256", required=True)
    args = cli.parse_args()
    if args.prepare_only or args.smoke or args.global_checkpoint:
        cli.error("checkpoint evaluation requires full data and no initialization override")
    if file_sha256(args.source_summary) != args.source_summary_sha256:
        cli.error("source summary SHA mismatch")
    minimum_epoch = args.minimum_checkpoint_epoch_exclusive
    source = json.loads(args.source_summary.read_text(encoding="utf-8"))
    for key, value in normalized_training_config(source["training_config"]).items():
        setattr(args, key, value)
    report = run(args, executor=partial(evaluate_checkpoint,
        source_summary=args.source_summary, source_sha256=args.source_summary_sha256,
        minimum_epoch=minimum_epoch))
    print(json.dumps({k: report[k] for k in (
        "status", "best_epoch", "test_metrics", "evaluation_seconds")}), flush=True)


if __name__ == "__main__":
    main()
