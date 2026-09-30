"""Compare global-experts algorithm-equivalent performance paths."""
from .global_experts_train import parser, run
from ..profiling.compare_optimized import compare_optimized


def select_profile_splits(dataset, selected, count):
    if count < len(selected["train"]) or count > len(dataset.split_manifest.train):
        raise ValueError("train-samples must include the selected representatives and fit the train split")
    ordered = tuple(dict.fromkeys((*selected["train"], *dataset.split_manifest.train)))
    return {**selected, "train": ordered[:count]}


def main():
    argument_parser = parser()
    argument_parser.add_argument("--train-samples", type=int)
    arguments = argument_parser.parse_args()
    if arguments.prepare_only:
        raise ValueError("comparison requires a model")
    selector = (None if arguments.train_samples is None else
                lambda dataset, selected: select_profile_splits(dataset, selected, arguments.train_samples))
    run(arguments, executor=compare_optimized, split_selector=selector)


if __name__ == "__main__":
    main()
