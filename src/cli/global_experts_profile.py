"""Profile the exact global-experts training model on a real prepared batch."""
from .global_experts_train import parser, run
from ..profiling.global_experts import profile_global_experts


def main():
    arguments = parser().parse_args()
    if arguments.prepare_only:
        raise ValueError("profiling requires a model; omit --prepare-only")
    report = run(arguments, executor=profile_global_experts)
    print({key: report[key] for key in ("status", "batch", "timings", "peak_allocated_bytes")}, flush=True)


if __name__ == "__main__":
    main()
