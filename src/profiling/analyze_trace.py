"""Attribute actual CUDA kernels using CPU ranges and profiler forward/backward flows."""
import argparse
from bisect import bisect_right
from collections import defaultdict
import json
from pathlib import Path
import re


def analyze(events):
    spans = [e for e in events if e.get("ph") == "X"]
    scopes = [e for e in spans if e.get("cat") == "user_annotation" and
              (e["name"].startswith("stage/") or re.fullmatch(r"module/experts\.\d+", e["name"])
               or e["name"] in {"module/router", "module/adapter", "module/input_map",
                                "module/output_map", "module/global_model.output_block"})]
    backward = next(e for e in spans if e["name"] == "pass/backward" and e.get("cat") == "user_annotation")

    def in_backward(ts):
        return backward["ts"] <= ts < backward["ts"] + backward["dur"]

    def scope_at(ts):
        matches = [e for e in scopes if e["ts"] <= ts < e["ts"] + e["dur"]]
        return min(matches, key=lambda e: e["dur"])["name"] if matches else "other"

    engines = defaultdict(list)
    for e in spans:
        if e["name"].startswith("autograd::engine::evaluate_function:"):
            engines[e["tid"]].append(e)
    for values in engines.values():
        values.sort(key=lambda e: e["ts"])
    starts = {tid: [e["ts"] for e in values] for tid, values in engines.items()}
    flow_sources = {e["id"]: scope_at(e["ts"]) for e in events
                    if e.get("cat") == "fwdbwd" and e["ph"] == "s"}
    engine_owner = {}
    for e in events:
        if e.get("cat") != "fwdbwd" or e["ph"] != "f" or e["tid"] not in engines:
            continue
        idx = bisect_right(starts[e["tid"]], e["ts"]) - 1
        if idx >= 0:
            node = engines[e["tid"]][idx]
            if e["ts"] < node["ts"] + node["dur"]:
                engine_owner[node["args"]["External id"]] = flow_sources.get(e["id"], "other")

    by_thread = defaultdict(list)
    for e in spans:
        if e.get("cat") in {"cpu_op", "cuda_runtime", "cuda_driver"}:
            by_thread[e["tid"]].append(e)
    owners = {}
    backward_cpu = defaultdict(float)
    for tid, values in by_thread.items():
        stack = []
        for e in sorted(values, key=lambda e: (e["ts"], -e["dur"])):
            while stack and stack[-1][0] <= e["ts"]:
                stack.pop()
            external = e.get("args", {}).get("External id")
            phase = "backward" if in_backward(e["ts"]) else "forward"
            if phase == "backward":
                owner = engine_owner.get(external, stack[-1][1] if stack else "other")
            else:
                owner = scope_at(e["ts"])
            if external is not None:
                owners[external] = (phase, owner)
            stack.append((e["ts"] + e["dur"], owner))
            if phase == "backward" and e["name"].startswith("autograd::engine::evaluate_function:"):
                backward_cpu[owner] += e["dur"] / 1000

    totals = defaultdict(lambda: {"kernels": 0, "kernel_ms": 0., "copy_ms": 0.})
    for e in spans:
        if e.get("cat") not in {"kernel", "gpu_memcpy", "gpu_memset"}:
            continue
        phase, owner = owners.get(e.get("args", {}).get("External id"), ("unknown", "other"))
        row = totals[(phase, owner)]
        if e["cat"] == "kernel":
            row["kernels"] += 1
            row["kernel_ms"] += e["dur"] / 1000
        else:
            row["copy_ms"] += e["dur"] / 1000
    return {"kernel_attribution": [{"phase": phase, "owner": owner, **values}
                                   for (phase, owner), values in sorted(totals.items())],
            "backward_engine_cpu_ms": dict(backward_cpu),
            "notes": "Actual kernel durations only; excludes GPU annotation spans. Backward ownership "
                     "uses profiler fwdbwd links to forward scopes; unmatched work remains other."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = analyze(json.loads(args.trace.read_text(encoding="utf-8"))["traceEvents"])
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
