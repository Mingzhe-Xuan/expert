"""One instrumented training pass plus uninstrumented synchronized timings."""

from contextlib import contextmanager
import json
from pathlib import Path
import time
import sys

import torch
import e3nn
from torch.profiler import ProfilerActivity, profile, record_function

from ..training.global_experts.runner import collate, masked_huber


@contextmanager
def module_ranges(model):
    targets = [(model, "encode_nodes", "stage/GMTNet_encoder"),
               (type(model.expert_irreps), "D_from_matrix", "stage/frame_representation")]
    for name, module in model.named_modules():
        if name in {"input_map", "output_map", "adapter", "router",
                    "global_model.output_block"} or name.startswith("experts."):
            targets.append((module, "forward", "module/" + name))
    originals = []
    try:
        for module, method, label in targets:
            original = getattr(module, method)
            originals.append((module, method, module.__dict__.get(method), method in module.__dict__))
            def wrapped(*args, _original=original, _label=label, **kwargs):
                with record_function(_label):
                    return _original(*args, **kwargs)
            setattr(module, method, wrapped)
        yield
    finally:
        for module, method, original, existed in reversed(originals):
            if existed:
                setattr(module, method, original)
            else:
                delattr(module, method)


def profile_global_experts(model, splits, *, output_dir, provenance, config,
                           device="cpu", warmup=2, repeats=3):
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("use a new empty profiler directory")
    output_dir.mkdir(parents=True, exist_ok=True)
    cuda = torch.device(device).type == "cuda"
    model.to(device).train()
    rows = splits["train"][:config.batch_size]
    data, mask, equality, routing, target, valid = collate(rows, device)

    def sync():
        if cuda:
            torch.cuda.synchronize(device)

    def step():
        model.zero_grad(set_to_none=True)
        sync()
        started = time.perf_counter()
        with record_function("pass/forward"):
            prediction = model(data, mask, equality, routing)
        with record_function("pass/loss"):
            loss = masked_huber(prediction, target, valid, delta=config.huber_delta)
        sync()
        middle = time.perf_counter()
        with record_function("pass/backward"):
            loss.backward()
        sync()
        ended = time.perf_counter()
        return loss, {"forward_loss_ms": (middle-started)*1000,
                      "backward_ms": (ended-middle)*1000,
                      "total_ms": (ended-started)*1000}

    for _ in range(warmup):
        step()
    timings = [step()[1] for _ in range(repeats)]
    model.zero_grad(set_to_none=True)
    if cuda:
        torch.cuda.reset_peak_memory_stats(device)
    activities = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if cuda else [])
    with module_ranges(model), profile(activities=activities, record_shapes=True,
                                      profile_memory=True) as prof:
        loss, instrumented_timing = step()
    gradients = {name: p.grad for name, p in model.named_parameters() if p.grad is not None}
    if not gradients or not all(torch.isfinite(g).all() for g in gradients.values()):
        raise ValueError("missing or nonfinite gradients")
    prof.export_chrome_trace(str(output_dir / "trace.json"))
    events = prof.key_averages()
    records = [{"name": e.key, "calls": e.count,
                "cpu_total_ms": e.cpu_time_total/1000,
                "cpu_self_ms": e.self_cpu_time_total/1000,
                "device_total_ms": getattr(e, "device_time_total", 0)/1000,
                "device_self_ms": getattr(e, "self_device_time_total", 0)/1000,
                "cpu_memory_bytes": e.cpu_memory_usage,
                "device_memory_bytes": getattr(e, "device_memory_usage", 0)} for e in events]
    for sort in ("self_cpu_time_total", "self_device_time_total"):
        (output_dir / (sort + ".txt")).write_text(
            events.table(sort_by=sort, row_limit=50), encoding="utf-8")
    counts = {}
    for row in routing:
        for pg in model.class_dag.ancestors(row.dag.current_point_group_number):
            counts[str(pg)] = counts.get(str(pg), 0) + 1
    report = {
        "status": "passed", "device": str(device), "torch_version": str(torch.__version__),
        "runtime": {"python": sys.executable, "torch_path": torch.__file__,
                    "e3nn_path": e3nn.__file__, "e3nn_version": e3nn.__version__},
        "device_name": torch.cuda.get_device_name(device) if cuda else "CPU",
        "model_metadata": model.metadata(), "provenance": provenance,
        "batch": {"samples": len(rows), "nodes": len(data.x), "edges": data.edge_index.shape[1],
                  "sample_ids": [r["sample_id"] for r in rows],
                  "point_groups": [r.dag.current_point_group_number for r in routing],
                  "expert_crystal_counts": counts},
        "warmup": warmup, "timings": timings, "instrumented_timing": instrumented_timing,
        "loss": float(loss.detach()), "parameters_with_grad": len(gradients),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if cuda else None,
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device) if cuda else None,
        "events": records,
        "notes": "Frozen DPA cache loading/model construction excluded; train mode, no optimizer step. "
                 "Module ranges are forward-only and nested: inclusive totals must not be summed. "
                 "Uninstrumented timings are authoritative; profiler adds overhead.",
    }
    (output_dir / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
