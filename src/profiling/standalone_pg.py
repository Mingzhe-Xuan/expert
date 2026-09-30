"""Paired complete pure-PG passes on cached native DPA graphs, without GMTNet."""
import copy
import json
from pathlib import Path
import statistics
import sys
import shutil
import tempfile
import time

import e3nn
import torch
from torch.profiler import ProfilerActivity, profile, record_function

from ..training.benchmark import CachedBackboneTensorModel, collate_frozen_examples, _predict
from ..experts.chain_routing import migrate_legacy_edge_scales


def _export_trace(prof, destination):
    # Kineto's Windows file API cannot open a non-ASCII absolute filename.
    if sys.platform == "win32" and not str(destination.resolve()).isascii():
        with tempfile.TemporaryDirectory(prefix="pg-profile-") as temporary:
            trace = Path(temporary) / "trace.json"
            prof.export_chrome_trace(str(trace))
            shutil.copyfile(trace, destination)
    else:
        prof.export_chrome_trace(str(destination))
    if not destination.is_file() or destination.stat().st_size == 0:
        raise RuntimeError("profiler did not export a non-empty trace")


def compare_standalone_pg(source_layout, architecture, rows, *, expert_point_groups,
                          edge_ids, output_dir, provenance, device="cuda", warmup=6, repeats=7):
    root = Path(output_dir)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError("use a fresh comparison directory")
    root.mkdir(parents=True, exist_ok=True)
    if warmup < 1 or repeats < 1:
        raise ValueError("warmup and repeats must be positive")
    torch.manual_seed(42)
    base = CachedBackboneTensorModel(source_layout, architecture, "dielectric", expert_point_groups,
        edge_ids, grouped_pg_gates=False, vectorized_pg_routing=False)
    state = copy.deepcopy(base.state_dict())
    del base
    batch = collate_frozen_examples(rows, source_layout, device=device)
    cuda = torch.device(device).type == "cuda"
    expected_output = expected_gradients = None
    results = {}
    for name, gates, routing, weighting in (("legacy", False, False, "legacy"),
            ("reference", False, False, "within_cross_chain"),
            ("grouped_gates", True, False, "within_cross_chain"),
            ("optimized", True, True, "within_cross_chain")):
        model = CachedBackboneTensorModel(source_layout, architecture, "dielectric", expert_point_groups,
            edge_ids, grouped_pg_gates=gates, vectorized_pg_routing=routing,
            pg_weighting=weighting).to(device).train()
        variant_state = (state if weighting == "legacy" else
            migrate_legacy_edge_scales(state, model.downstream.edge_gate, prefix="downstream.edge_gate."))
        model.load_state_dict(variant_state, strict=True)
        torch.manual_seed(42)

        def sync():
            if cuda:
                torch.cuda.synchronize(device)

        def step():
            model.zero_grad(set_to_none=True)
            sync()
            start = time.perf_counter()
            with record_function("pass/forward_loss"):
                prediction = _predict(model, batch).raw_cartesian
                loss = torch.nn.functional.huber_loss(prediction, batch.target_cartesian)
            sync()
            middle = time.perf_counter()
            with record_function("pass/backward"):
                loss.backward()
            sync()
            end = time.perf_counter()
            return prediction, loss, {"forward_loss_ms": (middle-start)*1000,
                                     "backward_ms": (end-middle)*1000, "total_ms": (end-start)*1000}

        output, loss, _ = step()
        gradients = {n: p.grad.detach().cpu().clone() for n, p in model.named_parameters() if p.grad is not None}
        if not gradients or not torch.isfinite(output).all() or not all(torch.isfinite(g).all() for g in gradients.values()):
            raise ValueError("nonfinite output or missing/nonfinite gradients")
        if weighting == "legacy":
            pass  # Algorithm changed: compare timings, never assert legacy/new prediction equality.
        elif expected_output is None:
            expected_output, expected_gradients = output.detach().cpu().clone(), gradients
        else:
            torch.testing.assert_close(output.detach().cpu(), expected_output, atol=2e-5, rtol=2e-4)
            if gradients.keys() != expected_gradients.keys():
                raise ValueError("gradient ownership differs")
            for key in gradients:
                torch.testing.assert_close(gradients[key], expected_gradients[key], atol=2e-4, rtol=2e-3,
                                           msg=lambda message: key + ": " + message)
        del output, loss, gradients
        model.load_state_dict(variant_state, strict=True)
        torch.manual_seed(42)
        for _ in range(warmup):
            step()
        timings = [step()[2] for _ in range(repeats)]
        model.zero_grad(set_to_none=True)
        if cuda:
            torch.cuda.reset_peak_memory_stats(device)
        activities = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if cuda else [])
        # Separate stage annotations without changing normal model/training interfaces.
        originals = []
        for module_name, module in model.named_modules():
            if module_name in {"interface", "downstream.adaptation", "downstream.readout"} or module_name.startswith("downstream.pg_experts."):
                original = module.forward
                originals.append((module, original))
                def wrapped(*args, _original=original, _name=module_name, **kwargs):
                    with record_function("module/" + _name):
                        return _original(*args, **kwargs)
                module.forward = wrapped
        try:
            with profile(activities=activities, record_shapes=True, profile_memory=True) as prof:
                output, loss, instrumented = step()
        finally:
            for module, original in originals:
                module.forward = original
        directory = root / name
        directory.mkdir()
        _export_trace(prof, directory / "trace.json")
        events = prof.key_averages()
        records = [{"name": e.key, "calls": e.count, "cpu_total_ms": e.cpu_time_total/1000,
                    "cpu_self_ms": e.self_cpu_time_total/1000,
                    "device_total_ms": getattr(e, "device_time_total", 0)/1000,
                    "device_self_ms": getattr(e, "self_device_time_total", 0)/1000} for e in events]
        report = {
            "status": "passed", "parity": "legacy_baseline" if weighting == "legacy" else "passed",
            "variant": name, "weighting": weighting, "provenance": provenance,
            "runtime": {"python": sys.executable, "torch": str(torch.__version__),
                        "torch_path": torch.__file__, "e3nn": e3nn.__version__, "e3nn_path": e3nn.__file__},
            "device_name": torch.cuda.get_device_name(device) if cuda else "CPU",
            "batch": {"samples": len(rows), "nodes": batch.graph.num_nodes,
                      "edges": batch.graph.edge_index.shape[1], "sample_ids": list(batch.sample_ids),
                      "source_dimension": source_layout.dimension},
            "dispatch": model.downstream.last_dispatch_stats,
            "warmup": warmup, "repeats": repeats, "timings": timings,
            "median_ms": {key: statistics.median(t[key] for t in timings) for key in timings[0]},
            "instrumented_timing": instrumented, "loss": float(loss.detach()),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if cuda else None,
            "cuda_kernel_calls": sum(e.count for e in events if str(e.device_type) == "DeviceType.CUDA"
                and not e.is_user_annotation and not e.key.startswith(("Memcpy", "Memset"))),
            "events": records,
        }
        if cuda and (report["dispatch"]["expert_buckets"] < 2 or not report["dispatch"]["asynchronous_cuda"]
                     or report["dispatch"]["cuda_streams"] != report["dispatch"]["expert_buckets"]):
            raise ValueError("expected multi-stream expert dispatch")
        (directory / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        results[name] = {key: value for key, value in report.items() if key != "events"}
        print(json.dumps(results[name]), flush=True)
        del model, output, loss, prof, events, originals
        if cuda:
            torch.cuda.empty_cache()
    summary = {"status": "passed", "variants": results, "warmup": warmup, "repeats": repeats,
               "notes": "Native frozen DPA O(3) input -> interface -> Adapter -> full_pg -> readout. "
                        "Huber Cartesian training loss; no optimizer updates. Data preparation excluded. "
                        "Same non-routing state/positive sigmas/batch/RNG. New-weighting variants "
                        "pass output/parameter-gradient parity; legacy algorithm is a separate baseline."}
    (root / "comparison.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
