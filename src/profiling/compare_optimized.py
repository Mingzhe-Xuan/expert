"""Paired reference/optimized runs with equal state, data, RNG and warmup policy."""
import copy
from dataclasses import replace
import json
from pathlib import Path
import statistics

import torch

from ..models.global_experts import GlobalExpertsModel
from ..training.global_experts.runner import collate, masked_huber
from .global_experts import profile_global_experts


def compare_optimized(model, splits, *, output_dir, provenance, config, device="cuda"):
    root = Path(output_dir)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError("use a fresh comparison directory")
    root.mkdir(parents=True, exist_ok=True)
    state = copy.deepcopy(model.state_dict())
    variants = [("reference", False, False, False), ("grouped_gates", True, False, False),
                ("gates_and_frames", True, True, False), ("optimized", True, True, True)]
    results = {}
    expected_output, expected_gradients = None, None
    for name, gates, frames, router in variants:
        candidate = GlobalExpertsModel(copy.deepcopy(model.global_model), model.equality_adjustment,
            expert_numbers=model.expert_numbers, edge_ids=model.router.edge_ids,
            config=replace(model.config, grouped_pg_gates=gates, cache_frames=frames,
                           vectorized_routing=router), class_dag=model.class_dag)
        candidate.load_state_dict(state, strict=True)
        candidate.to(device).train()
        data, mask, equality, routing, target, valid = collate(splits["train"][:config.batch_size], device)
        torch.manual_seed(config.seed)
        prediction = candidate(data, mask, equality, routing)
        loss = masked_huber(prediction, target, valid, delta=config.huber_delta)
        loss.backward()
        gradients = {n: p.grad.detach().cpu().clone() for n, p in candidate.named_parameters() if p.grad is not None}
        output = prediction.detach().cpu()
        if expected_output is None:
            expected_output, expected_gradients = output.clone(), gradients
        else:
            torch.testing.assert_close(output, expected_output, atol=2e-5, rtol=2e-4)
            if gradients.keys() != expected_gradients.keys():
                raise ValueError("gradient ownership differs from reference")
            for key, gradient in gradients.items():
                torch.testing.assert_close(gradient, expected_gradients[key], atol=2e-4, rtol=2e-3,
                                           msg=lambda message: key + ": " + message)
        candidate.zero_grad(set_to_none=True)
        candidate.load_state_dict(state, strict=True)
        del prediction, loss, data, mask, equality, target, valid
        torch.manual_seed(config.seed)
        report = profile_global_experts(candidate, splits, output_dir=root / name,
            provenance=provenance, config=config, device=device, warmup=6, repeats=7)
        results[name] = {
            "median_ms": {key: statistics.median(t[key] for t in report["timings"])
                          for key in ("forward_loss_ms", "backward_ms", "total_ms")},
            "timings": report["timings"], "peak_allocated_bytes": report["peak_allocated_bytes"],
            "cuda_kernel_calls": report["cuda_kernel_calls"], "parity": "passed",
            "batch": report["batch"], "loss": report["loss"], "runtime": report["runtime"],
            "device_name": report["device_name"],
        }
        print(json.dumps({"variant": name, **results[name]}), flush=True)
        del candidate
        if torch.device(device).type == "cuda":
            torch.cuda.empty_cache()
    summary = {"status": "passed", "warmup": 6, "repeats": 7, "variants": results,
               "notes": "Same initial state, real batch, RNG, training mode. No optimizer updates. "
                        "Forward and all parameter-gradient parity checked before measurement."}
    (root / "comparison.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
