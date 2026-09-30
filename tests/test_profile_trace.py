from src.profiling.analyze_trace import analyze


def test_trace_attribution_excludes_gpu_annotations():
    def span(name, cat, ts, dur, external=0, tid=1):
        return dict(name=name, cat=cat, ts=ts, dur=dur, tid=tid, ph="X",
                    args={"External id": external})
    events = [span("module/experts.8", "user_annotation", 0, 10),
              span("aten::mul", "cpu_op", 1, 2, 1),
              span("kernel", "kernel", 2, 1, 1),
              span("pass/backward", "user_annotation", 20, 20),
              span("autograd::engine::evaluate_function: MulBackward0", "cpu_op", 21, 10, 2, 2),
              span("aten::mul", "cpu_op", 22, 2, 3, 2),
              span("kernel", "kernel", 24, 2, 3),
              span("module/experts.8", "gpu_user_annotation", 0, 1000),
              dict(cat="fwdbwd", ph="s", id=1, ts=1, tid=1),
              dict(cat="fwdbwd", ph="f", id=1, ts=21, tid=2)]
    result = analyze(events)
    rows = result["kernel_attribution"]
    assert sum(r["kernels"] for r in rows) == 2
    assert sum(r["kernel_ms"] for r in rows) == .003
    assert all(r["owner"] == "module/experts.8" for r in rows)
    assert result["backward_engine_cpu_ms"]["module/experts.8"] == .01
