# Evaluation

Evaluation reports per-irrep physical metrics, graph and tensor equivariance, parameter
counts, FLOPs, latency, and peak memory. Reports are emitted as machine-readable data and
Markdown with architecture, task, PG, mode, and backend keys.

```python
report = evaluate_equivariance(model, batch, transforms)
```
