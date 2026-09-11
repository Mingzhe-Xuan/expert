# Evaluation

Evaluation reports per-irrep physical metrics, graph and tensor equivariance, parameter
counts, FLOPs, latency, and peak memory. Reports are emitted as machine-readable data and
Markdown with architecture, task, PG, mode, and backend keys.

`profile_model_efficiency` records registered plus externally owned frozen parameters,
trainable/active parameters, per-sample routed expert counts, one synchronized end-to-end
latency, and incremental CUDA peak allocation. FLOPs are explicitly scoped to Torch-dispatched
active-downstream operators on cached backbone features, so GRACE's TensorFlow execution is not
silently mislabeled as observed Torch FLOPs.

```python
report = evaluate_equivariance(model, batch, transforms)
```

`select_point_group_fixtures` deterministically chooses one checksum-traceable equilibrium
structure for each of the 32 crystallographic point groups. The final manifest is generated
from frozen JARVIS/MatTen resources through Slurm and re-detects every exact Hall setting;
prototype synthetic Wyckoff-orbit structures are not accepted as final fixtures.
