# Evaluation

`training_history.py` validates an accepted CGCNN current-group-only summary (status, routing,
model identity, SHA-256, contiguous epochs, finite fields, and best-validation epoch) and renders a
deterministic SVG/PNG optimization history. The plot shows train/validation Huber losses,
validation component MAE/Fnorm, epoch-end learning rate, and the selected checkpoint without
recomputing any benchmark metric.
When an accepted GMTNet summary is supplied, the same figure overlays its recorded training Huber
loss and validation MAE, marks both selected checkpoints, and collapses identical learning-rate
schedules into one explicitly shared curve. GMTNet validation loss/Fnorm are omitted because its
runner did not record them.

Evaluation reports per-irrep physical metrics, graph and tensor equivariance, parameter
counts, FLOPs, latency, and peak memory. Reports are emitted as machine-readable data and
Markdown with architecture, task, PG, mode, and backend keys.

`profile_model_efficiency` records registered plus externally owned frozen parameters,
trainable/active parameters, per-sample routed expert counts, one synchronized end-to-end
latency, and incremental CUDA peak allocation. FLOPs are explicitly scoped to Torch-dispatched
active-downstream operators on cached backbone features, so GRACE's TensorFlow execution is not
silently mislabeled as observed Torch FLOPs.

After all 58 point-group rows and 20 real-subset rows finish, aggregate their summaries
with `python -m src.cli.efficiency_report`. The command validates exact row-index coverage,
passed status, schemas, identities, numeric invariants, the `<5M` active budget, and the
declared FLOPs/latency scopes before publishing paired JSON and Markdown reports.

```python
report = evaluate_equivariance(model, batch, transforms)
```

`select_point_group_fixtures` deterministically chooses one checksum-traceable equilibrium
structure for each of the 32 crystallographic point groups. The final manifest is generated
from frozen JARVIS/MatTen resources through Slurm and re-detects every exact Hall setting;
prototype synthetic Wyckoff-orbit structures are not accepted as final fixtures.
