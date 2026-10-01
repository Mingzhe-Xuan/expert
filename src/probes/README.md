# Frozen point-group probe

`slurm/probe_fused_pg.sbatch` probes the frozen Global+PG528 best196 checkpoint using
existing diagnostics. Primary tap is32D crystal-level `global + sigmoid(logit)*auxiliary`,
BEFORE explicit feature mask; no additional node pooling needed. Same ridge protocol
as DPA probe. Controls: same-checkpoint global32D, projected PG auxiliary32D, fused
copy-norm8D (nonlinear), post-mask32D (explicit symmetry control), shuffled fused labels.
PG routing already uses the current PG, so this measures symmetry-conditioned decodability,
not independently learned point-group discovery. It also cannot isolate a training effect
versus a separately trained DPA-GMTNet. Extraction reproduces accepted test predictions,
checks model/checkpoint metadata and SHA, never updates backbone weights; saves small
pooled feature matrices for reuse under ignored results/fused-pg-probe/JOB.

Run `python -m src.probes.pg_linear --output results/pg-probe/RUN --device cuda`
inside Slurm using the existing DPA4 environment. Reads the 192 frozen reduced-total
DPA feature shards and the SHA-verified reduced manifest/JSONL. Never runs DPA or
uses dielectric targets, symmetry masks, PG experts or PG metadata as model inputs.

Primary labels are the seven source point groups used to define the reduced dataset.
Cached current-PG agreement is reported separately; it does not silently change labels.
Inputs: arithmetic mean over atoms of all 3200 feature coordinates; controls are
mean even scalar channels, irrep-copy norms AFTER pooling (nonlinear invariant
descriptor, not a strictly linear probe of the original representation), atomic
fractions, and training-label permutation. The cache was computed in canonical frames:
raw-coordinate success does not establish orientation-independent symmetry recognition.

Classifier: affine multiclass ridge least squares on one-hot labels, argmax scores;
this is a linear probe, not a neural MLP or logistic regression. Train-only centering
and standardization, constant coordinates discarded, intercept unpenalized. Objective:
mean squared one-hot error + alpha * squared weight norm. Seven predeclared alphas
1e-6..1, selected on validation macro-F1 (then accuracy, then larger alpha).
Test labels never enter fitting or model selection. No train+validation refit.
Output: per-alpha train/validation metrics, selected held-out metrics, per-class recall,
confusion matrices, predictions, split counts, runtime and provenance. Single split/seed
is exploratory; composition/canonicalization and DPA pretraining overlap limit causal claims.

## Reduced dielectric result

Job534 completed with Slurm COMPLETED/0:0 on 2026-10-01, source4a0adae,
wall23 seconds (probe14.28s, preflight3 tests passed). Frozen splits5001/637/677.
All192 cache shards and exactly6315 distinct IDs passed membership/identity checks.
Existing dpa4-py310 environment reused unchanged (Torch2.11.0+cu128, NumPy1.26.4).

| Input | Dimension | Alpha | Train accuracy | Validation accuracy | Test accuracy | Test macro-F1 |
|---|---:|---:|---:|---:|---:|---:|
| DPA raw mean pooling | 3200 | 0.1 | 92.66% | 63.27% | **63.66%** | **64.25%** |
| Mean even scalars only | 64 | 0.001 | 42.99% | 38.15% | 40.32% | 39.80% |
| Norm per copy AFTER pooling | 640 | 1 | 82.76% | 82.10% | **80.65%** | **81.67%** |
| Atomic fractions | 118 (85 variable) | 1e-5 | 36.57% | 29.98% | 31.17% | 30.08% |
| Raw mean with shuffled train labels | 3200 | 1 | 64.13%* | 15.86% | 16.10% | 13.45% |
| Training-majority class (mmm) | — | — | — | — | 17.87% | 4.33% |

*Shuffled-control training accuracy is against shuffled labels, not original labels.
Alpha is selected by validation macro-F1, not accuracy; the displayed validation accuracy
belongs to that selection. Norm-control alpha1 is the search boundary, so it is not an
exhaustively optimized classifier. No test-set-driven hyperparameter changes were made.

### Interpretation

The mean-pooled cached DPA representation contains linearly decodable point-group signal:
raw test accuracy63.66% substantially exceeds both composition31.17% and shuffled16.10%.
This is evidence of decodability on the frozen reduced split, not proof that DPA learns
an exact abstract symmetry algorithm. Raw3200D has a large train/test gap (92.66%/63.66%).
Even scalars alone carry weaker information. Norms of pooled equivariant copies expose
stronger signal, but that preprocessing is nonlinear:80.65% is NOT a strictly linear
probe of the original3200D vector. It is also not mean-pooling node-wise norms; norm and
mean do not commute. These results do not establish improved dielectric regression.

| Source PG | Test support | Raw mean recall | Pooled-norm recall |
|---|---:|---:|---:|
| 2/m |128|49.22%|81.25%|
| mm2 |98|43.88%|90.82%|
| mmm |121|59.50%|81.82%|
| 4/mmm |76|68.42%|55.26%|
| -3m |116|78.45%|69.83%|
| -43m |48|79.17%|89.58%|
| m-3m |90|80.00%|97.78%|

Labels use the source PG underlying dataset selection. Cache re-identification agrees
for4999/5001 training structures and all validation/test structures. No labels were
silently replaced. Inputs exclude PG labels, group operations, target tensors and router
weights. Nevertheless canonical-frame preprocessing already uses structure symmetry;
raw-coordinate decodability may depend on this convention. Norm/scalar controls reduce
orientation sensitivity but do not eliminate composition or dataset-source correlations.
Pretraining overlap was not audited; only one frozen split and one shuffled control were
used. Next diagnostic would be orientation perturbation / composition-disjoint evaluation,
not an unsupported claim of universal PG classification.

![Test confusion matrices](../../good_result/pg_probe_confusion.png)

Result: `results/pg-probe/534/summary.json`, SHA256
`51244f7fb8ed1d7ed2b4a8feb8c11d360bc8fcd0aa562e90c531c246a550a86f`.
Per-alpha scores, selected predictions and all confusion matrices are in that file.
Independent NumPy audit and figure: `python good_result/report_pg_probe.py`.
