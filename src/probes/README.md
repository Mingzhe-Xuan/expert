# Frozen point-group probe

## Four-model softmax probe

`slurm/probe_softmax_pg.sbatch` runs `src.probes.softmax_pg`. Reuses SHA-verified
PCA536,DPA-GMTNet537 and Global+PG535 feature caches; extracts original GMTNet443
with read-only hooks and strict reproduction of677 accepted dielectric predictions.
All primary vectors are32D before explicit mask. GMTNet443 epoch93 is the historical
baseline exception, not a new selection satisfying the later >100 checkpoint rule;
DPA-GMTNet498 and Global+PG528 are epoch196. Every split is ID-aligned to the frozen
manifest; same source PG labels and same train-only standardization for all models.

`src/probes/logistic.py` minimizes mean multiclass cross entropy plus
`0.5 * alpha * ||W||^2` (unpenalized intercept), no class weighting. Seven fixed
alphas1e-6..1, selected by validation macro-F1, then accuracy, then larger alpha.
Exact analytic float64 gradients and SciPy L-BFGS-B, up to5000 iterations; every
candidate must report success and final max-absolute gradient <=1e-6. No hidden
layer and no backbone updates. Test predictions/probabilities produced only for the
selected classifier; softmax output does not imply calibrated probabilities.
Each model also has a seed42 shuffled-training-label control. All groups use the
same sorted IDs/permutation. Saved classifiers include train mean/scale/keep mask,
weights/bias; summary includes convergence for every alpha and all test predictions.
Regularization grids have the same numeric values as ridge but different losses,
so selected alpha values are not directly comparable between classifier families.

## Standalone DPA-GMTNet probe

Job537 completed on2026-10-01, using independent DPA-GMTNet498 best196 (200epochs).
Same frozen train/validation/test5001/637/677 and source-PG ridge protocol.

| Representation | Dimensions (variable) | Train accuracy | Validation accuracy | Test accuracy | Test macro-F1 |
|---|---:|---:|---:|---:|---:|
| **Independent DPA-GMTNet, before mask** |32 (32)|27.83%|25.59%|**23.63%**|**13.83%**|
| Independent DPA-GMTNet, after mask |32 (29)|29.85%|30.46%|26.29%|16.52%|
| Before mask, shuffled train labels |32 (32)|18.88%*|25.90%|22.60%|11.45%|
| DPA PCA32 (536) |32 (32)|44.33%|42.07%|45.20%|44.61%|
| Global branch of Global+PG (535) |32 (32)|24.98%|25.12%|21.57%|12.02%|
| Global+PG fusion (535), before mask |32 (32)|27.27%|27.47%|25.85%|16.87%|
| Projected PG branch (535), before mask |32 (16)|36.25%|36.11%|36.19%|27.89%|

*Against shuffled labels. All three537 probes selected alpha0.001 via validation
macro-F1. Standalone DPA-GMTNet is only1.03 percentage points above its shuffled
control in test accuracy; this is weak evidence of PG linear separability, not a
statistically established improvement. It is2.22 points below Global+PG fusion and
21.57 below DPA PCA32. The majority-class bias remains (512/677 predictions are mmm).
This does NOT establish that GMTNet destroys all PG information or has worse dielectric
prediction. Learned compression, nonlinear encoding, scalarization before message
passing, and task-specific training differ from raw DPA pooling/PCA; their individual
effects have not been isolated. In particular, pooling node-wise irrep norms (GMTNet
input processing) is not the same operation as norms after pooling used in534.

Job537: COMPLETED/0:0,node221,21:51:27--21:52:12,45s wall /34.49s executor,
sourcecd4dc03. Four preflight tests pass. Strict checkpoint/config/input metadata,
split IDs and hashes verified; original677 dielectric predictions reproduce within
atol2e-5/rtol2e-4 (max absolute difference4.58e-5). No optimizer/model updates.
Independent NumPy audit matches all probe metrics/alpha choices and exact536 test labels.
Summary `results/gmtnet-pg-probe/537/summary.json`, SHA256
`fc07fecae1406aa441dc31255fef6e162b8ba63982fd7162c98f34d2f77bf31f`.
Frozen pooled features saved in the same directory as `features.pt`.

```bash
python good_result/report_pca_probe.py results/pca-pg-probe/536/summary.json --sha256 ba63673016aec2fc72b27871f962c7062da8d4d442d5a6fb52d450777fbfc2dd --gmtnet-summary results/gmtnet-pg-probe/537/summary.json --gmtnet-sha256 fc07fecae1406aa441dc31255fef6e162b8ba63982fd7162c98f34d2f77bf31f
```

`slurm/probe_gmtnet_pg.sbatch` / `src.probes.gmtnet_pg` freezes Job498 best196,
the independent200-epoch DPA-GMTNet (no optional equivariant attention). Reuses its
exact graph cache and DPA node scalarization. Read-only hooks capture equi_update
node outputs (then crystal mean pooling) and actual output_block input after mask.
Primary probe is before mask,32D; controls are post-mask32D and shuffled labels.
No forward replacement or optimizer; strict checkpoint/source identity and original
677 dielectric predictions must reproduce before any probe is accepted. Same ridge
protocol and source PG labels as534/535/536. PG-informed preprocessing/masks used in
training still limit claims of independent symmetry discovery.

## Dimension-matched DPA PCA probe

Job536 completed successfully on2026-10-01. The first32 principal components retain
**56.996% of training variance**. All comparisons below use the same6315 structures,
train/validation/test5001/637/677, source PG labels and validation-selected ridge.
Global/fused/PG rows reuse audited Job535, frozen528 best196; all before explicit mask.

| Representation | Nominal dimension | Train accuracy | Validation accuracy | Test accuracy | Test macro-F1 |
|---|---:|---:|---:|---:|---:|
| **DPA mean pooling + PCA32** |32|44.33%|42.07%|**45.20%**|**44.61%**|
| Global branch |32|24.98%|25.12%|21.57%|12.02%|
| Global + PG fusion |32|27.27%|27.47%|25.85%|16.87%|
| Projected PG branch |32 (16 variable)|36.25%|36.11%|36.19%|27.89%|
| PCA32, shuffled training labels |32|20.70%*|20.25%|22.30%|16.47%|
| DPA raw mean (reference, Job534) |3200|92.66%|63.27%|63.66%|64.25%|

*Shuffled training accuracy uses shuffled labels. PCA32 alpha0.01 was chosen solely
by validation macro-F1; no test-based PCA dimension or regularization selection.
PCA32 exceeds global/fused/PG test accuracy by23.63/19.35/9.01 percentage points.
Thus the observed weaker final-branch linear decoding cannot be attributed solely to
nominal dimensionality. PCA32 loses18.46 points versus uncompressed DPA, but also
has no large train/test gap. PCA maximizes total feature variance, not PG information;
56.996% explained variance is NOT the fraction of PG information retained.
These are descriptive single-split results, not causal attribution or regression rankings.
Global/PG have task-trained representations and PG-informed routing; DPA PCA does not.
Canonical-frame and dataset correlations remain possible for all these comparisons.

Job536: COMPLETED/0:0, node221,21:27:29--21:27:52 (23s wall,13.86s probe), source67ca5eb.
Eight preflight tests passed; reused dpa4-py310 without changes. All192 shards and6315
unique IDs verified; independent NumPy audit agrees on both new probes' predictions,
metrics, validation selection and exact test ID-label mapping versus535.
Result `results/pca-pg-probe/536/summary.json`, SHA256
`ba63673016aec2fc72b27871f962c7062da8d4d442d5a6fb52d450777fbfc2dd`.
Saved fitted basis/mean and split projections: `results/pca-pg-probe/536/pca_features.pt`.

```bash
python good_result/report_pca_probe.py results/pca-pg-probe/536/summary.json --sha256 ba63673016aec2fc72b27871f962c7062da8d4d442d5a6fb52d450777fbfc2dd
```

`slurm/probe_pca_pg.sbatch` runs `src.probes.pg_linear --pca-dimension 32` in the
existing dpa4-py310 environment. `src/probes/pca.py` fits exact float64 covariance
PCA using ONLY5001 training mean-pooled raw DPA vectors (3200D): subtract training
mean, no input variance scaling or whitening, retain top32 principal directions.
Validation/test only transform using this frozen basis. Subsequent train-only
standardization and ridge selection are identical to previous probes. No PG labels
enter PCA; it remains an affine linear map of raw mean-pooled DPA features.
Outputs include variance ratios, basis/mean and projected split matrices in ignored
`pca_features.pt`, metrics/predictions in `summary.json`, plus shuffled-label control.
Comparison uses Job535 global32D/fused32D/projected PG32D (16 nonconstant coordinates).
This matches nominal dimension, not irrep content, feature rank or training objective;
global and PG here are branches of528, not separately trained model checkpoints.

## Global + PG final fusion result (Job 535)

Frozen Job528 best196 checkpoint; reduced dielectric splits5001/637/677, seven
source-PG classes, same train-only standardized ridge protocol as Job534 below.
All rows except the norm control are affine linear probes of the indicated vector.

| Input | Raw / retained dimensions | Alpha | Train accuracy | Validation accuracy | Test accuracy | Test macro-F1 |
|---|---:|---:|---:|---:|---:|---:|
| Same-checkpoint global, before mask |32 /32|0.01|24.98%|25.12%|21.57%|12.02%|
| Projected PG auxiliary, before mask |32 /16|0.01|36.25%|36.11%|36.19%|27.89%|
| **Final fused, before mask** |32 /32|0.0001|27.27%|27.47%|**25.85%**|**16.87%**|
| Final fused, after explicit mask |32 /29|0.001|31.79%|33.28%|28.80%|20.19%|
| Fused copy norms (nonlinear control) |8 /8|0.0001|28.29%|28.57%|28.21%|19.46%|
| Fused, shuffled training labels |32 /32|0.001|19.68%*|23.86%|20.38%|10.06%|
| Training-majority class (mmm) |N/A|N/A|N/A|N/A|17.87%|4.33%|

*Shuffled-control train accuracy is against shuffled labels. Constant coordinates are
removed using training data only; retained dimension is not a learned rank estimate.

Fusion improves accuracy by4.28 percentage points over the same-checkpoint global
branch, but absolute linear decodability is weak. It is below the isolated projected
PG branch and the earlier raw-DPA mean probe (63.66%,3200D). Train accuracy27.27%
versus test25.85% does not resemble the earlier raw-DPA probe's large train/test gap.
This does not prove absence of PG information: nonlinear encoding, dimensional reduction,
and task-specific regression training are possible explanations, not isolated causes.
The classifier predicts mmm for514/677 test structures; recalls for4/mmm and m-3m
are zero. Macro-F1 therefore matters alongside accuracy. This is one exploratory split,
not a statistically established improvement or a dielectric-performance comparison.

The learned fusion coefficient is0.0246711 in
`fused = global + coefficient * auxiliary`. A small coefficient alone does not measure
the branch contribution: feature scales and their joint statistics also matter.
PG routing explicitly uses current PG, so even successful probing would be
symmetry-conditioned decodability, not independent discovery of symmetry.
The global row is from the same jointly trained528 model, not standalone DPA-GMTNet.

Job535: COMPLETED/0:0, node221, 2026-10-01 20:55:56--21:02:55; wall6m59s,
extraction/probe executor31.61s (wall also includes loading/construction), source5f6ed79.
Four preflight tests passed. Dataset/split/checkpoint hashes, metadata, all6315 IDs,
and unchanged checkpoint verified. All677 dielectric test predictions reproduce the
accepted528 outputs within atol2e-5/rtol2e-4 (max absolute difference4.58e-5).
An independent NumPy audit reproduces all six confusion matrices, accuracy/macro-F1,
validation-only alpha choices, and exact test ID-to-label mapping versus Job534.

Result: `results/fused-pg-probe/535/summary.json`, SHA256
`64b110cd63c84e7c0f86808cde3df72b0f0ae9b9c030db00fe77b74b1231cdae`.
Pooled matrices and IDs: `results/fused-pg-probe/535/features.pt` (ignored generated artifact).
Reproduce the audit/figure with:

```bash
python good_result/report_pg_probe.py --fused-summary results/fused-pg-probe/535/summary.json --sha256 64b110cd63c84e7c0f86808cde3df72b0f0ae9b9c030db00fe77b74b1231cdae
```

![Global versus fused test confusion matrices](../../good_result/fused_pg_probe_confusion.png)

## Implementation

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
