# Reduced dielectric-total: current-pg, parent-DAG variants, DPA4, and GMTNet

## Protocol

> Width note: the DPA4, CGCNN current-pg, and static all-ancestor full-PG runs predate the
> 2026-09-22 change from `[4, 1, 1, 1, 1]` to `[8, 2, 2, 2, 2]`. The relative-PG row is the first
> accepted result for the current 56-component configuration. Consequently, comparisons with the
> three historical full-PG rows measure the combined width-and-routing change, not a routing-only
> ablation.

- Dataset: `curated_reduced_total__dielectric`, 6,315 structures.
- Frozen splits: 5,001 train / 637 validation / 677 test.
- Dataset SHA-256: `6cfefc7c04734ceb4c7873e5aaa7d483251ea909a378e55034414d53f9110963`.
- Source-retained point groups: `2/m`, `mm2`, `mmm`, `4/mmm`, `-3m`, `-43m`, `m-3m`.
- DPA4 model (current-group only): `B+A+PGE+R`, `full_o3` adaptation/readout, `full_pg`
  experts; each sample activates only its detected current point-group expert.
- CGCNN full-PG model (current-group only): additive (non-DPA4) branch using GMTNet's fixed 92-component JARVIS CGCNN
  node descriptor, a learned `92 -> 128` even-scalar embedding, and the same
  `B+A+PGE+R/full_pg/full_o3` tensor architecture. Its expert registry is the deterministic union of
  canonical train/validation/test symmetries: `2/m`, `mm2`, `mmm`, `4/mmm`, `3m`, `-3m`, `6/mmm`,
  `-43m`, `m-3m`.
- CGCNN optimization follows GMTNet: Cartesian Huber loss (`delta=1`), AdamW, 200 full epochs,
  per-step linear learning-rate decay from `1e-3` to `1e-5`, and best-checkpoint selection by
  validation component MAE. The best checkpoint (epoch 159, validation MAE `4.3647098541`) was
  reloaded before test inference.
- CGCNN PG-parent-DAG uses the same feature cache, architecture, optimizer, split, seed, batch size,
  200-epoch schedule, and checkpoint rule. Its only intended model change is routing: each sample
  activates its current point-group expert plus every transitive parent in the frozen 32-group/
  80-cover-edge class DAG, with deduplicated equal-weight fusion. Test samples activate 1--11 experts
  (mean `5.2009`). Its best checkpoint is epoch 122 at validation MAE `4.3411664963`.
- CGCNN relative-PG parent-DAG uses the current 56-component hidden irreps and the same frozen data,
  optimizer, seed, batch size, 200-epoch schedule, and validation-MAE checkpoint rule. The offline DAG
  supplies every maximal current-PG-to-root path. Within each path, relative-position point-group
  edge residuals produce stick-breaking weights; path priors are normalized by node count, and
  duplicate destination PGs are summed then normalized. Test parent coverage is `86.71%`, with mean
  `4.2009` candidate parents and `3.6765` maximal paths per sample. Its best checkpoint is epoch 28 at
  validation MAE `4.3751330376`.
- GMTNet model: pinned official dielectric implementation.
- All rows use the same ordered 677 test IDs. The strict comparator additionally verifies symmetric
  target eigenvalue equivalence with `atol=2e-4`, `rtol=2e-5`.

## CGCNN current-group-only vs GMTNet training history

![CGCNN current-group-only and GMTNet 200-epoch training histories](cgcnn_current_group_only_training_curve.svg)

The curves come directly from the accepted 200-epoch summaries for CGCNN job 458 (SHA-256
`7070c4f66d57d5573b58d95a3219f780b0b776a662eab11426903544966fb536`) and GMTNet job 443
(SHA-256 `b4ded0b3696e87406fef4046b56685c0ae4d21b9bd5f9f3bd7cb1354abd9e6ae`). Dashed markers show
their validation-MAE-selected checkpoints at epochs 159 and 93. GMTNet's runner records training
Huber loss, validation MAE, and learning rate, but not validation loss or validation Fnorm; therefore
only its recorded series are overlaid. No metric was recomputed for this plot.

## CGCNN current-pg vs all-ancestor PG-DAG training history

![CGCNN current-pg and all-ancestor PG-DAG 200-epoch training histories](cgcnn_parent_dag_training_curve.svg)

This matched overlay uses accepted current-pg job 458 and parent-DAG job 472. Both summaries contain
200 contiguous epochs and the identical per-step learning-rate schedule. The parent-DAG run reaches
a slightly lower selected validation MAE (`4.3412` versus `4.3647`) and selects epoch 122 rather than
159. The figure shows only directly recorded train/validation Huber loss, validation MAE/Fnorm, and
learning rate; it does not derive or smooth any series. Source summary SHA-256 values are
`7070c4f66d57d5573b58d95a3219f780b0b776a662eab11426903544966fb536` and
`7c88bd2e7fd1eb9dd05f1dffe3f8f6cd05b5b73d06a218bf5d38f96fb9c37b50`.

## CGCNN current-pg vs relative-PG path-weighted training history

![CGCNN current-pg and relative-PG path-weighted 200-epoch training histories](cgcnn_relative_pg_56d_training_curve.svg)

This overlay uses accepted current-pg job 458 and 56D relative-PG job 478. The relative-PG run
selects epoch 28 at validation MAE `4.3751`; after that point its training objective continues to
decrease while validation loss, MAE, and Fnorm rise, so reloading the early best checkpoint is
material to the reported test result. The figure contains only recorded train/validation loss,
validation MAE/Fnorm, and learning-rate history. Job-478 summary and curve SVG/PNG SHA-256 values are
`3be671cb61a2b9c066046891ca5d98701cb8de200a88e5afd9b4043e3571debc`,
`d92632a5210bc4e6d7915943d5602d409ba2613ce5081ea2202474a1ed77f1a7`, and
`889937f8958092445790bf87e77bbc995fbdd3bba033f7bc6dbc59a5d18e069c`.

## Metrics

- RMSE: component-wise root mean squared error over the full test tensors.
- Fnorm: mean per-sample Frobenius norm of the tensor error.
- EwT x%: percentage of samples satisfying
  `Fnorm(prediction - target) / (Fnorm(target) + 1e-5) < x%`.

## Results

| Model | RMSE ↓ | Fnorm ↓ | EwT 25% ↑ | EwT 10% ↑ | EwT 5% ↑ |
|---|---:|---:|---:|---:|---:|
| DPA4 B+A+PGE+R full_pg (current-group only) | 26.166445 | 31.539129 | 12.26% | 2.51% | 1.03% |
| GMTNet | **25.449894** | **19.169209** | **53.03%** | **18.32%** | **7.39%** |
| CGCNN B+A+PGE+R full_pg (current-group only) | 26.186150 | 19.496103 | 40.77% | 10.64% | 4.28% |
| CGCNN B+A+PGE+R full_pg (PG parent-DAG all ancestors) | 26.144173 | 19.204304 | 41.51% | 11.52% | 4.87% |
| CGCNN B+A+PGE+R full_pg (relative-PG path-weighted, 56D) | 26.259335 | 19.407409 | 42.39% | 10.04% | 4.58% |

GMTNet is best on every reported test metric under this frozen protocol. Relative to the DPA4 row,
the additive CGCNN full-PG branch substantially improves Fnorm and every EwT threshold, while its
component RMSE is slightly higher (`+0.019705`). DPA4's best checkpoint was epoch 12
(`validation_loss=0.9333040631`); GMTNet's was epoch 93 (`validation_mae=4.1132789`); CGCNN's was
epoch 159 (`validation_mae=4.3647098541`) after completing all 200 epochs.

Relative to the matched-width historical CGCNN current-pg run, static all-ancestor routing improves
every metric: RMSE/Fnorm decrease by `0.041978`/`0.291799`, while EwT25/EwT10/EwT5 rise by
`0.74`/`0.89`/`0.59` percentage points. This remains the strongest CGCNN parent result on Fnorm,
EwT10, and EwT5, although it activates ancestors from class membership alone and does not establish
material-specific structural realizability.

The new 56D relative-PG result is mixed rather than uniformly better. Versus historical current-pg,
Fnorm improves by `0.088693`, EwT25 by `1.62` points, and EwT5 by `0.30` points, while RMSE worsens by
`0.073184` and EwT10 by `0.59` points. Versus static all-ancestor routing, it gains `0.89` points on
EwT25 but worsens RMSE/Fnorm by `0.115162`/`0.203105` and EwT10/EwT5 by `1.48`/`0.30` points. It is
substantially better than DPA4 on Fnorm and all EwT thresholds, but GMTNet remains best on all five
metrics; the relative-PG gaps to GMTNet are `+0.809441` RMSE, `+0.238200` Fnorm, and
`-10.64`/`-8.27`/`-2.81` EwT percentage points. Because the new run also doubles the hidden irrep
multiplicities, these cross-width deltas cannot isolate the causal effect of residual path weighting.

## Errors by source space group

![Test error versus training structures per source space group](reduced_dielectric_total_space_group_errors.svg)

Job 463 joins both prediction files to the frozen dataset by exact record ID and groups by the
curated **source space-group number**. All 69 space groups represented in the 677-sample test set
are retained in the [complete table](reduced_dielectric_total_space_group_errors.md); the
[machine-readable CSV](reduced_dielectric_total_space_group_errors.csv) additionally contains
relative Fnorm, EwT10/EwT5, and paired current-pg-minus-GMTNet deltas.

The primary correlation cohort contains the 35 space groups with at least five test structures
(621/677 test samples). The sensitivity cohort contains the 19 groups with at least ten test
structures (516/677 samples). Correlations are unweighted across space groups and use
`log10(train_count)`.

| Cohort | Model | Error | Spearman rho | p | Pearson r | p |
|---|---|---|---:|---:|---:|---:|
| test >= 5 | current-pg | RMSE | 0.4260 | 0.0107 | 0.3590 | 0.0342 |
| test >= 5 | current-pg | Fnorm | 0.3271 | 0.0551 | 0.3652 | 0.0310 |
| test >= 5 | current-pg | relative Fnorm | 0.1290 | 0.4603 | 0.0532 | 0.7617 |
| test >= 5 | GMTNet | RMSE | 0.4880 | 0.0029 | 0.3869 | 0.0217 |
| test >= 5 | GMTNet | Fnorm | 0.3782 | 0.0251 | 0.3781 | 0.0251 |
| test >= 5 | GMTNet | relative Fnorm | 0.3002 | 0.0798 | 0.0875 | 0.6172 |
| test >= 10 | current-pg | RMSE | 0.1352 | 0.5810 | 0.0650 | 0.7915 |
| test >= 10 | current-pg | Fnorm | 0.1387 | 0.5711 | 0.1375 | 0.5745 |
| test >= 10 | current-pg | relative Fnorm | -0.1273 | 0.6035 | -0.2702 | 0.2633 |
| test >= 10 | GMTNet | RMSE | 0.1835 | 0.4521 | 0.0991 | 0.6866 |
| test >= 10 | GMTNet | Fnorm | 0.1431 | 0.5589 | 0.1340 | 0.5845 |
| test >= 10 | GMTNet | relative Fnorm | -0.0342 | 0.8893 | -0.2954 | 0.2196 |

This does **not** support the simple hypothesis that more training structures per space group
automatically reduce test error. In the primary cohort the absolute errors instead increase with
training count, while the scale-normalized error has no significant relationship; all associations
also disappear under the stricter test-count threshold. Absolute error is strongly associated with
the space group's mean target norm (Spearman rho `0.8955/0.9185` for current-pg RMSE/Fnorm and
`0.8081/0.8084` for GMTNet, all `p < 1e-4`). The observed training-count trend is therefore mainly a
group-difficulty/target-scale and cohort-composition diagnostic, not evidence that additional data
hurts learning or that data volume alone explains the model gap.

Among the 35 primary-cohort groups, current-pg has lower Fnorm than GMTNet in 12 and GMTNet in 23.
The largest current-pg advantages are SG 74 (`-11.12` Fnorm, 5 test), SG 11 (`-5.83`, 23 test),
SG 62 (`-4.59`, 25 test), SG 166 (`-3.96`, 24 test), and SG 164 (`-2.08`, 71 test). The largest
GMTNet advantages are SG 141 (`+7.77`, 6 test), SG 129 (`+5.31`, 20 test), SG 216 (`+4.84`, 27
test), SG 58 (`+4.16`, 5 test), and SG 136 (`+3.44`, 9 test), where positive deltas mean larger
current-pg error. Small-test groups should be treated as hypotheses for targeted resampling rather
than stable rankings.

## Acceptance evidence

- DPA4 Slurm job 451: 64/64 recovery partitions; zero-failure JUnit; 677 predictions.
- GMTNet Slurm job 443: zero-failure JUnit; 677 predictions.
- CGCNN Slurm job 458: summary status `passed`; 200/200 epochs; zero-failure JUnit
  (`tests=1`, `failures=0`, `errors=0`, `skipped=0`); 677 predictions.
- Strict three-model comparator Slurm job 459: status `passed`; 677 identical ordered IDs; symmetric
  target eigenvalue-equivalence gate passed.
- Static all-ancestor PG-DAG Slurm job 472: status `passed`; 200/200 epochs; best epoch 122; JUnit
  1/0/0/0; exact 5,001/637/677 splits; 677 predictions; routing identity and DAG asset hash verified.
- Strict four-model comparator Slurm job 473: status `passed`; all four prediction files contain the
  identical ordered 677 IDs and pass the symmetric target eigenvalue-equivalence gate.
- Relative-PG Slurm job 478: status `passed`; 200/200 contiguous epochs; best epoch 28; exact
  5,001/637/677 splits; 677 predictions; JUnit 1/0/0/0; finite five-metric report; routing identity,
  complete offline path topology, edge stick-breaking, path prior, and duplicate-PG reduction
  metadata verified by strict acceptance job 482.
- Five-model comparator job 479: status `passed`; all five files contain the identical ordered 677
  IDs and pass the symmetric target eigenvalue-equivalence gate. Curve job 480 produced the accepted
  SVG/PNG. Replacement manifest job 484 (for the environment-only failure of job 483) completed with
  empty stderr and atomically verified 19/19 final artifact sizes and SHA-256 values.
- DPA4 prediction SHA-256:
  `a8745811e2f67a5810a2b862336898b0fb93049fb6905afaa58445a4e67436f5`.
- GMTNet prediction SHA-256:
  `484a4aa4137779013387f1470e5f9278d4f02c12523512c698128c7b4ca9d64c`.
- CGCNN prediction SHA-256:
  `ee22d732b810a1d75c442a29425bdbfc4c0c1c60a89675cdadf9cbaf7a1c48b4`.
- CGCNN summary SHA-256:
  `7070c4f66d57d5573b58d95a3219f780b0b776a662eab11426903544966fb536`.
- Parent-DAG prediction and summary SHA-256:
  `ecfd870541b8b43bf913b01d48e92bf8396fb6ce67f06fac9c4cb21d960cc6e2` and
  `7c88bd2e7fd1eb9dd05f1dffe3f8f6cd05b5b73d06a218bf5d38f96fb9c37b50`.
- Three-model comparison JSON SHA-256:
  `52091690cec9b61299a35a2f40acf4150f043f4bd7aed6ab338da4957c63e174`.
- Three-model comparison table SHA-256:
  `2a4ef5b8ef1fe82daa8a30e3389708f60e327cf93771e009fe96feb24fad07e7`.
- Four-model comparison JSON and table SHA-256:
  `d270599d45deaf2e98d51177c0b1c2f4eb47f0ae62fa7d5ed5b27c7a67fb2d58` and
  `2f6a02468f39eae24d280dcdae16f2e14c5121bc2b1a4eade9c76b9aae68d8e2`.
- Relative-PG prediction, summary, JUnit, and acceptance SHA-256:
  `666c32a449372998182143e5dec26f87ed3e09bf887efda73a681dd0eb31dec5`,
  `3be671cb61a2b9c066046891ca5d98701cb8de200a88e5afd9b4043e3571debc`,
  `93eb6b1943f9f741ccf4adfa9eb152331772a46e5ba78562aae91999211ae91c`, and
  `022cbf0edc64d9387d5eda5d979859b174278463ee56b3b25961647b118cf49d`.
- Five-model comparison JSON and table SHA-256:
  `ddf28f44be8f9272fb39ee6c813081f242c38c69cc39e7a6dd402be9e23f426e` and
  `f8751f301500a9d3c14d7d074a2c6351f59b4dc45ee3e65974dbc291d409f0ea`.
- Final 19-artifact manifest SHA-256:
  `06973993e5a43f6627fba07a2a77f958e452b6d974e4183ab6688adc14e7a01e`.
- Parent-DAG routing-comparison SVG/PNG SHA-256:
  `8d74a18cc87b9de49fb86fa7f037305a12995975abcfdb286ef7c2f849294226` and
  `c6499e9662e8f2d5eae7dca7c25872824d57739fa39d072b0024de7716d3e542`.
- Space-group analysis Slurm job 463: 69 observed test space groups and exact frozen counts
  5,001/637/677; stderr contains only known TorchScript annotation warnings and no traceback.
- Space-group JSON/CSV/Markdown/SVG SHA-256:
  `abebd7a9b07b68c0b257e4b2b97cef0f48ea35ee080ed344cf783a6460c1e474`,
  `f0e9e4b9c00ecbe5856a5400a6bed445cb046908954472d4ab423774cbd8da66`,
  `ef37f3bfcec074a433344e8cbda304e431bafc5eca0b9921f8a08dbbf8a04da2`, and
  `c052615fbfb7bc0d415a92ec492f7f3b4b1df8ee27dc81e84dd6bee4e8e89519`.

Raw generated evidence remains under ignored `results/reduced-benchmark/` on Guqq; all 19 final
artifacts were independently hash-checked after local transfer, and the accepted relative-PG curve
was promoted beside this report. Metric definitions follow the [GMTNet paper](../ref/GMTNet.pdf).
