# Reduced dielectric-total: DPA4 vs CGCNN full-PG vs GMTNet

## Protocol

- Dataset: `curated_reduced_total__dielectric`, 6,315 structures.
- Frozen splits: 5,001 train / 637 validation / 677 test.
- Dataset SHA-256: `6cfefc7c04734ceb4c7873e5aaa7d483251ea909a378e55034414d53f9110963`.
- Source-retained point groups: `2/m`, `mm2`, `mmm`, `4/mmm`, `-3m`, `-43m`, `m-3m`.
- DPA4 model: `B+A+PGE+R`, `full_o3` adaptation/readout, `full_pg` experts.
- CGCNN full-PG model: additive (non-DPA4) branch using GMTNet's fixed 92-component JARVIS CGCNN
  node descriptor, a learned `92 -> 128` even-scalar embedding, and the same
  `B+A+PGE+R/full_pg/full_o3` tensor architecture. Its expert registry is the deterministic union of
  canonical train/validation/test symmetries: `2/m`, `mm2`, `mmm`, `4/mmm`, `3m`, `-3m`, `6/mmm`,
  `-43m`, `m-3m`.
- CGCNN optimization follows GMTNet: Cartesian Huber loss (`delta=1`), AdamW, 200 full epochs,
  per-step linear learning-rate decay from `1e-3` to `1e-5`, and best-checkpoint selection by
  validation component MAE. The best checkpoint (epoch 159, validation MAE `4.3647098541`) was
  reloaded before test inference.
- GMTNet model: pinned official dielectric implementation.
- All rows use the same ordered 677 test IDs. The strict comparator additionally verifies symmetric
  target eigenvalue equivalence with `atol=2e-4`, `rtol=2e-5`.

## Metrics

- RMSE: component-wise root mean squared error over the full test tensors.
- Fnorm: mean per-sample Frobenius norm of the tensor error.
- EwT x%: percentage of samples satisfying
  `Fnorm(prediction - target) / (Fnorm(target) + 1e-5) < x%`.

## Results

| Model | RMSE ↓ | Fnorm ↓ | EwT 25% ↑ | EwT 10% ↑ | EwT 5% ↑ |
|---|---:|---:|---:|---:|---:|
| DPA4 B+A+PGE+R full_pg | 26.166445 | 31.539129 | 12.26% | 2.51% | 1.03% |
| GMTNet | **25.449894** | **19.169209** | **53.03%** | **18.32%** | **7.39%** |
| CGCNN B+A+PGE+R full_pg | 26.186150 | 19.496103 | 40.77% | 10.64% | 4.28% |

GMTNet is best on every reported test metric under this frozen protocol. Relative to the DPA4 row,
the additive CGCNN full-PG branch substantially improves Fnorm and every EwT threshold, while its
component RMSE is slightly higher (`+0.019705`). DPA4's best checkpoint was epoch 12
(`validation_loss=0.9333040631`); GMTNet's was epoch 93 (`validation_mae=4.1132789`); CGCNN's was
epoch 159 (`validation_mae=4.3647098541`) after completing all 200 epochs.

## Acceptance evidence

- DPA4 Slurm job 451: 64/64 recovery partitions; zero-failure JUnit; 677 predictions.
- GMTNet Slurm job 443: zero-failure JUnit; 677 predictions.
- CGCNN Slurm job 458: summary status `passed`; 200/200 epochs; zero-failure JUnit
  (`tests=1`, `failures=0`, `errors=0`, `skipped=0`); 677 predictions.
- Strict three-model comparator Slurm job 459: status `passed`; 677 identical ordered IDs; symmetric
  target eigenvalue-equivalence gate passed.
- DPA4 prediction SHA-256:
  `a8745811e2f67a5810a2b862336898b0fb93049fb6905afaa58445a4e67436f5`.
- GMTNet prediction SHA-256:
  `484a4aa4137779013387f1470e5f9278d4f02c12523512c698128c7b4ca9d64c`.
- CGCNN prediction SHA-256:
  `ee22d732b810a1d75c442a29425bdbfc4c0c1c60a89675cdadf9cbaf7a1c48b4`.
- CGCNN summary SHA-256:
  `7070c4f66d57d5573b58d95a3219f780b0b776a662eab11426903544966fb536`.
- Three-model comparison JSON SHA-256:
  `52091690cec9b61299a35a2f40acf4150f043f4bd7aed6ab338da4957c63e174`.
- Three-model comparison table SHA-256:
  `2a4ef5b8ef1fe82daa8a30e3389708f60e327cf93771e009fe96feb24fad07e7`.

Generated evidence remains under ignored `results/reduced-benchmark/` on Guqq. Metric definitions
follow the [GMTNet paper](../ref/GMTNet.pdf).
