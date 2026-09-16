# Reduced dielectric-total: DPA4 vs GMTNet

## Protocol

- Dataset: `curated_reduced_total__dielectric`, 6,315 structures.
- Frozen splits: 5,001 train / 637 validation / 677 test.
- Dataset SHA-256: `6cfefc7c04734ceb4c7873e5aaa7d483251ea909a378e55034414d53f9110963`.
- Retained point groups: `2/m`, `mm2`, `mmm`, `4/mmm`, `-3m`, `-43m`, `m-3m`.
- DPA4 model: `B+A+PGE+R`, `full_o3` adaptation/readout, `full_pg` experts.
- GMTNet model: pinned official dielectric implementation.
- Both rows use the same ordered 677 test IDs. The comparator additionally verifies symmetric-target
  eigenvalue equivalence with `atol=2e-4`, `rtol=2e-5`.

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

GMTNet is better on every reported test metric under this frozen protocol. DPA4's best checkpoint
was epoch 12 (`validation_loss=0.9333040631`); GMTNet's was epoch 93
(`validation_mae=4.1132789`).

## Acceptance evidence

- DPA4 Slurm job 451: 64/64 recovery partitions; zero-failure JUnit; 677 predictions.
- GMTNet Slurm job 443: zero-failure JUnit; 677 predictions.
- Comparator retry job 454: `COMPLETED`, exit `0:0`, runtime 4 seconds, status `passed`.
- DPA4 prediction SHA-256:
  `a8745811e2f67a5810a2b862336898b0fb93049fb6905afaa58445a4e67436f5`.
- GMTNet prediction SHA-256:
  `484a4aa4137779013387f1470e5f9278d4f02c12523512c698128c7b4ca9d64c`.
- Comparison JSON SHA-256:
  `da19fe542682e43019abfdc942b1f5e4ce144298cec66730e468e7ec62ebe223`.
- Comparison table SHA-256:
  `a7e7d30d910234d9b4733fb60cacee606dffa2841bd1683f389315b5c99d3ce7`.

Retrieved evidence is kept under ignored
`results/reduced-benchmark/final-evidence/`. Metric definitions follow the
[GMTNet paper](../ref/GMTNet.pdf).
