# Heavy standalone DPA relative-PG

Increase each Full-PG expert from two to twelve independently initialized finite-group
blocks, using the requested `[16,2,2,2,2]` carrier (64D). Every repeated block participates in inference
and backpropagation. This experiment uses the current within/cross-chain weighting.

## Capacity

The comparison target is non-attention **DPA-GMTNet**, matching the frozen feature source.
Its learned parameters total 738,661; ordinary 92D-input GMTNet has 668,517 (70,144 fewer
atom-projection weights). Counts exclude frozen DPA and non-trainable tensors.
The DPA-GMTNet count was audited against checkpoint 512 epoch 20: learned weight/bias
entries, excluding batch-normalization statistics, RBF centers, Wigner buffers/output masks
and the fixed dielectric output constant. Fifteen reachable PG experts and 24 edge scales
are instantiated from the reduced dataset's ancestor union.

| Blocks per expert | Standalone trainable parameters | Difference from DPA-GMTNet |
|---|---:|---:|
| 2 | 130,196 | -608,465 |
| 11 | 705,260 | -33,401 (-4.52%) |
| 12 | 769,156 | +30,495 (+4.13%) |
| 13 | 833,052 | +94,391 |

Uniform integer depth cannot give exact equality. Twelve is the closest depth to
DPA-GMTNet; ten (641,364) is closest to ordinary GMTNet (-4.06%). Each extra layer adds
63,896 parameters. These are actual constructor counts with the current 24-edge router.
This matches stored trainable capacity: only the experts selected for a material execute,
so active parameters and FLOPs are not matched to GMTNet.

## Training

`slurm/train_reduced_dpa4_relative_pg_heavy.sbatch` runs preflight unit tests and then one
full training run on the frozen 5,001/637/677 split. Settings: batch 64, seed 42,
200 epochs, Huber loss, AdamW with weight decay 1e-5, per-step LR 1e-3 to 1e-5,
checkpoint every 20 epochs, validation-MAE selection strictly at epoch >100.
Grouped PG gates, vectorized routing and asynchronous CUDA expert buckets stay enabled.
The runner rejects parameter counts other than 769,156 before the first training update.

Outputs are isolated under `results/reduced-benchmark/dpa4-relative-pg-heavy/JOB_ID/`.
Checkpoint depth metadata rejects wrong-depth restores; old checkpoints default to depth 2.

## Verification

Requested 64D/12-block tests: 41 passed (172.79s), covering independent blocks, grouped/reference
output and gradient parity, exact budget/nearest depth, full-model optimizer step and checkpoint
round trip, plus reduced-trainer and historical checkpoint regressions. Launcher bash syntax and
scoped diff checks passed. The Slurm job repeats heavy/checkpoint tests before training.
