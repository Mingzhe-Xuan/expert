# Standalone PG: shared chain weighting and execution optimization

## Algorithm

The DPA relative-PG training entry point now defaults to `within_cross_chain`, using
the exact same reference/vectorized router classes as the GMTNet wrapper. The generic
model/training APIs retain `legacy` as their default to preserve existing callers.
Explicitly pass `pg_weighting="within_cross_chain"` when constructing a new model.

For chain c = (H0, H1, ..., HL), H0 is the current point group and Hj are successive
parents. For edge j, let residual r_j be detached geometric metadata, sigma_j =
softplus(theta_j) + sigma_floor, u_j = (r_j / sigma_j)^2 and g_j = 1 - exp(-u_j).
Within-chain weights are:

- omega_c,0 = product_{j=1..L} g_j;
- omega_c,j = product_{h<j} g_h * exp(-u_j), for j >= 1.

For L=0, omega_c,0=1 and E_c=0; otherwise E_c=u_1. Cross-chain weights are
pi_c = softmax_c(-E_c / temperature). Each expert's merged coefficient is
alpha_H = sum_{c,j: H_j=H} pi_c * omega_c,j. The pure-PG dispatcher computes each
deduplicated expert once over its assigned structures, combines node features with
alpha_H, then applies the existing TensorReadout. There is no GMTNet/global fusion
coefficient in this standalone model. Chain and expert weights are normalized;
learnability comes from shared edge scales, not independent chain logits.

This deliberately differs from legacy root-to-current stick breaking with node-count
chain priors. Legacy and new predictions need not agree. Node pooling/readout and
canonical input preparation are not changed by this port.

## Interfaces and compatibility

The DPA CLI exposes `--pg-weighting`, `--chain-temperature`, `--initial-sigma`,
`--sigma-floor`, and positive/negative grouped-gate/vectorized-routing flags. Reference
execution is obtained by disabling both optimizations while keeping the same weighting.
The shared implementation is [chain_routing.py](../../src/experts/chain_routing.py);
old wrapper imports re-export these classes and retain their state-dict names.

Old standalone model state loads unchanged in legacy mode. Explicit scale conversion
via `migrate_legacy_edge_scales` preserves learned positive sigmas and non-routing
weights when switching algorithms. It is not an exact training resume: initialize a
new optimizer. New routing checkpoints record temperature, floor, edge order and
algorithm, and reject mismatches. Existing legacy checkpoint metadata stays valid.

## Measurement protocol

[compare_standalone_pg.sbatch](../../slurm/compare_standalone_pg.sbatch) runs regression
tests and real 7/64-crystal comparisons in one Slurm GPU allocation. Four variants:
legacy baseline; new weighting reference; new weighting grouped gates; new weighting
grouped gates plus vectorized chain routing and PG-only graph extraction shortcut.
All retain existing independent expert CUDA streams.

Six warmups and seven synchronized timed full forward/loss/backward passes per variant,
no optimizer updates. Same frozen DPA source features, native graphs, non-routing state,
positive sigmas, RNG and Cartesian Huber loss. Inputs are the original 3200D equivariant
source rather than the wrapper's 640D invariant embedding. Preparation is excluded.
New-weighting variants must match outputs (atol 2e-5, rtol 2e-4) and all parameter
gradients (atol 2e-4, rtol 2e-3). Legacy is a separate algorithm baseline, not a parity test.

## Verification and current measurement status

Implementation commit: `7750d43`; strict-double test setup fix: `ef167ba`. Both are
pushed to GitHub and synchronized to Guqq.

- Local final model/router/checkpoint suite: 131 passed.
- Local reduced benchmark/training regression suite: 38 passed.
- Strict-double construction/equivariance tests: 33 passed locally and on Guqq CPU
  Slurm job 525 (10.95 seconds); unchanged tolerances, including reference experts.
- Initial server job 521 stopped before profiling: 113 passed, 27 strict-double failures.
  Installed e3nn 0.5.9 builds Wigner generators in the global default dtype; float64
  angles and `.double()` alone retained float32 generator error (~1e-6). The fix scopes
  float64 default construction to strict-double tests and restores it afterward.
  Production float32 initialization and GPU parity thresholds are unchanged.

GPU retry **524 is queued for Resources behind existing job 523**, which is not modified
or interrupted. The retry reruns the full server suite before both four-variant batches.
Expected outputs: `results/standalone-pg-compare/524/{smoke,batch64}/comparison.json`,
per-variant `summary.json` and `trace.json`, plus `tests.xml` and `revision.txt`.
No successful GPU comparison has been obtained yet. In particular, **no pure-PG speedup
is claimed** from CPU timings, incomplete jobs, or the previous GMTNet-wrapper speedup.
