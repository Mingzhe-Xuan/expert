# Agent progress updates

- 2026-09-22: Began the requested training/test/curve/benchmark cycle for the new 56D full-PG
  relative-position parent-DAG model. The existing job-472 result is a historical static-router,
  old-width baseline and will not be relabelled. Next is a selective source audit/commit, followed by
  Slurm smoke and full-split preflight before the 200-epoch production run.
- 2026-09-22: Revision `1479bb7` is synchronized on Guqq and smoke job 474 passed every gate:
  7/7/7 splits, one epoch, seven unique predictions, finite metrics, zero JUnit failures/errors, and
  exact complete-path relative-PG stick-breaking metadata. Submitting the one-epoch full preflight next.
- 2026-09-22: While full preflight 475 runs, extended the strict result pipeline to accept and
  distinctly label relative-PG path-weighted histories/predictions alongside the historical static
  parent-DAG row. Focused tests pass 18/18 and comparison launcher syntax is valid.

- 2026-09-22: Started replacing the remaining full-PG hidden profile `[4, 1, 1, 1, 1]` with
  `[8, 2, 2, 2, 2]`. The implementation will make the default hidden O(3) layout uniform across
  modes, update the architecture documentation and exact layout tests, and revalidate the existing
  active-parameter ceiling before completion.
- 2026-09-22: The code, exact tests, module contract, GOAL/proposal, historical benchmark warning,
  and parameter analysis now use or distinguish the new 56-component full-PG layout. Focused
  expert/dispatcher verification passed 22/22; measured full-PG downstream totals remain under
  204k for all 32 experts and under 47k for the reduced expert sets. Full-suite verification is next.
- 2026-09-22: Completed the hidden-width unit. The full repository suite passed 298/298 in
  353.03 seconds; Python compilation, reference audit, and scoped whitespace checks passed. The
  default layout is now uniformly `[8, 2, 2, 2, 2]`, while pre-change benchmark metrics remain
  preserved and visibly marked as historical-width results.

- 2026-09-22: Began replacing Hall/space-group parent routing with canonical-frame point-group
  distance because the model embeddings use relative positions only. The existing subgroup-chain
  asset will supply both class topology and oriented rotation subsets; cache and report identities
  will change so Hall-edge artifacts cannot be reused silently.
- 2026-09-22: Finished the relative-position PG implementation and documentation migration. The
  runtime no longer accepts or requests a Hall embedding registry; it reconstructs oriented cover
  templates from `subgroup_chain.json`, computes minimum-orientation parent-minus-child residuals on
  species-labelled graph vectors, and writes lightweight schema-3 caches. All 298 project tests pass;
  the asset generator reproduces the tracked JSON exactly.
- 2026-09-22: Removed the stale asset-policy depth/index limits so metadata matches the complete-path
  implementation. One old assertion exposed the mismatch and was updated; the 32-PG path matrix plus
  asset contract rerun passed 33/33.

- 2026-09-22: Replaced the superseded node-residual/multi-`symprec` route with the requested offline
  Hall-edge design. Added explicit fractional operations/common-cell conventions, concrete
  `(Hall, setting)` paths, incremental parent-minus-child residuals, shared edge-template gates,
  chain stick-breaking, path-length priors, and unique-PG aggregation. Cross-module tests pass
  57/57; full-suite verification is next. No Guqq job was submitted because the required physical
  Hall embedding registry is absent.
- 2026-09-22: Completed local acceptance: clean compilation, parent-DAG Slurm syntax, scoped
  whitespace checks, 57/57 cross-module tests, and 271/271 complete project tests passed. The code
  is ready for a concrete embedding registry; generation or acquisition of that external artifact
  is the only gate before Slurm preparation/smoke.
- 2026-09-22: Added the final shared-edge-template collision and mutually exclusive router guards.
  The affected 16-test parent-DAG/dispatcher regression passed, as did repeated compilation, Slurm
  syntax, and whitespace checks. Test temporary directories were removed after verification.

- 2026-09-17: terminal audit completed after network recovery. Job 458 produced a passed summary,
  200/200 history through epoch 200, best epoch 159 by validation component MAE, zero-failure JUnit,
  and exactly 677 predictions. Job 459 produced a passed strict comparison with 677 ordered IDs and
  the symmetric-target eigenvalue-equivalence gate. CGCNN final metrics are RMSE `26.186150`, Fnorm
  `19.496103`, EwT25 `40.77%`, EwT10 `10.64%`, and EwT5 `4.28%`. Updated the tracked benchmark into
  a three-model table with artifact hashes; next step is final checks, selective commit, and push.

- 2026-09-17: user confirmed Guqq network recovery, reopening a fresh blocked audit. Next action is
  the mandated `net.sh` plus full three-minute wait, then an HTTP/1.1 pull-first check of jobs
  458/459 and their terminal artifacts; no completion or metric is assumed in advance.

- 2026-09-17: after recovery, the same Vlab observation-channel closure persisted across three
  consecutive resumed-goal turns. Immediate reconnection would violate the recorded bounded-retry
  lesson, and no local action can prove terminal metrics. Marked the goal blocked again while
  leaving Slurm jobs 458/459 unchanged.

- 2026-09-17: a corrected separator-free monitor pulled successfully and reconfirmed job 458
  `RUNNING` at 9:55:38; the best checkpoint had advanced to mtime 11:35. Vlab closed the SSH
  transport after roughly five minutes, affecting only observation and not Slurm. Deferred another
  recovery connection until a meaningful interval rather than immediately retrying.

- 2026-09-17: the corrected resumed-goal recovery completed its full wait and bounded HTTP/1.1
  pull successfully (`Already up to date`). Job 458 was healthy `RUNNING` at 9:53:18 on node221;
  job 459 remained correctly pending on its dependency. The fresh blocked audit resets, and
  low-frequency read-only monitoring resumes without changing either job.

- 2026-09-17: the blocked goal was resumed, starting a fresh blocked audit. After an approximately
  18-minute interval, prepared one bounded recovery attempt with `net.sh`, a full three-minute wait,
  and an HTTP/1.1 pull-first gate. Jobs 458/459 remain untouched; scheduler and artifact inspection
  are still conditional on pull success.

- 2026-09-17: the same mandatory pull-first network failure persisted across at least three
  consecutive goal turns after the last recovery attempt. No further local or remote action can
  establish job completion, retrieve artifacts, validate 677 predictions, or update the metric
  table without that external gate. Marked the goal blocked under the strict audit; jobs 458/459
  remain untouched and can be audited immediately after Guqq outbound access recovers.

- 2026-09-17: the persistent monitor observed job 458 continuously healthy through 7:26:17 on
  node221; the best checkpoint remained epoch 90, which is not a stall signal because publication
  occurs only on validation-MAE improvement. Interrupting the local wait later closed the SSH
  monitor but not the Slurm job. Three immediate recovery/pull-first connections failed, so the
  existing network lesson was consulted and retries stopped. A later retry after roughly 30 minutes
  completed `net.sh` and the full wait but still failed the bounded HTTP/1.1 pull; its `&&` gate
  prevented scheduler access. Final training/test evidence remains pending without weakening the
  200-epoch acceptance criterion.

- 2026-09-17: while job 458 continues, started a small independent implementation unit for an
  `afterok:458` comparator launcher. It will require explicit DPA4/GMTNet/CGCNN prediction paths and
  run the existing 677-ID/target-equivalence validator through Slurm, removing the completion-time
  manual race without changing training or model code.

- 2026-09-17: implemented the explicit-artifact, job-scoped comparison launcher. Ten focused tests,
  Bash syntax, compilation, and whitespace checks passed; the full project suite passed 248/248.
  Next: commit/push, pull first on Guqq, confirm accepted DPA4/GMTNet prediction paths, and submit the
  comparator with `afterok:458`.

- 2026-09-17: pushed/pulled comparator commit `2b2c65f`, verified accepted DPA4 job-451 and GMTNet
  job-443 prediction JSONLs, and submitted comparison job 459 with `afterok:458` plus the explicit
  future CGCNN job-458 prediction path. The persistent read-only monitor shows training healthy
  through 1:11 with checkpoint mtime continuing to advance.

- 2026-09-17: persistent monitoring advanced job 458 past five hours. A pull-first, CPU-only
  metadata check found best checkpoint epoch 90 at 5:15:22; job 459 remains correctly pending on
  `afterok:458`. The fixed 200-epoch training protocol is near its midpoint and remains healthy.

- 2026-09-17: full job 457 completed atomic feature materialization for every 5,001/637/677 record,
  then failed before training because a canonicalized structure routes to `6/mmm` while expert
  construction was hard-coded to the seven source-manifest retention labels. No predictions or
  metrics were emitted. Began a local repair to derive the expert domain deterministically from all
  cached canonical symmetries; the retry will reuse the complete caches and remain a Slurm job.

- 2026-09-17: implemented deterministic expert-domain discovery from all cached canonical train,
  validation, and test symmetries, while retaining the seven source-label smoke selection and the
  untouched DPA4 branch. The new `6/mmm` regression passed within 16 focused tests; the complete
  suite passed 247/247, and compilation, launcher syntax, and whitespace checks passed. Next:
  commit/push, pull-first sync, and submit the cache-reusing full retry.

- 2026-09-17: pushed repair `3dccf65`, pulled it first on Guqq, verified the complete 188/25/27 MiB
  train/validation/test caches, and submitted exact full retry job 458. Startup inspection showed
  the requested 200 epochs, batch 64, LR `1e-3 -> 1e-5`, weight decay `1e-5`, a live CUDA process,
  and nonzero GPU utilization; the prior pre-training `6/mmm` failure has not recurred.

- 2026-09-17: a later read-only monitor stalled at the Vlab banner. Ran the required `bash net.sh`,
  completed the full three-minute wait, and retried; the command channel still did not reach the
  mandatory `git pull`. Terminated the monitor without touching Slurm job 458 and stopped immediate
  retries in accordance with the existing network lesson. The last authoritative job state remains
  healthy `RUNNING`; terminal metrics are still pending.

- 2026-09-17: bounded verbose SSH proved both hops authenticate and Guqq accepts the command, but
  the mandatory default pull stalled. A later 45-second-bounded HTTP/1.1 pull also returned exit 1
  without output, and its `&&` gate correctly prevented scheduler access. This same external pull
  blocker has now repeated for three consecutive goal turns; job 458 remains untouched and the goal
  must wait for Guqq outbound network recovery before terminal verification can continue.

- 2026-09-17: resumed with a fresh `net.sh` and full 180-second wait; mandatory pull then succeeded.
  Job 458 was healthy `RUNNING` at 22:09 and had published a 3.3 MiB checkpoint, proving the dynamic
  canonical expert domain crossed the prior `6/mmm` failure and completed at least one validation
  boundary. Summary, JUnit, and predictions are correctly absent while training continues.

- 2026-09-17: a second recovery/wait plus HTTP/1.1 pull restored monitoring. Job 458 reached 35:57,
  its checkpoint advanced to 03:26, and a lightweight CPU-only metadata read reported epoch 10.
  This implies roughly 3.6 minutes/epoch and a projected ~12-hour 200-epoch run, within the 48-hour
  allocation. No terminal artifacts exist yet, so the job remains in progress.

- 2026-09-17: after two consecutive pull-gated monitor failures, the third recovery/wait succeeded.
  Job 458 was healthy at 49:04 and checkpoint epoch 14 (mtime 03:39), so the fresh blocked audit reset
  without marking the goal blocked. Training throughput remains near 3.5 minutes/epoch.

- 2026-09-16: started the additive CGCNN-feature `B+A+PGE+R/full_pg` branch. Froze the intended
  feature source to GMTNet's exact 92-component JARVIS/CGCNN descriptor and the optimization protocol
  to Cartesian Huber loss, AdamW, per-step linear decay to `1e-5`, and validation component MAE
  checkpoint selection. The existing DPA4 branch and its results remain unchanged. Planned local
  parity/regression gates followed by a real Slurm smoke and full 5,001/637/677 training/test run.

- 2026-09-16: implemented the independent fixed-feature provider/cache, learned GMTNet-shaped
  `92 -> 128` scalar embedding, full-PG model wrapper, dedicated CLI/Slurm launcher, explicit
  GMTNet optimization protocol, and optional third-model comparison-table input. Preserved the
  legacy DPA4 coefficient-MSE/scheduler/early-stop path behind its original default. Focused tests
  passed 26/26 and the complete project suite passed 246/246; compilation, CLI, launcher, and
  whitespace checks passed. Next: commit/push, Guqq pull-first sync, and real Slurm smoke.

- 2026-09-16: commit `90c9c45` was pushed and pulled on Guqq. First smoke job 455 failed safely
  before preprocessing because the DPA4 environment lacks jarvis-tools. Verified the existing
  GMTNet environment already has the compatible Torch 2.11/e3nn 0.5.9/jarvis-tools 2025.5.30 stack;
  split the launcher variable to `EXPERT_CGCNN_VENV` so the dependency contract is explicit before
  retrying. No DPA4 environment or accepted artifact was modified.

- 2026-09-16: corrected smoke job 456 passed on RTX 5090 with scheduler `COMPLETED/0:0`, a
  zero-failure JUnit, best-checkpoint reload, 7 prediction rows, finite common metrics, and final LR
  `1e-5`. Submitted full job 457 with exact 5,001/637/677 splits and 200 GMTNet-aligned epochs; its
  initial health check reached 25/5,001 feature records on node221 without traceback.

- 2026-09-13: started the explicit reduced dielectric-total DPA4-vs-GMTNet acceptance benchmark.
  Froze the common 5,001/637/677 group-preserving split and target tensor, selected the exact DPA4
  `B+A+PGE+R/full_pg/full_o3` configuration with seven current-point-group experts, and defined the
  primary output as component RMSE plus paper-compatible Fnorm and EwT 25/10/5 from saved predictions.
  Began auditing the official GMTNet implementation before defining its adapter and Slurm protocol.

- 2026-09-13: froze metric semantics from the GMTNet paper and official commit `7a606a4`: mean
  per-sample Frobenius distance and relative EwT 25/10/5, plus component RMSE. Implemented the
  manifest/hash/split/duplicate-group-gated reduced loader; architecture-aware DPA4 cached training
  with a trainable equivariant source interface and seven full-PG experts; pinned-official GMTNet
  preprocessing/training without WandB/path placeholders; per-ID prediction export; and a comparator
  that requires the same 677 IDs and frame-equivalent targets. Corrected the cached evaluator to
  compare canonical-frame predictions against canonical-frame targets and bumped cache schema to 2.
  Focused tests passed 27; next is final code review/commit and Guqq environment plus Slurm smoke.

- 2026-09-13: committed and pushed the locally verified pipeline as `e6c26e3` (full suite 228/228).
  Guqq's first pull failed with GnuTLS `-110`; the mandated `bash net.sh` plus three-minute wait was
  applied. A retry timed out, then a second recovery/wait followed by an HTTP/1.1 pull succeeded.
  Audit found 15 GiB free and confirmed DPA4 Torch 2.11/e3nn 0.5.9/DeepMD 3.2.0. Added a documented
  PyG scatter compatibility fallback so GMTNet can use the existing compiled PyG runtime without a
  risky login-node extension build; compilation and 9 focused tests passed before its follow-up commit.

- 2026-09-13: the official GMTNet transformer also imports `SparseTensor` only for typing. Added an
  inert fallback for that unused annotation so no retired compiled extension is required. Guqq's
  dedicated venv now imports Torch 2.4.1+cu121, e3nn 0.5.9, PyG 2.8.0.post1, pymatgen 2025.10.7,
  JARVIS-tools 2025.5.30, pandas 2.3.3 and spglib 2.6.0. Direct server clone and the first 31 MB
  recursive transfer both suffered transport failures; next transfer will use one compact Git bundle.

- 2026-09-13: replaced the fragile recursive vendor transfer with a 10,076,941-byte Git bundle
  (`8ab2a682...147c4`), which transferred successfully. Added a 21-record integration-smoke mode that
  deterministically takes one sample per retained point group from each frozen split and isolates
  smoke caches from full caches. Compilation, CLI/launcher checks, 10 focused tests and the full
  229-test suite passed. Three subsequent latest-source pulls/transport attempts failed; consulted the
  existing matching lesson in `docs/agents/lessons.md` and stopped blind retries before Slurm submit.

- 2026-09-13: Guqq was synchronized to `7cc7f85` through a Git bundle and the first real 21-record
  DPA4/GMTNet Slurm smokes were submitted as jobs 437/438. DPA4 exposed a frozen-target scope bug;
  GMTNet completed graph preprocessing but proved Torch 2.4.1+cu121 cannot execute `sm_120` kernels
  on the RTX 5090. Corrected the source lookup, added regression coverage, made optional compiled
  extension fallbacks handle ABI-loader errors, and recorded the layered-environment lesson. Next is
  a fully revalidated Torch 2.11-first GMTNet layer followed by repeat Slurm smokes.

- 2026-09-13: repaired the GMTNet environment ordering to Torch 2.11.0+cu128 and submitted jobs
  439/440. GMTNet job 440 passed complete GPU training/evaluation on the 7/7/7 real smoke; DPA4 job
  439 cached all 21 backbone examples but failed its finite-group invariant check. Server sweeps
  isolated the required numerical threshold: all seven retained groups fail at 5e-8, only `2/m`
  passes at 1e-7, four pass at 2e-7, and all seven pass at 5e-7. Separated the subduction tolerance
  from the stricter CG tolerance and added backend-residue regression coverage; next is full local
  regression and a DPA4-only Slurm retry.

- 2026-09-13: DPA4 retry 441 passed all real GPU smoke stages at `b7c0bb8`; together with GMTNet
  440 this establishes both exact-model pipelines on the same 7/7/7 point-group-stratified subset.
  Submitted full jobs 442/443 for the frozen 5,001/637/677 protocol. DPA4 442 is running full feature
  extraction on node221; GMTNet 443 is queued for that same single GPU. Final acceptance remains the
  common 677-ID prediction comparison, not the one-epoch smoke metrics.

- 2026-09-13: measured full DPA4 job 442 at a sustained 8% SM utilization and 1.37 GiB GPU memory;
  sequential per-structure dispatch, not device capacity, is the throughput limit. Began a
  deterministic four-shard cache mode that runs multiple DPA4 extractors inside the same Slurm GPU
  allocation and validates/reorders their outputs back to the untouched frozen split. Current 442
  remains running until the replacement passes local tests and a real concurrent smoke.

- 2026-09-13: the first four-way smoke (job 444) proved all workers could coexist on the RTX 5090,
  but each failed at cache publication because the serializer accepted only bare split names while
  the sharded CLI supplied an explicit partition identity. The failure was repeated across all four
  workers, so consulted `docs/agents/lessons.md` and added the missing schema-boundary lesson. The
  fix extends validated split identities to bounded `split:shard:index/count` labels and adds real
  serializer round-trip/malformed-label tests before retry.

- 2026-09-13: corrected four-way smoke 445 passed and reproduced serial smoke 441 after exact shard
  merge, while raising GPU SM utilization from 8% to 99% at 2.44 GiB. Cancelled serial full job 442
  at 100/5,001 because its 47.5-minute measured rate made the 48-hour limit infeasible. Released
  GMTNet full job 443, which is now preprocessing the common training split, and submitted replacement
  four-way DPA4 full job 446 behind it.

- 2026-09-13: GMTNet full job 443 completed with exit `0:0` after all 200 epochs and exact 677-row
  test inference. Its epoch-93 best checkpoint reports RMSE 25.4498959, Fnorm 19.1692104 and EwT
  25/10/5 of 53.0281%/18.3161%/7.3855%; output line count, JUnit and artifact hashes were verified.
  DPA4 full job 446 then started automatically with all four shard workers active. Submitted
  dependency-gated comparison job 447 so common-ID metric recomputation cannot run until both full
  training jobs succeed.

- 2026-09-13: exact job-446 timestamps disproved the four-monolithic-shard throughput assumption:
  two shards reached 50 train records in 89--97 minutes, while another required 98 minutes for only
  25, projecting beyond the 48-hour allocation. Reworked the launcher to separate 64 deterministic,
  reusable cache partitions from four persistent GPU workers, limiting long-tail imbalance without
  changing any sample, feature, model, seed, or final order. Fourteen focused tests, Bash/Python
  checks, and the complete 244-test suite passed; next is commit/sync, cancel the provably infeasible
  task-owned jobs 446/447, and submit a real fine-grained smoke before the replacement full run.

- 2026-09-13: fine-grained smoke 448 completed all seven partition caches, exact merge, training,
  reload and test export in 00:13:45. Partition timestamps plus two-worker diagnostic 449 exposed a
  remaining scheduler issue: fixed round-robin ownership can strand a worker behind one slow crystal
  while another exhausts its assigned partitions and idles. Replaced static ownership with atomic,
  job-local dynamic claims; next is local regression and a fresh five-partition real smoke.

- 2026-09-13: dynamic-claim scheduler passed 9 focused tests plus Bash syntax, Python compilation
  and whitespace checks. The complete isolated suite passed 239 tests in 313.50 seconds with no
  failures or skips; five removed cases exclusively tested the retired static assignment helper.
  Next is commit/push, Guqq sync, and the uncached five-partition Slurm smoke.

- 2026-09-13: pushed dynamic scheduler `0b5ce27`; Guqq pulled it after the required recovery wait.
  Real job 450 completed five uncached partitions with two workers, demonstrated cross-worker dynamic
  claiming, passed exact merge/reload/testing, and wrote a zero-failure JUnit plus seven predictions.
  Submitted full DPA4 job 451 with 64 partitions/two workers/three-day walltime and comparison job
  452 held by `afterok:443:451` against the completed GMTNet full run.

- 2026-09-14: DPA4 full job 451 published its first recovery-safe cache partition at 01:23:51
  runtime: partition 0 passed 79/10/11 train/validation/test examples and a zero-failure JUnit with
  the expected dataset/checkpoint/runtime provenance. Its worker immediately claimed partition 2
  while the slower partition 1 continued, proving dynamic scheduling in the real full run. Current
  aggregate throughput projects inside the three-day allocation; comparator 452 remains gated.

- 2026-09-14: after a jump-session interruption and the mandatory recorded `net.sh` plus full
  three-minute recovery wait, Guqq pulled first and showed DPA4 job 451 healthy at 09:56:50 runtime.
  Partitions 0--11 were complete, 12/13 active, GPU utilization 99%, and bounded error scans empty.
  The observed steady throughput projects about 53 hours for all 64 partitions, retaining margin for
  final training/testing and dependency-gated comparison 452 within the three-day allocation.

- 2026-09-14: a later pull-first bounded check found DPA4 job 451 healthy at 18:13:13 runtime with
  23/64 partitions complete and 25/64 claimed. GPU utilization remained 99%, bounded error scans
  remained empty, and the steady projection improved to about 51 hours for all cache partitions.
  Comparator 452 remains dependency-held; no job or model setting was changed.

- 2026-09-16: after repeated slow GitHub responses, the mandated recovery/wait completed and the
  HTTP/1.1 pull-first attempt eventually returned `Already up to date`. DPA4 job 451 was healthy at
  2-00:52:45 with 61/64 partitions complete, 63/64 claimed, 99% GPU utilization and no bounded error
  marker. Only three cache units remain before merge/training/testing; comparator 452 stays gated.

- 2026-09-16: DPA4 full job 451 completed all 64 caches and the exact 5,001/637/677 merge/train/test
  pipeline with a zero-failure JUnit and 677 predictions. Its summary reports RMSE 26.1664448, Fnorm
  31.5391254 and EwT 25/10/5 12.2600%/2.5111%/1.0340%. Comparator 452 failed before Python because
  Slurm `--wrap` ran `/bin/sh`, which rejected `set -o pipefail`; next is an explicit-Bash comparator
  retry over unchanged prediction artifacts, followed by common-ID/hash verification and retrieval.

- 2026-09-16: explicit-Bash comparator job 454 completed with exit `0:0` in 4 seconds. It verified
  the identical ordered 677 IDs and frame-equivalent symmetric targets, then produced the requested
  table: DPA4 RMSE/Fnorm/EwT25/10/5 26.166445/31.539129/12.26%/2.51%/1.03%, versus GMTNet
  25.449894/19.169209/53.03%/18.32%/7.39%. Retrieved predictions, summaries, JUnits, table and logs
  passed independent local line-count, unique/order, hash, metric-parity and error-marker checks.
  Added the compact tracked report under `docs/benchmarks/`; the benchmark acceptance goal is met.

- 2026-09-13: started strict per-property `>5%` point-group reduction over the verified recommended
  datasets. The dielectric properties select seven groups and elastic stiffness selects six; planned
  a streaming hash-gated reducer, explicit per-record/property eligibility metadata, deterministic
  manifests, fixture tests, and Slurm-only full extraction.

- 2026-09-13: implemented the streaming reduced-dataset builder and Slurm launcher. Inputs are
  size/SHA/count/split gated, the four outputs are promoted only after all validate, original records
  and splits are retained, and each selected row carries its property's available-point-group list
  plus count/frequency. Focused tests passed 8; compilation, CLI and launcher syntax checks passed.

- 2026-09-13: Slurm job 436 completed the strict `>5%` real-data extraction in 5 seconds. Retrieved
  34,559 rows across four subtype files and independently verified every byte size, SHA-256 and line
  count. Promoted the compact manifest and usage/schema summary; full JSONL outputs remain ignored.

- 2026-09-13: started a reproducible point-group frequency visualization using the tracked
  recommended-set counts. Planned a dependency-free four-panel SVG renderer and strict count/schema/
  determinism tests so the figure remains regenerable without loading the full curated JSONL files.

- 2026-09-13: completed the recommended-set point-group frequency figure as a deterministic
  dependency-free SVG. The four-panel design keeps subtype count axes independent, orders all 32
  groups by crystal system, and annotates each dominant group. Visual inspection passed; focused
  tests passed 6, deterministic regeneration retained SHA-256 `d1bc875d...`, and compilation/CLI
  checks passed.

- 2026-09-13: started unified physical curation of DTNet/GMTNet dielectric and GMTNet/MatTen
  elastic data. Audited local schemas and primary papers, separated dielectric electronic/ionic/
  total semantics, identified both elastic targets as stiffness tensors in GPa, and planned a new
  `data/curation` module with reason-coded physical/numerical/outlier screening, conservative
  structure deduplication, conflict-safe per-subtype merging, deterministic group splits, and
  descriptive plus 32-point-group statistics. Full processing is reserved for Slurm.

- 2026-09-13: implemented the curation module, fixture tests, CLI, and Slurm launcher. A real-data
  spot audit exposed JARVIS/VASP's `XX,YY,ZZ,XY,YZ,ZX` shear order; corrected both the curation
  adapter and existing training loader to standard Voigt order, reducing a hexagonal sample's false
  PG residual from about 3.9% to `4.3e-8`. Focused tests passed 15 and the complete suite passed 221
  with zero failures. Next step is commit/push followed by pull-first Guqq synchronization and full
  Slurm curation.

- 2026-09-13: Guqq pull-first synchronization succeeded at `94591b6`; GMTNet and MatTen resources
  are present, while DTNet raw/processed resources are not yet on the server. Adjusted the curation
  launcher so its Slurm allocation performs the official hash-gated DTNet download/conversion into
  ignored data paths and writes only a candidate preparation manifest under `results/` before the
  full audit. No batch processing was run on the login node.

- 2026-09-13: job 434 ended during DTNet preparation with a fail-closed path error: the downloaded
  raw file has the expected 37,774,263-byte size, but a manifest under `results/` made the converter
  treat `results/` as its data root. No processed/merged output was written. Revised the launcher to
  use a unique temporary manifest under `data/manifests/`, move it to the ignored job results after
  conversion, and avoid redownloading an already present raw file.

- 2026-09-13: replacement Slurm job 435 completed the unified curation at `06d63cd`. Retrieved and
  hash/size/row-count verified all eight physical-valid and recommended datasets, promoted the
  compact manifest and descriptive/point-group reports, and confirmed 32/32 point-group coverage
  for every subtype. Dielectric remains moderately imbalanced (CV about 1.0); elastic remains
  strongly imbalanced (CV 1.836, `m-3m` share 28.4%). After two mandatory-pull GitHub failures, a
  spaced HTTP/1.1 retry succeeded; the 59,708-row audit was retrieved and all nine artifacts pass
  manifest SHA-256, byte-size, and line-count verification.

- 2026-09-13: completed DTNet dataset integration. Downloaded and verified the 37,774,263-byte
  official raw JSON, generated a deterministic 16,576,244-byte normalized JSONL plus tracked
  provenance/split/quality manifest, and registered `dtnet × dielectric` with total-tensor symmetric
  target loading, smoke selection, and structure candidates. Real validation loaded all 6,648 rows
  with 5,318/665/665 splits. Focused integration passed 26 tests; final project suite passed 217 with
  zero failures; compile, CLI help, deterministic regeneration, hashes, and diff checks passed.

- 2026-09-13: started DTNet dielectric dataset integration from the official
  `pfnet-research/dielectric-pred` repository. Scoped a separate `dtnet × dielectric` unit, a strict
  raw-to-normalized converter retaining electronic/ionic/total tensors, a total-tensor canonical
  training target, frozen provenance/checksum/split manifest, loader integration, and real-data
  validation. Raw/processed data will remain Git-ignored.
- 2026-09-13: inspected all 6,648 upstream records and froze the public seed-3 split counts as
  5,318/665/665. Found material electronic/total antisymmetry in respectively 2,133/2,131 rows above
  `1e-8` (maximum about 2.6925), while the official model symmetrizes its output. Adjusted the data contract to retain
  raw tensors/residuals but train this repository's symmetric `0e+2e` target on `(T + T^T)/2`.

- 2026-09-13: the third consecutive resumed-Goal recovery audit reached Guqq, but the mandatory
  HTTP/1.1 GitHub pull failed with GnuTLS `-110`. All later status and Slurm commands remained gated
  and did not execute. Because the same pre-synchronization external blocker has now repeated for
  three Goal turns and all remaining acceptance work requires Guqq/Slurm, the benchmark Goal is
  formally blocked with no score claim; EquiformerV2 remains paused.

- 2026-09-13: the spaced second resumed-Goal recovery check was again closed by the Vlab SSH endpoint
  before Guqq command output. The mandatory pull and all guarded status/submission commands did not
  run. This is recovery turn 2 for the same external condition; no job or metric is claimed, and the
  Goal remains active pending the strict third-turn blocked audit.

- 2026-09-13: pushed the tested prerequisite repairs as `b8b9583` and began the pull-first Guqq
  resubmission gate. Three bounded connections produced one GitHub GnuTLS `-110` failure followed by
  two Vlab SSH endpoint closures. Because all scheduler commands were guarded behind the mandatory
  pull, no MACE diagnostic, elastic manifest replacement, or training job was submitted. The existing
  outbound-network lesson was consulted; further blind retries are paused and EquiformerV2 remains
  untouched.

- 2026-09-13: completed both repairs. Elastic manifest generation now uses deterministic ordered
  `ProcessPoolExecutor` workers with progress and strict CLI validation; its Slurm job requests 8 CPU,
  48 GiB and passes all CPUs to the generator. Native backbone graph conversion recomputes vector
  lengths and filters every edge failing strict `< cutoff`, preserving zero-copy topology when no
  filtering is needed. Targeted runs passed 31 and 30 tests; the full suite passed 208 tests with
  0 failures in 521.21 seconds. `py_compile`, both launcher `bash -n`, and diff checks passed.

- 2026-09-13: terminal audit after network recovery found job 415 cancelled by its four-hour time
  limit with no candidate, and job 414 failed after feature 2,275/3,770 because a native MACE edge
  violated the strict cutoff check at a numerical boundary. Failed JSON/JUnit/log evidence is retained;
  no metrics exist. Entered two repair units: deterministic multi-worker protocol generation with
  progress, and tolerance-safe native graph cutoff conversion. EquiformerV2 remains paused.

- 2026-09-12: after the user confirmed recovery, two pull-first sessions succeeded at `ec1a58a`.
  Protocol job 415 was monitored from 2:11:49 through 2:50:44; its Python process sustained about
  114% CPU and reached roughly 10.4 GiB RSS under a 16 GiB request, with no traceback or candidate.
  MACE dielectric advanced from feature 1,375 to 1,775/3,770 and free space held at 16 GiB. The Goal
  is resumed; both jobs remain active and unchanged, and EquiformerV2 remains paused.

- 2026-09-12: a third consecutive Goal turn hit the same Guqq-to-GitHub mandatory-pull timeout before
  any scheduler read. The candidate algorithm was inspected locally and still requires exact
  per-structure symmetry screening; without terminal evidence, do not restart or replace healthy
  jobs. All remaining acceptance work requires Guqq/Slurm, so the Goal is externally blocked after
  the required threshold. Jobs 415/414 remain untouched and EquiformerV2 remains paused.

- 2026-09-12: a follow-up terminal-monitor connection reached Guqq but timed out at the mandatory
  pull after 120 seconds, before its scheduler loop. No job state was inferred or changed; retain the
  preceding verified 415/414 progress and retry only after another interval.

- 2026-09-12: the spaced pull-first check recovered and synchronized Guqq to `4563521`. A bounded
  twelve-sample monitor followed protocol job 415 from 58:21 to 1:09:22 without a traceback or
  candidate; MACE dielectric job 414 advanced from feature 750 to 825/3,770. Free space decreased
  from 7.0 to 6.8 GiB. Both jobs remain active and unchanged; no metric is claimed.

- 2026-09-12: resumed the benchmark monitor after the user reported network recovery. SSH reached
  Guqq three times, but the mandatory first pull failed with GnuTLS `-110`, then a 133-second GitHub
  port-443 timeout, then the explicit 90-second bound. `set -e` prevented all later scheduler and
  artifact reads, so jobs 415/414 were neither queried nor changed. The existing outbound-network
  lesson applies; stop blind retries and retain the last verified Slurm evidence. EquiformerV2 was
  not accessed.

- 2026-09-12：开始四个本地训练单元的 `frequency × PG` 描述性统计；冻结使用
  manifest split 与模型相同 spglib 容差，并将 current-PG 样本平衡和 parent-DAG 实际路由
  负载分开解释。
- 2026-09-12：完成 29,478 条正式 split/本地成功结构的 32-PG 频数统计，发布 Markdown
  与机器可读 JSON；三个完整 train split 均严重不平衡，BEC 因仅 10 条本地样本不可判定。

- 2026-09-12：用户确认 Guqq 网络恢复，继续资源与 Slurm 阶段；按指示暂停
  EquiformerV2 checkpoint 下载，只推进十文件非门控资源及不依赖 Eq 的验收任务。
- 2026-09-12：Guqq 可达但服务器 GitHub HTTPS 仍报 GnuTLS `-110`；改用本地 committed
  `main` Git bundle 经 SCP 传入，再以该 bundle 的 `git pull --ff-only` 作为首个远程操作。
- 2026-09-12：23.2 MB complete-history bundle 的 SCP 被跳板中断，未采信远端副本；
  改为以服务器已有 `6395e15` 为 prerequisite 的最小增量 bundle，再由 Git pull 验证。
- 2026-09-12：6.9 KB 增量 bundle pull 成功并创建 staging；四个小资源上传成功，
  17.85 MB shard 中断。剩余资源改为 8 MiB 分块 SCP，重组后全量 size/SHA 原子提升。
- 2026-09-12：已传完 DPA4、JARVIS elastic、MatTen elastic 和 MACE 前六个 8 MiB 分块；
  MACE 第七块连续三次被跳板重置，按规范停止盲重试并补充经验，改为该块的 4 MiB
  唯一命名子块。远端部分文件继续视为不可信 staging，整文件哈希通过后才原子提升。
- 2026-09-12：非 Eq 分块已全部传至 staging，bundle pull 成功；整体验证在首个 dataset
  SHA 即因缩略记录转抄错误而 fail closed，未提升任何最终文件。现改用 committed manifest
  原值与本地独立哈希双重核对，再执行 all-verify-before-any-rename。
- 2026-09-12：十个非 Eq 数据/checkpoint 文件已全部通过 manifest size/SHA 并在 Guqq
  原子提升；下一阶段提交 full tests、fixture builder、三 standalone smokes 及排除
  `index % 4 == 3` 的两组 Slurm arrays。完整 BEC 仍因 12 GiB 余量不足不提交。
- 2026-09-12：Guqq 已提交 jobs `360`–`365`：full tests、MACE/GRACE/DPA4 standalone、
  fixture builder 及 12 个 non-Eq/non-BEC real rows；Eq 和未完成 BEC rows 均未提交。
  下一步提交依赖 `afterok:364` 的 44-row 非 Eq PG array 并监控全部终态。
- 2026-09-12：已提交 non-Eq PG array `366`（44 rows，`afterok:364`）。集群明确返回
  `Slurm accounting storage is disabled`，因此 strict `sacct` audit 无法执行；后续使用
  `scontrol` 与 stdout/stderr、JSON/JUnit 证据交叉监控，并显式保留此验收限制。
- 2026-09-12：`360/361` 已 `COMPLETED 0:0`；`362/363` 和多个 real rows 非零失败，
  row 10 通过，`364` 仍运行、`366` 等待依赖。因同类 real smoke 连续失败超过三次，
  已查阅 separate-runtime 经验并转入 JSON/JUnit/日志诊断，不盲目重提。
- 2026-09-12：诊断确认 real float32 frame 逆矩阵阈值过严、DPA4 neighbor schema 混用
  CPU/CUDA，以及 TensorFlow 2.20 不支持 RTX 5090 sm_120 CUDA kernel。先取消待运行的
  365/366 工作，再本地修复并以 targeted/full tests 验证后重提。
- 2026-09-12：364 fixture builder 已 `COMPLETED 0:0`；366 在依赖解除后已启动，现已取消
  余下任务。完成 dtype-aware frame tolerance、DPA4 schema device 归一和显式 GRACE
  TF-CPU fallback；targeted 42、完整 182 tests、compile/shell/diff checks 全部通过。
- 2026-09-12：`6e7043f` 已同步 Guqq，并提交最小修复探针 402（GRACE）、403（DPA4）、
  404（real 0–2）。初始均 pending；五项全部通过前不恢复大数组。
- 2026-09-12：real 0 已 `COMPLETED 0:0`，frame 修复成立；GRACE CPU fallback 越过 PTX
  后发现 `float_dtype` 字符串契约，DPA4 尚余一处 device mismatch。先读取精确 traceback，
  不重提或扩大数组。
- 2026-09-12：403 traceback 确认 DPA4 adapter forward 已通过，余错位于 e3nn validation
  matrix 的 CPU/CUDA 边界；404_2 则独立失败于 O(2) CG。Slurm 407 在 e3nn 0.5.9 下复现
  992/1000 组合失败，定位为 Wigner generator 默认 float32 污染。现已实现显式 float64
  generator、CPU-first smoke matrix 及 GRACE 字符串 dtype；targeted 51、full 194 tests 全绿。
- 2026-09-12：`5472bb2` 已推送并同步 Guqq，提交 408 GRACE、409 DPA4 与 410 real 0–2。
  后续三次监控连接均只返回跳板欢迎信息；已查阅既有网络 lesson 并停止盲重连，等待跨轮
  间隔后再收集终态。Eq checkpoint 与 Eq index 始终未访问。

- 2026-09-12：完成两项本地 DoD 补缺并推送后，登记一次有界 Guqq 资源同步恢复检查；
  仍以 HTTP/1.1 `git pull --ff-only` 为首个远程操作，失败即停止且不写资源。
- 2026-09-12：Guqq pull 已恢复并同步至 `6395e15`；远端现有 dielectric 文件大小不符
  manifest，未采信也未覆盖。登记同一持久 transport 下的逐文件临时上传、校验与原子替换。
- 2026-09-12：SSH control-master 握手被跳板重置且未写文件；调整为许可内的逐文件
  `.upload-part` SCP，之后在 pull-first 校验连接中统一核对 size/SHA 并原子替换。
- 2026-09-12：SCP staging 的 pull-first 连接再次以 GnuTLS `-110` 结束，guarded mkdir
  未执行、传输未开始；停止本轮即时重试，保留已核验 10-file contract 与原子上传方案。

- 2026-09-12：开始 efficiency report aggregation 单元；将 58/20 smoke 内嵌记录汇总为
  同源、稳定排序且 fail-closed 的 JSON/Markdown 交付物。
- 2026-09-12：完成严格 58/20 coverage 与 efficiency schema/numeric/CUDA/scope 校验、
  稳定 JSON/Markdown 渲染和 CLI；targeted 14、完整项目 178 tests 全绿。

- 2026-09-12：开始 runtime efficiency evidence 单元；补齐 README 已声明但代码缺失的参数、
  active experts、active downstream FLOPs、end-to-end latency 与 CUDA peak memory 报告。
- 2026-09-12：完成并接入两套 smoke 的 efficiency profiler；GRACE external frozen variables
  纳入 total parameter 口径，FLOPs 显式限定 active downstream scope。Targeted 16、完整 170 tests 全绿。

- 2026-09-12：开始 strict Slurm terminal-state audit 单元；将用 versioned job manifest 精确
  展开并核验全部单作业与 array tasks，替代无法形成完成证明的人工 `sacct` 浏览。
- 2026-09-12：完成 strict sacct CLI、manifest schema 与数组完整性校验；targeted 12、完整
  167 tests 全绿，CLI help/compile/diff 通过。真实 audit JSON 待 Guqq jobs 结束后生成。

- 2026-09-12：开始 required data-job evidence 单元；目标是让 32-PG fixture 构建和完整 BEC
  prepare/validate 作业也具备失败保留的 summary/JUnit、Git/env 与 Slurm-ID 证据。
- 2026-09-12：完成 32-PG builder 与 BEC prepare/validate job-level evidence；修正 BEC resume
  的旧错误污染并强制全索引计数。Targeted 27、完整 161 tests、scoped compile/shell/diff 全绿。
- 2026-09-12：间隔两个本地实现单元后做一次 pull-first 恢复检查，仍以 GnuTLS `-110`
  失败且未开始上传；遵守单次界限，本回合不再连接 Guqq。

- 2026-09-12：DoD 审计发现 standalone adapter jobs 缺少失败 JSON/JUnit 与独立 Git/env
  fingerprints，并发现 DPA4 smoke 的 source-layout 符号未导入；已冻结对应行为与静态测试。
- 2026-09-12：完成共用 fail-preserving adapter runner、四套 sbatch Git/env/JSON/JUnit 证据链
  与 DPA4 layout import 修复；targeted 17、完整 157 tests 全绿，compile/shell/diff checks 通过。

- 2026-09-12：开始 Guqq 资源同步单元；已复核三份 benchmark 数据和三套非门控骨干的
  10 个 contract 文件。首个 tar stream 在成功 pull 后被跳板重置；改为先 pull 的 multiplex
  transport 加逐文件原子 scp，且剔除未进入 GRACE loader contract 的额外 GMM artifact。
- 2026-09-12：资源同步后两次 pull 分别以 GnuTLS `-110` 和 GitHub 443 超时失败；连续
  三次未形成完整上传，已查阅 `lessons.md` 并暂停盲连。保留已验证本地资源和原子复制方案。

- 2026-09-12：用 scoped HTTP/1.1 恢复 Guqq pull，拦截 wheel-only 导致的旧 Hydra 回退；
  完成 Equiformer venv（fairchem-core 1.10.0、Torch 2.4.1+cu121、e3nn 0.5.9），
  `pip check` clean，freeze SHA 已记录，剩余 15 GiB。
- 2026-09-12：固化 Equiformer 119-entry exact lock 和 modern Hydra/OmegaConf guard；
  targeted 9 passed，完整本地 suite 153 passed、0 failed/skip，compile/diff checks 通过。

- 2026-09-12：Equiformer runtime 三次尝试均被强制 pull 的出站网络故障拦截，pip 未执行；
  本轮停止盲连。前三套环境及 locks 已验证，最后 runtime 与 Slurm evidence 尚待完成。

- 2026-09-12：完成 `requirements/guqq/dpa4.txt`：80 个唯一 exact pins 与 7 个关键版本
  断言；targeted 7、完整 151 tests 全绿，compile/diff 通过，准备 Equiformer runtime。

- 2026-09-12：跨回合间隔后 Guqq pull 恢复；完成 DPA4 venv（DeepMD-kit 3.2.0、Torch
  2.11.0+cu128、e3nn 0.5.9），`pip check` clean，freeze SHA 已记录，剩余 24 GiB。

- 2026-09-12：DPA4 安装三次均在强制 pull 阶段无输出终止，pip 未执行；按三次失败规则
  暂停本轮 Guqq 重连。MACE/GRACE 已验证，DPA4/Equiformer 与 Slurm evidence 仍待完成。

- 2026-09-12：完成 `requirements/guqq/grace.txt`：95 个唯一 exact pins 与 8 个关键版本
  断言；targeted 5、完整 149 tests 全绿，compile/diff 通过，准备 DPA4 安装。

- 2026-09-12：Guqq 间隔重试恢复；完成 GRACE venv（TensorPotential 0.6.0、TensorFlow
  2.20.0、Torch 2.11.0+cu128、e3nn 0.5.9），`pip check` clean，剩余 31 GiB。

- 2026-09-12：完成并验证 `requirements/guqq/mace-core.txt`：71 个唯一 exact pins、官方
  cu128 index、9 个关键版本断言；targeted 3、完整 147 tests 全绿，compile/diff 通过。

- 2026-09-12：GRACE 安装三次均由首个强制 pull 的 Guqq→GitHub TLS/443 故障安全拦截；
  环境未改变，停止盲目重连并启动已验证 MACE/core 完整 freeze lock 固化。

- 2026-09-12：完成 Guqq MACE/core venv：Torch 2.11.0+cu128、MACE 0.3.16、e3nn 0.4.4
  与冻结科学栈安装成功，`pip check` clean；安装后根文件系统尚余 41 GiB。

- 2026-09-12：MACE/core 首次依赖解析在安装前因 `python-hostlist` 无 wheel 而安全停止；
  同时识别 PyPI Torch 2.11 默认 CUDA 13 偏离冻结 CUDA 12.8，改用官方 cu128 wheel 索引。

- 2026-09-12：Guqq 已按规定先 pull；MACE/core、GRACE、DPA4、EquiformerV2 四套独立
  Python 3.10.12 venv 均已创建并确认不继承 system site packages，进入逐套安装阶段。

- 2026-09-12：官方 PyPI metadata 审计确认四 backbone runtime 不能共用一个 venv；启动
  `EXPERT_MACE_VENV|EXPERT_GRACE_VENV|EXPERT_DPA4_VENV|EXPERT_EQUIFORMERV2_VENV`
  Slurm contract 与 frozen array-index selector 实现。
- 2026-09-12：完成四 venv Slurm dispatch；58/20 rows 的 modulo-four mapping、缺变量失败、
  standalone/core launchers 与 shell syntax 均通过，完整项目 144 passed、0 skip。

- 2026-09-12：验收审计发现 global head 仍使用代表 Hall orientation，而非材料实际
  `SymmetryRecord.rotations`；启动 operation-aware Reynolds projector 修复与非代表 orientation
  回归测试，保持 BEC raw head 不依赖 symmetry projector。
- 2026-09-12：完成逐材料 operation-aware global projector；非代表 `mm2` orientation 与
  mixed-crystal readout 回归证明实际 fixed space、不变性、幂等和 backward，完整套件
  133 passed、0 skip，compile/diff checks 通过。

- 2026-09-12：开始 32-PG real-equilibrium fixture pipeline；将从已冻结 JARVIS/MatTen
  structures 经 Slurm 选择每点群一条，保存 Hall/source/checksum，并拒绝 synthetic prototype
  充当最终 fixture。保留同工作区 published benchmark 审计的全部并发改动。
- 2026-09-12：完成 equilibrium structure iterator、32-group deterministic fixture manifest、
  58-row real-checkpoint smoke、periodic graph/BEC joint audit 和完整 Slurm evidence wrappers；
  targeted 11 passed，完整本地套件 129 passed、0 skip，compile/sbatch syntax/diff checks 通过。
  未生成 synthetic acceptance asset；真实 fixture 与 GPU/Slurm 证据仍待 Guqq。

- 2026-09-12：开始 JARVIS tensor、MatTen elastic 与 JARVIS-DFPT BEC 的 published
  score 审计；将只从 primary sources 提取数值，下载公开来源到 `docs/`，并按数据与
  split 可比性生成表格。
- 2026-09-12：完成 `docs/benchmarks/README.md` 跑分汇总与 `SOURCES.md` 来源清单；
  下载 MatTen、CEITNet、JARVIS-DFPT 原文和 ALIGNN 官方 README 快照，区分原始报告、
  后续统一重跑与不同 target/split。链接、表格、PDF、checksum 和 whitespace 检查通过。

- 2026-09-11：开始 MACE/GRACE/DPA4/EquiformerV2 正式 adapters；测试计划冻结为
  manifest/checksum/runtime fail-closed、scalar readout 前真实 tap、统一 `O3FeatureBatch`、
  graph reuse、checkpoint frozen policy，以及 SO(3) 模型的双前向 parity Reynolds wrapper。
- 2026-09-11：完成 backbone 基础提交候选：补齐 MACE-MP medium-0b3 revision/size/SHA
  和四资源 feature/parity/graph metadata；实现严格 resource registry、周期反演双调用
  Reynolds wrapper 与 O(3) interface projector。目标 5 passed，合并 112 passed、1 旧 skip；
  四个真实 checkpoint feature taps 仍是下一实现单元。
- 2026-09-11：开始真实 MACE-MP medium-0b3 adapter；验收边界为严格 resource/runtime、
  first-interaction parity layout、MACE graph reuse、mixed-batch node mapping、冻结 backbone
  与仅 interface projector 可训练。真实 checkpoint 数值测试将通过 Guqq Slurm 执行。
- 2026-09-11：完成 MACE adapter 候选与 `mace_adapter_smoke.sbatch`；本地目标 7 passed、
  合并 114 passed、1 旧 skip。Guqq 首条仓库 `git pull` 等待 60 秒后中止，因此真实
  MACE Slurm 尚未运行；下一步先推送候选，再做有界 pull/环境检查/作业提交。
- 2026-09-11：开始 DPA4-Plus adapter；精确源码审计确认公开 sparse-edge builder、
  `src=neighbor/dst=center` 方向、e3nn-compatible packed real `(l,m)` order，以及 descriptor
  直接返回 scalar 前 final latent，避免依赖脆弱的 module forward hook。
- 2026-09-11：完成 DPA4-Plus adapter 候选、共享 parity edge-geometry 透传与
  `dpa4_adapter_smoke.sbatch`；目标测试 9 passed、合并 116 passed、1 旧 skip，compile/diff
  检查通过。真实 checkpoint 的 DeePMD 3.2/GPU 数值验收仍需在 Guqq 经 Slurm 执行。
- 2026-09-11：开始 GRACE adapter；测试计划先冻结 archive metadata、官方 restore、
  scalar readout 前 `rho` feature tap、真实 bond geometry、冻结梯度与 O(3) 数值验收。
- 2026-09-11：完成 GRACE adapter 候选；确认 `rho` 实为 17-channel scalar-only 并将 tap
  修正为 4512-component `AA`，实现 27 个 history copy 的显式 layout、runtime-derived
  TensorPotential→e3nn 基变换、artifact 逐文件 checksum 与 Slurm smoke。目标 11 passed、
  合并 118 passed、1 旧 skip；真实 checkpoint restore 尚待 Guqq。
- 2026-09-11：开始 EquiformerV2 adapter；先审计官方 fairchem checkpoint loader、最终
  node SO(3) coefficient tap 和 radius graph contract，同时保持 OMat24 gated resource fail closed。
- 2026-09-11：完成 EquiformerV2 adapter 候选；实现官方 checkpoint loader、final-normalized
  3200-component SO(3) tap、fairchem graph geometry 复用、paired-inversion O(3) completion、
  frozen/interface gradient smoke 与 Slurm 入口。目标 13 passed、完整项目 114 passed；受限
  checkpoint 的 checksum、restore 和数值验收仍待用户完成 OMat24 access 后在 Guqq 执行。
- 2026-09-11：开始四个独立 real-data modules；测试计划先冻结统一 tensor sample、三种
  source schema、published split、resource checksum、BEC site order 和 3/1/1 smoke selection。
- 2026-09-11：完成 real-data loader 候选；JARVIS dielectric/elastic、MatTen elastic 与
  JARVIS-DFPT BEC 统一为 immutable `TensorSample`，支持 frozen resource gate、published/
  seeded split、target irrep conversion、node order 和 3/1/1 smoke selection。目标 11 passed、
  完整项目 119 passed；三个本地真实源实读通过，完整 BEC output 继续 fail closed。
- 2026-09-12：开始 end-to-end model/smoke runner；先冻结 canonical frame target、backbone
  graph reuse、3/1/1 optimizer/checkpoint/metrics 和真实-vs-test-seam 边界，再实现 Slurm CLI。
- 2026-09-12：完成真实 backbone→原生 graph→dispatcher composition、canonical target batch、
  3/1/1 optimizer/checkpoint/metrics runner、20-run coverage schedule 与 Slurm array；定向
  4 passed、完整项目 123 passed。真实 jobs、GPU 指标与 checkpoint evidence 仍待 Guqq。

- 2026-09-11：恢复续跑并确认工作树 clean；开始 deterministic spglib
  canonicalization，实现范围包括显式 Hall setting、input/canonical frame、完整
  `(W,t)` operations 与 species-preserving audit permutations。
- 2026-09-11：canonicalization 已通过 6 项目标测试；测试曾捕获 improper 群操作
  被误强制为 det=+1 的错误，修复后 `m-3m` 保留完整 48 操作。随后开始 Hall-level
  physical parent embedding 的 affine group、species mapping、checksum 与 DAG 验证。
- 2026-09-11：完成 explicit-Hall spglib canonicalization 与严格 ParentEmbedding/DAG
  validation：frame/site/tensor round-trip、proper-input rotation covariance、完整
  `(W,t)` species-preserving permutations、affine group、Wyckoff split mapping、domain
  variants/checksum/cycle/connectivity 均有测试；完整本地结果 69 passed、1 个旧 skip。
- 2026-09-11：开始 Full-PG 数学 registry：逐 labelled O(3) copy 做实有限群
  commutant decomposition，生成可逆 subduction/copy provenance；finite-group CG
  使用 Hom-space Reynolds projector 并逐 operation 验证 intertwining。
- 2026-09-11：完成 Full-PG subduction/CG registry：symmetric commutant 确定性拆分
  全部实 irreducible copies，正交 inverse 保证 feature round-trip，Hom-space Reynolds
  生成允许/禁止 CG paths；32 群逐操作 tests、provenance、非平凡 paths 和 checksums
  全绿。完整本地结果 75 passed、1 个旧 opt-in skip。
- 2026-09-11：开始 Phase C `full_o3 | o2_tp` backends；Full-O(3) 使用 e3nn 固定
  convention，local-O(2) 将按 degree/copy triples 枚举完整 bandlimited Hom paths，
  并验证 global O(3)、local gauge、improper/parity 与 edge reversal。
- 2026-09-11：完成双 TP backend：`full_o3` 使用完整 e3nn paths；`o2_tp` 对每个
  degree/copy triple 以非混叠 O(2) quadrature 枚举完整 Hom paths 并保留独立权重，
  支持默认 `mmax=2` 与 full-m。目标测试 12 passed，完整本地 87 passed、1 个旧 skip。
- 2026-09-11：开始 shared adaptation、routed O3E、A1/Full-PG experts、continuous
  routing 与 O(3)-space fusion；测试覆盖双 backend placements、empty edges、O(3)/32-PG
  equivariance、两层权重独立、parent limits、path dedup、梯度与 `<5M`。
- 2026-09-11：完成 Phase C 核心 modules：adaptation/O3E 在完整 PBC edges 上放置
  独立选择的 TP backend；A1 与 Full-PG experts 各含两个独立 blocks，C1 可显式
  bypass；continuous residual gates 对 Hall active set 去重归一，fusion 回公共 O(3)
  space。目标测试 16 passed，完整本地 103 passed、1 个旧 skip。
- 2026-09-11：开始五分支 dispatcher 与 dielectric/elastic/BEC final readouts；将对
  冻结 26 configs 逐项验证 conditional construction、forward/loss/backward、active
  `<5M` 和 checkpoint，同时覆盖 mixed-size global/node scopes 与 PG constraints。
- 2026-09-11：完成统一五分支 dispatcher 与三类 final readout；global heads 做
  per-crystal pooling/PG fixed-space/decanonicalization，BEC 保留逐节点 raw 并独立给出
  ASR。修复 float32 fixed-space 判秩污染；4 项目标测试覆盖全部 26 configs 的
  forward/loss/backward/checkpoint 与 `<5M`，下一单元进入真实 backbone adapters。

- 2026-09-11：启动 `GOAL.md` Phase A 正式实现；审计确认仓库尚无 `src/` 包，
  本实现单元将建立规定模块边界、核心 typed contracts、五分支配置校验与冻结的
  26-config manifest。测试范围已先写入 `docs/agents/test.md`。
- 2026-09-11：完成首个 Phase A 实现单元：正式 `src/` 模块骨架和 README、
  `PeriodicGraph`/`O3FeatureBatch`/symmetry/parent DAG/`TensorPrediction` contracts、
  三类 target layouts、checkpoint convention checksum，以及与运行时枚举严格一致的
  26-config manifest；新增测试 15 passed，合并本地套件 27 passed、1 个旧 opt-in skip。
- 2026-09-11：开始实现正式 32 点群 registry、`l=0..4` O(3) 表示和 invariant
  subspace；Guqq 同步命令经 ProxyJump 仅返回 vlab 欢迎信息后悬挂，相关 SSH
  进程已终止，尚无 Guqq checkout 同步证据。
- 2026-09-11：完成正式 32 点群 registry 和 invariant-subspace 单元：构造时验证
  operation count/identity/inverse/closure，生成正交 Cartesian operations、O(3)
  表示、幂等 projector、确定性 basis 与 checksum；测试覆盖全部群和三类 targets，
  合并本地结果 31 passed、1 个既有 opt-in skip。
- 2026-09-11：开始 target Cartesian transforms 与 BEC controls 实现；测试范围已
  覆盖三类 target round-trip/O(3) covariance、32 群 fixed-space、ASR、optional
  joint projector 及两者交换性。
- 2026-09-11：完成三类 target 的高精度 Cartesian↔irrep/O(3) transforms、global
  PG fixed-space projection、de-canonicalization，以及 BEC 独立 ASR 和显式 optional
  joint projector；使用 float64 极分解修复 e3nn change-of-basis 的单精度遗留误差，
  未放宽测试标准。目标测试 8 passed，合并本地结果 39 passed、1 个既有 opt-in skip。
- 2026-09-11：开始 Phase B cutoff PBC graph constructor/collation；测试范围先冻结为
  image multiedges、完整等距 shell、skew cells、严格 boundary、mixed batches、空边
  fallback、device/dtype transfer 和 O(3) covariance。
- 2026-09-11：完成 Phase B PBC graph 基础：通过 reciprocal bounds 自适应枚举全部
  strict-cutoff image edges，保留 directed multiedges/cell shifts，支持 mixed-size
  collation、空边和统一 `.to()`；目标测试 6 passed，合并本地结果 45 passed、
  1 个既有 opt-in skip。
- 2026-09-11：开始四个独立训练单元的 config/split/normalizer 基础；测试计划已覆盖
  合法 pairing、seed 20260911、duplicate-group 防泄漏、5-group 3/1/1、train-only
  fitting、RMS/variance inverse 和 state compatibility。
- 2026-09-11：完成四个合法 `dataset × property` contract、seed 20260911 的
  duplicate-group-safe fallback split，以及只允许 train fitting 的 copy-aware
  coefficient normalizer；支持 RMS/variance、物理单位 inverse 和跨单元/layout
  fail-closed loading。目标测试 6 passed，合并本地结果 51 passed、1 个既有 skip。
- 2026-09-11：开始 copy-aware coefficient loss/metrics 与 checkpoint 实现；测试范围
  覆盖三 targets、normalizer weighting、optimizer update、输出 round-trip、原子保存
  和 architecture/unit/layout/convention 全部 fail-closed gates。
- 2026-09-11：完成逐 labelled irrep-copy coefficient MSE/raw MAE-RMSE、optimizer
  update 和 atomic checkpoint save/load；checkpoint 用安全 tensor-only load，且在
  模型/优化器 mutation 前严格验证 architecture、training unit、完整 conventions
  与 normalizer layout。目标测试 8 passed，合并本地结果 59 passed、1 个既有 skip。

- 2026-09-11：完成 `GOAL.md` 与 `assets/docs/subgroup_chain.json`：Goal 将完整模型验收冻结为四真实 backbone、五分支/26 合法配置、A1/Full-PG、full-O3/O2、三类性质、四独立训练单元、严格 `<5M`、全部等变/单元测试及 Guqq Slurm 两级 smoke；subgroup asset 保留 32 群、80 cover edges、433 oriented instances 和 222 maximal chains，并由新增测试验证。

- 2026-09-11：冻结 dataset × property 独立训练、官方 split/8:1:1 fallback、backbone graph/6 Å fallback 规则；定义 32 点群单结构 forward/backward 与每训练单元 5 结构 train/test 两级 smoke，并将 unnatural-parity carrier 改为逐 target 审计（仅 BEC 加 `1x1e`）。

- 2026-09-11：在 proposal 中明确 BEC 默认 forward 不输入 \(\pi_g\)、不依赖联合空间群 projector；\(\pi_g\) 仅用于图/等变性审计和非默认 hard-projection control，ASR 作为无需 \(\pi_g\) 的独立物理约束。

- 2026-09-11：修订 `proposal.md` 的正式任务范围：保留 `a1_only | full_pg` 两种 PG hidden 实现；将 JARVIS-DFPT BEC 纳入第一阶段 benchmark；增加 node-wise pooling/head 分流、BEC `1e` input carrier、PBC graph automorphism、联合等变性/可选投影/ASR 约束、32 点群 smoke tests 与参数预算重统计要求。

- 2026-09-11：启动 GRACE、DPA4/SeZM、EquiformerV2 checkpoint 与 JARVIS-DFPT
  BEC 数据准备；建立资源审计和测试记录。
- 2026-09-11：完成可追溯资源准备单元：新增 backbone/BEC manifests、可恢复并发
  下载与逐原子 BEC 提取器、流式校验汇总器及 Slurm 作业脚本；GRACE 和
  DPA4-Plus 已落盘，BEC 通过 10 个真实样本验证，EquiformerV2 明确记录为门控阻塞。
- 2026-09-11：经用户明确授权，将资源准备提交 `e65c0e0` 推送到
  `origin/main`；随后确认 Guqq SSH 在密钥交换前由服务端关闭，完整 BEC Slurm
  作业尚未提交，并按连续三次失败规则记录网络诊断经验。

- 2026-09-11：将 proposal 顶层架构收敛为五个固定分支：`B+R`、`B+A+R`、`B+A+O3E+R`、`B+PGE+R`、`B+A+PGE+R`；统一 PG expert 的 `a1_only | full_pg` 与 adaptation/O3E/readout 的 `full_o3 | o2_tp` 配置轴，并同步方法图、前向伪代码、参数口径、实验矩阵、phase/MVP 和总结。
- 2026-09-11：按用户要求复测 Guqq 连接；verbose SSH 与 `ssh-keyscan` 均确认
  TCP 22 可达但服务端在 SSH banner/key exchange 前主动断开，本地 alias、用户、
  端口和 identity 配置正确。故障继续定位为服务端 sshd/pre-auth 网络策略问题，
  `git pull` 和后续 Slurm 操作均未能执行。
- 2026-09-11：完成本机 SSH 跳板配置：备份用户级 config，为 `Host Guqq` 增加
  `ProxyJump vlab`，并通过 `ssh -G` 静态解析、vlab BatchMode 登录及 Guqq
  BatchMode 端到端状态 0 验收。首次 Guqq 远程操作按规范执行 `git pull`，但因
  默认登录目录不是 Git 仓库返回状态 1；SSH 免密链路本身已确认正常。
# 2026-09-12 — Full JARVIS backbone + direct-readout benchmark runner

- Added full published-split training for JARVIS dielectric and elastic with deterministic
  minibatches, train-only copy-aware normalization, AdamW, validation scheduling/early stopping,
  strict best-checkpoint restore, and original-frame Fnorm/EwT25/10/5 reporting.
- Added one-time frozen pre-interface feature extraction for all four adapters and tensor-only,
  atomic caches gated by backbone checkpoint SHA, training unit, split, exact sample IDs, and layout.
- Expanded readout edge harmonics to each target's maximum degree, restoring a legal elastic l=4
  path for the MACE l<=1 source tap. Added a non-Eq-only Slurm launcher while EqV2 is paused.
- Verification: benchmark targeted tests 5 passed; complete suite 199 passed; compile, CLI help,
  Slurm shell syntax, and diff checks passed. No full benchmark score has been claimed yet.
- Committed and pushed as `1e4958a`. Three subsequent pull-first Guqq attempts failed at the Vlab
  jump-host boundary, so probe/capacity gates could not be verified and no full training job was
  submitted; EquiformerV2 remained untouched.
# 2026-09-12 — Correct GMTNet elastic benchmark population

- Audited frozen official GMTNet source and found the existing 14,480-record manifest represented
  only its first magnitude screen, while the reported benchmark uses a second structural-symmetry
  forbidden-component screen before the seed-32 split and contains 14,220 records.
- Reproduced the official O(3)-probe support-mask algorithm, persisted compact per-record masks,
  applied the same zero projection in the loader, and made full benchmark training reject any split
  other than 11,376/1,422/1,422 for elastic (3,770/471/471 for dielectric).
- Added a CPU Slurm candidate-manifest job so server-side batch processing never edits tracked source.
  Verification: targeted 13 passed, full suite 202 passed, relevant compile/shell/diff checks passed.
  Full candidate generation awaits Guqq connectivity; no incomparable elastic job was submitted.
- Audited the official evaluation loop and corrected elastic Fnorm/EwT to contract predictions to
  the released 6x6 Voigt order before flattening, using the published `norm(label)+1e-5` denominator.
  The final regression state is 203 passed; another Guqq sync attempt failed at the same jump host.
- Pushed the protocol and metric fixes as `f209a1a` and `13c4485`. The final bounded connection was
  explicitly closed by the Vlab SSH endpoint, so no remote pull or job submission occurred.

# 2026-09-12 — Benchmark Goal externally blocked after third goal turn

- A third consecutive Goal-turn pull-first attempt again stopped at the Vlab→Guqq SSH boundary with
  no Guqq command output. No manifest, extraction, training, or evaluation job was submitted.
- All remaining acceptance work requires Slurm, so the active benchmark Goal is formally blocked
  rather than represented as complete. Resume from `main` after connectivity returns: pull first,
  collect 408–410, verify capacity, submit the elastic protocol candidate, then launch training.

# 2026-09-12 — Post-block recovery attempt 1

- The automatic Goal continuation triggered a fresh recovery audit. After correcting a local quoting
  error, a bounded pull-first SSH attempt remained at the Vlab banner for 60 seconds and was stopped;
  no Guqq command, scheduler query, or Slurm submission ran.
- The benchmark remains externally blocked. EquiformerV2 access stayed excluded.

# 2026-09-12 — Post-block recovery attempt 2

- A loop-free, fixed-argument pull-first SSH command again stopped after the Vlab banner. It produced
  no Guqq output for 60 seconds and was terminated; no remote mutation or Slurm submission is
  claimed.
- This is the second consecutive same-condition turn in the fresh recovery audit. Benchmark metrics
  remain unavailable and EquiformerV2 remains excluded.

# 2026-09-12 — Post-block recovery attempt 3 and renewed blocker

- Effective host configuration confirmed the intended Vlab proxy jump. Verbose evidence showed that
  both SSH authentications and Guqq command acceptance succeeded, followed by `Broken pipe` before
  mandatory pull output. Pull completion remains unknown and the guarded later commands have no
  execution evidence.
- This is the third consecutive same-condition recovery turn. All remaining benchmark work requires
  Slurm behind the mandatory pull gate, so the Goal is formally blocked again without any metric
  claim. EquiformerV2 remained untouched.

# 2026-09-12 — User-confirmed Guqq recovery

- The benchmark Goal is reactivated after the user confirmed network recovery. The next connection
  remains pull-first and will collect exact probe/capacity evidence before any Slurm submission.
- EquiformerV2 remains paused. Elastic training remains gated on generating, returning, validating,
  and committing the exact 14,220-record protocol manifest.
- The first recovery turn tried both non-PTY and PTY pull-first sessions. Each stopped at the Vlab
  banner without Guqq output and was boundedly terminated; no job was submitted. This starts a fresh
  blocked audit at attempt 1 while the Goal remains active.
- Guqq subsequently recovered. Persisted evidence proves GRACE 408, DPA4 409, and MACE B+R 410_0
  passed. Jobs 413 (elastic protocol) and 414 (MACE dielectric) were submitted; 414 is extracting
  features, while 413 exposed an e3nn 0.4.4/PyTorch 2.6 safe-load compatibility boundary.
- Implemented a narrow compatibility fix that allowlists only builtin `slice` before the standalone
  e3nn import. Fresh-process targeted tests passed 8/8; the project suite passed 204/204. Relevant
  compile, launcher syntax, and diff checks also passed.
- Pushed and synchronized `20ebc15`, then submitted replacement protocol job 415. It remained healthy
  beyond the former import failure; MACE dielectric job 414 reached train feature 250/3,770. Free
  space was 7.2 GiB and no EquiformerV2 resource was accessed.
- A single pull-first monitor followed 415 through 39:46 runtime with no traceback. During the same
  window, MACE job 414 advanced from feature 325 to 575/3,770 and free space remained near 7.0 GiB.
  Closing the monitor did not modify either Slurm job.

# 2026-09-13 — 网络故障恢复规范补充

- 在 `AGENTS.md` 中新增网络故障恢复顺序：SSH 成功连接后先执行 `bash net.sh`，等待 3 分钟，再执行 `git pull`、集群查询或任务提交等进一步操作。
- 同步澄清原有 pull-first 规则在网络故障场景下应位于恢复脚本和等待步骤之后；相关内容检索与 Markdown whitespace 检查通过。
# 2026-09-17 — Parent-DAG reduced benchmark started

- The accepted CGCNN full-PG run is confirmed to be current-group-only: its training path supplies
  neither `parent_dags` nor `parent_residuals`, and its summary records
  `routing=current_point_group_only`.
- Began a separate matched experiment rather than modifying the accepted model. The new path will
  derive material-specific Hall parents only from higher-symmetry detections of each actual
  structure, validate complete parent operations/species correspondence/checksums, and route with
  continuous geometric residual gates.
- Tests and module boundaries were recorded before implementation. No server connection, job, or
  new metric has been claimed in this stage.
- Implemented the material parent detector, versioned routing cache, batched DAG/residual plumbing,
  separate parent-DAG CLI and Slurm launcher, and four-model strict comparator. Existing CGCNN and
  DPA4 table rows now explicitly say `current-group only`; their numeric results are unchanged.
- Local verification passed: 32 focused tests and all 252 repository tests, plus Python compilation
  and Slurm shell syntax checks. No remote training metric is claimed yet.
# 2026-09-18 — Current-group-only curve requested

- Started a bounded visualization unit for CGCNN job 458. The plot will consume its recorded
  200-epoch summary, label the routing as `current-group only`, and show optimization/validation
  signals plus the learning-rate schedule. No model, split, prediction, or accepted metric changes.
- Retrieved and hash-verified the accepted job-458 summary, implemented a validation/rendering CLI,
  and generated deterministic SVG plus PNG assets. The benchmark report now embeds the SVG and
  states the source hash/best epoch. Visual QA and 12 focused/regression tests passed.
# 2026-09-18 — GMTNet overlay requested

- Began extending the accepted CGCNN current-group-only training figure with GMTNet job 443's native
  history. The comparison will use recorded values only and will not infer missing validation
  metrics from test artifacts.
- Completed the paired render and report update. It marks CGCNN epoch 159/MAE 4.3647 and GMTNet
  epoch 93/MAE 4.1133, distinguishes all recorded series, and states why GMTNet validation
  loss/Fnorm are absent. Deterministic rendering, visual QA, and 13 focused/regression tests passed.
# 2026-09-18 — Curve label refinement started

- Updating only the paired figure's CGCNN-facing display label to `current-pg`. No history,
  checkpoint, metric, model, or routing implementation changes.
- Completed the label-only rerender. All visible series/checkpoint labels and the paired title now
  use `current-pg`; focused tests pass 3/3 and visual QA passed.
# 2026-09-20 — Space-group stratified benchmark started

- The requested evaluation is now scoped to the frozen reduced dielectric-total split. Each source
  space group will receive train/validation/test counts and separate current-pg/GMTNet metrics,
  instead of only dataset-wide aggregates.
- The primary relationship analysis is predeclared as group-level Spearman/Pearson correlation
  between log10 training count and test error for groups with at least five test structures, with a
  >=10-test sensitivity check. Target magnitude is recorded to expose scale as a confounder.
- The curated JSONL already contains the required labels. GMTNet predictions are local; accepted
  current-pg per-sample predictions still need recovery before numerical results can be claimed.
- Implemented the strict join, per-group metrics/deltas, two correlation cohorts, target-scale
  diagnostic, deterministic CSV/Markdown/SVG writers, CLI, and Slurm entry point. Focused tests
  passed 24/24 and the complete repository suite passed 258/258. No numerical relationship is
  claimed until the immutable current-pg predictions are recovered and the real analysis runs.
- Pushed implementation commit `cde6a3c`. After the user-confirmed recovery, three compliant Guqq
  sessions each ran `net.sh` and waited at least three minutes, but their following HTTP/1.1 pulls
  timed out silently. The chained gates prevented path inspection and `sbatch`; no job ID or
  per-space-group result is claimed. `docs/agents/lessons.md` already records this exact three-strike
  outbound-GitHub failure and requires a meaningful interval or network change before retrying.
- A later renewed recovery confirmation did not change the observable Guqq-to-GitHub failure:
  `net.sh` and the wait succeeded, while the bounded pull produced no output and failed. The
  pull-first guard again prevented all scheduler actions.
- Bypassed only GitHub—not Git—using a hash-verified, chunked Git bundle. Guqq pulled commit
  `cde6a3c` from the bundle and completed Slurm job 463. Retrieved and verified JSON/CSV/Markdown/SVG
  results for all 69 test space groups, then promoted the full table, CSV, plot, and interpretation
  into the benchmark documentation.
- Returned to the active parent-DAG objective. Auditing job 462 and the separate parent launcher;
  no parent-DAG metric is accepted yet. A missing or incomplete old smoke will be replaced rather
  than inferred from scheduler disappearance.
- Smoke job 464 passed the fallback path but selected no actual parents. Full one-epoch preflight 465
  failed during train routing detection with `affine operation group is not multiplication closed`.
  The repair will reject only the invalid relaxed candidate, preserve all strict validation, search
  later candidates, and invalidate the old routing-cache convention before rerunning.
- Implemented the fail-closed candidate loop and bumped the routing convention to v2. Focused tests
  passed 34/34 and the complete suite 259/259; strict parent validation was not relaxed. Next step is
  commit, incremental bundle synchronization, and replacement full-data preflight.
- Implemented cached-Hall-first child reproduction and bumped the convention to v3. Focused tests
  passed 35/35 and the full suite 260/260. Next: commit/push, transfer a small incremental bundle,
  and rerun the same full-data one-epoch preflight.
- Committed the v3 fix as `951b682`, transferred its 1,144-byte incremental Git bundle by SCP,
  verified SHA-256 and prerequisite on Guqq, and pulled it into the tracked worktree. Submitted
  full-split one-epoch activation preflight job 467; no full training will be accepted or launched
  until its routing coverage and strict artifacts pass review.
- Job 467 failed after confirming nonzero parent selections: one cached float32 graph still could not
  reproduce its cached child point group. Per the three-failure lesson rule, reviewed and extended the
  parent-detection lesson. The next fix preserves fatal validation for inconsistent cache metadata but
  makes numerical child non-reproduction a current-only fallback for that material.
- Implemented convention-v4 per-material fallback with explicit cached Hall metadata validation.
  Focused tests passed 22/22 and the full project suite passed 262/262; the next gate is a small source
  commit, hash-verified SCP bundle synchronization, and the same one-epoch full-split preflight.
- Committed and pushed the verified v4 change as `a7437cc`; created an incremental bundle requiring
  `951b682`, verified its Git prerequisites and SHA-256 locally, and prepared direct SCP synchronization.
- SCP synchronization completed, Guqq verified and pulled `a7437cc`, and Slurm job 468 now runs the
  unchanged one-epoch full-split activation protocol. Formal training remains gated on its artifacts.
- Job 468 passed. Untruncated local artifact review confirms exact splits, zero-failure JUnit, finite
  metrics, and active parent routing in all three splits (46/4/6 samples). The cached routing can now
  support the formal 200-epoch run after the prediction-line count is checked remotely.
- Confirmed exactly 677 preflight predictions and submitted formal parent-DAG training as Slurm job 469
  with the matched default 200-epoch protocol. Monitoring and final strict comparison remain active.
- User superseded job 469's mechanism before completion. Audited the existing subgroup asset: it already
  contains 32 point groups and 80 parent-to-child maximal-cover class edges, so the new branch can use a
  validated transitive reverse lookup rather than online relaxed-spglib detection or `max_parents`.
- Cancelled obsolete job 469 after the mandatory recovery gate. Implemented the versioned offline DAG,
  PG-number cache, all-ancestor online routing/equal fusion, report/comparator identities, and module
  documentation. Expanded focused tests pass 55/55; full-suite verification is next.
- Full local verification passed 267/267 after compilation, Slurm syntax, and whitespace checks. The
  implementation activates 15 reachable expert classes globally and 1–11 deduplicated experts per
  retained PG sample, with the exact per-sample set determined only by its stored PG number.
- Pushed implementation `69f61fe` and formatting follow-up `52d0c88`. The only post-commit test hiccup
  was pytest system-temp permission during fixture setup; the same DAG test file passed 3/3 with a
  workspace basetemp, and the committed diff is whitespace-clean.
- Guqq verified/pulled the incremental bundle and submitted new-mechanism smoke job 470. This smoke uses
  only cached PG numbers and the offline DAG; it does not invoke material relaxed-Hall discovery.
- Smoke 470 passed with all 15 reachable expert classes instantiated and 1–11 experts activated per
  sample. Its routing identity and finite 7-sample metrics are correct; full-data preflight is next.
- Direct artifact parsing confirmed smoke JUnit and seven prediction rows. After replacing the brittle
  nested-shell grep gate, submitted full-split one-epoch static PG-DAG preflight as job 471.
- Full preflight 471 passed in 613.38 seconds with 677 unique, structurally valid all-ancestor prediction
  rows and zero JUnit failures. The class-DAG route is now accepted for formal 200-epoch training.
- Submitted the matched default 200-epoch static all-ancestor PG-DAG run as Slurm job 472. No protocol
  or hyperparameter differs from current-pg except the requested offline-DAG expert activation.
- Job 472 passed strict terminal validation: 200 contiguous epochs, best epoch 122, zero-failure JUnit,
  exact 5,001/637/677 splits, and 677 unique all-ancestor prediction rows. Starting the requested
  four-model comparison and deterministic curve/report unit; no accepted training artifact is changed.
- Completed result integration. Comparator 473 passed all four-model identity/target gates; the new
  curve overlays matched 200-epoch current-pg and parent-DAG histories and passed deterministic/visual
  QA. Parent routing modestly improves all five test metrics over current-pg but remains behind GMTNet.
  Focused tests pass 15/15 and the complete project suite passes 269/269.
# 2026-09-21 — Material-Hall path-weighted parent-DAG started

- Audited the current parent-DAG implementation: job-472's production path activates a deduplicated
  static PG ancestor closure and assigns every expert `1/N`; the separate material Hall contract
  retains residual gates but currently fuses its whole active set in one softmax and its detector
  emits only a star.
- Froze the replacement behavior and tests: enumerate material-specific current-to-root Hall paths,
  weight paths in proportion to node count and normalize across paths, apply residual gates inside
  each path, then sum duplicate destination contributions.  The reduced parent-DAG launcher will
  return to material Hall caches; static all-ancestor routing remains a labelled ablation.
- Implemented and locally accepted the replacement. Material discovery now transitive-reduces
  validated Hall operation sets into cover edges (routing convention v5); `ParentDAGSpec` exposes
  deterministic maximal paths; fusion applies length-normalized path priors and per-path residual
  gates, then combines duplicate PG destinations. The parent training CLI again builds material
  routing caches and records an unambiguous routing/path-fusion contract. Test totals: 17 focused,
  47 cross-module, and 272 complete project tests passed; compilation and diff checks passed.
- User corrected the algorithm before commit: the accepted target uses a stable offline full-path
  topology, concrete Hall/common-cell edge embeddings, incremental `H \\ K` residuals, analytic edge
  gates, and chain-wise stick-breaking. The just-tested node-softmax/multi-`symprec` route is now
  superseded and will not be committed or trained. Asset audit confirms the class/orientation layer
  exists but the physical Hall translation/origin embedding registry must be supplied separately;
  implementation will enforce that boundary fail closed.
