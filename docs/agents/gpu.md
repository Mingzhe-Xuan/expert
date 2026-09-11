# GPU / server activity

## 2026-09-11 — Sync Phase A checkpoint and locate checkout

- Intended connection: `ssh Guqq` through the configured `vlab` jump host.
- Authorized purpose: synchronize pushed commit `eb6ad94`, identify the existing project
  checkout if its path is not already known, and perform lightweight `git`/Slurm status
  inspection only.
- The first attempted remote operation will be `git pull` in the most likely project
  checkout. If that path is absent, only read-only directory discovery will follow; a
  later connection will begin with `git pull` in the resolved checkout.
- No training, inference, evaluation, compilation, batch processing, or other
  computational workload will run on the login node.
- Result: the ProxyJump command reached the jump path and printed `Welcome to Vlab`, but
  produced no Guqq-side Git or checkout-discovery output within 30 seconds. The two SSH
  processes started by this attempt were terminated locally. No login-node computation,
  repository mutation, or Slurm submission occurred; commit `6ec34e2` remains pushed but
  is not yet proven synchronized to Guqq.

## 2026-09-11 — JARVIS-DFPT BEC preparation

- Intended connection: `ssh Guqq`.
- Authorized purpose: pull the committed dataset-preparation utility, download
  the official JARVIS-DFPT archives, and submit full BEC extraction/validation
  through Slurm.
- No training or batch processing may run directly on the login node.
- Expected source download: 5,000 archives, approximately 9.95 GiB total.
- Expected artifact to copy back: compact JSONL plus validation summary and
  error log; raw archives remain outside Git.

## 2026-09-11 — Retry after authorized push

- Intended connection: `ssh Guqq` after commit `e65c0e0` was explicitly
  authorized and pushed to `origin/main`.
- First remote operation: `git pull` in the project checkout.
- Purpose remains limited to cluster inspection, resource download, virtual
  environment preparation and Slurm submission for the JARVIS-DFPT BEC job.

## 2026-09-11 — SSH diagnostic retry

- Port 22 is reachable and the approved SSH configuration resolves to user
  `xmz`, host `211.86.155.221`, identity `~/.ssh/id_ed25519_codex`.
- Intended connection uses verbose transport diagnostics but still makes
  `git pull` the first remote operation; no compute runs on the login node.

## 2026-09-11 — User-requested Guqq connection diagnosis

- Intended connection: retry `ssh Guqq` to identify the current failure layer.
- Permission check: limited to `git pull` as the first remote command, SSH
  transport diagnostics, and lightweight cluster/status inspection if login
  succeeds; no training, inference, compilation, batch processing, or other
  compute workload will run on the login node.
- Diagnostic plan: use a bounded connection timeout and verbose SSH logging to
  distinguish local configuration, TCP reachability, key exchange,
  authentication, and remote-shell failures.
- Result: TCP connection to `211.86.155.221:22` was established, but the remote
  side closed immediately after the client version string and before sending
  its SSH banner; `git pull` therefore could not execute.
- Cross-check: `ssh-keyscan -T 10` received no host key/banner, while `ssh -G`
  still resolved the expected user `xmz`, port 22, and
  `~/.ssh/id_ed25519_codex` identity.

## 2026-09-11 — Guqq connection through vlab

- Intended connection: first validate passwordless access to `vlab`, then connect
  to `Guqq` through `ProxyJump vlab`.
- Permission check: SSH configuration and lightweight connectivity diagnostics
  only; if Guqq login succeeds, the first remote operation is `git pull`.
- No training, inference, evaluation, compilation, batch processing, or other
  compute workload will run directly on either login host.
- Result: passwordless BatchMode access to both `vlab` and `Guqq` succeeded.
  The first Guqq remote operation was `git pull`; it reached the remote shell
  but returned `not a git repository` because the login directory is not the
  project checkout. A subsequent no-op `true` command exited with status 0.

## 2026-09-11 — Locate checkout and prepare real MACE Slurm smoke

- Intended connection: `ssh Guqq` through the validated `vlab` jump after local commit
  `b108a7f`; first remote operation remains `git pull`, followed by lightweight checkout,
  environment and Slurm status discovery only.
- Permission check: no model loading, inference, compilation or tests on the login node.
  Once the repository and venv are known, the real MACE checkpoint feature test will be
  submitted through Slurm; resource/checkpoint transfer may use `scp` if required.
- Result: connection succeeded through `vlab`; the first home-directory `git pull` failed as
  expected because `/home/xmz` is not a checkout. Lightweight discovery located the repository
  at `/home/xmz/expert`; Slurm reports the `compute` partition.

## 2026-09-11 — Inspect Guqq project environment for MACE smoke

- Intended connection: first run `git pull` inside `/home/xmz/expert`, then inspect only Git
  status, existing virtual environments, Python/package metadata, checkpoint paths and Slurm
  status needed to construct the real MACE job.
- Permission check: all actions are lightweight login-node management; no import-heavy model
  load, inference, test suite, compilation or batch processing will run outside Slurm.
- Result: SSH reached Guqq, but the required first command `git pull` inside `/home/xmz/expert`
  produced no response for 60 seconds and was interrupted. No later inspection command ran and
  no login-node compute was performed. Retry will use a bounded Git timeout after the candidate
  adapter is pushed.

## 2026-09-11 — Pull MACE candidate and inspect runtime

- Intended connection: enter `/home/xmz/expert` and make bounded `git pull` the first remote
  operation after pushing commit `804e391`; then inspect commit status, venv metadata, checkpoint
  presence and Slurm GPU resources.
- Permission check: inspection and dependency metadata only on the login node. The adapter import,
  checkpoint load, feature extraction and equivariance/backward smoke will run only via Slurm.
- Result: the jump host connected, but the Guqq leg closed before a remote shell was established;
  therefore no command, including the planned first `git pull`, executed. No immediate blind retry;
  local work continues with exact-version source/API audit.

## 2026-09-11 — Bounded retry for MACE candidate sync

- Intended connection: after a pause, retry the Guqq leg; inside `/home/xmz/expert`, run
  `timeout 30s git pull` first, then only inspect HEAD, existing venv/package metadata,
  checkpoint availability and exact Slurm GRES configuration.
- Permission check: no checkpoint import or inference on the login node. A successful inspection
  will be followed by a separate recorded connection that submits the smoke via `sbatch`.
- Result: the jump host connected, but the Guqq leg again closed before a remote shell was
  established. The required first `git pull` therefore did not execute, and no remote command or
  login-node compute ran. Further implementation continues locally before another recorded retry.

## 2026-09-11 — Sync DPA4 adapter and inspect Slurm runtime

- Intended connection: after pushing commit `607c057`, connect through the configured vlab jump,
  enter `/home/xmz/expert`, and run `timeout 30s git pull` as the first substantive remote operation.
- Authorized purpose after a successful pull: inspect HEAD, existing virtual-environment package
  metadata, local MACE/DPA4 checkpoint presence, filesystem capacity, and Slurm GPU resources.
- Permission check: these are lightweight login-node management operations only. Checkpoint load,
  feature extraction, equivariance checks, and backward passes will be submitted with `sbatch`.
- Result: the client received only the vlab jump-host welcome line and then exited with status 1.
  No Guqq shell output was received, so there is no evidence that the required `git pull` or any
  later inspection command executed. No login-node compute ran; another blind retry is deferred.

## 2026-09-12 — Sync end-to-end runner and inspect acceptance prerequisites

- Intended connection: enter `/home/xmz/expert` through the configured jump host and execute
  `timeout 30s git pull` as the first substantive remote operation, synchronizing commit
  `0eb1d89` before any inspection.
- Authorized purpose after the pull: inspect HEAD, Slurm availability, existing project virtual
  environments, checkpoint/data presence, and filesystem capacity needed for the three acceptance
  suites. These are lightweight login-node management checks only.
- Permission check: no checkpoint loading, inference, tests, compilation, extraction, or batch
  processing will run on the login node. Any such work will be submitted through `sbatch` after a
  separate recorded submission step.
- Result: the safely quoted bounded SSH call returned only `Welcome to Vlab` and completed without
  any Guqq shell output. Consequently there is no evidence that the required first `git pull`, HEAD,
  `sinfo`, or disk check executed. No remote compute ran; the repeated jump-host-to-Guqq failure
  remains external while local fixture/runner implementation continues.
