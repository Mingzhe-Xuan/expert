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
