# GPU / server activity

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
