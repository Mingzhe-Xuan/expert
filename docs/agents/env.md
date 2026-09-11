# Guqq environments

## 2026-09-12 — Acceptance environment plan

- Path: `/home/xmz/expert-envs/acceptance-py310`
- Creation command: `python3 -m venv /home/xmz/expert-envs/acceptance-py310`
- Isolation: `include-system-site-packages = false`; the path is task-specific and must be absent
  before creation, so no unrelated environment is overwritten.
- Initial runtime expectation: Guqq `/usr/bin/python3` (Python 3.10.x), with pip supplied by `venv`.
- Installation status: not started. Exact installed versions, CUDA/PyTorch build, scientific stack,
  four backbone runtimes, source revisions where wheels are unavailable, and `pip freeze` fingerprint
  must be appended here after installation and before any acceptance job is submitted.
- Resource status: MACE, GRACE and DPA4 artifacts exist locally and may be transferred by `scp` after
  checksum verification; EquiformerV2 remains blocked on the user-gated OMat24 license/access flow.
