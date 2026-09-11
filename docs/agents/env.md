# Guqq environments

## 2026-09-12 — Acceptance environment plan

- Path: `/home/xmz/expert-envs/acceptance-py310`
- Creation command: `python3 -m venv /home/xmz/expert-envs/acceptance-py310`
- Isolation: `include-system-site-packages = false`; the path is task-specific and must be absent
  before creation, so no unrelated environment is overwritten.
- Initial runtime expectation: Guqq `/usr/bin/python3` (Python 3.10.x), with pip supplied by `venv`.
- Creation status: unconfirmed. Three post-plan connections ended at the Vlab jump without Guqq
  shell output, so no claim is made that the directory exists.
- Installation status: not started. Exact installed versions, CUDA/PyTorch build, scientific stack,
  four backbone runtimes, source revisions where wheels are unavailable, and `pip freeze` fingerprint
  must be appended here after installation and before any acceptance job is submitted.
- Resource status: MACE, GRACE and DPA4 artifacts exist locally and may be transferred by `scp` after
  checksum verification; EquiformerV2 remains blocked on the user-gated OMat24 license/access flow.
- Local pre-transfer audit: the three equilibrium datasets and MACE/GRACE/DPA4 artifacts match the
  SHA-256 prefixes and exact byte sizes frozen in their manifests. The local test environment that
  passed 129 tests uses Torch 2.11.0, e3nn 0.4.4, spglib 2.6.0, ASE 3.26.0, MACE 0.3.16 and
  fairchem-core 2.3.0; it lacks TensorPotential and DeepMD. This is audit input, not the Guqq lock:
  Equiformer acceptance remains pinned to fairchem-core 1.10.0 and GRACE/DPA4 require their recorded
  runtimes before jobs may be submitted.
