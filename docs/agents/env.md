# Guqq environments

## 2026-09-13 — GMTNet reduced-benchmark environment plan

- Planned path: `/home/xmz/expert-envs/gmtnet-py310`, created with
  `python3 -m venv /home/xmz/expert-envs/gmtnet-py310` before installing packages.
- The environment is independent from the four backbone environments because official GMTNet adds
  PyTorch-Geometric compiled extensions, pymatgen, JARVIS tools and pandas and must not perturb their
  frozen dependency locks.
- Official source path: `/home/xmz/expert-data/vendor/GMTNet`, pinned and runtime-verified at commit
  `7a606a459ee48a320ed38450e391811fb43d5e19`; source remains outside this repository.
- Required runtime families: CUDA PyTorch, e3nn, torch-geometric/torch-scatter, pymatgen, jarvis-tools,
  pandas, spglib and NumPy. Exact installed versions and `pip freeze` SHA-256 will be appended after
  installation and before the first GMTNet Slurm smoke.

## 2026-09-12 — Acceptance environment plan

- MACE/core path: `/home/xmz/expert-envs/acceptance-py310` (created and verified after the
  intermittent SSH retry; retained rather than needlessly replacing the empty task-specific venv).
- GRACE path: `/home/xmz/expert-envs/grace-py310`
- DPA4 path: `/home/xmz/expert-envs/dpa4-py310`
- EquiformerV2 path: `/home/xmz/expert-envs/equiformerv2-py310`
- Creation command pattern: `python3 -m venv /home/xmz/expert-envs/<name>`
- Isolation: `include-system-site-packages = false`; the path is task-specific and must be absent
  before creation, so no unrelated environment is overwritten.
- Initial runtime expectation: Guqq `/usr/bin/python3` (Python 3.10.x), with pip supplied by `venv`.
- Creation status: all four task-specific venvs exist with Python 3.10.12, pip 22.0.2,
  `home = /usr/bin`, and `include-system-site-packages = false`.
- Installation status: MACE/core is installed and `pip check` is clean: Torch 2.11.0+cu128,
  CUDA toolkit 12.8.1, e3nn 0.4.4, MACE 0.3.16, NumPy 1.26.4, SciPy 1.15.3, spglib 2.6.0,
  ASE 3.26.0, matscipy 1.1.1, pytest 8.4.2, and python-hostlist 2.3.0. The complete verified MACE snapshot is committed as
  `requirements/guqq/mace-core.txt`.
- GRACE is installed and `pip check` is clean: TensorPotential 0.6.0, TensorFlow 2.20.0,
  Torch 2.11.0+cu128, CUDA toolkit 12.8.1, e3nn 0.5.9, NumPy 1.26.4, SciPy 1.15.3,
  spglib 2.6.0, ASE 3.26.0, and pytest 8.4.2. Its freeze SHA-256 is
  `8e07b1621156789b07e806a8e9fc4645fc2da757ec019eb7c9c723714ec84765`.
- DPA4 is installed and `pip check` is clean: DeepMD-kit 3.2.0, Torch 2.11.0+cu128,
  CUDA toolkit 12.8.1, e3nn 0.5.9, NumPy 1.26.4, SciPy 1.15.3, spglib 2.6.0, and ASE 3.26.0.
  Its freeze SHA-256 is `92a818e296e05dd0fe4a3d50cf3cb0fe6fbcdb538a2eefbcab0378ad44eb7f8b`.
- EquiformerV2 is installed and `pip check` is clean: fairchem-core 1.10.0, Torch 2.4.1+cu121,
  e3nn 0.5.9, Hydra 1.3.2, OmegaConf 2.3.0, NumPy 1.26.4, SciPy 1.15.3, spglib 2.6.0,
  ASE 3.26.0, and pytest 8.4.2. Its freeze SHA-256 is
  `acf2cb6c7f575339e392ec45337b8015313dc8e7d0ec3f73e08a97f5d60efa76`.
- All four runtime installations are dependency-clean. Exact installed versions,
  scientific stack,
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

### Frozen incompatibility boundary

Official PyPI `Requires-Dist` metadata makes a single combined venv invalid:

- MACE 0.3.16 requires `e3nn==0.4.4`;
- fairchem-core 1.10.0 requires `e3nn>=0.5`, `torch~=2.4.0`, and NumPy 1.26.x;
- DeepMD 3.2.0 `[torch]` requires `e3nn>=0.5.9` and `torch==2.11.0`;
- TensorPotential 0.6.0 installs a TensorFlow CUDA stack independently.

The Slurm contract therefore exposes `EXPERT_MACE_VENV`, `EXPERT_GRACE_VENV`,
`EXPERT_DPA4_VENV`, and `EXPERT_EQUIFORMERV2_VENV`. Mixed arrays select the corresponding
environment from the frozen backbone order; no dependency solver override may weaken these pins.
