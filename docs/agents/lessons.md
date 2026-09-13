# Agent lessons

## 2026-09-13 — Long-tail feature extraction needs fine-grained recovery units

- Equal record counts do not imply equal accelerator work for periodic equivariant models. Crystal
  neighbor topology and model-internal work can make equally sized shards differ by more than 2x,
  so a job gated on the slowest monolithic shard can exceed its walltime even when aggregate GPU
  utilization is high.
- Separate cache partition count from concurrent worker count. Use many deterministic, provenance-
  checked cache partitions to bound imbalance and preserve partial progress, but keep only enough
  persistent workers to saturate the GPU. Workers must dynamically claim the next unclaimed
  partition; fixed round-robin assignment can still strand a slow worker while others go idle.
  Restore the original manifest order only after validating exact coverage across all partitions.

## 2026-09-13 — Model cache partition identity in the cache schema

- If feature extraction is partitioned, do not encode shard identity only in directory names or
  bypass the cache's split validation. Extend the schema validator to represent both the canonical
  split role and a bounded `shard_index/shard_count` partition, and require the loader to match that
  full identity together with the exact ordered sample IDs.
- Exercise the actual cache serializer in the shard test. Testing only partition arithmetic and merge
  logic can miss a boundary mismatch where extraction succeeds but every worker fails on publication.

## 2026-09-13 — Layered virtual environments require ABI-level validation

- A `.pth`-layered environment must put the GPU-compatible PyTorch installation before environments
  that contribute pure-Python packages such as PyG or pymatgen. Import success alone does not prove
  CUDA compatibility; submit a minimal Slurm GPU smoke and execute a real CUDA tensor operation.
- Never expose an entire user site merely to satisfy a missing lightweight dependency: stale compiled
  extensions such as `torch_scatter` can shadow compatible fallbacks and fail with an `OSError`, not
  only `ImportError`. Link the smallest package set (including matching `.dist-info`) and make optional
  compiled-extension fallbacks tolerate loader/ABI errors.
- Run `pip check` after constructing the layers, then record a fresh freeze and checksum after every
  ordering change so the effective environment remains auditable.

## 2026-09-13 — JARVIS/VASP elastic Voigt ordering

JARVIS `elastic_total_kbar` retains VASP's component order `XX, YY, ZZ, XY, YZ, ZX`, which differs
from the commonly assumed standard Voigt order `xx, yy, zz, yz, xz, xy`. Directly expanding a
JARVIS matrix with the standard index map silently permutes all shear components. This can survive
shape/symmetry tests but causes false crystal point-group violations (about 3.9% relative residual
for the audited hexagonal JVASP-20502 example). Reorder both matrix axes by `[0, 1, 2, 4, 5, 3]`
before converting to a Cartesian fourth-rank tensor; regression tests should use distinct shear
diagonal values so the convention cannot pass accidentally.

## 2026-09-13 — Symmetric tensor datasets may contain material antisymmetric labels

- Do not assume a dielectric dataset is numerically symmetric because the property and model output
  are symmetric by construction. DTNet's published MP file contains electronic/total tensors with
  substantial antisymmetric components, while its official readout explicitly symmetrizes outputs.
- For an irrep target containing only `0e+2e`, preserve the raw label for audit, record the
  antisymmetric residual, and explicitly define the training label as `(T + T^T)/2`. Silently dropping
  entries or applying a tiny-tolerance rejection would either lose many official samples or obscure
  a real source-data convention.

## 2026-09-13 — Pin pytest import and temp roots on this Windows workspace

- `uv run pytest` may resolve an entry point whose import path omits the repository root; invoke
  `uv run python -m pytest` with `PYTHONPATH=.` for repository tests.
- The user-level `C:\Users\asus\AppData\Local\Temp\pytest-of-asus` and stale workspace pytest
  directories can be unreadable even when the test itself is valid. Give each acceptance run a new,
  task-scoped `--basetemp` below ignored `results/` and disable the cache provider with
  `-p no:cacheprovider`.
- Treat collection/import/temp-root failures as environment failures, not product-test failures, and
  retain the last successful code-level suite until the invocation is corrected.

## 2026-09-11 — Verify the angular support of a named readout input

- A configuration field such as `allowed_l_p` states what an instruction may support, not what a
  particular checkpoint actually emits. GRACE's `rho` lists degrees beyond zero but its
  `ls_max=[0,0,0,0]` makes it an invariant-only, 17-channel scalar tensor.
- Freeze a backbone tap from the instantiated instruction metadata: inspect `only_invar`, `lmax`,
  every `(l,m,parity,history)` row, and its downstream graph position. Here the correct common
  equivariant tap is `AA`, before the `AA1` and `AA2` higher-order scalar branches.
- Distinct real-spherical-harmonic implementations need an explicit per-degree change of basis.
  Derive it from deterministic samples, require an orthogonal map and a tight reconstruction
  residual, then apply it independently to every labelled copy.


## 2026-09-11 — Periodic inversion for SO(3)-to-O(3) feature projection

- Do not negate the Cartesian cell to construct the inversion partner: that changes its
  handedness and can send a backbone down a different cell convention. Keep the same cell and
  map fractional sites as `(-s) mod 1`, preserving the original species/node order explicitly.
- Parity projection must combine two actual frozen-backbone evaluations of the paired structures.
  Relabelling one SO(3) feature tensor with even/odd metadata provides no reflection guarantee.

## 2026-09-11 — Determine symmetry fixed-space rank in reference precision

- A point-group Reynolds projector may have forbidden eigenvalues at float32 round-off scale.
  Applying a tighter rank threshold directly in model precision can promote those residues to
  basis vectors and silently reintroduce forbidden tensor components.
- Treat the fixed-space basis as convention metadata: determine its rank and deterministic basis
  in float64, then cast the completed basis to the model tensor's dtype/device for projection.

## 2026-09-11 — Preserve determinant when repairing O(3) operations

- A polar/SVD cleanup used for numerical rotation matrices must not always force
  determinant `+1`. Doing so is valid for an oriented canonical frame, but corrupts an
  O(3) point group by mapping reflections/inversion into proper rotations.
- Keep two explicit policies: canonical frame repair may request `force_proper=True`;
  symmetry-operation repair must preserve the determinant sign of its input. A group-order
  assertion (e.g. `m-3m` has 48 unique operations) catches this error reliably.

## 2026-09-11 — e3nn 0.5 with PyTorch 2.6+ and float64 tensor transforms

- PyTorch 2.6+ defaults `torch.load` to `weights_only=True`; e3nn 0.5 packaged Wigner
  constants contain Python `slice` objects. Permit only builtin `slice` with
  `torch.serialization.add_safe_globals` before importing e3nn; do not disable safe
  loading globally.
- e3nn `CartesianTensor` can produce a float64 `change_of_basis` whose values were first
  generated through default-float intermediates. Casting does not remove the resulting
  roughly `1e-7` round-trip error.
- For float64 reference tests, compute the closest orthonormal basis via polar/SVD repair
  once and induce coefficient representations from that same repaired basis. Using an
  independently generated representation with the repaired basis introduces a subtle
  convention mismatch, especially for repeated elastic irrep copies.

## 2026-09-11 — SSH closes before key exchange

- Symptom: TCP port 22 succeeds, but OpenSSH reports
  `kex_exchange_identification: Connection closed by remote host` immediately
  after sending the local version string.
- Diagnosis: `ssh -G Guqq` in the approved host environment confirmed user
  `xmz`, host `211.86.155.221` and the intended Ed25519 identity. `ssh -vv`
  showed that the connection closes before authentication or key exchange.
- Consequence: changing repository paths, keys, Slurm scripts or application
  code cannot fix this class of failure. Do not repeatedly submit connections
  once the pre-authentication close is established.
- Recovery: wait for the SSH service/network policy to recover, or ask the
  server administrator to inspect `sshd` availability, `MaxStartups`, source-IP
  filtering and tools such as fail2ban. On the next successful connection,
  still make `git pull` the first remote operation.
# 2026-09-12 — Point-group acceptance must use detected operations

- A crystallographic point-group symbol does not by itself freeze the orientation of a material's
  Hall setting. Acceptance checks for forbidden tensor components must construct representations
  from the fixture's spglib-detected Cartesian operations; using only the registry's representative
  Hall group can silently test a differently oriented fixed subspace. The model's global readout
  projector must use those same per-material operations, not merely leave them to the audit layer.
- Slurm smoke commands are not self-auditing merely because they return nonzero on failure. Each
  command must write failure JSON and JUnit before re-raising, and the aggregate pytest wrapper must
  convert skip/xfail counts into a nonzero acceptance result.

## 2026-09-12 — Frozen backbone runtimes require separate environments

- MACE 0.3.16 pins e3nn 0.4.4, while fairchem-core 1.10 requires e3nn 0.5+ with Torch 2.4 and
  DeepMD 3.2 requires e3nn 0.5.9+ with Torch 2.11. These constraints cannot be faithfully resolved
  in one venv. Mixed-backbone Slurm arrays must select a recorded per-backbone environment from the
  frozen schedule; dependency overrides or a shared environment would invalidate checkpoint evidence.

## 2026-09-12 — Gate remote mutations behind the mandatory pull

- Guqq may accept SSH while its outbound GitHub connection fails with GnuTLS termination or port-443
  timeout. Keep `set -e` and the mandatory `git pull --ff-only` before package or job commands so a
  stale checkout cannot proceed accidentally.
- After three consecutive outbound failures, stop blind SSH retries. Record the unchanged environment,
  prepare reproducible commands locally, and retry only after a meaningful interval or network change.
- When default Guqq HTTPS pulls repeatedly fail with GnuTLS `-110` or silent timeouts, a scoped
  `git -c http.version=HTTP/1.1 pull --ff-only` can recover without changing persistent Git config.
  GitHub SSH is not a fallback on this host because it has no accepted GitHub public key.
- A Vlab banner alone is ambiguous. Use one bounded `ssh -vv` diagnostic after spaced failures to
  distinguish jump authentication, target authentication, command acceptance, and command-channel
  transport. If both hosts authenticate and Guqq accepts the command but the channel breaks before
  pull output, record pull completion as unknown. Keep every later command behind `&&`; do not bypass
  the mandatory pull with a separate scheduler or submission connection. A later recovery starts
  with another idempotent `pull --ff-only`.

## 2026-09-12 — Wheel-only resolution can silently select obsolete pure-Python dependencies

- `--only-binary=:all:` caused fairchem's unconstrained Hydra dependency to backtrack from modern
  Hydra/OmegaConf to `hydra-core==0.11.3` and `omegaconf==1.4.1`, because the modern chain includes
  a source-only pure-Python antlr runtime. A dependency-clean solve is not sufficient if it violates
  the runtime generation expected by the frozen backbone release.
- Pin Hydra 1.3.2 and OmegaConf 2.3.0 explicitly, allow only their pure-Python antlr4 packaging
  exception, then retain wheel-only policy for native dependencies and finish with `pip check`.

## 2026-09-12 — Vlab jump-host resets during SCP payloads

- A successful SSH handshake and several successful chunks do not guarantee that a later SCP
  payload will finish: the jump path can reset mid-transfer and leave a partial remote filename.
- After the same 8 MiB chunk fails three consecutive times, stop blind retries and subdivide only
  that chunk under a distinct prefix. This preserves already completed chunks while shortening the
  failure window without confusing a partial file with accepted input.
- Treat staging files as untrusted until reconstruction into a new temporary path passes both the
  frozen byte count and SHA-256. Only then atomically rename it to the manifest path; incomplete or
  superseded chunks may be removed only after the verified promotion succeeds.
- Never promote from a digest copied out of an abbreviated progress summary. Read exact values from
  the committed machine-readable manifest (and independently recompute them locally) when building
  the final server-side check. A guarded all-files-before-any-rename sequence kept this transcription
  error from publishing even one unverified resource.

## 2026-09-12 — GPU smoke paths must respect dtype, device, and runtime kernel boundaries

- An orthogonal frame produced in float64 and cast to float32 cannot be validated with a fixed
  `1e-8` inverse tolerance. Scale the structural invariant check from the tensor dtype epsilon while
  retaining a strict floor for float64, and test that a materially perturbed inverse still fails.
- Third-party neighbor-list builders may return host tensors even when their model is on CUDA.
  Normalize indices, vectors, and masks to the model device (and their contract dtypes) before both
  descriptor calls and geometry arithmetic.
- TensorFlow 2.20 cannot execute GRACE kernels on Guqq's RTX 5090 compute capability 12.0; its PTX
  fallback fails before checkpoint inference. Use an explicit, reported TensorFlow-CPU fallback for
  this frozen runtime while keeping the trainable PyTorch interface on the requested CUDA device.
- e3nn 0.5.9 constructs SO(3) generators using the process-wide default dtype before multiplying
  them by caller-provided angles. Passing float64 rotations alone therefore does not guarantee a
  float64 representation when the default is float32; Reynolds projectors can turn that error into
  spurious finite-group paths. For convention-defining O(2) CG construction, generate the real-basis
  Lie algebra explicitly in float64 and test the sampled group law across pinned runtime generations.
- e3nn representation checks should construct `D_from_matrix` from a CPU rotation and only then cast
  the finished matrix to the feature device/dtype. This avoids older/newer e3nn internal CPU constants
  colliding with CUDA inputs while preserving a device-matched comparison.
