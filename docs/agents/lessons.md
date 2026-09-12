# Agent lessons

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
