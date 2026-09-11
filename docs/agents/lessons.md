# Agent lessons

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
