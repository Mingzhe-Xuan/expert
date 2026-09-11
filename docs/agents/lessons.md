# Agent lessons

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
