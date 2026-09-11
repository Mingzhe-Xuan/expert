# Test plan and results

## 2026-09-11 — Backbone/BEC resource preparation

计划检查：

- 下载文件大小和发布方 checksum（若提供）一致；另记录 SHA-256。
- checkpoint 可由其官方包或安全的 metadata-only 方法识别，不执行大规模推理。
- 每个 backbone 记录具体模型版本、feature-tap 候选、连续群/parity 约定和元素覆盖。
- BEC 数据逐样本具有结构、与原子数一致的 N x 3 x 3 tensor、稳定样本 ID；统计
  缺失值、形状错误和 split 情况。
- raw 数据、checkpoint 和克隆的上游仓库均被 Git 忽略。

实际结果：

- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_prepare_jarvis_bec.py -q`：
  2 passed。
- 10 个真实 JARVIS-DFPT archive 下载、MD5 校验和 XML 提取：10/10 成功，
  0 个提取错误；原子数范围 3–27，结构站点数与 BEC 第一维全部一致。
- 10 样本 summary validator：`valid=true`、无重复 ID、无非有限值或形状错误。
- JVASP-29884 单样本：14 个原子，BEC 形状 14 x 3 x 3，ASR Frobenius
  residual 为 3.1623e-8 e。
- GRACE 文件 85,894,694 bytes，SHA-256
  `19a006ab5668e6aa3426de6a78ee4a9a9da373680ec9a6171787b0259a67bc20`，
  与官方 LFS checksum 一致。
- DPA4-Plus 文件 35,550,605 bytes，SHA-256
  `6820a320dc241a4002e6c257ad21983950df178b1f80271a6e6af199adb18567`；
  配置 JSON 语法和 `lmax=4, mmax=1` 已核验。
- JARVIS raw index MD5
  `0bbf6fb41120357b3ff8b19eb3690b8b`，与 Figshare 元数据一致。
- 两个 manifest 均通过 `python -m json.tool`；新 Python 文件通过
  `python -m compileall -q`。
- EquiformerV2 下载返回 HTTP 401/GatedRepo；本机 `hf auth whoami` 为
  `Not logged in`，故未伪造完成状态。
- Guqq 连通性诊断：TCP 22 成功；`ssh -G Guqq` 在批准环境中解析为用户 `xmz`
  和指定 Ed25519 identity；`ssh -vv Guqq git pull` 在认证前返回
  `kex_exchange_identification: Connection closed by remote host`，因此没有远端
  命令得到执行，也没有绕过 Slurm 在登录节点运行批处理。
