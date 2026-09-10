# Agent state

## 当前状态

GRACE 与 DPA4-Plus checkpoint 已下载并校验；JARVIS-DFPT 官方 raw-file 索引
已下载，完整 BEC 提取器通过 10 样本验证。EquiformerV2 权重仍受 Hugging Face
人工门控限制；完整 5,000 档案提取等待服务器连接恢复后通过 Slurm 执行。

## 当前计划

1. 提交资源 manifest、BEC 提取器、验证器与 Slurm 脚本。
2. 恢复 `ssh Guqq` 连接后先 `git pull`，下载 raw index 并提交完整 Slurm 作业。
3. 将生成的 JSONL、summary 和 errors 文件复制回本地并更新 manifest。
4. 用户完成 `facebook/OMAT24` 访问申请和 `hf auth login` 后下载 EquiformerV2。

## 变更记录

- 2026-09-11：开始其他 backbone 与 BEC 数据准备阶段；先进行官方资源审计。
- 2026-09-11：锁定并下载 GRACE-1L-OMAT-medium-ft-E；内部 YAML 确认中间
  feature `lmax=4`，权重 SHA-256 与 Hugging Face LFS 值一致。
- 2026-09-11：DPA4 发布配置审计显示 Nano/Mini/Neo/Air 的 `lmax` 分别为
  1/2/3/3，改选唯一满足条件的 Plus（`lmax=4, mmax=1`）并下载。
- 2026-09-11：确认 JARVIS 冻结 raw index 只有 5,000 个 DFPT 档案、约
  9.95 GiB，比论文的 5,015 少 15 个；完成 calculation-matched BEC 提取器。
- 2026-09-11：本地 10 样本提取验证通过；服务器 SSH 首次连接被远端关闭，
  完整提取暂未提交，下一步重试连接而不在本地执行重批处理。
