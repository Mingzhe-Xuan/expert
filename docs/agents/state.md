# Agent state

## 当前状态

commit `e65c0e0` 已按用户明确授权推送到 `origin/main`。GRACE 与 DPA4-Plus
checkpoint 已下载并校验，JARVIS-DFPT BEC 提取器通过 10 样本验证。完整
5,000 档案提取被服务器 SSH 在密钥交换前主动断开所阻塞；EquiformerV2 另受
Hugging Face 人工门控限制。

## 当前计划

1. 等待 Guqq SSH 服务端的 pre-authentication close 状态解除。
2. 恢复连接后先 `git pull`，下载 raw index 并提交完整 Slurm 作业。
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
- 2026-09-11：用户明确授权后已将 `e65c0e0` 推送至 `origin/main`。SSH 连续
  三次在 key exchange 前由远端关闭；端口 22 可达且本地 alias/用户/密钥配置
  正确，因此停止盲目重试并将完整 Slurm 提取标为外部阻塞。
