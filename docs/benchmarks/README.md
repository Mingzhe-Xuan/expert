# Tensor benchmark published scores

更新日期：2026-09-12。

本页汇总本项目四个独立训练单元对应的公开模型跑分：JARVIS tensor dielectric、
JARVIS tensor elastic、MatTen elastic，以及 JARVIS-DFPT Born effective charge（BEC）。
数值只取自原论文、官方论文页面或官方仓库；搜索结果和综述不作为数值来源。

## 如何阅读这些表

- **A（直接可比）**：相同数据、相同 split 和相同指标口径已被来源明确确认。
- **B（论文内可比）**：同一篇论文中的模型可直接比较，但与其他论文存在 baseline
  重训、样本数小差异或未发布 split ID，因此不应跨表排序。
- **C（不可直接比较）**：只是同一数据库或同类性质，target、清洗、split 或指标不同。
- `Fnorm` 是 test samples 上的平均 Frobenius distance，越低越好；`EwT x%` 是相对
  Frobenius error 小于阈值的样本比例，越高越好。

最重要的结论是：**不要从每张表挑最低数值拼成单一排行榜**。例如 GMTNet elastic
在不同论文中出现 67.38、68.12 和 117.62；这些是不同论文版本或重跑协议下的结果，
不是可互换的重复测量。

## 1. JARVIS tensor dielectric

### 1.1 GMTNet 原始 ICML 2024 协议（A）

数据为 calculation-matched JARVIS-DFT dielectric tensors，4,713 samples，8:1:1 split。

| Model | Fnorm ↓ | EwT 25% ↑ | EwT 10% ↑ | EwT 5% ↑ | 数值来源 |
|---|---:|---:|---:|---:|---|
| MEGNet | 4.16 | 74.9% | 38.9% | 19.1% | GMTNet Table 4 |
| ETGNN | 3.92 | 81.3% | 41.6% | 23.8% | GMTNet Table 4 |
| GMTNet | 3.50 | 84.5% | 57.1% | 27.8% | GMTNet Table 4 |

来源：[GMTNet paper](../ref/GMTNet.pdf)。

### 1.2 CEITNet 汇总的同 benchmark 结果（A/B）

CEITNet 明确使用 GMTNet 数据和 8:1:1 protocol；dielectric test set 为 471。前三行取
各原论文报告值，CEITNet 为该文结果。因此它适合当前项目的第一版公开基线主表。

| Model | Fnorm ↓ | EwT 25% ↑ | EwT 10% ↑ | EwT 5% ↑ | 备注 |
|---|---:|---:|---:|---:|---|
| ETGNN | 3.92 | 81.3% | 41.6% | 23.8% | 原始 GMTNet 数值 |
| GMTNet | 3.50 | 84.5% | 57.1% | 27.8% | 原始 GMTNet 数值 |
| GoeCTP / GeoCTP | 3.23 | 83.2% | 56.8% | 35.5% | GoeCTP 原文数值 |
| CEITNet | **2.87** | **86.1%** | **63.8%** | **39.3%** | CEITNet Table 4 |

来源：[CEITNet paper](sources/CEITNet.pdf)，Table 2--4。

### 1.3 IrredNet OpenReview submission（B）

该文声称沿用 GMTNet preprocessing，dielectric 样本数同为 4,713；目前 OpenReview
附件受防机器人页面限制，数值由其公开 HTML/PDF 索引中的 Table 3 复核。

| Model | Fnorm ↓ | EwT 25% ↑ | EwT 10% ↑ | EwT 5% ↑ |
|---|---:|---:|---:|---:|
| MEGNet | 4.16 | 74.9% | 38.9% | 19.1% |
| ETGNN | 3.92 | 81.3% | 41.6% | 23.8% |
| GMTNet | 3.50 | 84.5% | 57.1% | 27.8% |
| IrredNet | **3.23** | **85.8%** | 53.2% | **29.8%** |

在线来源：[OpenReview submission](https://openreview.net/forum?id=65s5CctH7I)，Table 3。
由于其 elastic sample count 写作 14,200，而 GMTNet 原文为 14,220，本页把整组标为 B。

### 1.4 GoeCTP 论文中的统一重跑（B）

这些 baseline 是 GoeCTP 作者按自己的训练设置重跑，因此只能在本表内部比较。

| Model | Fnorm ↓ | EwT 25% ↑ | EwT 10% ↑ | EwT 5% ↑ |
|---|---:|---:|---:|---:|
| MEGNet | 3.71 | 75.8% | 38.9% | 18.0% |
| ETGNN | 3.40 | 82.6% | 49.1% | 25.3% |
| GMTNet | 3.28 | 83.3% | 56.0% | 30.5% |
| GoeCTP | **3.23** | 83.2% | **56.8%** | **35.5%** |

来源：[GoeCTP paper](../ref/GoeCTP_into_equivariant.pdf)，Table 1。

## 2. JARVIS tensor elastic

### 2.1 GMTNet 原始协议（A）

数据为 14,220 个 calculation-matched JARVIS-DFT elastic tensors，8:1:1 split，单位 GPa。

| Model | Fnorm ↓ | EwT 25% ↑ | EwT 10% ↑ | EwT 5% ↑ | 数值来源 |
|---|---:|---:|---:|---:|---|
| GMTNet | **67.38** | **66.1%** | **21.8%** | **7.7%** | GMTNet Table 5 |

GMTNet 原文 Table 5 没有在 elastic 列同时给出 MEGNet/ETGNN 的完整同行结果，因此不从
后续重跑表倒填。来源：[GMTNet paper](../ref/GMTNet.pdf)。

### 2.2 CEITNet 同协议评测（A）

CEITNet 使用 14,220 samples、1,422 test instances；`*` 表示 CEITNet 作者在相同协议下
重新评测，因为对应原论文没有完全相同的结果。

| Model | Fnorm ↓ | EwT 25% ↑ | EwT 10% ↑ | EwT 5% ↑ | 结果类型 |
|---|---:|---:|---:|---:|---|
| ETGNN* | 102.76 | 49.6% | 9.8% | 2.3% | CEITNet 重跑 |
| GMTNet | **67.38** | 66.1% | 21.8% | 7.7% | 原论文 |
| GoeCTP / GeoCTP* | 85.00 | 59.6% | 23.9% | 10.4% | CEITNet 重跑 |
| CEITNet | 70.11 | **70.6%** | **32.2%** | **14.4%** | CEITNet Table 6 |

来源：[CEITNet paper](sources/CEITNet.pdf)，Table 6。CEITNet 在 Fnorm 上没有超过
GMTNet，但在三个 EwT 指标上更高。

### 2.3 IrredNet OpenReview submission（B）

| Model | Fnorm ↓ | EwT 25% ↑ | EwT 10% ↑ | EwT 5% ↑ |
|---|---:|---:|---:|---:|
| MEGNet | 132.21 | 22.8% | 3.0% | 0.3% |
| ETGNN | 90.84 | 42.0% | 12.8% | 1.2% |
| GMTNet | 68.12 | 65.0% | 21.6% | 7.7% |
| IrredNet | **64.23** | **66.5%** | **26.5%** | **12.1%** |

在线来源：[OpenReview submission](https://openreview.net/forum?id=65s5CctH7I)，Table 3。
该文报告 14,200 samples，因此不要把 64.23 与 GMTNet 原始 14,220-sample 表直接排名。

### 2.4 GoeCTP 论文中的统一重跑（B）

| Model | Fnorm ↓ | EwT 25% ↑ | EwT 10% ↑ | EwT 5% ↑ |
|---|---:|---:|---:|---:|
| MEGNet | 143.86 | 23.6% | 3.0% | 0.5% |
| ETGNN | 123.64 | 32.0% | 3.8% | 0.5% |
| GMTNet | 117.62 | 36.0% | 7.6% | 2.0% |
| GoeCTP | **107.11** | **42.5%** | **15.3%** | **7.2%** |

来源：[GoeCTP paper](../ref/GoeCTP_into_equivariant.pdf)，Table 3。这组 baseline 与
GMTNet 原始 elastic 跑分差异很大，必须作为独立 replication family 展示。

## 3. MatTen elastic benchmark

MatTen 数据集含 10,276 个 Materials Project elasticity tensors，按 crystal system
stratified 8:1:1 split，test set 为 1,021。原论文训练目标是完整 tensor，但主表公开的
稳定数值是从预测 tensor 推导出的 Hill-averaged bulk/shear/Young's moduli。

| Model | Bulk K MAE ↓ | Shear G MAE ↓ | Young E MAE ↓ | 说明 |
|---|---:|---:|---:|---|
| MatTen | 7.37 ± 0.10 | **8.38 ± 0.16** | 20.59 ± 0.35 | 一个完整 tensor 模型导出三指标 |
| MatSca | **7.32 ± 0.09** | 8.63 ± 0.07 | **19.87 ± 0.43** | 每种 scalar modulus 独立模型 |
| AutoMatminer | 9.84 ± 0.34 | 9.27 ± 0.32 | 22.10 ± 0.77 | 每种 scalar modulus 独立模型 |

单位均为 GPa；`±` 是五个不同初始化模型的标准差。来源：[MatTen paper](sources/MatTen_main.pdf)，
Table 1。MatTen 同时报告 MAE/MAD：K=0.130±0.002、G=0.280±0.005、E=0.275±0.005。

注意：这些是**导出标量指标**，不是完整 (6\times6) tensor 的 Fnorm。当前项目应在
相同官方 split 上额外报告完整 Voigt/tensor MAE、Fnorm 和 irrep-wise error，不能只用
K/G/E 数字宣称超过 MatTen。

## 4. JARVIS-DFPT BEC benchmark

### 4.1 最接近本项目 target 的公开结果（B）

| Model | Target | Data / split | Metric | Score ↓ | 可比性说明 |
|---|---|---|---|---:|---|
| ETGNN | atom-wise full asymmetric BEC tensor | ~5,000 JARVIS-DFT; 70/20/10 | test component MAE (e) | **0.045** | 同数据库和完整逐原子 tensor；split IDs、清洗和 site matching 未公开到可直接复用 |

来源：[ETGNN paper](../ref/ETGNN.pdf)，正文 JARVIS-DFT BEC experiment。它是当前找到的
最合适 full-tensor baseline，但应在本项目冻结 split 上重跑后才升级为 A 级。

### 4.2 相关但不是本项目 full-tensor target（C）

| Model | Target | N train/val/test | Metric | Score ↓ | 为什么不可直接比较 |
|---|---|---:|---|---:|---|
| CFID/GBDT | material maximum BEC scalar | 3,411 total | MAE (e) | 0.60 | 每材料一个最大值，不保留 atom axis 或 (3\times3) tensor |
| ETGNN scalar control | material maximum BEC component | ~5,000; 70/20/10 | MAE (e) | 0.12 | ETGNN 为与 CFID 比较而另训的 scalar model |
| ALIGNN 2.0 | “Born effective charge” target，官方表未注明 full/site-wise tensor contract | 4,472/248/249 | held-out MAE (e) | 0.234 | target schema 未达到本项目 calculation-matched (N\times3\times3) contract |

CFID 来源：[JARVIS-DFPT paper](sources/JARVIS_DFPT_2020.pdf)；ETGNN scalar control 来源：
[ETGNN paper](../ref/ETGNN.pdf)；ALIGNN 来源：
[official README snapshot](sources/ALIGNN_README_snapshot.md)。

因此，本项目 BEC 主表应至少同时报告 atom-wise component MAE、atom-wise Frobenius
MAE/RMSE、`0e/1e/2e` error 和 ASR residual；其中只有 component MAE 可以谨慎地与
ETGNN 的 0.045 e 对照。

## 5. 同任务但不同 benchmark 的模型

以下论文很相关，但不能填入上述四个 benchmark 的直接排名：

| Model | 使用的数据 | 与本项目关系 |
|---|---|---|
| PGEqNN | Materials Project 2026 release，按 PG 子集做 5-fold CV | 适合 PG representation 消融，不是 GMTNet JARVIS 或 MatTen 官方 split |
| AnisoNet | 约 6,700 个 Materials Project dielectric tensors | full dielectric tensor，但不是 JARVIS tensor benchmark |
| BEC EGCNN | in-house substituted perovskites、Li3PO4、ZrO2 | full atom-wise BEC，但化学空间和 split 完全不同 |

PGEqNN 原稿已在 [docs/ref/PGEqNN.pdf](../ref/PGEqNN.pdf) 归档。上述结果可作为方法背景，
不能作为相同 test set 上的横向 SOTA 数字。

## 6. 建议本项目最终对照列

正式跑完后，建议直接追加到本页各 A 级表，而不是新建另一套指标：

| Training unit | 必报公开对照 | 本项目需要匹配的主指标 |
|---|---|---|
| JARVIS dielectric | ETGNN、GMTNet、GoeCTP、CEITNet | Fnorm、EwT25/10/5 |
| JARVIS elastic | GMTNet、CEITNet；可重跑 ETGNN/GoeCTP | Fnorm、EwT25/10/5，GPa |
| MatTen elastic | MatTen、MatSca、AutoMatminer | full tensor/Voigt MAE + K/G/E MAE |
| JARVIS-DFPT BEC | ETGNN；ALIGNN 仅作辅助 | component MAE + atom Frobenius + irrep-wise + ASR |

下载文件、URL、SHA-256 和访问状态见 [SOURCES.md](SOURCES.md)。
