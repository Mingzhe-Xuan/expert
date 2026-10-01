# pretrain 与 O(3)：数据效率实验

## 当前进度

实验协议与脚本已实现，正在进行 Slurm 冒烟验证；正式的 48 次训练尚未完成。
`report.html` 是进度报告，缺失指标显示 Pending，不代表已有完整比较结果。

已生成并检查的图：

- [总介电：点群与晶系分布](dielectric_dataset_distribution.png)
- [弹性：点群与晶系分布](elastic_dataset_distribution.png)

## 数据与比较设计

本轮使用质量筛选后的 JARVIS-DFT 来源记录，排除其他数据库来源，并保留全部可用点群。
目标分别为总介电张量（电子＋离子，无量纲）和弹性刚度（GPa）。沿用既有按重复结构分组的
划分；数据经过物理有效性筛选及晶体对称性投影，因此结果不能直接当作原论文未修改数据集的复现。

| 任务 | train | validation | test | 合计 |
|---|---:|---:|---:|---:|
| dielectric-total | 3,791 | 476 | 504 | 4,771 |
| elastic | 8,720 | 1,048 | 1,103 | 10,871 |

两个模型共享训练子集，25% ⊂ 50% ⊂ 75% ⊂ 100%；子集抽样 seed 固定为 20261001，
训练 seeds 为 42、43、44。两类任务独立训练。默认 200 epochs，原版每任务架构、Huber、
AdamW、batch 64、学习率 1e-3 线性衰减到 1e-5。仅替换节点输入特征，不增加额外注意力模块。
对称约束保留原版的操作顺序，按 batch 并行实现；每次运行均对照原版输出和梯度。
因此这里的训练时间对应该共同实现，不直接代表未优化原代码的速度。
所有模型统一按后半程最小 validation Fnorm 选择 checkpoint，再评估固定测试集。
这是预先指定的选择规则，与历史介电实验按 validation MAE 选择的规则不同。

## 产出

- `report.html`、`metrics.json`：各比例的 Fnorm、EwT 5/10/25%、RMSE，以及特征准备、图准备、
  缓存后训练、新结构推理、总时间；均值与训练 seed 标准差。
- `*_training_validation.png/svg`：各比例的 Training Huber Loss 与 Validation Fnorm。
- `*_groups_p{25,50,75,100}.png/svg`：固定测试集按点群、元素数、原子数、目标幅值分组的 Fnorm。
  目标幅值分箱仅用完整训练集四分位数确定，各组标注样本数。
- `*_dataset_distribution.png/svg`：全部 train/validation/test 结构的点群和晶系分布。
- `audit.json`：来源、划分、曲线、独立重算指标及风险统计。
- `efficiency_comparisons.json`：pretrain 50% 与 O(3) 100% 的配对 seed Fnorm 差值，
  以及达到共同 validation Fnorm 阈值的 epoch／训练秒数；未达到阈值的运行明确保留。

当前协议检验同一固定划分上的数据效率；没有成分隔离或结构族隔离，不能据此宣称域外泛化改善。

Fnorm 为逐结构矩阵 Frobenius 误差的均值；elastic 使用 6×6 Voigt 矩阵。
EwT 使用 `||error|| / (||target|| + 1e-5)`，表内为百分比。

时间说明：特征与图准备按选定 train＋validation 记录的实测耗时分摊，加初始化及完整缓存写入开销。
同一缓存跨 seed 复用，这些是分摊成本，不是每个 seed 重新执行的墙钟时间。
总时间＝特征准备＋图准备＋训练；训练包括验证与 checkpoint 写入。
新结构推理包含独立重算的特征、图与约束构建、batch-one 下游预测，各阶段在兼容环境分别计时后相加；
模型常驻，不包含模型冷启动，不能把它解释成完整在线服务延迟。

## 复现与验收

从服务器取回相同目录结构的 `summary.json` 和 `predictions.jsonl` 后，在仓库根目录执行：

```bash
python good_result/pretrain/report.py
```

默认要求 48 次正式运行齐全；缺少运行时报错。生成进度版使用 `--allow-partial`。
不使用历史实验补齐新协议的空缺，不由 MAE 推算 Fnorm。图表模型标签仅为 `pretrain` 与 `O(3)`。

训练模块、Slurm 命令和计时细节见[实验说明](../../src/experiments/pretrain/README.md)。
数据、缓存、权重和图表生成物默认不进入普通 Git；本目录保存脚本与实际生成文件。
