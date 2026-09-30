# Global + PG experts：完整前后向 profiling

## 结论

2026-09-30，Guqq Slurm **Job 518 已完成**完整 CUDA 前向、loss、反向采集。
本次小批次的首要瓶颈是 **PG experts 带来的大量小算子与 autograd/CPU 调度开销**，
不是显存容量。旋转表示构造和逐标量路由也有明显前向开销。
这是诊断结果，不是优化后的加速结果，也不能直接外推为 batch=64 的训练吞吐。

## 配置与验收

- 源码：`c1a9fdcf58c774549851cfe19f85a536832db9af`；GPU：RTX 5090。
- PyTorch `2.11.0+cu128`，e3nn `0.5.9`。解释器来自 `gmtnet-py310/bin/python`，
  实际 torch/e3nn 模块来自 `dpa4-py310/lib/python3.10/site-packages`；保留了原环境，未修改依赖。
- 冻结 DPA4：3200 维源特征转为 640 维 O(3) 不变量；缓存 64 shards/split。
- 默认 56 维 PG carrier；full_o3 Adapter；attention 关闭；GMTNet 不冻结；
  Adapter gate 初值 0，融合 sigmoid logit 初值 -4，路由尺度初值 0.08。
- 真实分层批次：7 晶体、70 节点、1276 条边；当前 PG 编号为 5/7/8/15/20/31/32。
  15 个活跃专家各执行一次，不随 subchain 重复执行。
- `status=passed`，loss=`6.1202301979`，188 个参数张量具有有限梯度。
  训练模式，无 optimizer.step；预热会更新 BatchNorm buffers。
- 2 次预热、3 次不带 profiler 的完整前后向，再采集 1 次带 profiler 的完整前后向。
  不计 DPA 特征提取、磁盘加载、模型构造和数据预处理。

先前 Job 517 在 GPU Wigner-D 构造时混用 CPU 常量与 CUDA angles。
修复仅将不参与训练的 frame 表示在 CPU 构造后传到特征 device；未 detach 学习特征，
未修改 PG 算法或原 GMTNet 训练入口。修复后的本地模型测试 22/22 通过。

## 不带 profiler 的端到端耗时

| 次数 | 前向 + loss（ms） | 反向（ms） | 合计（ms） |
|---|---:|---:|---:|
| 1 | 840.79 | 414.51 | 1255.30 |
| 2 | 199.57 | 342.14 | 541.71 |
| 3 | 201.78 | 341.88 | 543.66 |
| 中位数 | 201.78 | 342.14 | 543.66 |

第一条仍明显偏慢，可能存在延迟初始化/JIT 或运行时波动；本次没有进一步隔离原因。
后三列的中位数分别计算，不能要求相加严格等于合计中位数。
后两次约 542–544 ms，仅作为本批次参考；正式吞吐基准应增加预热和重复次数。
峰值 allocated **94.77 MiB**，reserved **102 MiB**，不含其他进程或完整训练优化器状态。

## 插桩前向分解

带 profiler 时前向+loss 为 397.69 ms，反向为 715.87 ms，合计 1113.56 ms；
采集显著放大了耗时。下表是**该次插桩运行**的 CPU 区间，不是无插桩运行的分摊估计。

| 区间 | 调用数 | CPU 区间累计（ms） | 占前向区间 |
|---|---:|---:|---:|
| PG experts（顶层 15 个，排除其嵌套 blocks 重复求和） | 15 | 164.25 | 41.4% |
| frame 表示矩阵构造 | 7 | 70.39 | 17.7% |
| chain router | 7 | 47.88 | 12.1% |
| GMTNet encoder | 1 | 35.32 | 8.9% |
| GMTNet output block | 1 | 23.46 | 5.9% |
| PG output map | 15 | 11.99 | 3.0% |
| Adapter | 1 | 8.53 | 2.2% |

其余包括检查、派发、旋转应用、pooling、融合等，未全部设立独立标记。
PG 前向区间中，编号 8/7/5 最大，分别 14.63/13.25/13.25 ms；
32/29/31 约 8.01/8.00/7.46 ms。不能仅以点群阶数判断开销，copy 数量、门控分支、
节点数和 CPU 发射成本都会影响结果。

## 实际 CUDA kernels 与反向归因

原始 trace 中有 **28,985 个真实 kernel 事件**，对应 CUDA runtime 28,536 次
`cudaLaunchKernel` 与 driver 侧 449 个调用。Kernel 时长累计 **33.066 ms**；
GPU memcpy/memset 累计约 **1.759 ms**。这些累计值不是端到端 wall time，也不是硬件利用率。

| PG experts 范围 | Kernel 数 | Kernel 时间累计（ms） |
|---|---:|---:|
| 前向 | 4,930 | 5.955 |
| 反向 | 17,644 | 19.068 |
| 合计 | 22,574（全模型 77.9%） | 25.023（全模型 75.7%） |

使用 trace 的 `fwdbwd` 流，将 backward autograd 节点关联到对应 forward 模块区间，
再用 CPU `External id` 对应真实 CUDA kernel。未匹配的工作保留为 `other`。
PG 对应的反向 engine 节点 CPU 时间累计 **482.07 ms**，占统计到的 engine 节点时间
612.11 ms 的 **78.8%**；相当于完整插桩 backward 区间的约 67.3%。
这是沿 autograd 关联的归因，不是逐模块独立重跑测得的反向 wall time。

关键陷阱：PyTorch trace 同时包含 `gpu_user_annotation`，其持续时间覆盖发射间隙，
不能当作 kernel 计算时间。摘要中同名 module 可能各有 CPU 和 GPU annotation 记录。
本分析只将 `cat=kernel` 求和；不把 GPU annotation 和嵌套 blocks 叠加进去。

## 代码层面的瓶颈解释与建议顺序

1. **先向量化 PG copy 门控与对应反向。** `_FiniteGroupBlock.forward` 对每个 copy
   分别 slice、norm、sigmoid、multiply、cat；15 个专家各两层放大了小算子数。
   整个 pass 的统计包括 7843 次 `aten::mul`、5580 次 select、4796 次 slice、
   808 次 vector_norm、800 次 sigmoid。不是所有这些调用都来自 PG，但 trace 的
   模块归因已经独立确认 PG 占大多数 kernel。可按 irrep 维度/门控类型分组批量计算。
2. **缓存固定 frame 的表示矩阵。** 当前每次 forward 都重新做 Wigner-D；它只依赖
   已缓存的晶体 frame 与 carrier，不依赖模型权重。缓存必须绑定 frame、irreps、
   dtype/device；若后续引入旋转增广，则必须相应更新，不能错误复用。
3. **向量化 router，减少重复校验和小张量创建。** 逐边/逐 chain 操作、反复 scales
   softplus 和 Python 标量判断会产生 launch/sync。可缓存静态 DAG 索引、批量计算残差
   能量与 gate，并将可预先验证的几何数据移至数据准备阶段；仍须保留异常检测契约。
4. **再评估 PG 的群平均线性层。** 当前每次前向做群平均投影；可考虑预计算线性
   投影基或 commutant 参数化，但这需要独立等变/梯度验证。W 是可学习的，不能直接
   把投影后的 W 跨 optimizer step 缓存。小批次中不是首先堆更多 GPU 并行流就能解决。

本轮仅提出优化方向，未实施这些性能修改，也未测任何优化收益。
未覆盖 batch=64、大晶体、所有 32 点群、训练后权重、optimizer 更新或 DPA 在线反传；
也未做独占 GPU/锁频实验，不能将一次 trace 推广为所有训练条件的结论。

## 复现与结果文件

启动器：[`profile_global_experts.sbatch`](../../slurm/profile_global_experts.sbatch)。
分析器：[`analyze_trace.py`](../../src/profiling/analyze_trace.py)。

```bash
python -m src.profiling.analyze_trace \
  results/global-pg-profile/518/trace.json \
  results/global-pg-profile/518/attribution.json
```

本地生成物（已被 Git 忽略）：[summary.json](../../results/global-pg-profile/518/summary.json)、
[trace.json](../../results/global-pg-profile/518/trace.json)、
[attribution.json](../../results/global-pg-profile/518/attribution.json)。
服务器原件保留在 `results/global-pg-profile/518/profile/`。
summary/trace 的本地 SHA-256 与远端逐一匹配：

- summary：`7ae46c03dc4b2215194e2e8bc3dffb8f023e8da2c1bb45d9054758454c3cdcf8`
- trace：`00bb19d48911b892fc6816cb9ff66c98adb274428d71b926204fa6dd34bb0b6c`

远端临时源码 bundle 已删除，可从本地提交重新生成；所有作业结果和日志均保留。
