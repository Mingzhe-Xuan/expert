# Global + PG experts：三项优化实测

## 结论

Slurm **Job 520 完成并通过校验**。源码 `471dfc3a3d48758ae62218ec26e1996a6a1b0895`，
同一 RTX 5090 GPU 分配内，按相同初始权重、buffers、输入批次和 RNG 比较四条执行路径。

- 64 晶体：前后向中位耗时 **1145.06 → 412.65 ms，2.775×**；耗时降低 **63.96%**。
- 7 晶体：**541.87 → 196.99 ms，2.751×**；耗时降低 **63.65%**。
- 64 晶体 CUDA kernels：**50,786 → 13,844，减少 72.74%**。
- 峰值 allocated 显存基本不变：64 晶体 **650.82 → 649.67 MiB**。

这些是固定真实批次、缓存预热后的完整 forward + loss + backward 测量，
不包括 DPA 提取、数据加载、optimizer.step 或完整 epoch；不是训练精度提升结论。

## 累积优化对比

每条路径预热 6 次，计时 7 次，下面是无 profiler 插桩的中位数。
前向、反向和合计的中位数分别计算，因此前两列之和不一定等于第三列。

### 64 晶体：982 节点、17,882 条边、15 个活跃 PG experts

| 执行路径 | 前向+loss ms | 反向 ms | 合计 ms | 相对旧路径加速 | CUDA kernels |
|---|---:|---:|---:|---:|---:|
| 旧路径 | 673.98 | 470.29 | 1145.06 | 1.00× | 50,786 |
| 1：PG copy 门控分组批量化 | 678.48 | 316.93 | 995.84 | 1.15× | 31,005 |
| 1+2：再加 frame 表示缓存 | 400.19 | 322.04 | 722.62 | 1.58× | 31,001 |
| 1+2+3：再加向量路由及融合后统一投影 | 219.13 | 193.52 | 412.65 | 2.77× | 13,844 |

三项全部开启相对旧路径：前向约 3.08×，反向约 2.43×。
七次合计耗时范围：旧路径 1117.97–1146.70 ms，优化路径 411.65–413.45 ms。
门控分组在此批次并没有加速前向（678.48 vs 673.98 ms），其主要收益来自反向；
不应把 kernel 数减少直接等同为同比例 wall-time 加速。
这些是累积消融，不能当作各项完全独立、可直接相加的加速倍数。

### 7 晶体：70 节点、1,276 条边、15 个活跃 PG experts

| 执行路径 | 前向+loss ms | 反向 ms | 合计 ms | 加速 | CUDA kernels |
|---|---:|---:|---:|---:|---:|
| 旧路径 | 200.28 | 342.50 | 541.87 | 1.00× | 28,985 |
| 1 | 147.69 | 151.76 | 298.83 | 1.81× | 9,211 |
| 1+2 | 117.56 | 155.85 | 273.52 | 1.98× | 9,207 |
| 1+2+3 | 87.06 | 110.12 | 196.99 | 2.75× | 6,235 |

小批次 kernels 减少 78.49%；峰值 allocated 显存 94.77 → 94.09 MiB。

## 已实现内容

1. [`optimized.py`](../../src/models/global_experts/optimized.py)：按 copy 的维度和门控类型
   分组，批量 norm/gate，之后恢复原坐标顺序。保持每个 copy 的独立参数及 state-dict 键。
2. 同文件的 frame LRU 缓存：以 frame 数值、irreps、dtype/device 为键，默认容量 8192；
   原位修改 frame、旋转增广会改变 key，模型设备/dtype 转换会清空缓存。
3. [`vectorized_routing.py`](../../src/models/global_experts/vectorized_routing.py)：缓存经校验
   的不可变 DAG 静态索引，通过 padded prefix products 批量计算 chain 权重；保留全零和
   部分零 gate 的梯度行为。每次 forward 重新计算可学习尺度和 residual 对应权重，不缓存它们。
   在 [`model.py`](../../src/models/global_experts/model.py) 中先汇合 expert 特征，再调用一次
   无偏置线性 output map；请求诊断时仍提供各 PG 投影特征。

原有专家和 GMTNet baseline 实现未被替换。新模型默认开启三项；恢复旧执行路径：

```bash
--no-grouped-pg-gates --no-cache-frames --no-vectorized-routing
```

## 正确性与性能证据

- 全 32 PG 的 grouped/loop 输出、输入梯度、参数梯度及群作用等变性对照通过。
- 路由普通值、全零、混合零、singleton、非法 DAG/residual、detachment 对照通过。
- frame 缓存命中、容量驱逐、数值改变、dtype 失效及完整模型输出/梯度对照通过。
- 模型/优化首轮组合 72 项通过；补充边界 9 项、benchmark/profiler 3 项、
  batch64 选样及 DPA/CGCNN 生命周期 3 项通过（含重复验收项，不当作独立总测试数）。
- 旧 GMTNet attention、DPA-GMTNet、expert suites **41/41 通过**。
- GPU 两种批次、四条路径都先验证输出与全部参数梯度再测性能。
  输出容差 atol=2e-5、rtol=2e-4；梯度容差 atol=2e-4、rtol=2e-3。
  CUDA 并行归约和线性融合换序可能有浮点差异，不要求逐位一致。
- 独立读取 10 份 JSON，确认状态、各优化开关、相同 batch/运行环境、唯一训练 IDs、
  7 条有限计时、逐项重算中位数、loss/kernel/memory 与详细 summary 一致。
- 压缩包为普通 tar，11 个指定成员；本地 SHA-256 与远端一致：
  `3ef23a26233396aa808a0de259798790ac52754229cbdc4d47ed019868384a6d`。

## 优化后的剩余开销

64 晶体的**插桩前向 CPU 区间**（不能与上面的无插桩 wall time 混算）：

| 区间 | 旧路径 ms | 优化路径 ms |
|---|---:|---:|
| PG experts 合计 | 162.51 | 46.50 |
| Router 合计（仍每晶体调用一次，共 64 次） | 389.13 | 129.77 |
| Wigner-D 构造 | 650.85（64 次） | 缓存命中，无构造调用 |
| Output map | 11.80（15 次） | 0.87（1 次） |

路由目前只是每个晶体内部向量化，尚未跨晶体批量化，因此仍有明显调用开销。
本轮按约定没有进一步改群平均线性参数化，也没有引入 torch.compile 或并行 streams。

## 范围与限制

- frame 和 DAG 是热缓存。真实训练首轮遇到新 frame 时仍有构造成本；长期收益取决于
  数据重用、增广和缓存命中率。本批次显存变化不代表整个 8192 项缓存装满后的占用。
- batch64 保留 7 个 PG 代表，再从固定训练 split 顺序补足；不是整个训练集的随机分布估计。
- 四种变体按固定顺序测量，同一 GPU 分配但未锁频、未重复多轮随机顺序实验。
- 输入仍是冻结 DPA 不变量；结果不覆盖在线训练 DPA。无 optimizer.step，无完整训练质量对照。

## 复现与文件

启动器：[`compare_global_experts.sbatch`](../../slurm/compare_global_experts.sbatch)。
比较逻辑：[`compare_optimized.py`](../../src/profiling/compare_optimized.py)。
原始 JSON（Git 忽略的结果文件）：
[7 晶体](../../results/global-pg-compare/520/smoke/comparison.json)、
[64 晶体](../../results/global-pg-compare/520/batch64/comparison.json)。
各变体完整 trace 保留在 Guqq `results/global-pg-compare/520/{smoke,batch64}/` 内。
