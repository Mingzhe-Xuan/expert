# Training structures vs. learnable parameters

## 统计口径

本文统计当前 `src` 中以下模型配置：

- architecture branch：`B+A+PGE+R`；
- O(3) adaptation backend：`full_o3`；
- point-group expert：分别统计 `a1_only` 和 `full_pg`；
- O(3) readout backend：`full_o3`；
- backbone 参数全部冻结；
- backbone 后的可训练 `O3InterfaceProjector` 计入 learnable parameters；
- Recommended 为每个性质实例化全部 32 个 PG experts；
- Reduced 显式将该性质的 `available_point_groups` 传给 `expert_point_groups`，从而只实例化
  dielectric 的 7 个或 elastic 的 6 个专家。

参数通过实际实例化 `PointGroupTensorModel`、`O3InterfaceProjector` 并对
`requires_grad=True` 的参数执行 `numel()` 得到，不包含冻结 backbone。

## 数据规模与激活点群

| Dataset | Training structures | Instantiated PG experts |
|---|---:|---:|
| Recommended dielectric electronic | 8,455 | 32 |
| Reduced dielectric electronic | 4,924 | 7 |
| Recommended dielectric ionic | 8,680 | 32 |
| Reduced dielectric ionic | 4,933 | 7 |
| Recommended dielectric total | 8,808 | 32 |
| Reduced dielectric total | 5,001 | 7 |
| Recommended elastic stiffness | 16,948 | 32 |
| Reduced elastic stiffness | 12,699 | 6 |

Reduced dielectric 的可用点群为：

```text
2/m, mm2, mmm, 4/mmm, -3m, -43m, m-3m
```

Reduced elastic stiffness 的可用点群为：

```text
2/m, mmm, 4/mmm, -3m, 6/mmm, m-3m
```

## `a1_only` PG expert

`a1_only` 使用 56 维 hidden layout。下表的 Downstream 包含 adaptation、完整 expert bank、
routing gate 和 readout；其后四列再加入对应 backbone 的可训练 interface，因此是完整的
learnable parameter 数量。

| Dataset | Training structures | Experts | Downstream | + DPA4 | + MACE | + EquiformerV2 | + GRACE |
|---|---:|---:|---:|---:|---:|---:|---:|
| Recommended electronic | 8,455 | 32 | 47,324 | 48,348 | 48,604 | 49,372 | 50,012 |
| Reduced electronic | 4,924 | 7 | 9,284 | 10,308 | 10,564 | 11,332 | 11,972 |
| Recommended ionic | 8,680 | 32 | 47,324 | 48,348 | 48,604 | 49,372 | 50,012 |
| Reduced ionic | 4,933 | 7 | 9,284 | 10,308 | 10,564 | 11,332 | 11,972 |
| Recommended total | 8,808 | 32 | 47,324 | 48,348 | 48,604 | 49,372 | 50,012 |
| Reduced total | 5,001 | 7 | 9,284 | 10,308 | 10,564 | 11,332 | 11,972 |
| Recommended elastic | 16,948 | 32 | 47,414 | 48,438 | 48,694 | 49,462 | 50,102 |
| Reduced elastic | 12,699 | 6 | 6,974 | 7,998 | 8,254 | 9,022 | 9,662 |

模块分解如下：

| Dataset scope/task | Adaptation | Expert bank | Gate | Readout | Downstream total |
|---|---:|---:|---:|---:|---:|
| Recommended dielectric | 306 | 46,952 | 1 | 65 | 47,324 |
| Reduced dielectric | 306 | 8,912 | 1 | 65 | 9,284 |
| Recommended elastic | 306 | 46,952 | 1 | 155 | 47,414 |
| Reduced elastic | 306 | 6,512 | 1 | 155 | 6,974 |

Reduced 后，dielectric expert bank 减少 `81.0%`，elastic expert bank 减少 `86.1%`。

## `full_pg` PG expert

`full_pg` 与 `a1_only` 一样使用 `[8, 2, 2, 2, 2]`、共 56 维的 hidden layout，
但 finite-group blocks 比 A1-only blocks 更完整。下列数字已按当前实现重新实例化统计。

| Dataset | Training structures | Experts | Downstream | + DPA4 | + MACE | + EquiformerV2 | + GRACE |
|---|---:|---:|---:|---:|---:|---:|---:|
| Recommended electronic | 8,455 | 32 | 203,556 | 204,580 | 204,836 | 205,604 | 206,244 |
| Reduced electronic | 4,924 | 7 | 46,292 | 47,316 | 47,572 | 48,340 | 48,980 |
| Recommended ionic | 8,680 | 32 | 203,556 | 204,580 | 204,836 | 205,604 | 206,244 |
| Reduced ionic | 4,933 | 7 | 46,292 | 47,316 | 47,572 | 48,340 | 48,980 |
| Recommended total | 8,808 | 32 | 203,556 | 204,580 | 204,836 | 205,604 | 206,244 |
| Reduced total | 5,001 | 7 | 46,292 | 47,316 | 47,572 | 48,340 | 48,980 |
| Recommended elastic | 16,948 | 32 | 203,646 | 204,670 | 204,926 | 205,694 | 206,334 |
| Reduced elastic | 12,699 | 6 | 39,822 | 40,846 | 41,102 | 41,870 | 42,510 |

模块分解如下：

| Dataset scope/task | Adaptation | Expert bank | Gate | Readout | Downstream total |
|---|---:|---:|---:|---:|---:|
| Recommended dielectric | 306 | 203,184 | 1 | 65 | 203,556 |
| Reduced dielectric | 306 | 45,920 | 1 | 65 | 46,292 |
| Recommended elastic | 306 | 203,184 | 1 | 155 | 203,646 |
| Reduced elastic | 306 | 39,360 | 1 | 155 | 39,822 |

Reduced 后，dielectric expert bank 减少 `77.4%`，elastic expert bank 减少 `80.6%`。

## Backbone interface 参数

| PG hidden mode | DPA4 | MACE | EquiformerV2 | GRACE |
|---|---:|---:|---:|---:|
| `a1_only` | 1,024 | 1,280 | 2,048 | 2,688 |
| `full_pg` | 1,024 | 1,280 | 2,048 | 2,688 |

例如，Reduced total + MACE 的 learnable parameters 为：

- `a1_only`：`9,284 + 1,280 = 10,564`；
- `full_pg`：`46,292 + 1,280 = 47,572`。

## 数据是否足够

从总规模看，Reduced 对当前模型是更容易训练的配置：

- dielectric 约 5,000 个训练结构对应约 10k--49k learnable parameters；
- elastic 约 12,700 个训练结构对应约 8k--42.5k learnable parameters；
- 相比 Recommended，Reduced 只减少约 25%--43% 的训练结构，却减少约 75%--85% 的
  expert parameters。

Recommended 的总参数量本身并不过大。约 48k--206k learnable parameters 对 8.5k--17k
训练结构仍属于可训练范围。主要风险来自条件路由，而不是全局参数/样本比：

- adaptation 和 readout 能从所有训练结构获得梯度；
- 在默认 `parent_dags=None` 的情况下，每个结构只更新当前 PG expert；
- Recommended 中部分低频点群只有个位数或十几条结构，无法可靠拟合其独立 expert；
- 总训练结构数不能补偿某个 expert 自身缺少有效样本的问题。

因此，当前实现下的推荐判断是：

1. Reduced 足以训练 `a1_only` 和 `full_pg` 两种完整的 `B+A+PGE+R`，适合作为稳定的架构
   benchmark；
2. Recommended + `a1_only` 整体可训练，但低频专家需要更强正则化或冻结；
3. Recommended + `full_pg` 的共享 adaptation/readout 数据充足，但独立低频专家明显欠定，
   更适合结合 parent-DAG sharing、祖先 expert 初始化或低频 PG 共享路径；
4. 需要同时报告 registered learnable parameters、每个点群实际激活的参数和 expert update
   counts，不能只报告全模型总参数。

## 当前实现注意事项

`PointGroupTensorModel` 已支持 `expert_point_groups` 参数，因此 Reduced 可以真正裁掉未使用
experts。但是调用方必须把 reduced manifest 中的 `available_point_groups` 显式传入。
若仍使用默认 `expert_point_groups=None`，模型会实例化全部 32 个 experts，Reduced 只改变
数据而不会减少 learnable parameters。

此外，当前通用完整数据训练入口 `train_cached_backbone_readout` 仍是 direct `B+R` readout；
`B+A+PGE+R` 已在 `build_real_system`、smoke 和模型层实现，但将其用于完整 Recommended/Reduced
训练仍需要训练入口显式接入相同的 expert-point-group 配置。
