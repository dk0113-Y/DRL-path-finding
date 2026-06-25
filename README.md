# DRL Path Finding

面向移动机器人未知环境自主探索的深度强化学习仿真实验框架。项目在二维占据栅格环境中完成随机地图生成、局部观测、累计 belief map、Double-DQN 训练、固定协议评估和实验产物记录。

当前维护主线为 `A_new` / `final_4ch_no_frontier_raster`：局部 advantage 分支使用 4 通道行为状态，value 分支使用结构化 frontier-block tree，并通过 dueling Q head 输出 8 邻域动作价值。

> 本仓库验证的是栅格仿真中的算法与实验流程，不包含 ROS 部署、物理机器人实验或真实传感器验证。ROS2/Gazebo 迁移与接口验证位于独立的 ROS2 仓库。

## 研究问题

智能体从未知二维地图中的一个可行位置出发，每一步：

1. 获取半径受限的雷达式局部观测。
2. 更新累计占据栅格与访问历史。
3. 从 `N / NE / E / SE / S / SW / W / NW` 中选择动作。
4. 以有效覆盖率达到阈值作为成功条件。

当前正式配置的主要环境参数为：

| 参数 | 值 |
|---|---:|
| Grid size | `40 x 60` |
| Obstacle ratio | `0.20` |
| Scan radius | `10` |
| Max episode steps | `600` |
| Coverage threshold | `0.95` |
| Formal training budget | `650000` environment steps |

环境真值地图仅用于仿真推进、局部传感、终止条件和指标计算，不作为策略的全局决策输入。

## 方法概览

```text
Random occupancy grid
  -> local radar observation
  -> cumulative belief map
  -> shared semantic extraction
       |-> 4-channel local advantage canvas
       `-> structured frontier-block value tree
  -> semantic dueling Q network
  -> Double-DQN update / greedy evaluation
```

### 4-channel local advantage canvas

`env/advantage_state_builder.py` 构造以智能体为中心的局部状态：

1. `free`
2. `obstacle`
3. `visit_count_log_norm`
4. `recent_trajectory_decay`

当前主线不在 advantage canvas 中使用 frontier raster。

### Structured frontier-block value tree

`env/shared_semantic_layer.py` 从累计 belief map 中提取：

- `UnknownBlock`
- `FrontierCluster`
- `SupportGeometry`

`env/value_state_builder.py` 将未知区域、入口方向、入口宽度和支撑障碍密度等信息组织为结构化 value-tree tensors。

### Double-DQN

`agents/q_value_agent.py` 中的 `ExplorationQNetwork` 由以下部分组成：

- `AdvantageCanvasEncoder`
- `ValueTreeEncoder`
- `SemanticDuelingHead`

`training/learner.py` 实现 Double-DQN target、hard target-network sync、replay sampling、n-step target、Smooth L1 TD loss、action masking 和可选 CUDA AMP 路径。

## 技术栈

- Python
- PyTorch
- NumPy
- Matplotlib / Pillow
- PowerShell experiment launchers
- CSV / JSON experiment records

当前仓库没有依赖锁文件。运行前需在独立 Python 环境中安装 PyTorch、NumPy、Matplotlib 和 Pillow。

## 仓库结构

```text
.
|-- train_q_agent.py                  # 训练、评估和 artifact 写出入口
|-- agents/                           # Q network 与状态适配
|-- encoders/                         # advantage/value encoders
|-- heads/                            # semantic dueling head
|-- env/                              # 地图、局部观测、belief 与语义状态
|-- training/                         # collector、learner、replay、evaluator
|-- experiments/final_method/         # 主方法、baseline、ablation launchers
|-- experiment_records/               # 轻量 CSV/JSON 实验记录
|-- scripts/                          # PowerShell orchestration
|-- demos/                            # 交互式语义与状态可视化
|-- tools/                            # backfill、plot、artifact utilities
`-- docs/                             # 协议与实验说明
```

训练 checkpoint、raw outputs 和 replay/profiling artifacts 默认不进入版本控制。

## 快速开始

检查环境和网络前向：

```powershell
python --version
python agents\q_value_agent.py
```

打印 A_new smoke contract，不启动训练：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_a_new_final_4ch.ps1 `
  -RunStage smoke -Device cpu -DryRun
```

执行 CPU smoke：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_a_new_final_4ch.ps1 `
  -RunStage smoke -Device cpu
```

正式 CUDA 训练：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_a_new_final_4ch.ps1 `
  -RunStage formal -Device cuda
```

统一 held-out probe：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_a_new_unified_final_probe.ps1 `
  -Device cuda
```

`checkpoint_store/` 中必须存在对应 checkpoint 才能复现依赖权重的 probe。

## 实验矩阵

当前 final-method 实验矩阵包含：

| Label | Method | 作用 |
|---|---|---|
| A | `final_4ch_no_frontier_raster` | 完整方法 |
| B | `Anew_B_classical_frontier_greedy` | 非学习 classical frontier baseline |
| C | `Anew_C_local_state_ddqn` | 仅局部状态的 DDQN baseline |
| D | `Anew_D_no_value_tree` | 移除结构化 value tree |
| E | `Anew_E_no_dual_state_split` | 移除 dual-state split |
| F_key | `Anew_F3_no_behavior_memory` | 移除行为记忆通道 |
| R_key | `Anew_R5` | 移除效率类 reward penalties |

相关 launcher 与记录位于：

- `experiments/final_method/`
- `experiment_records/final_method/`

## 实验结果

仓库中已提交的统一对比记录：

```text
experiment_records/final_method/unified_final_probe/
  unified_final_probe_20260527_103303/
```

该记录使用 100 个 episode，seed 范围为 `20261323..20261422`。

| Label | Coverage | Success rate | Avg. steps | Timeout rate |
|---|---:|---:|---:|---:|
| B | 0.642881 | 0.21 | 515.59 | 0.79 |
| A | 0.937350 | 0.92 | 291.98 | 0.08 |
| C | 0.496706 | 0.00 | 600.00 | 1.00 |
| D | 0.931999 | 0.82 | 323.52 | 0.18 |
| E | 0.925461 | 0.81 | 362.62 | 0.19 |
| F_key | 0.587208 | 0.00 | 600.00 | 1.00 |
| R_key | 0.927040 | 0.86 | 333.37 | 0.14 |

指标含义：

- `Coverage`：episode 结束时的平均有效覆盖率。
- `Success rate`：覆盖率达到 `0.95` 的 episode 比例。
- `Avg. steps`：平均 episode 步数。
- `Timeout rate`：达到 `600` 步上限仍未成功的比例。

结果边界：

- 表中数值来自已提交 CSV/JSON 记录，本次 README 修改没有重新训练或重跑 probe。
- checkpoint 二进制未提交；部分记录保留原训练机器的绝对路径。
- 这些结果只支持当前二维栅格协议下的方法比较，不能外推为真实机器人性能或普遍优于其他 DRL 方法。

## 输入与输出

### 输入

- 随机矩形障碍地图及 episode seed
- 局部雷达式 observation
- 训练/评估配置
- 可选 checkpoint

### 输出

默认实验目录通常包含：

```text
outputs/<run>/
|-- logs/
|   |-- train_episodes.csv
|   |-- final_probe.csv
|   |-- metric_snapshot.json
|   |-- benchmark_summary.json
|   |-- config_snapshot.json
|   `-- artifact_index.json
|-- checkpoints/
|   `-- last.pt
|-- plots/                            # 可选
`-- trajectories/                    # 可选
```

正式比较应同时保留 config、seed、checkpoint provenance、episode-level metrics 和 summary artifacts。

## 当前状态与限制

- 当前主线为 A_new 4-channel advantage canvas + structured value tree。
- 训练、smoke、formal probe、baseline/ablation launcher 和轻量结果记录已存在。
- 仓库未提供 `requirements.txt` 或锁定环境，跨机器复现需人工建立一致依赖。
- checkpoint 和原始训练输出未提交，checkpoint-dependent probe 不能仅靠 clone 直接复现。
- 部分历史记录含原机器绝对路径，公开发布前应决定是否脱敏。
- 本仓库不包含 ROS2/Gazebo、物理机器人、真实 LiDAR 或板端部署验证。

## 仓库数据管理

默认不要提交：

- `outputs/`
- `checkpoint_store/`
- `checkpoints/`
- `*.pt`, `*.pth`, `*.ckpt`
- replay buffers
- profiling/debug dumps
- `__pycache__/`

公开实验记录前，应检查绝对路径、数据来源、checkpoint 许可和可能包含环境信息的 artifact。
