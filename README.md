# Grid Exploration DDQN

这是一个基于 PyTorch 的栅格地图自主探索强化学习工程。项目当前已经实现了从随机地图生成、局部观测、累计 belief map、frontier 候选提取，到 DDQN 训练、周期评估、checkpoint 保存和离线绘图的一整套训练闭环。

## 当前主要功能

- 在带障碍的随机二维栅格地图上生成可行起点。
- 用雷达式局部观测逐步更新 agent 的累计认知地图，而不是直接使用全局真值地图作为输入。
- 从累计地图中构造三类策略输入：
  - `near_map`：固定大小的局部 belief window，包含未知/障碍/空闲、轨迹新近性、全局归一化位置等通道。
  - `mid_map`：agent-centered 的中尺度区域密度图，编码未知区域、障碍密度和粗粒度访问密度。
  - `frontier_tokens`：从 frontier 连通区域中抽取的稀疏候选 token，描述相对位置、簇大小和局部障碍复杂度。
- 使用双分支 Q 网络做 8 邻域动作决策：
  - 局部窗口由 near encoder 编码；
  - 中尺度地图和 frontier token 先融合成 global context；
  - 再进入 split-input dueling head 输出 Q 值。
- 使用 Double DQN + n-step transition 的方式训练。
- 训练过程中自动记录 CSV 日志、周期 greedy evaluation、保存 `last.pt` / `best.pt`，并在结束后生成训练与评估曲线。

## 代码结构

- `train_q_agent.py`
  - 主训练入口，负责参数配置、系统组装、训练循环、评估、checkpoint 与绘图。
- `agents/`
  - Q 网络与状态适配逻辑。
- `encoders/`
  - 局部与全局输入的神经网络编码器。
- `heads/`
  - dueling Q head。
- `env/`
  - 地图生成、局部观测、belief map、frontier 提取、拓扑与状态构造。
- `training/`
  - collector、replay buffer、learner、evaluator、logger、plotting。

## 训练流程概览

1. `RandomMapGenerator` 生成随机障碍地图和起点。
2. `RadarSensor` + `LocalObservationModel` 提供局部观测。
3. `CumulativeBeliefMap` 维护 agent 的累计认知地图与访问历史。
4. `StateTensorAdapter` 将状态转成网络输入张量。
5. `TransitionCollector` 用 epsilon-greedy 与环境交互，并写入 replay buffer。
6. `DDQNLearner` 从 replay 中采样，执行 Double DQN 更新。
7. `GreedyEvaluator` 周期性评估当前策略。
8. `CSVMetricLogger`、`CheckpointManager`、`generate_all_plots()` 输出结果。

## 运行方式

最直接的训练命令：

```bash
python train_q_agent.py --device cuda
```

快速 smoke 测试：

```bash
python train_q_agent.py --smoke --device cpu
```

常用可调参数包括：

- `--total-env-steps`
- `--batch-size`
- `--rows` `--cols`
- `--scan-radius`
- `--reward-info-scale`
- `--eval-interval-env-steps`

## 依赖

核心依赖：

- Python
- `numpy`
- `torch`

可选依赖：

- `scipy`：用于更快的 frontier 连通域提取；没有它也有 fallback 实现。
- `matplotlib`：用于训练结束后的曲线绘图。

## 输出

默认运行结果会写到 `outputs/`。其中通常包含：

- `logs/`：训练与评估 CSV
- `checkpoints/`：`last.pt`、`best.pt`
- `plots/`：离线生成的指标曲线
- `trajectories/`：评估轨迹图

这些目录属于实验产物，不适合作为源码仓库的默认提交内容，因此已经在 `.gitignore` 中排除。
