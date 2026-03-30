# DRL-path-finding

这是一个基于 PyTorch 的栅格地图自主探索强化学习工程。项目当前主线已经收口到“共享语义双状态架构”：从随机地图生成、局部观测、累计 belief map、共享环境语义解析，到 DDQN 训练、周期评估、checkpoint 保存与可选离线绘图，形成了一整套训练闭环。

## 当前主要功能

- 在带障碍的随机二维栅格地图上生成可行起点。
- 用雷达式局部观测逐步更新 agent 的累计认知地图，而不是直接使用全局真值地图作为输入。
- 基于累计地图构造新的共享语义层：
  - `Accessible Unknown Block`：可进入未知块，是 value 分支的基本对象。
  - `Frontier Entry Cluster`：未知块的子入口簇，是进入方式而不是主机会对象。
- 使用新的 dueling Q 网络做 8 邻域动作决策：
  - advantage 分支读取 agent-centered 的 `local decision canvas`；
  - value 分支读取 block-tree 结构化状态；
  - 再由语义分离的 dueling head 输出 Q 值。
- 使用 Double DQN + n-step transition 的方式训练。
- 训练过程中默认记录 CSV 日志、周期 greedy evaluation、保存 `last.pt` / `best.pt`。
- 训练结束后的离线绘图、轨迹图导出、额外可视化产物属于可选开关，当前默认关闭以降低 wall-clock 开销。

## 当前主线说明

- 当前训练 / 评估 / final_probe 主链已经不再使用旧的 `near / mid / token` 三分支语义。
- 当前主输入主线是：
  - `SharedSemanticLayer`
  - `Advantage State`（local decision canvas）
  - `Value State`（Accessible Unknown Block tree）
  - `Semantic Dueling Head`
- 仓库里仍保留部分旧模块文件作为历史参考或底层算子来源，但它们不再是当前训练主路径。

## 代码结构

- `train_q_agent.py`
  - 主训练入口，负责参数配置、系统组装、训练循环、评估、checkpoint 与可选绘图。
- `agents/`
  - Q 网络与状态适配逻辑。
- `encoders/`
  - advantage canvas encoder 与 value tree encoder。
- `heads/`
  - dueling Q head。
- `env/`
  - 地图生成、局部观测、belief map、shared semantic layer、advantage/value 状态构造。
- `training/`
  - collector、replay buffer、learner、evaluator、logger、plotting。

当前主线之外的历史参考模块包括但不限于：

- `env/frontier_token_builder.py`
- `env/local_state_builder.py`
- `encoders/global_encoder.py`
- `encoders/local_encoder.py`
- `heads/q_head.py`

这些文件目前不作为训练入口的主语义解释路径。

## 训练流程概览

1. `RandomMapGenerator` 生成随机障碍地图和起点。
2. `RadarSensor` + `LocalObservationModel` 提供局部观测。
3. `CumulativeBeliefMap` 维护 agent 的累计认知地图、analysis box 与 revisit/recency 状态。
4. `SharedSemanticLayer` 从累计地图提取 accessible blocks 与 entry clusters。
5. `StateTensorAdapter` 将 advantage canvas 和 value block-tree 转成网络输入张量。
6. `TransitionCollector` 用 epsilon-greedy 与环境交互，并写入 replay buffer。
7. `DDQNLearner` 从 replay 中采样，执行 Double DQN 更新。
8. `GreedyEvaluator` 周期性评估当前策略。
9. `CSVMetricLogger`、`CheckpointManager`、`generate_all_plots()` 输出结果。

## 运行方式

推荐的常规主训练命令：

```bash
python train_q_agent.py --device cuda
```

这条命令对应当前默认主训练基线：

- `enable_amp = False`
- `enable_inference_amp = False`
- `enable_torch_compile = False`
- `enable_channels_last = False`
- `enable_tf32 = True`
- `enable_cudnn_benchmark = True`

快速 smoke 测试：

```bash
python train_q_agent.py --smoke --device cpu
```

常用 profiling 命令：

```bash
python train_q_agent.py --device cuda --profile --total-env-steps 24000 --warmup-steps 4000 --eval-interval-env-steps 24000 --eval-episodes 4 --final-greedy-episodes 1 --timing-log-interval 4000 --episode-print-interval 0 --no-save-eval-trajectories --no-save-train-representative-trajectories --no-save-final-probe-trajectories --no-generate-plots-on-finish
```

实验性快速 CUDA 路径：

- 可通过 `--fast-cuda` 或 `build_fast_cuda_config()` 启用。
- 该路径会打开 AMP / inference AMP / torch.compile / channels-last 等运行时性能开关。
- 目前这条路径在当前机器与当前模型上**未验证形成稳定净收益**，因此保留为实验性入口，不作为默认推荐主训练配置。

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

- `scipy`：用于更快的 unknown block / entry cluster 连通域提取；没有它也有 fallback 实现。
- `matplotlib`：用于训练结束后的曲线绘图。

## 输出

默认运行结果会写到 `outputs/`。其中通常包含：

- `logs/`：训练与评估 CSV
- `checkpoints/`：`last.pt`、`best.pt`
- `plots/`：离线生成的指标曲线，仅在启用相关开关时生成
- `trajectories/`：评估轨迹图，仅在启用相关开关时生成

这些目录属于实验产物，不适合作为源码仓库的默认提交内容，因此已经在 `.gitignore` 中排除。
