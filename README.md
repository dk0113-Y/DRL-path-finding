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
- 正式训练 run 结束后会额外写出结构化 formal artifact，供 exchange/control-plane 直接消费。
- 训练结束后的离线绘图、轨迹图导出、额外可视化产物属于可选开关，当前默认关闭以降低 wall-clock 开销。

## 当前主线说明

- 当前训练 / 评估 / final_probe 主链已经不再使用旧的 `near / mid / token` 三分支语义。
- 当前真实训练输入主线是 shared semantic dual-state 架构：
  - `SharedSemanticLayer`
  - `Advantage State`（local decision canvas / advantage canvas）
  - `Value State`（Accessible Unknown Block tree / value block tree）
  - `Semantic Dueling Head`
- 当前网络实际接收的状态张量为 `advantage_canvas`、`value_block_features`、`value_entry_features`、`value_block_mask`、`value_entry_mask`。
- Some legacy files may remain in the repository for historical reference, but they are not part of the current training path.

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
- `final_greedy_episodes = 100`

快速 smoke 测试：

```bash
python train_q_agent.py --smoke --device cpu
```

常用 profiling 命令：

```bash
python train_q_agent.py --device cuda --profile --total-env-steps 24000 --warmup-steps 4000 --eval-interval-env-steps 24000 --eval-episodes 4 --final-greedy-episodes 1 --timing-log-interval 4000 --episode-print-interval 0 --no-save-eval-trajectories --no-save-train-representative-trajectories --no-save-final-probe-trajectories --no-generate-plots-on-finish
```

当前正式支持两种训练 budget 组织方式：

- `budget_mode=env_steps`
  - 兼容旧主线；训练停止、周期 eval 与 step-level log 仍按 `total_env_steps` / `eval_interval_env_steps` / `log_interval` 组织。
  - `warmup_steps` 在该模式下继续有效。
- `budget_mode=episodes`
  - 训练停止条件改为 `total_train_episodes`。
  - 周期 eval / 训练 step snapshot / stdout train print 可分别由 `eval_interval_episodes`、`log_interval_episodes`、`train_print_interval_episodes` 控制。
  - `warmup_episodes` 在该模式下作为正式 warmup 入口；单图内部步数上限仍由 `max_episode_steps` 控制。

## Formal protocol / final_probe

当前 formal protocol revision 为 `formal_last_checkpoint_v2_3`。

- 当前默认 formal lane 的 held-out `final_probe` 为 `final_greedy_episodes=100`。
- 不显式传入 `--final-greedy-episodes` 时，主训练脚本和 VSCode 默认正式训练 preset 都使用 100 episodes。
- 历史 `formal_last_checkpoint_v2_2` lane 使用过 16-episode final_probe；这些结果保留为历史 evidence。
- `final_greedy_episodes` 是 frozen comparability field，新旧 episode 数和协议 revision 会自然进入不同 strict comparability lane，不能简单混写为同一正式可比组。

`tools/supplementary_multi_checkpoint_probe.py` 仍保留为 `supplementary_confidence_check` 工具。它适合用于多 checkpoint 同 seed 对比、非默认 episode 数检查、恢复性评估和额外置信度分析；但 100-episode held-out 评估本身已经是当前默认 formal final_probe，不再天然等同于 supplementary evidence。

训练 episode 现在也支持固定 seed 序列：

- `use_fixed_train_episode_seeds`
- `fixed_train_episode_seed_base`

当该开关开启时，相同配置下相同 episode 编号会绑定相同地图生成 seed。`logs/train_episodes.csv` 会额外记录：

- `train_episode_idx`
- `phase_episode_idx`
- `episode_seed`
- `map_fingerprint`

其中 `map_fingerprint` 用于直接审计“同编号 episode 是否确实对应同一地图/起点”。

如需进一步压缩运行时数值路径的不确定性，可选开启：

```bash
python train_q_agent.py --strict-reproducibility
```

该开关会尽量使用 deterministic runtime guard，并在 CUDA 下关闭 `cudnn_benchmark` 与 TF32。它是可选增强项，不替代固定 train episode seed 序列。

实验性快速 CUDA 路径：

- 可通过 `--fast-cuda` 或 `build_fast_cuda_config()` 启用。
- 该路径会打开 AMP / inference AMP / torch.compile / channels-last 等运行时性能开关。
- 目前这条路径在当前机器与当前模型上**未验证形成稳定净收益**，因此保留为实验性入口，不作为默认推荐主训练配置。

常用可调参数包括：

- `--total-env-steps`
- `--budget-mode`
- `--total-train-episodes`
- `--warmup-episodes`
- `--eval-interval-episodes`
- `--use-fixed-train-episode-seeds`
- `--fixed-train-episode-seed-base`
- `--batch-size`
- `--rows` `--cols`
- `--scan-radius`
- `--reward-info-scale`
- `--eval-interval-env-steps`
- `--final-greedy-episodes`

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

- `logs/`：训练与评估 CSV，以及 formal structured JSON
- `checkpoints/`：`last.pt`、`best.pt`
- `plots/`：离线生成的指标曲线，仅在启用相关开关时生成
- `trajectories/`：评估轨迹图，仅在启用相关开关时生成

正式训练的 `logs/` 目录当前会额外生成：

- `metric_snapshot.json`
  - 统一导出 `recent_train`、`last_eval`、`best_eval`、`final_probe`
  - 包含主指标、次指标、稳定性指标、semantic monitoring 汇总
  - 同时记录 best / last checkpoint 对应的 train-episode 索引（如果可得）
- `benchmark_summary.json`
  - 导出总运行时、运行模式、runtime/timing 开关、可用 timing summary、`env_steps_to_best`
  - 在 episode-budget 模式下也会导出 `budget_mode`、`train_episodes_to_best`、`total_train_episodes_completed`
- `config_snapshot.json`
  - 导出完整训练配置、git sha、comparability 相关字段、`observed_run_contract`
- `artifact_index.json`
  - 列出本 run 实际存在的 csv / checkpoint / structured summary / plots / trajectories
- `training_summary.txt`
  - 面向人工复核的轻量文本摘要

当前 `config_snapshot.json` 里的 `observed_run_contract` 至少包含：

- `budget_mode`
- `final_env_steps`
- `final_train_episode_idx`
- `train_episodes_header`
- `train_steps_header`
- `eval_metrics_header`
- `final_probe_header`

这些字段来自真实当前 run 的产物，不依赖历史 backfill。

如果需要对历史 runs 做 formal backfill 与 bootstrap 阈值汇总，可执行：

```bash
python tools/backfill_formal_run_artifacts.py
python tools/generate_historical_baseline_summary.py
```

第二条命令会生成：

- `formal_artifacts/historical_baseline_summary.json`

该文件用于 formal_train 的 bootstrap 阈值与 comparability 校准说明；如果历史 run 仍不足，会显式写出 `insufficient_history_for_calibration=true`。

这些目录属于实验产物，不适合作为源码仓库的默认提交内容，因此已经在 `.gitignore` 中排除。
