# DRL-path-finding

这是一个基于 PyTorch 的栅格地图自主探索强化学习工程。项目当前主线已经收口到“共享语义双状态架构”：从随机地图生成、局部观测、累计 belief map、共享环境语义解析，到 DDQN 训练、checkpoint validation、best-checkpoint final_probe、formal artifact 写出与可选离线绘图，形成了一整套正式训练闭环。

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
- 训练过程中默认记录 train / model-selection / recheck / final_probe CSV 日志，并保存 formal `best.pt` 与 diagnostic `last.pt`。
- 正式训练 run 使用专用 model-selection seed set 周期性评估候选 checkpoint，top-k recheck 后选择 `best.pt`，再只对 `best.pt` 执行 held-out `final_probe`，并写出结构化 formal artifact，供 exchange/control-plane 直接消费。
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
  - 主训练入口，负责参数配置、系统组装、训练循环、checkpoint validation、best.pt 选择、best-only final_probe 与可选绘图。
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
8. 从默认 300k env steps 开始，每 20k steps 对当前 online net 做 checkpoint validation，并保存候选 checkpoint。
9. 训练结束后保存 diagnostic `checkpoints/last.pt`，对 validation top-k 候选做 recheck，并将最终胜出者保存为 formal `checkpoints/best.pt`。
10. `GreedyEvaluator` 只对 `best.pt` 执行独立 held-out `final_probe`；`last.pt` 仅作为训练终点诊断对象。
11. `CSVMetricLogger` 与 `training.formal_artifacts` 写出 CSV、formal structured JSON 和可选绘图。

## 运行方式

推荐的常规正式训练命令：

```powershell
python .\train_q_agent.py --device cuda
```

这条命令对应当前默认主训练基线，并会在不显式传入 `--final-greedy-episodes` 时使用 100-episode formal final_probe：

- `enable_amp = False`
- `enable_inference_amp = False`
- `enable_torch_compile = False`
- `enable_channels_last = False`
- `enable_tf32 = True`
- `enable_cudnn_benchmark = True`
- `total_env_steps = 500000`
- `epsilon_decay_steps = 400000`
- `epsilon_end = 0.03`
- `formal_protocol = formal_posthoc_trainselect_v1`
- `periodic_checkpoint_interval_env_steps = 20000`
- `posthoc_candidate_start_env_steps = 200000`
- `posthoc_selection_window_env_steps = 40000`
- `posthoc_final_probe_topk = 3`
- `enable_best_checkpoint_selection = False`（仅 legacy v3 兼容路径使用）
- `final_greedy_episodes = 100`

快速 smoke 测试：

```powershell
python .\train_q_agent.py --smoke --device cpu
```

常用 profiling 命令（用于性能排查，不作为正式 comparable run）：

```powershell
python .\train_q_agent.py --device cuda --profile --total-env-steps 24000 --warmup-steps 4000 --final-greedy-episodes 1 --timing-log-interval 4000 --episode-print-interval 0 --no-save-train-representative-trajectories --no-save-final-probe-trajectories --no-generate-plots-on-finish
```

当前正式支持两种训练 budget 组织方式：

- `budget_mode=env_steps`
  - 训练停止与 step-level train snapshot 按 `total_env_steps` / `log_interval` 组织。
  - `warmup_steps` 在该模式下继续有效。
- `budget_mode=episodes`
  - 训练停止条件改为 `total_train_episodes`。
  - 训练 step snapshot / stdout train print 可分别由 `log_interval_episodes`、`train_print_interval_episodes` 控制。
  - `warmup_episodes` 在该模式下作为正式 warmup 入口；单图内部步数上限仍由 `max_episode_steps` 控制。

## Formal protocol / final_probe

当前 formal protocol revision 为 `formal_posthoc_trainselect_v1`。

- 当前默认正式训练预算为 `total_env_steps=500000`。
- 训练 epsilon 调度为 `epsilon_start=1.0`，`epsilon_decay_steps=400000`，`epsilon_end=0.03`。训练阶段保留低但非零的 exploration tail；评估阶段仍使用 greedy policy。
- Training dynamics 现在是 first-class ranking evidence。后续对比应优先审查 full/early/mid/late/last-window 训练趋势，再解释 final formal test。
- 训练期间不跑 checkpoint validation / recheck / final probe；collector / learner 不会因为中间评估暂停。
- 从 `posthoc_candidate_start_env_steps=200000` 开始，每 `periodic_checkpoint_interval_env_steps=20000` 只保存训练侧 checkpoint，例如 `checkpoints/ckpt_step_200000.pt`。
- 训练结束后，post-hoc selector 只读取训练侧 artifact，在每个候选 checkpoint 前 `posthoc_selection_window_env_steps=40000` 的窗口内计算平滑指标。
- 默认训练侧复合分数为 `0.35*reward_z + 0.25*coverage_z + 0.20*success_z - 0.10*length_z - 0.10*repeat_visit_ratio_z`，选出最多 `posthoc_final_probe_topk=3` 个候选。
- final formal test 只对 post-hoc top-k 候选执行一次 held-out final probe，使用 `fixed_final_probe_seed_base=20261323`，默认 `final_greedy_episodes=100`。
- held-out final probe 按 `success_rate -> coverage -> reward` 选出 formal winner，并将其复制/登记为 `checkpoints/best.pt`。
- `checkpoints/last.pt` 仍保存，但仅是 diagnostic-only training endpoint；除非它进入 post-hoc top-k，否则不会额外单独跑 held-out final probe。
- 历史 `formal_last_checkpoint_*` lane 与 legacy `formal_best_checkpoint_v3` lane 保留为 historical / compatibility evidence，不能和 `formal_posthoc_trainselect_v1` 混写为同一 strict comparability lane。

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

```powershell
python .\train_q_agent.py --strict-reproducibility
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
- `--log-interval-episodes`
- `--train-print-interval-episodes`
- `--use-fixed-train-episode-seeds`
- `--fixed-train-episode-seed-base`
- `--formal-protocol`
- `--periodic-checkpoint-interval-env-steps`
- `--posthoc-candidate-start-env-steps`
- `--posthoc-candidate-end-env-steps`
- `--posthoc-selection-window-env-steps`
- `--posthoc-final-probe-topk`
- `--enable-best-checkpoint-selection`（legacy `formal_best_checkpoint_v3` only）
- `--best-checkpoint-selection-start-env-steps`（legacy only）
- `--best-checkpoint-selection-interval-env-steps`（legacy only）
- `--best-checkpoint-validation-episodes`（legacy only）
- `--best-checkpoint-topk-recheck`（legacy only）
- `--best-checkpoint-recheck-episodes`（legacy only）
- `--use-fixed-model-select-seeds`（legacy only）
- `--fixed-model-select-seed-base`（legacy only）
- `--batch-size`
- `--rows` `--cols`
- `--scan-radius`
- `--reward-info-scale`
- `--log-interval`
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

- `logs/`：训练 CSV、`posthoc_candidate_scores.csv`、`final_probe.csv`，以及 formal structured JSON
- `checkpoints/`：当前 formal 主线保存 `best.pt`；`last.pt` 保留为训练终点诊断对象；`ckpt_step_<env_steps>.pt` 保存 post-hoc 候选 checkpoint
- `plots/`：离线生成的指标曲线，仅在启用相关开关时生成
- `trajectories/`：训练或 final_probe 轨迹图，仅在启用相关开关时生成

正式训练的 `logs/` 目录当前会额外生成：

- `metric_snapshot.json`
  - 统一导出 `recent_train`、post-hoc candidate selection summary、final probe winner summary、`final_probe`，并补充 diagnostic last-checkpoint summary
  - 包含主指标、次指标、稳定性指标、semantic monitoring 汇总
  - 当前正式 acceptance object 是 `best.pt` 上的 `final_probe`
  - 同时记录 best checkpoint 与 last checkpoint 对应的 env steps / train-episode 索引
  - 包含 full/early/mid/late/last-window training dynamics slope-friendly summary，以及 best-vs-last gap summary
- `benchmark_summary.json`
  - 导出总运行时、运行模式、runtime/timing 开关、可用 timing summary
  - 导出 `best_checkpoint_env_steps`、`last_checkpoint_env_steps`，并确认 legacy model-selection eval count / recheck eval count 在新协议下为 0
- `config_snapshot.json`
  - 导出完整训练配置、git sha、comparability 相关字段、`observed_run_contract` 和新的 evaluation contract
- `artifact_index.json`
  - 列出本 run 实际存在的 csv / checkpoint / structured summary / plots / trajectories
- `posthoc_selection_summary.json`
  - 记录候选范围、窗口、z-score 权重、top-k 排名和全部候选训练侧指标
- `final_probe_summary.json`
  - 记录 top-k 候选的一次 held-out final probe 结果和 formal winner
- `best_vs_last_gap_summary.json`
  - 记录 formal winner 与 last checkpoint 的诊断性 gap；last 未进入 top-k 时使用训练终点 recent window 作诊断参照
- `formal_selection_manifest.json`
  - 记录协议名、候选范围、checkpoint interval、winner checkpoint、`best.pt` 和 `last.pt`
- `training_summary.txt`
  - 面向人工复核的轻量文本摘要

当前 `config_snapshot.json` 里的 `observed_run_contract` 至少包含：

- `budget_mode`
- `final_env_steps`
- `final_train_episode_idx`
- `train_episodes_header`
- `train_steps_header`
- `model_select_eval_header`（legacy 兼容字段；新协议下通常为空）
- `best_recheck_eval_header`（legacy 兼容字段；新协议下通常为空）
- `final_probe_header`

当前正式结论以 post-hoc top-k 候选的一次 held-out `final_probe` winner 为准，并登记为 `checkpoints/best.pt`。`checkpoints/last.pt` 保留为训练终点诊断对象；除非它进入 post-hoc top-k，否则不额外单独跑 final test。

如果需要对历史 runs 做 formal backfill 与 bootstrap 阈值汇总，可执行：

```powershell
python .\tools\backfill_formal_run_artifacts.py
python .\tools\generate_historical_baseline_summary.py
```

第二条命令会生成：

- `formal_artifacts/historical_baseline_summary.json`

该文件用于 formal_train 的 bootstrap 阈值与 comparability 校准说明；如果历史 run 仍不足，会显式写出 `insufficient_history_for_calibration=true`。

这些目录属于实验产物，不适合作为源码仓库的默认提交内容，因此已经在 `.gitignore` 中排除。
