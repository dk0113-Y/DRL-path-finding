# F/R 消融实验基础设施

本目录只用于 F/R 输入层和配置层消融，不包含 D `no_value_tree`、E `no_semantic_dual_state_split`、C local DDQN baseline，也不包含 classical frontier greedy baseline。

## 设计边界

- 不复制主工程。
- 不重写训练系统。
- F 组只做 `advantage_canvas` 通道置零。
- R 组只做 reward 参数覆盖。
- full method 默认训练入口 `train_q_agent.py` 不受影响；只有通过 `experiments/ablations/run_ablation_train.py` 显式启动时才启用消融。
- pilot/smoke run 不能进入论文 Results。
- formal run 才能作为结果候选。

## run-stage 与 smoke 约束

- `smoke`：功能测试，不进入论文 Results。
- `pilot`：小规模试跑，不进入论文主结果表。
- `formal`：正式实验，不能携带 `--smoke`。
- 如果命令中使用 `--run-stage formal` 或 `--run-stage pilot`，不能在 passthrough 参数里再传 `--smoke`；例如 `--run-stage formal -- --smoke` 会被启动器拒绝。

## F 组

F 组保持 `advantage_canvas shape = [B, 5, H, W]`，按通道名称置零：

- F1：`no_frontier_channel`
- F2：`no_visit_count_channel`
- F3：`no_recent_trajectory_channel`
- F4：`no_visit_traj_channels`
- F5：`occupancy_only_canvas`

## R 组

R 组只覆盖 `TrainConfig` 中已有的 reward 字段：

- R1：`no_step_penalty`
- R2：`no_revisit_penalty`
- R3：`no_turn_penalty`
- R4：`no_timeout_penalty`
- R5：`no_efficiency_penalties`
- R6：`sparse_reward_variant`

## 示例命令

列出规格：

```powershell
python experiments\ablations\run_ablation_train.py --list
```

dry-run F1：

```powershell
python experiments\ablations\run_ablation_train.py --ablation-id F1 --dry-run
```

smoke F5：

```powershell
python experiments\ablations\run_ablation_train.py --ablation-id F5 --run-stage smoke
```

formal R5：

```powershell
python experiments\ablations\run_ablation_train.py --ablation-id R5 --run-stage formal -- --device cuda --total-env-steps 500000 --final-greedy-episodes 100
```

## 批量运行

批量入口 `run_ablation_batch.py` 会默认从 `experiment_records/full_method_main/logs/config_snapshot.json` 读取 A Full method 的 `full_train_config`，用它对齐地图、训练预算、学习超参数、seed policy、formal protocol 和 reward 基准参数，然后按顺序调用单个消融启动器。

当前 A config 如果记录了 `train_side_only_tuning=true`，批量 run 也会默认保持该模式。这种模式不会生成完整 final probe，不能直接作为最终论文 Results。后续需要 final_probe 时，应使用完整 formal protocol 重新评估，或通过额外参数显式关闭 train-side-only 模式，例如在确认协议后使用 `--extra-train-args "--no-train-side-only-tuning"`。

真实批量运行要求上述 base config 文件存在并能解析；dry-run 在本地缺少该文件时只会打印 warning 并使用当前 `TrainConfig` 默认值预览命令。

dry-run 推荐组：

```powershell
python experiments\ablations\run_ablation_batch.py --preset recommended_first_batch --dry-run
```

正式跑推荐组：

```powershell
python experiments\ablations\run_ablation_batch.py --preset recommended_first_batch --run-stage formal --device cuda
```

跑完整 F/R 组：

```powershell
python experiments\ablations\run_ablation_batch.py --preset full_fr_batch --run-stage formal --device cuda
```

仅跑指定实验：

```powershell
python experiments\ablations\run_ablation_batch.py --ablation-ids F1,F4,F5,R5 --run-stage formal --device cuda
```

每个 run 完成后，脚本只会把 curated logs 复制到 `experiment_records/ablations/<ablation_dir>/logs/`，并更新对应的 `run_record.md`；不会复制 checkpoints、模型权重、完整 outputs、replay buffer、plots 或 debug/profile 文件。
