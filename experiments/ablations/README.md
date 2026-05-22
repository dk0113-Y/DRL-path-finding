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
