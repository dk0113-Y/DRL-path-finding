# F/R 消融结果记录

本目录保存 F/R 消融实验的 formal logs。这里只保存 curated logs 和人工记录，不保存 outputs 原目录、checkpoints、模型权重、replay buffer 或大体积调试文件。

## F 组目录

- `ablation_no_frontier_channel`
- `ablation_no_visit_count_channel`
- `ablation_no_recent_trajectory_channel`
- `ablation_no_visit_traj_channels`
- `ablation_occupancy_only_canvas`

## R 组目录

- `ablation_no_step_penalty`
- `ablation_no_revisit_penalty`
- `ablation_no_turn_penalty`
- `ablation_no_timeout_penalty`
- `ablation_no_efficiency_penalties`
- `ablation_sparse_reward_variant`

每个目录的 `logs/` 子目录用于放置经过筛选的 formal artifacts。smoke/pilot run 不进入论文 Results，也不应作为主表 evidence。
