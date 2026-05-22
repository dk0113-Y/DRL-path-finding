# F/R 消融结果记录

本目录保存 F/R 消融实验的 curated formal logs 和人工记录，不保存 outputs 原目录、checkpoints、模型权重、replay buffer 或大体积调试文件。

canonical `ablation_id` 和 `short_id` 保持不变；文件系统目录统一使用：

```text
<short_id>_ablation_<canonical_id>
```

outputs run_name、`experiment_records` 目录和 checkpoint_store 推荐文件名应使用同一个 slug。checkpoint_store 被 Git 忽略，不应提交 `.pt`、`.pth` 或 `.ckpt` 文件；本目录只提交 curated logs。

## F 组目录

- `F1_ablation_no_frontier_channel`
- `F2_ablation_no_visit_count_channel`
- `F3_ablation_no_recent_trajectory_channel`
- `F4_ablation_no_visit_traj_channels`
- `F5_ablation_occupancy_only_canvas`

## R 组目录

- `R1_ablation_no_step_penalty`
- `R2_ablation_no_revisit_penalty`
- `R3_ablation_no_turn_penalty`
- `R4_ablation_no_timeout_penalty`
- `R5_ablation_no_efficiency_penalties`
- `R6_ablation_sparse_reward_variant`

每个目录的 `logs/` 子目录用于放置经过筛选的 formal artifacts。smoke/pilot run 不进入论文 Results，也不应作为主表 evidence。
