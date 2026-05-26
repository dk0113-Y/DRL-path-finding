# Anew_D No-Value-Tree Structural Ablation

`Anew_D_no_value_tree` implements the planned D group for the current A_new
matrix. It is an A_new-aligned structural ablation, not a restoration of the
legacy D experiment.

## Contract

- `experiment_id = Anew_D`
- `method_id = Anew_D_no_value_tree`
- `method_name = no_value_tree`
- `ablation_group = structural`
- `ablation_name = no_value_tree`
- `advantage_canvas_schema = final_4ch_no_frontier_raster`
- `frontier_raster_used = false`
- `value_tree_enabled = false`
- `value_replacement_strategy = zero_value_state`
- `reward_override = {}`
- `train_side_only_tuning = true` by default

The advantage canvas remains the current A_new four-channel schema:

- `free`
- `obstacle`
- `visit_count_log_norm`
- `recent_trajectory_decay`

The value-tree branch receives zero block and entry feature tensors with the
same shapes as A_new and all value masks set false. This removes structured
frontier-block value-tree information while preserving the
`ExplorationQNetwork` interface and parameter count.

## Boundaries

This implementation does not restore the old 5-channel canvas, does not restore
`frontier_block_area_map`, and does not inherit legacy D artifacts. It keeps the
current A_new matched default training parameters:

- `reward_info_scale = 3.1`
- `reward_obstacle_weight = 0.2`
- `learner_updates_per_iter = 1`
- `min_replay_size = 8000`
- `total_env_steps = 650000`
- `epsilon_end = 0.03`
- `epsilon_decay_steps = 300000`
- `reward_revisit_penalty = 0.12`
- `reward_turn_penalty_scale = 0.06`
- `reward_timeout_penalty = 10.0`
- `train_side_only_tuning = true`

Smoke and pilot runs are local checks only and are not paper Results evidence.
Formal train-side-only artifacts can be compared against the current A_new
train-side contract, but they cannot replace unrun final-probe evidence.

## Commands

Dry-run:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_a_new_no_value_tree_ablation.ps1 -RunStage formal -Device cuda -DryRun
```

Smoke:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_a_new_no_value_tree_ablation.ps1 -RunStage smoke -Device cpu
```

Formal train-side-only:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_a_new_no_value_tree_ablation.ps1 -RunStage formal -Device cuda
```
