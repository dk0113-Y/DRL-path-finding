# A_new F_key No-Behavior-Memory Ablation

`Anew_F3_no_behavior_memory` is the A_new-aligned F_key input-state ablation. It
tests the aggregate contribution of the behavior-memory channels inside the
advantage branch while preserving the current A_new method contract.

## Contract

- `experiment_id = Anew_F`
- `method_id = Anew_F3_no_behavior_memory`
- `method_name = no_behavior_memory`
- `ablation_group = input_state`
- `ablation_name = no_behavior_memory`
- `channel_ablation = no_behavior_memory`
- `zeroed_advantage_channels = ["visit_count_log_norm", "recent_trajectory_decay"]`
- `advantage_canvas_schema = final_4ch_no_frontier_raster`
- `advantage_canvas_channel_count = 4`
- `frontier_raster_used = false`
- `value_tree_enabled = true`
- `value_tree_unchanged = true`
- `reward_override = {}`
- `train_side_only_tuning = true` by current default

## Scope

The advantage canvas schema remains the A_new four-channel no-frontier-raster
schema:

1. `free`, kept unchanged.
2. `obstacle`, kept unchanged.
3. `visit_count_log_norm`, zeroed after canvas construction.
4. `recent_trajectory_decay`, zeroed after canvas construction.

The tensor shape remains 4-channel and the `ExplorationQNetwork` parameter count
stays aligned with A_new. The structured frontier-block value tree remains
enabled and is still built from `SharedSemanticSnapshot`; this row does not use
the D-group `zero_value_state` path.

Under the current A_new schema, this operation is equivalent to an occupancy-only
advantage canvas. `Anew_F4_occupancy_only` is therefore an alias-level
description only and is not kept as a separate formal experiment row, run name,
or artifact row.

This row does not restore the legacy 5-channel advantage canvas, does not
restore `frontier_block_area_map`, and does not inherit legacy F artifacts.

Smoke and pilot runs are local checks only, not paper Results. Formal
train-side-only outputs can be compared against the current A_new train-side
contract, but they do not automatically replace unrun final-probe evidence.

## Commands

Dry-run:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_a_new_no_behavior_memory_ablation.ps1 -RunStage formal -Device cuda -DryRun
```

Smoke:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_a_new_no_behavior_memory_ablation.ps1 -RunStage smoke -Device cpu
```

Formal train-side-only:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_a_new_no_behavior_memory_ablation.ps1 -RunStage formal -Device cuda
```
