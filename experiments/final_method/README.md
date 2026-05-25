# Final Method Launchers

This directory is the active experiment entry point for main. It keeps only the
A_new final method and the Anew_R1-Anew_R5 reward ablations.

## A_new Final 4-Channel Method

`A_new` is the current full_method_main:

- `method_id = A_new`
- `method_name = final_4ch_no_frontier_raster`
- `advantage_canvas_schema = final_4ch_no_frontier_raster`
- advantage canvas channels: `free`, `obstacle`, `visit_count_log_norm`, `recent_trajectory_decay`
- `frontier_raster_used = false`
- `value_tree_enabled = true`
- `value_branch_source = SharedSemanticSnapshot`
- `value_branch_representation = structured_frontier_block_value_tree`
- `model_class = ExplorationQNetwork`
- `advantage_encoder.canvas_in_channels = 4`

The advantage branch no longer uses a frontier raster. Frontier and unknown-block
semantics remain in the structured value-tree branch built from
`SharedSemanticSnapshot`.

Legacy A/F1/F6/F7/ABCDEFR experiment entries and records were removed from active
main and archived before cleanup at:

- branch: `legacy/pre-a-new-cleanup`
- tag: `legacy-pre-a-new-cleanup-20260525`

## Reward Ablations

The supported reward ablation launchers are Anew_R1 through Anew_R5. Every one of
them keeps the A_new final 4-channel schema and changes only the reward override:

- `Anew_R1`: `reward_step_penalty = 0.0`
- `Anew_R2`: `reward_revisit_penalty = 0.0`
- `Anew_R3`: `reward_turn_penalty_scale = 0.0`
- `Anew_R4`: `reward_timeout_penalty = 0.0`
- `Anew_R5`: all four efficiency penalties above set to `0.0`

Smoke is the default stage. Formal 500000-step training must be requested
explicitly.

Dry-run A_new:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_a_new_final_4ch.ps1 -RunStage smoke -Device cpu -DryRun
```

Smoke A_new:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_a_new_final_4ch.ps1 -RunStage smoke -Device cpu
```

Dry-run reward ablations:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_a_new_reward_ablations.ps1 -RunStage smoke -Device cpu -DryRun
```
