# Final Method Launchers

This directory is the active experiment entry point for main. It keeps the A_new
final method, Anew_R1-Anew_R5 reward ablations, the A_new-aligned B classical
frontier greedy baseline, and A_new-aligned D/F_key ablation launchers.

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

## Formal Defaults

The current A_new formal defaults are aligned to the matched legacy A/F1
training configuration for a controlled A_new rerun. This is a candidate formal
configuration pending validation, not an optimal or proven setting.

- `reward_info_scale = 3.1`
- `reward_obstacle_weight = 0.2`
- `learner_updates_per_iter = 1`
- `min_replay_size = 8000`
- `epsilon_end = 0.04`
- `epsilon_decay_steps = 240000`
- `train_side_only_tuning = true`

A_new still uses the final 4-channel no-frontier-raster schema; no legacy
5-channel frontier raster or F1 compatibility experiment is restored.

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

## Anew_B Classical Frontier Greedy Baseline

`Anew_B_classical_frontier_greedy` is the A_new-aligned classical frontier greedy
baseline for the B group. It is a traditional non-learning policy and does not
train a neural model, load a checkpoint, use `ExplorationQNetwork`, restore
legacy B artifacts, or inherit the old baseline framework.

The policy decision path uses only belief-derived frontier geometry,
`SharedSemanticSnapshot` candidates, the current pose, and valid action indices.
It selects the reachable frontier target with minimum BFS cost over known-free
belief cells and takes one valid action toward that target. If no reachable
frontier exists, it chooses the valid next action with the largest immediate
unknown-belief footprint; remaining ties use the repository `ACTIONS_8` order:
N, NE, E, SE, S, SW, W, NW.

The runner uses the current A_new environment, reward, seed, and metric
contract. Simulator-side map access is limited to environment stepping, sensor
updates, termination, and metric computation; the policy is not given the full
map. Smoke and pilot runs are not Results. Formal B benchmark artifacts can
support a classical baseline comparison after artifact review, but B cannot
replace D/F/R internal ablations or neural representation evidence.

Dry-run B:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_a_new_classical_frontier_baseline.ps1 -RunStage formal -Device cpu -DryRun
```

Smoke B:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_a_new_classical_frontier_baseline.ps1 -RunStage smoke -Device cpu
```

Formal benchmark B:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_a_new_classical_frontier_baseline.ps1 -RunStage formal -Device cpu
```

## Anew_D No-Value-Tree Ablation

`Anew_D_no_value_tree` is the A_new-aligned D structural ablation for the planned
matrix. It removes structured frontier-block value-tree information by feeding a
shape-compatible zero value state with all value masks set false. The
`ExplorationQNetwork` interface and parameter count stay aligned with A_new.

The D row keeps the current A_new advantage canvas unchanged:

- `advantage_canvas_schema = final_4ch_no_frontier_raster`
- channels: `free`, `obstacle`, `visit_count_log_norm`, `recent_trajectory_decay`
- `advantage_canvas_channel_count = 4`
- `frontier_raster_used = false`

It does not restore legacy 5-channel inputs, does not restore
`frontier_block_area_map`, and does not inherit legacy D artifacts. It also keeps
the current matched A_new default training parameters and uses `reward_override =
{}`. Smoke and pilot runs are not Results. Formal train-side-only outputs can be
used for contract-aligned comparison against A_new train-side-only runs, but they
do not automatically substitute for unrun final probe evidence.

Dry-run D:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_a_new_no_value_tree_ablation.ps1 -RunStage formal -Device cuda -DryRun
```

Smoke D:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_a_new_no_value_tree_ablation.ps1 -RunStage smoke -Device cpu
```

## Anew_F3 No-Behavior-Memory F_key Ablation

`Anew_F3_no_behavior_memory` is the A_new-aligned F_key input-state ablation. It
keeps the A_new final 4-channel no-frontier-raster schema and changes only the
behavior-memory channels in the advantage branch:

- `free`: kept unchanged
- `obstacle`: kept unchanged
- `visit_count_log_norm`: zeroed
- `recent_trajectory_decay`: zeroed

The row keeps the structured frontier-block value tree enabled and unchanged,
uses `reward_override = {}`, and inherits the current matched A_new default
training parameters, including `train_side_only_tuning = true`.

In the current schema, this is equivalent to an occupancy-only advantage canvas.
`Anew_F4_occupancy_only` is therefore not a separate formal experiment row, run
name, or artifact row. This launcher does not restore legacy 5-channel inputs,
does not restore `frontier_block_area_map`, and does not inherit legacy F
artifacts.

Smoke and pilot runs are local checks only, not Results. Formal train-side-only
outputs can be compared to current A_new train-side-only runs under the same
contract, but they do not automatically replace unrun final-probe evidence.

Dry-run F_key:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_a_new_no_behavior_memory_ablation.ps1 -RunStage formal -Device cuda -DryRun
```

Smoke F_key:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_a_new_no_behavior_memory_ablation.ps1 -RunStage smoke -Device cpu
```

Formal train-side-only F_key:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_a_new_no_behavior_memory_ablation.ps1 -RunStage formal -Device cuda
```
