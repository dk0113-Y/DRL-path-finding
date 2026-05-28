# Final Method Launchers

This directory is the active experiment entry point for main. It keeps the A_new
final method, Anew_R1-Anew_R5 reward ablations, the A_new-aligned B classical
frontier greedy baseline, the A_new-aligned C local-state DDQN learning
baseline, and A_new-aligned D/F_key ablation launchers.

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

## Frozen V1 Formal Defaults

The current A_new formal defaults are frozen to the AN_tuned_v1 last.pt-oriented
training contract. The formal experiment matrix inherits these defaults without
per-run hyperparameter overrides. Paper-facing held-out comparison is recorded
by the unified final probe under
`experiment_records/final_method/unified_final_probe/`.

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

A_new still uses the final 4-channel no-frontier-raster schema; no legacy
5-channel frontier raster or F1 compatibility experiment is restored.

Legacy A/F1/F6/F7/ABCDEFR experiment entries and records were removed from active
`main`. The remote repository is intended to maintain `main` as the only branch;
the historical cleanup state is preserved by tag only:

- tag: `legacy-pre-a-new-cleanup-20260525`

## Reward Ablations

The supported reward ablation launchers are Anew_R1 through Anew_R5. Every one of
them keeps the A_new final 4-channel schema and changes only the reward override:

- `Anew_R1`: `reward_step_penalty = 0.0`
- `Anew_R2`: `reward_revisit_penalty = 0.0`
- `Anew_R3`: `reward_turn_penalty_scale = 0.0`
- `Anew_R4`: `reward_timeout_penalty = 0.0`
- `Anew_R5`: all four efficiency penalties above set to `0.0`

Smoke is the default stage. Formal 650000-step training must be requested
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
train a neural model, load a checkpoint, or use `ExplorationQNetwork`.

The policy restores the legacy `classical_frontier_greedy_v1` decision logic
from `DRL_PF` while running inside the current A_new environment and metric
contract. The decision path uses only belief-derived frontier geometry,
`SharedSemanticSnapshot` candidates, current pose, valid action indices,
frontier cache, visit counts, and recent trajectory. BFS is used only to select
the reachable frontier target; the next action follows the legacy
squared-Euclidean-distance, recent-revisit, visit-count, fixed-action tie-break.
If no reachable frontier exists, it chooses the valid next action with the
largest belief-only radar line-of-sight immediate information gain.

The runner uses the current A_new environment, reward, seed, and metric
contract. It restores legacy B policy logic only; it does not restore legacy B
artifacts, old checkpoint flows, `baselines/`, or `experiments/ablations/`.
Simulator-side map access is limited to environment stepping, sensor updates,
termination, and metric computation; the policy is not given the full map. Smoke
and pilot runs are not Results. Formal B benchmark artifacts support the
classical baseline comparison, but B cannot replace D/F/R internal ablations or
neural representation evidence.

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

## Anew_C Local-State DDQN Learning Baseline

`Anew_C_local_state_ddqn` is the A_new-aligned C group learning baseline. It
tests A_new against a simpler DDQN agent whose model input is only a local
belief patch:

- channels: `known_free`, `known_obstacle`, `unknown`
- patch size: `2 * scan_radius + 1`; under the current `scan_radius = 10`, this is `21 x 21`
- source: cumulative belief map sampled around the current agent pose
- carrier key: `advantage_canvas`, with `local_state_canvas_role = baseline_local_state_input`
- model: `LocalStateQNetwork`

This row does not use the structured frontier-block value tree,
`SharedSemanticSnapshot` as a value branch, behavior-memory channels,
frontier raster input, full-map decision input, or any legacy C artifact. It
also does not restore the old `baselines/` or `experiments/ablations/`
frameworks. Zero value tensors are supplied only to satisfy the existing DDQN
interface, and `LocalStateQNetwork` ignores them.

C keeps the current A_new environment, reward, seed, and train-side-only metric
contract, including `reward_override = {}` and `train_side_only_tuning = true`.
Smoke and pilot are local checks only. Formal train-side-only artifacts can
support a simpler learning-baseline comparison after artifact review, but C does
not replace B classical comparison or D/F/R internal ablations.

Dry-run C:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_a_new_local_state_ddqn_baseline.ps1 -RunStage formal -Device cuda -DryRun
```

Smoke C:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_a_new_local_state_ddqn_baseline.ps1 -RunStage smoke -Device cpu
```

Formal train-side-only C:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_a_new_local_state_ddqn_baseline.ps1 -RunStage formal -Device cuda
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
used for contract-aligned comparison against A_new train-side-only runs.
Paper-facing held-out comparison is recorded by the unified final probe.

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
contract. Paper-facing held-out comparison is recorded by the unified final
probe.

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

## A_new Minimum-Closure Batch

`run_a_new_minimum_closure_batch.py` and
`scripts/run_a_new_minimum_closure_batch.ps1` provide a one-command batch for
the staged minimum-closure train-side runs under the frozen A_new formal
defaults.

Default run order:

1. `Anew_C_local_state_ddqn`
2. `Anew_D_no_value_tree`
3. `Anew_E_no_dual_state_split`
4. `Anew_F3_no_behavior_memory` as `F_key`
5. `Anew_R5` as `R_key` / `no_efficiency_penalties`

The batch launcher follows the current A_new default training configuration at
execution time. It does not hardcode final training parameter values. A_new
parameters are frozen to the AN_tuned_v1 last.pt-oriented formal training
contract.

The default batch does not run `A_new`, because A_new is tuned separately. It
also does not run `Anew_B_classical_frontier_greedy`; B can be run independently
or included with `-IncludeB`, and the batch launches B as a CPU non-learning
benchmark. `R_key` means `Anew_R5`; `-IncludeAllRewardAblations` expands the R
command to `Anew_R1` through `Anew_R5` for full reward analysis.

Successful C/D/E/F_key runs archive all top-level `logs/` files to
`experiment_records/final_method/A_new_minimum_closure/<method_id>/logs/` and
copy `checkpoints/last.pt` to
`checkpoint_store/final_method/A_new_minimum_closure/<method_id>.pt`. R_key
keeps the existing reward-ablation archive roots under
`experiment_records/final_method/A_new_reward_ablations/` and
`checkpoint_store/final_method/A_new_reward_ablations/`, including the audited
formal training CSVs and benchmark manifest files.

Dry-run the formal plan:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_a_new_minimum_closure_batch.ps1 -RunStage formal -Device cuda -DryRun
```

## Unified Final Probe

After the `last.pt` checkpoints are available, run the unified held-out final
probe with:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_a_new_unified_final_probe.ps1 -Device cuda
```

The runner evaluates B first with `ClassicalFrontierGreedyPolicy`, then evaluates
`A`, `C`, `D`, `E`, `F_key`, and `R_key` from checkpoint-store `last.pt` files.
All methods use the same fixed final-probe seed block. The default is 100
episodes with `seed_base = 20261323`.

Outputs are written under
`experiment_records/final_method/unified_final_probe/<run_id>/`, including one
`final_probe.csv` per method and a global
`unified_final_probe_summary.csv` / `.json`.

Smoke and pilot stages are local checks only, not paper Results. Formal
train-side-only outputs can support contract-aligned comparisons after artifact
audit. Paper-facing held-out comparison is recorded by the unified final probe.
Do not commit `outputs/`, `checkpoint_store/`, `checkpoints/`, or checkpoint
files.

## Environment-Shift Final Probe

The environment-shift final probe is a controlled sensitivity check for the
same trained checkpoints. It does not retrain, change reward logic, change model
architecture, or change the training loop. The runner only overrides evaluation
environment fields after loading each checkpoint's `train_config`; checkpoint
files and training configs are not modified.

Scenario definitions live in
`experiments/final_method/environment_shift_scenarios.json`:

| Scenario | rows | cols | obstacle_ratio | max_episode_steps | coverage_stop_threshold | seed_base |
|---|---:|---:|---:|---:|---:|---:|
| `S1_low_density` | 40 | 60 | 0.10 | 600 | 0.95 | 20271323 |
| `S2_high_density` | 40 | 60 | 0.30 | 600 | 0.95 | 20271423 |
| `S3_larger_same_density` | 50 | 70 | 0.20 | 600 | 0.95 | 20271523 |

The default matrix is `S1/S2/S3 x A/B/D/E/R x 100 episodes`. Labels are
paper-facing labels. Internally, `R` maps to the `R_key` / `Anew_R5`
checkpoint provenance, and `F` maps to `F_key` if requested, but `F` is not part
of the default environment-shift matrix.

Dry-run the full matrix:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_a_new_environment_shift_final_probe.ps1 -Device cuda -DryRun
```

Run smoke validation only:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_a_new_environment_shift_final_probe.ps1 -Device cpu -Smoke
```

Smoke runs use 2 episodes per scenario/group and write under
`experiment_records\final_method\environment_shift_final_probe\smoke\`. Smoke
and pilot outputs are local execution checks only and must not enter paper
Results.

Run the formal environment-shift matrix:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_a_new_environment_shift_final_probe.ps1 -Device cuda
```

Formal outputs are written under
`experiment_records\final_method\environment_shift_final_probe\<run_id>\`, with
one scenario-specific run directory per scenario. Each run writes
`run_manifest.json`, `scenario_manifest.json`,
`unified_final_probe_summary.csv`, `unified_final_probe_summary.json`, and one
`final_probe.csv` per requested paper-facing group.

Do not commit `checkpoint_store/`, `checkpoints/`, `outputs/`, formal or smoke
raw probe outputs, heavy logs, or `.pt` / `.pth` / `.ckpt` files. After the
formal environment-shift run finishes, audit and summarize the artifacts first;
only then copy curated lightweight table or figure candidates into the paper
workspace.
