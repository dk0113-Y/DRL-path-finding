# DRL-path-finding

这是一个基于 PyTorch 的二维栅格地图自主探索强化学习工程。当前 `main`
已经清理为 A_new final 4-channel 主线：advantage 分支只接收局部 4 通道
canvas，value 分支继续接收由 `SharedSemanticSnapshot` 构造的 structured
frontier-block value tree。

## 当前主线

`A_new` 是新的 full_method_main：

- `method_id = A_new`
- `method_name = final_4ch_no_frontier_raster`
- `advantage_canvas_schema = final_4ch_no_frontier_raster`
- `advantage_canvas_channels = ["free", "obstacle", "visit_count_log_norm", "recent_trajectory_decay"]`
- `advantage_canvas_channel_count = 4`
- `frontier_raster_used = false`
- `value_tree_enabled = true`
- `value_branch_source = SharedSemanticSnapshot`
- `value_branch_representation = structured_frontier_block_value_tree`
- `model_class = ExplorationQNetwork`

## Candidate Formal Defaults

A_new formal defaults are aligned to the matched legacy A/F1 training
configuration for a controlled A_new rerun. This is a candidate formal
configuration pending validation, not an optimal, best, or frozen formal
setting.

- `reward_info_scale = 3.1`
- `reward_obstacle_weight = 0.2`
- `learner_updates_per_iter = 1`
- `min_replay_size = 8000`
- `epsilon_end = 0.04`
- `epsilon_decay_steps = 240000`
- `train_side_only_tuning = true`

The method contract remains A_new: final 4-channel no-frontier-raster advantage
canvas, structured value tree from `SharedSemanticSnapshot`, and
`ExplorationQNetwork`. Legacy 5-channel frontier raster inputs are not restored.

Advantage canvas 不再包含 frontier raster。frontier、unknown block 和 frontier
cluster 语义仍保留在 shared semantic layer 与 value tree 中，用于 value branch。

旧 A/F1/F6/F7/ABCDEFR 实验入口、旧 frontier-raster diagnostics 和旧结果记录已从
active main 移除，并在清理前归档到：

- branch: `legacy/pre-a-new-cleanup`
- tag: `legacy-pre-a-new-cleanup-20260525`

## 代码结构

- `train_q_agent.py`: 主训练入口，负责配置、系统组装、训练循环、checkpoint
  selection、final probe 和 formal artifacts。
- `agents/q_value_agent.py`: `ExplorationQNetwork` 与 `StateTensorAdapter`。
- `env/advantage_state_builder.py`: A_new final 4-channel advantage canvas。
- `env/shared_semantic_layer.py`: shared semantic snapshot，包括 frontier / unknown
  block / cluster 构造逻辑。
- `env/value_state_builder.py`: structured frontier-block value tree。
- `experiments/final_method/`: A_new final method, Anew_R1-Anew_R5 reward
  ablations, Anew_B classical frontier greedy baseline, Anew_C local-state DDQN
  learning baseline, Anew_D no-value-tree, and Anew_F3 no-behavior-memory
  launchers.

## 运行方式

A_new smoke dry-run：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_a_new_final_4ch.ps1 -RunStage smoke -Device cpu -DryRun
```

A_new smoke：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_a_new_final_4ch.ps1 -RunStage smoke -Device cpu
```

A_new formal：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_a_new_final_4ch.ps1 -RunStage formal -Device cuda
```

Anew_R1-Anew_R5 reward ablation dry-run：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_a_new_reward_ablations.ps1 -RunStage smoke -Device cpu -DryRun
```

Anew_R1-Anew_R5 formal：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_a_new_reward_ablations.ps1 -RunStage formal -Device cuda -NoCopyCheckpoints
```

`Anew_R1` through `Anew_R5` all bind to the A_new final 4-channel schema. Their
reward overrides are:

- `Anew_R1`: `reward_step_penalty = 0.0`
- `Anew_R2`: `reward_revisit_penalty = 0.0`
- `Anew_R3`: `reward_turn_penalty_scale = 0.0`
- `Anew_R4`: `reward_timeout_penalty = 0.0`
- `Anew_R5`: all four efficiency penalties above set to `0.0`

## Anew_B Classical Frontier Greedy Baseline

`Anew_B_classical_frontier_greedy` is the A_new-aligned B group classical
frontier greedy baseline. It is a traditional non-learning baseline: it does
not train a model, does not load a checkpoint, and does not use
`ExplorationQNetwork`.

The policy uses only belief-derived frontier/shared-semantic state, the current
pose, and valid action indices. It selects the lowest-BFS-cost reachable
frontier target over known-free belief cells, then takes a valid action toward
that target. If no reachable frontier exists, it falls back to deterministic
immediate information-gain over currently unknown belief cells, then to the
first valid action in `ACTIONS_8` order.

The runner keeps the current A_new environment, reward, seed, and metric
contract, including the default reward parameters. It does not restore legacy B
artifacts, does not inherit the old baseline framework, and does not restore
`baselines/` or `experiments/ablations/`. Simulator internals may use the map for
stepping, sensing, termination, and metrics, but the policy decision path does
not receive the full map.

Smoke and pilot runs are local checks only, not paper Results. A B formal
benchmark can support a classical baseline comparison after artifact review, but
it cannot replace D/F/R internal ablations or explain neural representation
contributions.

Anew_B dry-run:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_a_new_classical_frontier_baseline.ps1 -RunStage formal -Device cpu -DryRun
```

Anew_B smoke:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_a_new_classical_frontier_baseline.ps1 -RunStage smoke -Device cpu
```

Anew_B formal benchmark:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_a_new_classical_frontier_baseline.ps1 -RunStage formal -Device cpu
```

## Anew_C Local-State DDQN Learning Baseline

`Anew_C_local_state_ddqn` is the A_new-aligned C group learning baseline. It is a
simpler local-state DDQN comparison row, not an A_new ablation and not a
replacement for B/D/F/R.

- `experiment_id = Anew_C`
- `method_id = Anew_C_local_state_ddqn`
- `method_name = local_state_ddqn`
- `baseline_group = learning`
- `baseline_type = learning_ddqn`
- `model_class = LocalStateQNetwork`
- local input: `known_free`, `known_obstacle`, `unknown`
- local patch size: `2 * scan_radius + 1`, currently `21`
- input source: cumulative belief patch around the current agent pose
- no structured value tree, no SharedSemanticSnapshot value branch, no behavior-memory channels, no frontier raster
- no full map or ground-truth map is used for model decisions
- no legacy C artifact, legacy `baselines/`, or `experiments/ablations/` framework is restored

C uses the current A_new environment, reward, seed, and train-side-only metric
contract, including the matched default reward/training parameters and
`train_side_only_tuning = true`. Smoke and pilot runs are local checks only, not
paper Results. A formal train-side-only C run can support a simpler learning
baseline comparison only after its artifact package is audited.

Anew_C dry-run:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_a_new_local_state_ddqn_baseline.ps1 -RunStage formal -Device cuda -DryRun
```

Anew_C smoke:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_a_new_local_state_ddqn_baseline.ps1 -RunStage smoke -Device cpu
```

Anew_C formal train-side-only:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_a_new_local_state_ddqn_baseline.ps1 -RunStage formal -Device cuda
```

## Anew_D No-Value-Tree Structural Ablation

`Anew_D_no_value_tree` is the A_new-aligned D group implementation from the
planned matrix. It tests the overall contribution of the structured
frontier-block value tree while preserving the current A_new local advantage
canvas:

- `method_id = Anew_D_no_value_tree`
- `method_name = no_value_tree`
- `ablation_group = structural`
- `ablation_name = no_value_tree`
- `advantage_canvas_schema = final_4ch_no_frontier_raster`
- `advantage_canvas_channel_count = 4`
- `frontier_raster_used = false`
- `value_tree_enabled = false`
- `value_replacement_strategy = zero_value_state`
- `reward_override = {}`
- `train_side_only_tuning = true` by default

This row does not restore the legacy 5-channel advantage canvas, does not
restore `frontier_block_area_map`, and does not inherit any legacy D artifacts.
It uses the current A_new matched default training parameters. Smoke and pilot
runs are local checks only, not paper Results. Formal train-side-only outputs can
be compared to the current A_new train-side contract, but they do not replace
unrun final-probe evidence.

Anew_D dry-run:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_a_new_no_value_tree_ablation.ps1 -RunStage formal -Device cuda -DryRun
```

Anew_D smoke:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_a_new_no_value_tree_ablation.ps1 -RunStage smoke -Device cpu
```

Anew_D formal train-side-only:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_a_new_no_value_tree_ablation.ps1 -RunStage formal -Device cuda
```

## Anew_F3 No-Behavior-Memory F_key Ablation

`Anew_F3_no_behavior_memory` is the A_new-aligned F_key input-state ablation. It
keeps the current A_new 4-channel no-frontier-raster advantage canvas schema,
retains `free` and `obstacle`, and zeros the behavior-memory channels
`visit_count_log_norm` and `recent_trajectory_decay`.

- `experiment_id = Anew_F`
- `method_id = Anew_F3_no_behavior_memory`
- `method_name = no_behavior_memory`
- `ablation_group = input_state`
- `channel_ablation = no_behavior_memory`
- `zeroed_advantage_channels = ["visit_count_log_norm", "recent_trajectory_decay"]`
- `advantage_canvas_schema = final_4ch_no_frontier_raster`
- `advantage_canvas_channel_count = 4`
- `frontier_raster_used = false`
- `value_tree_enabled = true`
- `value_tree_unchanged = true`
- `reward_override = {}`
- `train_side_only_tuning = true` by current default

Under the current schema, this is equivalent to an occupancy-only advantage
canvas, so `Anew_F4_occupancy_only` is only an alias-level explanation and is not
kept as a separate formal row, run name, or artifact row. This row does not
restore legacy 5-channel inputs, does not restore `frontier_block_area_map`, and
does not inherit legacy F artifacts. It keeps the current A_new matched default
training parameters and does not change reward defaults.

Smoke and pilot runs are local checks only, not paper Results. Formal
train-side-only outputs can be used for contract-aligned comparison against the
current A_new train-side-only runs, but they do not automatically substitute for
unrun final-probe evidence.

Anew_F3 dry-run:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_a_new_no_behavior_memory_ablation.ps1 -RunStage formal -Device cuda -DryRun
```

Anew_F3 smoke:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_a_new_no_behavior_memory_ablation.ps1 -RunStage smoke -Device cpu
```

Anew_F3 formal train-side-only:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_a_new_no_behavior_memory_ablation.ps1 -RunStage formal -Device cuda
```

## A_new Minimum-Closure Batch

`scripts/run_a_new_minimum_closure_batch.ps1` is a batch orchestration launcher
for starting the staged minimum-closure train-side experiments after the final
A_new candidate training configuration has been reviewed and frozen.

Default formal run set:

- `Anew_C_local_state_ddqn`
- `Anew_D_no_value_tree`
- `Anew_E_no_dual_state_split`
- `Anew_F3_no_behavior_memory` as `F_key`
- `Anew_R5` as `R_key` / `no_efficiency_penalties`

The batch follows the current A_new default training configuration at execution
time and does not hardcode final training parameter values. A_new parameters are
still candidate / tuning; the formal configuration is not yet frozen. After the
A_new final candidate is determined, update the default `TrainConfig` or the
relevant A_new runner defaults, then run the batch.

The default run set does not include `A_new`, because A_new is tuned separately.
It also does not include `Anew_B_classical_frontier_greedy`; B is an optional CPU
non-learning benchmark available with `-IncludeB` or as a separate run. Use
`-IncludeAllRewardAblations` only when running the full R1-R5 reward-analysis
enhancement.

Minimum-closure dry-run:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_a_new_minimum_closure_batch.ps1 -RunStage formal -Device cuda -DryRun
```

Smoke and pilot runs are local checks only, not paper Results. Formal
train-side-only outputs require artifact audit before being written into
paper_work, and they do not automatically replace unrun final-probe evidence.

## Repository Hygiene

Do not commit `outputs/`, `checkpoint_store/`, `checkpoints/`, or checkpoint files
such as `.pt`, `.pth`, and `.ckpt`. Smoke outputs are local artifacts only.
