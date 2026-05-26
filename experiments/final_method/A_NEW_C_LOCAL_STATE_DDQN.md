# Anew_C Local-State DDQN Baseline

`Anew_C_local_state_ddqn` is the A_new-aligned C group learning baseline. It is
a simpler local-state DDQN comparison row, not an A_new structural/input/reward
ablation.

## Contract

- `experiment_id = Anew_C`
- `method_id = Anew_C_local_state_ddqn`
- `method_name = local_state_ddqn`
- `baseline_group = learning`
- `baseline_type = learning_ddqn`
- `model_class = LocalStateQNetwork`
- `is_learning_baseline = true`
- `is_ablation = false`

## Input

The model input is a local 3-channel cumulative-belief patch centered on the
current agent pose:

- `known_free`
- `known_obstacle`
- `unknown`

Patch size is `2 * scan_radius + 1`. Under the current A_new default
`scan_radius = 10`, C uses `21 x 21`.

C does not use visit-count memory, recent-trajectory decay, a frontier raster,
frontier-block area maps, structured value-tree tensors, oracle paths, or
full-map decision inputs. The simulator may still use its internal map for
stepping, sensing, termination, and metrics.

## Training Alignment

C reuses the current A_new DDQN training loop through explicit factories:

- local-state tensor adapter: `experiments/final_method/a_new_local_state_ddqn.py`
- model: `agents/local_state_q_network.py`
- runner: `experiments/final_method/run_a_new_local_state_ddqn_baseline.py`
- launcher: `scripts/run_a_new_local_state_ddqn_baseline.ps1`

The runner keeps the current A_new environment, reward, seed, and metric
contract:

- `reward_override = {}`
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

Value tensors are all-zero interface tensors with false masks. They exist only
because the current DDQN replay/learner interface expects value tensor keys.
`LocalStateQNetwork` ignores these tensors, and metadata records
`value_tensors_used_by_model = false`.

## Evidence Boundary

Smoke and pilot runs are local checks only and are not paper Results. A formal
train-side-only C run can support a simpler learning baseline comparison only
after its artifact package is checked. C cannot replace:

- B classical frontier-greedy baseline
- D no-value-tree structural ablation
- F_key no-behavior-memory input-state ablation
- R reward ablations

C does not restore legacy C artifacts, legacy `baselines/`, or the old
`experiments/ablations/` framework.

## Commands

Dry-run:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_a_new_local_state_ddqn_baseline.ps1 -RunStage formal -Device cuda -DryRun
```

Smoke:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_a_new_local_state_ddqn_baseline.ps1 -RunStage smoke -Device cpu
```

Formal train-side-only:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_a_new_local_state_ddqn_baseline.ps1 -RunStage formal -Device cuda
```
