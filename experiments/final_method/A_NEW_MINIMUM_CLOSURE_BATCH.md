# A_new Minimum-Closure Batch Launcher

This batch launcher is an orchestration entry point for starting the staged
minimum-closure train-side runs after the final A_new candidate training
configuration has been reviewed and frozen.

Default run set:

1. `Anew_C_local_state_ddqn`
2. `Anew_D_no_value_tree`
3. `Anew_E_no_dual_state_split`
4. `Anew_F3_no_behavior_memory`
5. `Anew_R5` as `R_key` / `no_efficiency_penalties`

The default run set does not include `A_new`, because A_new is being tuned
separately. It also does not include `Anew_B_classical_frontier_greedy`, because
B is a CPU non-learning benchmark that can be run independently or with
`-IncludeB`.

## Parameter Boundary

The batch launcher follows the current A_new default training configuration at
execution time. It does not hardcode final training parameter values. Current
A_new training parameters are frozen to the AN_tuned_v1 last.pt-oriented formal
training contract.

If the A_new final candidate changes later, update the default `TrainConfig` or
the relevant A_new runner defaults before running a new matrix. Final-probe
evaluation is intentionally deferred to a later unified last.pt evaluation pass.

## Commands

Dry-run the default formal plan without starting training:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_a_new_minimum_closure_batch.ps1 -RunStage formal -Device cuda -DryRun
```

Run the default formal plan after the A_new candidate has been frozen:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_a_new_minimum_closure_batch.ps1 -RunStage formal -Device cuda
```

Include the optional B CPU benchmark:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_a_new_minimum_closure_batch.ps1 -RunStage formal -Device cuda -IncludeB
```

Run the full reward-analysis enhancement instead of only `R_key`:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_a_new_minimum_closure_batch.ps1 -RunStage formal -Device cuda -IncludeAllRewardAblations
```

Use `-ContinueOnFailure` to continue later commands after a child failure. The
default behavior is to stop at the first failure.

## Artifact Archive

The formal child runners write full raw run directories under `outputs/`. In
addition, successful C/D/E/F_key runs copy all top-level `logs/` files to:

```text
experiment_records/final_method/A_new_minimum_closure/<method_id>/logs/
```

They also copy `checkpoints/last.pt` to:

```text
checkpoint_store/final_method/A_new_minimum_closure/<method_id>.pt
```

R_key keeps its existing reward-ablation archive paths:
`experiment_records/final_method/A_new_reward_ablations/<method_id>/logs/` and
`checkpoint_store/final_method/A_new_reward_ablations/<method_id>.pt`.

## Evidence Boundary

Smoke and pilot runs are local checks only, not paper Results. Formal
train-side-only outputs can support contract-aligned comparisons after artifact
audit, but they do not automatically replace unrun final-probe evidence.

Do not commit `outputs/`, `checkpoint_store/`, `checkpoints/`, or checkpoint
files such as `.pt`, `.pth`, and `.ckpt`.
