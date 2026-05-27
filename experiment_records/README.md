# Experiment Records

`experiment_records/` stores lightweight, paper-facing records for the active
A_new final method, B/C baselines, minimum-closure ablations, reward ablations,
and the unified final probe.

- Active lightweight records belong under `experiment_records/final_method/`.
- The current full-method train-side record is under
  `experiment_records/full_method_main/A_full_method/`.
- The paper-facing held-out comparison is under
  `experiment_records/final_method/unified_final_probe/`.
- Do not commit raw smoke outputs, raw run directories, checkpoint bodies, or
  large logs.
- Do not commit generated plot images; commit only plot manifests if they are
  needed to reproduce a figure.
- Do not commit `outputs/`, `checkpoint_store/`, `checkpoints/`, `.pt`, `.pth`, or
  `.ckpt` files.

`Anew_B_classical_frontier_greedy` records, when formal benchmark runs are
performed, must be treated as A_new-aligned traditional non-learning baseline
records. B does not train a model, does not use a checkpoint, does not use
`ExplorationQNetwork`, and restores only the legacy
`classical_frontier_greedy_v1` policy logic from `DRL_PF`. The policy uses
belief-derived frontier/shared-semantic state, valid action indices, visit
counts, recent trajectory, and frontier cache only; simulator map access is for
stepping, sensing, termination, and metrics. Smoke and pilot outputs are not
Results evidence and should not be written here as formal records.

`Anew_C_local_state_ddqn` records, when formal train-side-only runs are
performed, must be treated as A_new-aligned simpler learning baseline records.
C trains `LocalStateQNetwork` from scratch with a three-channel cumulative-belief
patch (`known_free`, `known_obstacle`, `unknown`) and the current A_new
environment, reward, seed, and train-side-only metric contract. C does not use a
structured value tree, behavior-memory channels, frontier raster, full-map
decision input, legacy C artifacts, legacy `baselines/`, or the old
`experiments/ablations/` framework. Smoke and pilot outputs are not Results
evidence and should not be written here as formal records. Formal C artifacts
can support only a simpler learning-baseline comparison after audit; they cannot
replace B, D, F, or R evidence.

`Anew_D_no_value_tree` records, when formal runs are performed, must be treated
as A_new-aligned no-value-tree structural ablation records. Smoke and pilot
outputs are not Results evidence and should not be written here as formal
records. Formal train-side-only artifacts can support comparison to the current
A_new train-side contract; paper-facing held-out comparison is recorded by the
unified final probe.

`Anew_E_no_dual_state_split` records, when formal runs are performed, must be
treated as A_new-aligned structural ablation records for the dual-state split.
Smoke and pilot outputs are not Results evidence and should not be written here
as formal records. Formal train-side-only artifacts can support comparison to
the current A_new train-side contract; paper-facing held-out comparison is
recorded by the unified final probe.

`Anew_F3_no_behavior_memory` records, when formal runs are performed, must be
treated as A_new-aligned F_key input-state ablation records. The row keeps the
current 4-channel no-frontier-raster schema, preserves `free` and `obstacle`,
zeros `visit_count_log_norm` and `recent_trajectory_decay`, keeps the value tree
enabled, and uses `reward_override = {}`. Under the current schema this is
equivalent to an occupancy-only advantage canvas, but `Anew_F4_occupancy_only`
must not be written as a separate formal row, run name, or artifact row. Smoke
and pilot outputs are not Results evidence and should not be written here as
formal records.

`Anew_R5` records are the current `R_key` reward-ablation evidence row. They
belong under `experiment_records/final_method/A_new_reward_ablations/Anew_R5/`
and should include the audited formal training artifacts, including
`benchmark_summary.json`, `final_method_manifest.json`, `metric_snapshot.json`,
`config_snapshot.json`, `reproducibility_contract.json`, `train_episodes.csv`,
`train_steps.csv`, and `training_summary.txt`. `R_key` keeps the A_new final
4-channel schema and value tree, and changes only the efficiency-penalty reward
terms through the Anew_R5 reward override.

`A_new_minimum_closure_batch` is orchestration only. Its default train-side run
set is `Anew_C_local_state_ddqn`, `Anew_D_no_value_tree`,
`Anew_E_no_dual_state_split`, `Anew_F3_no_behavior_memory`, and `Anew_R5` as
`R_key`. It does not run `A_new` by default because A_new is tuned separately,
and it does not run `Anew_B_classical_frontier_greedy` unless B is explicitly
included as an optional CPU benchmark. The batch follows the frozen V1 A_new
formal defaults at execution time and does not hardcode final training
parameter values. Formal train-side artifacts from the batch can be recorded
after artifact audit; paper-facing held-out comparison is recorded by the
unified final probe.

Legacy A/F1/F6/F7/ABCDEFR records, older full-method reference records, baseline
records, and final-probe matrices were removed from active `main`. The remote
repository is intended to keep `main` as the only maintained branch; historical
cleanup state is preserved by tag only:

- tag: `legacy-pre-a-new-cleanup-20260525`
