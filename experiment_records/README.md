# Experiment Records

`main` reserves tracked record space for the active A_new final method, the
Anew_R1-Anew_R5 reward ablations, the A_new-aligned B classical baseline, and
A_new-aligned structural ablations such as `Anew_D_no_value_tree`.

- Active lightweight records belong under `experiment_records/final_method/`.
- Do not commit raw smoke outputs, raw run directories, checkpoint bodies, or
  large logs.
- Do not commit `outputs/`, `checkpoint_store/`, `checkpoints/`, `.pt`, `.pth`, or
  `.ckpt` files.

`Anew_B_classical_frontier_greedy` records, when formal benchmark runs are
performed, must be treated as A_new-aligned traditional non-learning baseline
records. B does not train a model, does not use a checkpoint, does not use
`ExplorationQNetwork`, and must not inherit legacy B or old baseline artifacts.
The policy uses belief-derived frontier/shared-semantic state and valid action
indices only; simulator map access is for stepping, sensing, termination, and
metrics. Smoke and pilot outputs are not Results evidence and should not be
written here as formal records.

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
A_new train-side contract, but they do not replace unrun final-probe evidence.

`Anew_F3_no_behavior_memory` records, when formal runs are performed, must be
treated as A_new-aligned F_key input-state ablation records. The row keeps the
current 4-channel no-frontier-raster schema, preserves `free` and `obstacle`,
zeros `visit_count_log_norm` and `recent_trajectory_decay`, keeps the value tree
enabled, and uses `reward_override = {}`. Under the current schema this is
equivalent to an occupancy-only advantage canvas, but `Anew_F4_occupancy_only`
must not be written as a separate formal row, run name, or artifact row. Smoke
and pilot outputs are not Results evidence and should not be written here as
formal records.

Legacy A/F1/F6/F7/ABCDEFR records, older full-method reference records, baseline
records, and final-probe matrices were archived before cleanup at:

- branch: `legacy/pre-a-new-cleanup`
- tag: `legacy-pre-a-new-cleanup-20260525`
