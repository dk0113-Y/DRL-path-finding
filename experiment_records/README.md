# Experiment Records

`main` reserves tracked record space for the active A_new final method, the
Anew_R1-Anew_R5 reward ablations, and A_new-aligned structural ablations such as
`Anew_D_no_value_tree`.

- Active lightweight records belong under `experiment_records/final_method/`.
- Do not commit raw smoke outputs, raw run directories, checkpoint bodies, or
  large logs.
- Do not commit `outputs/`, `checkpoint_store/`, `checkpoints/`, `.pt`, `.pth`, or
  `.ckpt` files.

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
