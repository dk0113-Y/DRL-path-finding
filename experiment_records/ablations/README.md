# Ablation Result Records

This directory stores curated ablation logs and manual records only. It does not store raw `outputs/`, checkpoints, model weights, replay buffers, or large debug artifacts.

Canonical `ablation_id` and `short_id` stay stable. Filesystem directories use:

```text
<short_id>_ablation_<canonical_id>
```

When the canonical id already starts with `<short_id>_ablation_`, that canonical id is the slug.

## Structural Ablations

- `D_ablation_no_value_tree`: removes value-tree information by zero/false replacement while preserving the training interface.
- `E_ablation_no_semantic_dual_state_split`: keeps value-tree and advantage-canvas inputs, but removes the explicit `value_state` / `advantage_state` semantic split decision structure.

E checkpoints remain local under `checkpoint_store/ablations/E_ablation_no_semantic_dual_state_split.pt` and must not be committed. E requires its own retraining; A, D, F, or R checkpoints are not valid final E performance evidence.

## F Group Directories

- `F1_ablation_no_frontier_channel`
- `F2_ablation_no_visit_count_channel`
- `F3_ablation_no_recent_trajectory_channel`
- `F4_ablation_no_visit_traj_channels`
- `F5_ablation_occupancy_only_canvas`

## R Group Directories

- `R1_ablation_no_step_penalty`
- `R2_ablation_no_revisit_penalty`
- `R3_ablation_no_turn_penalty`
- `R4_ablation_no_timeout_penalty`
- `R5_ablation_no_efficiency_penalties`
- `R6_ablation_sparse_reward_variant`

Each directory's `logs/` subdirectory is for curated artifacts. `run_record.md` records source output path, copied artifact list, missing artifact list, checkpoint_store path, and eligibility verdict.

Smoke and pilot runs do not enter paper Results and must not be used as main-table evidence. Formal runs are result candidates only after artifact completeness, seed policy, final probe protocol, and run stage are reviewed.
