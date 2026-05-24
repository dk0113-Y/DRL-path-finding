# Ablation Experiment Infrastructure

This directory contains controlled ablation launchers for structural, input/channel, and reward experiments. It does not contain learning baselines such as C `C_baseline_local_state_ddqn`; those stay under `baselines/` and `experiments/baselines/`.

## Design Boundaries

- Do not duplicate the main training system.
- Do not rewrite environment, replay, collector, learner, or reward semantics except through explicit reward-ablation config overrides.
- Full method default training through `train_q_agent.py` is unchanged; ablations activate only through `experiments/ablations/run_ablation_train.py`.
- Smoke and pilot runs are functional checks only and do not enter paper Results.
- Formal runs are the only result candidates, after artifact and protocol review.

## Experiment Groups

Structural ablations:

- D `D_ablation_no_value_tree`: removes value-tree information by replacing value branch tensors with zero/false tensors while preserving the full network interface.
- E `E_ablation_no_semantic_dual_state_split`: keeps advantage canvas and value-tree information, but removes the explicit `value_state` / `advantage_state` split decision structure by replacing `SemanticDuelingHead` with a fused single action-value pathway.

Input/channel ablations:

- F1 `no_frontier_channel`
- F2 `no_visit_count_channel`
- F3 `no_recent_trajectory_channel`
- F4 `no_visit_traj_channels`
- F5 `occupancy_only_canvas`
- F6 `local_frontier_binary_map`
- F7 `local_frontier_global_area_map`

Reward ablations:

- R1 `no_step_penalty`
- R2 `no_revisit_penalty`
- R3 `no_turn_penalty`
- R4 `no_timeout_penalty`
- R5 `no_efficiency_penalties`
- R6 `sparse_reward_variant`

## E Structural Ablation

E tests whether explicitly separating value-state and action-conditioned advantage-state before dueling fusion is beneficial. It does not test whether value-tree information exists; value-tree tensors remain enabled and are consumed by `ValueTreeEncoder`.

The E model uses:

- `AdvantageCanvasEncoder` for full advantage canvas inputs.
- `ValueTreeEncoder` for full value-tree block/entry tensors.
- `NoSemanticDualStateSplitQNetwork`, which projects both encoder outputs into a fused per-action latent and predicts Q values through one action-value head.

The E model does not call `SemanticDuelingHead`, and its aux output marks `no_semantic_dual_state_split=1`, `semantic_dual_state_split_used=0`, and `value_tree_used_by_model=1`.

E requires retraining. A, D, F, or R checkpoints cannot be reused as final E performance evidence.

## F6/F7 Frontier Channel Variants

F6 and F7 are advantage frontier-channel diagnostics. They are not reward ablations, do not alter the value tree, and keep the network tensor interface unchanged: the advantage canvas still has the same five channel names, with channel 2 remaining `frontier_block_area_map` for compatibility.

- F6 `F6_ablation_local_frontier_binary_map` tests whether the advantage branch needs a local frontier position cue. Channel 2 is an agent-centered local crop of `cum_map.get_frontier_u8(refresh=False)` from the cumulative belief map. Frontier cells are `1.0`; all other cells are `0.0`.
- F7 `F7_ablation_local_frontier_global_area_map` tests whether local-indexed frontier positions still benefit from global unknown-block area attributes. Channel 2 uses the same cumulative-map local frontier crop as the primary spatial index, then assigns matched local frontier cells `block.block_area / total_accessible_unknown_area` from `SharedSemanticSnapshot`. Local frontier cells without a semantic block match remain `0.0` and are counted in state meta.

The full method A keeps the default `semantic_block_area_raster` behavior: semantic frontier cluster geometry is projected into the local canvas with block-area ratios. F1 still remains a zeroed-channel ablation and should not be interpreted as either F6 or F7.

## Naming

Canonical `ablation_id` and `short_id` stay stable. Filesystem slugs use:

```text
<short_id>_ablation_<canonical_id>
```

When the canonical id already starts with `<short_id>_ablation_`, the canonical id itself is the slug. Examples:

- D: `D_ablation_no_value_tree`
- E: `E_ablation_no_semantic_dual_state_split`
- F5: `F5_ablation_occupancy_only_canvas`
- F6: `F6_ablation_local_frontier_binary_map`
- F7: `F7_ablation_local_frontier_global_area_map`
- R5: `R5_ablation_no_efficiency_penalties`

Default outputs run names use `<slug>_<run_stage>`. Curated logs are archived to `experiment_records/ablations/<slug>/logs/`. Checkpoint bodies are copied to `checkpoint_store/ablations/<slug>.pt` by the batch runner and are ignored by Git.

## Single Experiment

List specs:

```powershell
python experiments\ablations\run_ablation_train.py --list
```

Dry-run E:

```powershell
python experiments\ablations\run_ablation_train.py --ablation-id E --run-stage smoke --dry-run
```

Smoke E:

```powershell
python experiments\ablations\run_ablation_train.py --ablation-id E --run-stage smoke
```

Formal R5:

```powershell
python experiments\ablations\run_ablation_train.py --ablation-id R5 --run-stage formal -- --device cuda --total-env-steps 500000 --final-greedy-episodes 100
```

## Batch Runner

`run_ablation_batch.py` reads `experiment_records/full_method_main/logs/config_snapshot.json` and aligns map, budget, learning hyperparameters, seed policy, formal protocol, and reward base parameters before invoking individual ablation launches.

Available structural presets include:

- `structural_core_batch`: D only, preserving the original minimum structural check semantics.
- `semantic_core_batch`: E only.
- `structural_extended_batch`: D and E.
- `frontier_channel_variant_batch`: F6 and F7.

Dry-run E through the batch runner:

```powershell
python experiments\ablations\run_ablation_batch.py --ablation-ids E --run-stage smoke --device cpu --dry-run
```

Run E smoke through the batch runner:

```powershell
python experiments\ablations\run_ablation_batch.py --ablation-ids E --run-stage smoke --device cpu
```

Dry-run F6/F7 frontier channel variants:

```powershell
python experiments\ablations\run_ablation_batch.py --preset frontier_channel_variant_batch --run-stage smoke --device cpu --dry-run
```

Or use the smoke-default launcher:

```powershell
.\scripts\run_f6_f7_frontier_channel_variants.ps1 -RunStage smoke -Device cpu -DryRun
```

Wait for a running C baseline formal job and then launch E formal through the
batch runner:

```powershell
.\scripts\wait_then_run_e_ablation_after_c_baseline.ps1 -Device cuda
```

Preview the watcher without waiting or launching:

```powershell
.\scripts\wait_then_run_e_ablation_after_c_baseline.ps1 -DryRun
```

Each completed run copies curated logs to `experiment_records/ablations/<slug>/logs/`, writes `run_record.md`, and by default copies `run_dir/checkpoints/last.pt` to `checkpoint_store/ablations/<slug>.pt`. Raw `outputs/`, replay buffers, plots, debug files, and checkpoint files are not tracked.
