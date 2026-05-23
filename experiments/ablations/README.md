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

## Naming

Canonical `ablation_id` and `short_id` stay stable. Filesystem slugs use:

```text
<short_id>_ablation_<canonical_id>
```

When the canonical id already starts with `<short_id>_ablation_`, the canonical id itself is the slug. Examples:

- D: `D_ablation_no_value_tree`
- E: `E_ablation_no_semantic_dual_state_split`
- F5: `F5_ablation_occupancy_only_canvas`
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

Dry-run E through the batch runner:

```powershell
python experiments\ablations\run_ablation_batch.py --ablation-ids E --run-stage smoke --device cpu --dry-run
```

Run E smoke through the batch runner:

```powershell
python experiments\ablations\run_ablation_batch.py --ablation-ids E --run-stage smoke --device cpu
```

Each completed run copies curated logs to `experiment_records/ablations/<slug>/logs/`, writes `run_record.md`, and by default copies `run_dir/checkpoints/last.pt` to `checkpoint_store/ablations/<slug>.pt`. Raw `outputs/`, replay buffers, plots, debug files, and checkpoint files are not tracked.
