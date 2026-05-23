# C Local-State DDQN Learning Baseline

`C_baseline_local_state_ddqn` is a learning baseline, not an ablation. It is
not an F channel ablation, R reward ablation, D value-tree structural ablation,
or B classical frontier-greedy baseline.

## Method Boundary

- Uses the existing DDQN training protocol, 8-neighborhood action space, valid
  action mask, map generator, reward definitions, termination rules, and final
  probe seed policy.
- Does not use the shared semantic dual-state representation as policy input.
- Does not use value tree, value block-tree, frontier cluster, accessible
  unknown block, or ground-truth full map information for decisions.
- Carries the local-state patch through the existing `advantage_canvas` replay
  key only for interface compatibility.

## State And Model

Local state channels:

1. `known_free`
2. `known_obstacle`
3. `unknown`

`LocalStateQNetwork` is a small CNN over the 3-channel local belief patch. It
keeps the existing trainer-compatible forward signature:

```python
model(
    advantage_canvas,
    value_block_features,
    value_entry_features,
    value_block_mask,
    value_entry_mask,
    return_aux=False,
)
```

The model ignores all `value_*` tensors. The C adapter emits zero dummy value
tensors with all-false masks only so the existing replay and learner interfaces
remain unchanged.

## Smoke Commands

```powershell
python .\scripts\check_local_state_ddqn_baseline.py
python .\experiments\baselines\run_local_state_ddqn_train.py --baseline-id C_baseline_local_state_ddqn --run-stage smoke -- --device cpu --output-root outputs
```

Smoke artifacts are diagnostic only and must not be copied into formal paper
result tables.

## Formal Command Template

Run this later on formal hardware; do not commit checkpoints or output folders.

```powershell
python .\experiments\baselines\run_local_state_ddqn_train.py `
  --baseline-id C_baseline_local_state_ddqn `
  --run-stage formal `
  -- `
  --device cuda `
  --rows 40 `
  --cols 60 `
  --obs-size 6 `
  --scan-radius 10 `
  --obstacle-ratio 0.20 `
  --max-episode-steps 600 `
  --coverage-stop-threshold 0.95 `
  --trajectory-history-steps 10 `
  --final-greedy-episodes 100 `
  --fixed-final-probe-seed-base 20261323 `
  --reward-info-scale 3.1 `
  --reward-obstacle-weight 0.2 `
  --reward-step-penalty 0.02 `
  --reward-terminal-bonus 20.0 `
  --reward-revisit-penalty 0.1 `
  --reward-turn-penalty-scale 0.05 `
  --reward-timeout-penalty 8.0 `
  --output-root outputs
```

`outputs/`, `checkpoint_store/`, and `checkpoints/` are local artifacts and
must stay out of Git.
