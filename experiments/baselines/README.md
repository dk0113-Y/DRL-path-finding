# Learning Baselines

This directory contains entrypoints for learning baselines. For the C group,
use `run_baseline_batch.py` as the main workflow so training outputs are
archived in the same way as D/F/R ablation runs.

## Supported Baseline

- `C_baseline_local_state_ddqn`: local-state DDQN learning baseline.

Other baseline IDs are not supported by the batch wrapper until their training
entrypoints and archive contracts are implemented.

## Recommended Commands

Dry-run:

```powershell
python .\experiments\baselines\run_baseline_batch.py --baseline-id C_baseline_local_state_ddqn --run-stage pilot --device cuda --dry-run --extra-train-args "--total-env-steps 24000 --warmup-steps 4000 --final-greedy-episodes 2 --rows 40 --cols 60 --obs-size 6 --scan-radius 10 --obstacle-ratio 0.20 --max-episode-steps 600 --coverage-stop-threshold 0.95 --fixed-final-probe-seed-base 20261323 --reward-info-scale 3.1 --reward-obstacle-weight 0.2 --reward-step-penalty 0.02 --reward-terminal-bonus 20.0 --reward-revisit-penalty 0.1 --reward-turn-penalty-scale 0.05 --reward-timeout-penalty 8.0"
```

Pilot:

```powershell
python .\experiments\baselines\run_baseline_batch.py --baseline-id C_baseline_local_state_ddqn --run-stage pilot --device cuda --extra-train-args "--total-env-steps 24000 --warmup-steps 4000 --final-greedy-episodes 2 --rows 40 --cols 60 --obs-size 6 --scan-radius 10 --obstacle-ratio 0.20 --max-episode-steps 600 --coverage-stop-threshold 0.95 --fixed-final-probe-seed-base 20261323 --reward-info-scale 3.1 --reward-obstacle-weight 0.2 --reward-step-penalty 0.02 --reward-terminal-bonus 20.0 --reward-revisit-penalty 0.1 --reward-turn-penalty-scale 0.05 --reward-timeout-penalty 8.0"
```

Formal runs use the same wrapper with `--run-stage formal` and formal training
budgets.

## Archive Behavior

`run_baseline_batch.py` calls
`experiments/baselines/run_local_state_ddqn_train.py`, parses
`baseline_manifest_json: <path>` from the child process output, and copies
curated logs into:

```text
experiment_records/baselines/C_baseline_local_state_ddqn/logs/
```

By default, it also copies:

```text
outputs/<run_dir>/checkpoints/last.pt
```

to:

```text
checkpoint_store/baselines/C_baseline_local_state_ddqn.pt
```

Use `--no-copy-checkpoints` when the local checkpoint copy should be skipped.
`outputs/`, `checkpoint_store/`, and model checkpoint files are ignored by Git.

## Trajectory History Setting

`TrainConfig.trajectory_history_steps` currently defaults to `10`, and the C
baseline recommended commands do not pass `--trajectory-history-steps` because
that option is not exposed by `train_q_agent.py`'s CLI parser.
