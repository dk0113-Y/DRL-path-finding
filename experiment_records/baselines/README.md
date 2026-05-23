# Baseline Experiment Records

`C_baseline_local_state_ddqn` is a learning baseline, not an ablation. Its
curated records live under:

```text
experiment_records/baselines/C_baseline_local_state_ddqn/
```

This tree stores only curated logs and `run_record.md`. Raw training outputs,
checkpoint directories, and model weights stay out of Git.

## Allowed Contents

- `logs/final_probe.csv`
- `logs/final_probe_summary.json`
- `logs/metric_snapshot.json`
- `logs/config_snapshot.json`
- `logs/reproducibility_contract.json`
- `logs/posthoc_selection_summary.json`
- `logs/formal_selection_manifest.json`
- `logs/artifact_index.json`
- `logs/baseline_manifest.json`
- `logs/training_summary.txt`
- `run_record.md`

## Not Tracked

- `outputs/`
- `checkpoint_store/`
- `checkpoints/`
- `*.pt`
- `*.pth`
- `*.ckpt`

Checkpoint files are local runtime artifacts only. The baseline batch wrapper
may copy `last.pt` into `checkpoint_store/baselines/`, but that file must not
be committed.

## Results Eligibility

Smoke and pilot runs cannot enter paper Results. A formal C baseline run is
only a final-results candidate after the final probe artifacts are complete and
the run record reports an eligibility verdict of `pass`.
