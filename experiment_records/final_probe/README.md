# Final Probe Matrix

This directory is reserved for the unified final probe matrix artifacts.

The runner is `tools/run_final_probe_matrix.py`. It audits all required inputs before formal execution, runs the B classical frontier greedy baseline first, then evaluates checkpoint-backed A/C/D/E/F/R methods with the same `seed_base + episode_index` episode seeds.

Default protocol:

- episodes: 100
- seed base: 20261323
- device: cuda
- output root: `experiment_records/final_probe`
- R6: excluded unless `--include-r6` is passed

Expected formal outputs:

- `final_probe_summary.csv`
- `final_probe_per_episode.csv`
- `final_probe_protocol.json`
- `method_registry.json`
- `run_manifest.json`
- per-method `methods/<method_id>/per_episode.csv`
- per-method `methods/<method_id>/summary.json`

Checkpoints remain in `checkpoint_store` and are referenced by path; they are not copied here.
