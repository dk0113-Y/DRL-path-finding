# Final Method Launchers

This directory contains final-method candidate launchers that are separate from the legacy full-method and ablation entries.

## A_new Final 4-Channel Method

`A_new` is the final 4-channel no-frontier-raster method:

- `method_id = A_new`
- `method_name = final_4ch_no_frontier_raster`
- `advantage_canvas_schema = final_4ch_no_frontier_raster`
- advantage canvas channels: `free`, `obstacle`, `visit_count_log_norm`, `recent_trajectory_decay`
- `frontier_raster_used = false`
- `value_tree_enabled = true`
- `value_branch_source = SharedSemanticSnapshot`
- `value_branch_representation = structured_frontier_block_value_tree`
- `model_class = ExplorationQNetwork`
- `advantage_encoder.canvas_in_channels = 4`

F1 provided the diagnostic evidence that the frontier-raster input is unnecessary or harmful in the advantage branch. F1 remains a legacy 5-channel zero-frontier diagnostic: it keeps the old tensor slot and zeros `frontier_block_area_map`. `A_new` is the final engineering structure: it removes that unused raster slot and must be retrained from scratch because the first advantage-encoder convolution changes from 5 input channels to 4.

F6 and F7 also remain legacy 5-channel frontier-channel diagnostics. They are not `A_new` and continue to use `legacy_5ch_frontier_raster`.

Smoke is the default stage. Formal 500000-step training is never the default and must be requested explicitly.

Dry-run:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_a_new_final_4ch.ps1 -RunStage smoke -Device cpu -DryRun
```

Smoke:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_a_new_final_4ch.ps1 -RunStage smoke -Device cpu
```
