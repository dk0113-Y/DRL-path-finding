# Engineering Overview

## Main Method

Current `main` is scoped to A_new:

- final method: `final_4ch_no_frontier_raster`
- active reward ablations: `Anew_R1` through `Anew_R5`
- archived legacy evidence: `legacy/pre-a-new-cleanup` and
  `legacy-pre-a-new-cleanup-20260525`

Legacy A/F1/F6/F7/ABCDEFR launchers and frontier-raster diagnostics are no longer
active workflow entries on `main`.

## State Construction

`StateTensorAdapter` still builds five tensors for the Q network:

- `advantage_canvas`
- `value_block_features`
- `value_entry_features`
- `value_block_mask`
- `value_entry_mask`

The advantage branch uses the final A_new 4-channel local canvas:

1. `free`
2. `obstacle`
3. `visit_count_log_norm`
4. `recent_trajectory_decay`

This local canvas is tied to `cum_map.local_shape` and intentionally contains no
frontier raster channel.

## Shared Semantics And Value Tree

The shared semantic layer remains frontier-first:

- `UnknownBlock`
- `FrontierCluster`
- `SupportGeometry`

These semantics are not removed by the A_new cleanup. They feed the value branch
through `SharedSemanticSnapshot` and the structured frontier-block value tree.

Current value features are:

- block: `block_area_ratio`, `frontier_cluster_count`
- entry: `delta_r_ratio`, `delta_c_ratio`, `entry_width_ratio`,
  `support_obstacle_density`

The important boundary is that frontier / unknown-block semantics stay in the
value tree, while the advantage branch stays a 4-channel local occupancy,
revisit-pressure, and trajectory canvas.
