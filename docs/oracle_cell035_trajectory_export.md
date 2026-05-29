# Oracle Cell035 Trajectory Export

`scripts/export_oracle_cell035_trajectory.py` exports an oracle/ideal
cell035 exploration trajectory for downstream Gazebo waypoint replay.

This is not a LaserScan-conditioned deployment path. The exporter uses the
training-side oracle observation stack:

- `RadarSensor(scan_radius=10)`
- `LocalObservationModel`
- `CumulativeBeliefMap`
- `StateTensorAdapter`
- `ExplorationQNetwork`
- `select_greedy_action`

The output is intended for the ROS2 route named:

```text
Oracle-planned trajectory replay with SLAM mapping
```

## Run

Linux cell035 export:

```bash
cd /home/dk/drl_repos/DRL-path-finding
python scripts/export_oracle_cell035_trajectory.py \
  --checkpoint /home/dk/drl_repos/DRL-path-finding/deploy_checkpoints/A_full_method_last.pt \
  --true-grid /home/dk/ros2_repos/ROS2/assets/cell035/grids/random_train_like_seed20260513_true_grid.npy \
  --start-rc 20 36 \
  --cell-size 0.35 \
  --rows 40 \
  --cols 60 \
  --world-x 21.0 \
  --world-y 14.0 \
  --scan-radius-cells 10 \
  --coverage-goal 0.95 \
  --max-steps 400 \
  --output-dir /home/dk/ros2_repos/ROS2/assets/cell035/trajectories
```

The default output files are:

- `cell035_oracle_trajectory.csv`
- `cell035_oracle_trajectory_summary.json`

Do not commit full generated trajectories, checkpoint files, or run logs unless
they are explicitly curated as small evidence artifacts.

## CSV Semantics

The first CSV row is the initial pose with `step=0` and blank action fields.
Each later row records the executed target cell after one greedy oracle action.

Fields:

- `step`: zero-based row in the exported trajectory, with step 0 as the start.
- `row`, `col`: current grid cell after the action for steps greater than 0.
- `x`, `y`: world-frame center of `row`, `col`.
- `action_idx`, `action_name`: greedy valid action selected at the previous
  cell.
- `target_row`, `target_col`, `target_x`, `target_y`: executed target cell and
  its world-frame center. For steps greater than 0 these match `row`, `col`,
  `x`, and `y`.
- `coverage`, `best_coverage`: cumulative oracle coverage after the row.
- `valid_actions`: JSON array of valid action indices before the action.
- `q_values`: JSON array of the eight Q-values before masking.
- `stop_reason`: populated on the final row.

Coordinate conversion matches the ROS2 bridge:

```text
x = -world_x / 2.0 + (col + 0.5) * cell_size
y =  world_y / 2.0 - (row + 0.5) * cell_size
```

For cell035 the defaults are `cell_size=0.35`, `rows=40`, `cols=60`,
`world_x=21.0`, and `world_y=14.0`.

## Research Boundary

Use this export to prove that an oracle/ideal DRL exploration trajectory can be
replayed by a continuous Gazebo robot and paired with real `/scan` SLAM
mapping. Do not describe it as closed-loop LaserScan-conditioned DRL
exploration, and do not use it to replace final probe results.
