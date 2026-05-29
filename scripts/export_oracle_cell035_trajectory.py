from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import deque
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch


if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents.q_value_agent import (  # noqa: E402
    ExplorationQNetwork,
    StateTensorAdapter,
    select_greedy_action,
)
from env.agent_version import LocalObservationModel  # noqa: E402
from env.core_cummap import CumulativeBeliefMap  # noqa: E402
from env.core_radar import RadarSensor  # noqa: E402
from env.grid_topology import ACTIONS_8, EMPTY, GridTopology  # noqa: E402


ACTION_NAMES = ("N", "NE", "E", "SE", "S", "SW", "W", "NW")
CSV_NAME = "cell035_oracle_trajectory.csv"
SUMMARY_NAME = "cell035_oracle_trajectory_summary.json"


def grid_rc_to_cell_center_xy(
    row: int,
    col: int,
    *,
    cell_size: float,
    world_x: float,
    world_y: float,
) -> tuple[float, float]:
    x = -float(world_x) / 2.0 + (int(col) + 0.5) * float(cell_size)
    y = float(world_y) / 2.0 - (int(row) + 0.5) * float(cell_size)
    return float(x), float(y)


def json_compact(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        number = float(value)
        return None if not math.isfinite(number) else number
    if isinstance(value, float):
        return None if not math.isfinite(value) else value
    return value


def load_policy(checkpoint_path: Path, device: torch.device) -> ExplorationQNetwork:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if not isinstance(payload, Mapping):
        raise RuntimeError(f"checkpoint payload is not a mapping: {checkpoint_path}")
    if "online_state_dict" not in payload:
        raise RuntimeError(f"checkpoint missing online_state_dict: {checkpoint_path}")

    net = ExplorationQNetwork().to(device)
    net.load_state_dict(payload["online_state_dict"], strict=True)
    net.eval()
    return net


def trajectory_row(
    *,
    step: int,
    row: int,
    col: int,
    action_idx: int | None,
    target_row: int | None,
    target_col: int | None,
    coverage: float,
    best_coverage: float,
    valid_actions: list[int],
    q_values: list[float],
    stop_reason: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    x, y = grid_rc_to_cell_center_xy(
        row,
        col,
        cell_size=float(args.cell_size),
        world_x=float(args.world_x),
        world_y=float(args.world_y),
    )
    if target_row is None or target_col is None:
        target_x, target_y = "", ""
    else:
        target_x, target_y = grid_rc_to_cell_center_xy(
            target_row,
            target_col,
            cell_size=float(args.cell_size),
            world_x=float(args.world_x),
            world_y=float(args.world_y),
        )

    return {
        "step": int(step),
        "row": int(row),
        "col": int(col),
        "x": x,
        "y": y,
        "action_idx": "" if action_idx is None else int(action_idx),
        "action_name": "" if action_idx is None else ACTION_NAMES[int(action_idx)],
        "target_row": "" if target_row is None else int(target_row),
        "target_col": "" if target_col is None else int(target_col),
        "target_x": target_x,
        "target_y": target_y,
        "coverage": float(coverage),
        "best_coverage": float(best_coverage),
        "valid_actions": json_compact([int(v) for v in valid_actions]),
        "q_values": json_compact([float(v) for v in q_values]),
        "stop_reason": str(stop_reason),
    }


def export_trajectory(args: argparse.Namespace) -> dict[str, Any]:
    checkpoint_path = Path(args.checkpoint).expanduser()
    true_grid_path = Path(args.true_grid).expanduser()
    output_dir = Path(args.output_dir).expanduser()

    if not true_grid_path.exists():
        raise FileNotFoundError(f"true_grid not found: {true_grid_path}")
    true_grid = np.load(true_grid_path).astype(np.int8)
    expected_shape = (int(args.rows), int(args.cols))
    if true_grid.shape != expected_shape:
        raise ValueError(f"true_grid shape mismatch: got {true_grid.shape}, expected {expected_shape}")

    start = (int(args.start_rc[0]), int(args.start_rc[1]))
    if not GridTopology.in_bounds(true_grid.shape, start[0], start[1]):
        raise ValueError(f"start_rc outside grid: {start}")
    if int(true_grid[start[0], start[1]]) != int(EMPTY):
        raise ValueError(f"start_rc is not EMPTY/free: {start} value={int(true_grid[start[0], start[1]])}")

    device = torch.device(str(args.device))
    net = load_policy(checkpoint_path, device=device)
    adapter = StateTensorAdapter(device="cpu")
    sensor = RadarSensor(scan_radius=int(args.scan_radius_cells))
    obs_model = LocalObservationModel(true_grid, start, sensor=sensor)
    local_snap = obs_model.local_snap
    cum_map = CumulativeBeliefMap(true_grid, start, local_snap)
    free_mask = GridTopology.free_mask(true_grid)

    agent = start
    best_coverage = float(cum_map.coverage_rate)
    recent_positions: deque[tuple[int, int]] = deque(
        [(int(agent[0]), int(agent[1]))],
        maxlen=max(1, int(args.recent_traj_limit)),
    )

    rows: list[dict[str, Any]] = [
        trajectory_row(
            step=0,
            row=agent[0],
            col=agent[1],
            action_idx=None,
            target_row=None,
            target_col=None,
            coverage=float(cum_map.coverage_rate),
            best_coverage=best_coverage,
            valid_actions=[],
            q_values=[],
            stop_reason="",
            args=args,
        )
    ]

    stop_reason = "max_steps"
    q_values_last: list[float] = []
    valid_last: list[int] = []

    for step in range(1, int(args.max_steps) + 1):
        valid = tuple(int(v) for v in GridTopology.valid_action_indices_fast(free_mask, agent))
        valid_last = list(valid)
        if len(valid) <= 0:
            stop_reason = "no_valid_actions"
            rows[-1]["stop_reason"] = stop_reason
            break

        state_batch, _state_meta = adapter.build_single_state_tensors(
            cum_map,
            agent,
            recent_trajectory_positions=tuple(recent_positions),
            return_state_meta=True,
        )
        policy_state = adapter.move_state_batch(state_batch, target_device=device, non_blocking=True)
        with torch.inference_mode():
            q_tensor = net(
                policy_state["advantage_canvas"],
                policy_state["value_block_features"],
                policy_state["value_entry_features"],
                policy_state["value_block_mask"],
                policy_state["value_entry_mask"],
                return_aux=False,
            )
            action = select_greedy_action(q_tensor, valid_action_indices=valid)

        action_idx = int(action.item())
        q_values_last = [float(v) for v in q_tensor.squeeze(0).detach().cpu().tolist()]
        if action_idx not in valid:
            raise RuntimeError(f"selected invalid action {action_idx}; valid={valid}")

        dr, dc = ACTIONS_8[action_idx]
        target = (int(agent[0]) + int(dr), int(agent[1]) + int(dc))
        if not GridTopology.can_step(free_mask, agent[0], agent[1], target[0], target[1]):
            raise RuntimeError(f"target cell is not a legal step: start={agent} target={target}")

        agent = target
        recent_positions.append((int(agent[0]), int(agent[1])))
        local_snap = obs_model.observe_fast(agent)
        cum_map.update(agent, local_snap)
        best_coverage = max(best_coverage, float(cum_map.coverage_rate))

        reached_goal = bool(float(cum_map.coverage_rate) >= float(args.coverage_goal))
        row_stop_reason = "coverage_goal" if reached_goal else ""
        rows.append(
            trajectory_row(
                step=step,
                row=agent[0],
                col=agent[1],
                action_idx=action_idx,
                target_row=target[0],
                target_col=target[1],
                coverage=float(cum_map.coverage_rate),
                best_coverage=best_coverage,
                valid_actions=valid_last,
                q_values=q_values_last,
                stop_reason=row_stop_reason,
                args=args,
            )
        )
        if reached_goal:
            stop_reason = "coverage_goal"
            break
    else:
        rows[-1]["stop_reason"] = stop_reason

    if rows[-1]["stop_reason"] == "":
        rows[-1]["stop_reason"] = stop_reason

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / CSV_NAME
    summary_path = output_dir / SUMMARY_NAME

    fieldnames = [
        "step",
        "row",
        "col",
        "x",
        "y",
        "action_idx",
        "action_name",
        "target_row",
        "target_col",
        "target_x",
        "target_y",
        "coverage",
        "best_coverage",
        "valid_actions",
        "q_values",
        "stop_reason",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    final_row = rows[-1]
    summary = {
        "schema_version": "oracle_cell035_trajectory_export/v1",
        "route_name": "oracle-planned trajectory replay with SLAM mapping",
        "stop_reason": str(stop_reason),
        "steps": int(final_row["step"]),
        "best_coverage": float(best_coverage),
        "final_coverage": float(final_row["coverage"]),
        "start_rc": [int(start[0]), int(start[1])],
        "final_rc": [int(final_row["row"]), int(final_row["col"])],
        "trajectory_csv": str(csv_path),
        "summary_json": str(summary_path),
        "checkpoint_path": str(checkpoint_path),
        "true_grid_path": str(true_grid_path),
        "cell_size": float(args.cell_size),
        "rows": int(args.rows),
        "cols": int(args.cols),
        "world_x": float(args.world_x),
        "world_y": float(args.world_y),
        "scan_radius_cells": int(args.scan_radius_cells),
        "coverage_goal": float(args.coverage_goal),
        "max_steps": int(args.max_steps),
        "recent_traj_limit": int(args.recent_traj_limit),
        "final_valid_actions": valid_last,
        "final_q_values": q_values_last,
        "note": (
            "This export uses oracle/ideal training-side observations. It is not "
            "closed-loop LaserScan-conditioned DRL exploration."
        ),
    }
    summary_path.write_text(
        json.dumps(json_safe(summary), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export a cell035 oracle/ideal DRL exploration trajectory for Gazebo "
            "waypoint replay. The first CSV row is the initial pose; subsequent "
            "rows are executed target cell centers after each greedy action."
        )
    )
    parser.add_argument(
        "--checkpoint",
        default="/home/dk/drl_repos/DRL-path-finding/deploy_checkpoints/A_full_method_last.pt",
        help="Path to A_full_method_last.pt or a compatible checkpoint.",
    )
    parser.add_argument(
        "--true-grid",
        default="/home/dk/ros2_repos/ROS2/assets/cell035/grids/random_train_like_seed20260513_true_grid.npy",
        help="Path to the cell035 true_grid .npy file.",
    )
    parser.add_argument("--start-rc", nargs=2, type=int, default=(20, 36), metavar=("ROW", "COL"))
    parser.add_argument("--cell-size", type=float, default=0.35)
    parser.add_argument("--rows", type=int, default=40)
    parser.add_argument("--cols", type=int, default=60)
    parser.add_argument("--world-x", type=float, default=21.0)
    parser.add_argument("--world-y", type=float, default=14.0)
    parser.add_argument("--scan-radius-cells", type=int, default=10)
    parser.add_argument("--coverage-goal", type=float, default=0.95)
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--recent-traj-limit", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--output-dir",
        default="experiment_records/cell035_oracle_trajectory",
        help="Directory for CSV and summary JSON outputs.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    summary = export_trajectory(args)
    print(json.dumps(json_safe(summary), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
