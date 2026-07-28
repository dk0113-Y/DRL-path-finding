from __future__ import annotations

"""Export reusable paper assets for online exploration decisions and environment interaction."""

import json
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap, to_rgba
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch, FancyBboxPatch, Polygon
from matplotlib.transforms import Affine2D

from env.agent_version import LocalObservationModel
from env.block_random_g import RandomMapGenerator
from env.core_cummap import CumulativeBeliefMap
from env.core_radar import RadarSensor
from env.grid_topology import ACTIONS_8, EMPTY, INVISIBLE, OBSTACLE, GridTopology
from tools.export_architecture_pictures import (
    ACTION_TO_KEY,
    ExportConfig,
    FIXED_ACTION_PREFERENCES,
    KEY_TO_ACTION,
    Snapshot,
    WorldCanvas,
    _agent_world_to_canvas,
    _build_method_world_canvas,
    _capture_snapshot,
    _format_clean_axis,
    _format_output_path,
    _project_belief_to_canvas,
    _select_fallback_action,
    _set_global_seed,
    _trajectory_world_to_canvas,
)


LOCAL_CMAP = ListedColormap(("#B4BCC4", "#F8F8F6", "#1E1E1E"))
LOCAL_NORM = BoundaryNorm((-1.5, -0.5, 0.5, 1.5), LOCAL_CMAP.N)


@dataclass(frozen=True, slots=True)
class OnlineWorkflowStyle:
    """Centralized cell-relative styles shared by all online-workflow assets."""

    max_inches: float = 5.6
    min_inches: float = 3.4
    cell_inches: float = 0.205
    body_width: float = 0.64
    body_length: float = 0.88
    body_color: str = "#55966B"
    body_edge_color: str = "#2F5940"
    wheel_color: str = "#30363B"
    lidar_color: str = "#E99D4E"
    lidar_edge_color: str = "#2F5940"
    heading_color: str = "#F8F8F6"
    trajectory_color: str = "#5185C0"
    scan_color: str = "#5185C0"
    legal_color: str = "#5185C0"
    illegal_color: str = "#99AABB"
    selected_color: str = "#C96144"
    new_free_color: str = "#99C290"
    new_obstacle_color: str = "#8281B9"
    action_start_radius_cells: float = 0.48
    action_length_cells: float = 1.50


@dataclass(frozen=True, slots=True)
class OnlineWorkflowAssets:
    """One decision cycle with observation fusion separated from motion."""

    step: int
    action_index: int
    action_key: str
    valid_action_indices: tuple[int, ...]
    trajectory_display_world: np.ndarray | None
    local_observation: Snapshot
    belief_before_update: Snapshot
    belief_after_update: Snapshot
    environment_after_action: Snapshot


def _figure_size(shape: tuple[int, int], style: OnlineWorkflowStyle) -> tuple[float, float]:
    rows, cols = max(1, int(shape[0])), max(1, int(shape[1]))
    width = float(cols) * float(style.cell_inches)
    height = float(rows) * float(style.cell_inches)
    scale = min(1.0, float(style.max_inches) / max(width, height))
    width *= scale
    height *= scale
    if min(width, height) < float(style.min_inches):
        scale = float(style.min_inches) / max(min(width, height), 1e-6)
        width *= scale
        height *= scale
    if max(width, height) > float(style.max_inches):
        scale = float(style.max_inches) / max(width, height)
        width *= scale
        height *= scale
    return width, height


def _create_axis(shape: tuple[int, int], style: OnlineWorkflowStyle):
    fig = plt.figure(figsize=_figure_size(shape, style), frameon=False)
    fig.patch.set_facecolor("white")
    ax = fig.add_axes((0.0, 0.0, 1.0, 1.0))
    ax.set_facecolor("white")
    _format_clean_axis(ax, shape)
    return fig, ax


def _save_online_figure(
    fig: plt.Figure,
    png_path: Path,
    *,
    dpi: int,
    include_svg: bool,
) -> Path | None:
    """Save one asset with a consistent opaque background and close its figure."""

    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        png_path,
        dpi=int(dpi),
        bbox_inches=None,
        pad_inches=0.0,
        facecolor="white",
        transparent=False,
    )
    svg_path = png_path.with_suffix(".svg") if include_svg else None
    if svg_path is not None:
        fig.savefig(
            svg_path,
            bbox_inches=None,
            pad_inches=0.0,
            facecolor="white",
            transparent=False,
        )
    plt.close(fig)
    return svg_path


def _heading_angle_deg(action_index: int) -> float:
    dr, dc = ACTIONS_8[int(action_index)]
    return float(math.degrees(math.atan2(float(dc), -float(dr))))


def _draw_topdown_robot(
    ax,
    *,
    row: float,
    col: float,
    heading_action: int,
    style: OnlineWorkflowStyle,
    alpha: float = 1.0,
    zorder: int = 12,
) -> None:
    """Draw the shared robot glyph at a row/column grid-cell center."""

    body_w = float(style.body_width)
    body_h = float(style.body_length)
    transform = (
        Affine2D()
        .rotate_deg_around(float(col), float(row), _heading_angle_deg(heading_action))
        + ax.transData
    )
    body = FancyBboxPatch(
        (float(col) - body_w / 2.0, float(row) - body_h / 2.0),
        body_w,
        body_h,
        boxstyle="round,pad=0.02,rounding_size=0.13",
        facecolor=style.body_color,
        edgecolor=style.body_edge_color,
        linewidth=0.075,
        alpha=float(alpha),
        transform=transform,
        zorder=zorder,
    )
    ax.add_patch(body)

    wheel_width = 0.18
    wheel_height = 0.34
    for dx in (-0.43, 0.43):
        for dy in (-0.27, 0.27):
            ax.add_patch(
                Ellipse(
                    (float(col) + dx, float(row) + dy),
                    width=wheel_width,
                    height=wheel_height,
                    facecolor=style.wheel_color,
                    edgecolor="none",
                    alpha=float(alpha),
                    transform=transform,
                    zorder=zorder - 1,
                )
            )

    ax.add_patch(
        Circle(
            (float(col), float(row)),
            radius=0.18,
            facecolor=style.lidar_color,
            edgecolor=style.lidar_edge_color,
            linewidth=0.075,
            alpha=float(alpha),
            transform=transform,
            zorder=zorder + 1,
        )
    )
    ax.add_patch(
        Polygon(
            (
                (float(col), float(row) - 0.47),
                (float(col) - 0.13, float(row) - 0.29),
                (float(col) + 0.13, float(row) - 0.29),
            ),
            closed=True,
            facecolor=style.heading_color,
            edgecolor=style.body_edge_color,
            linewidth=0.045,
            alpha=float(alpha),
            transform=transform,
            zorder=zorder + 2,
        )
    )


def _draw_action_selection(
    ax,
    *,
    center_row: float,
    center_col: float,
    valid_action_indices: tuple[int, ...],
    chosen_action_index: int,
    style: OnlineWorkflowStyle,
) -> None:
    """Draw ACTIONS_8 legality and the selected action in canonical index order."""

    valid = set(int(idx) for idx in valid_action_indices)
    chosen = int(chosen_action_index)
    if chosen not in valid:
        raise ValueError(f"chosen action {chosen} is not legal: {sorted(valid)}")

    for action_idx, (dr, dc) in enumerate(ACTIONS_8):
        norm = math.hypot(float(dr), float(dc))
        unit_r, unit_c = float(dr) / norm, float(dc) / norm
        selected = int(action_idx) == chosen
        color = (
            style.selected_color
            if selected
            else style.legal_color
            if int(action_idx) in valid
            else style.illegal_color
        )
        start_radius = float(style.action_start_radius_cells)
        end_radius = start_radius + float(style.action_length_cells)
        arrow = FancyArrowPatch(
            posA=(
                float(center_col) + start_radius * unit_c,
                float(center_row) + start_radius * unit_r,
            ),
            posB=(
                float(center_col) + end_radius * unit_c,
                float(center_row) + end_radius * unit_r,
            ),
            arrowstyle="-|>",
            mutation_scale=15.0 if selected else 12.0,
            linewidth=3.1 if selected else 1.75,
            color=color,
            alpha=1.0 if selected else 0.92,
            capstyle="round",
            joinstyle="round",
            zorder=11 if selected else 9,
        )
        ax.add_patch(arrow)


def _draw_recent_trajectory(
    ax,
    snapshot: Snapshot,
    canvas: WorldCanvas,
    trajectory_world: np.ndarray,
    style: OnlineWorkflowStyle,
) -> None:
    rows, cols = _trajectory_world_to_canvas(
        snapshot,
        canvas,
        trajectory_world=np.asarray(trajectory_world, dtype=np.int32),
    )
    if rows.size <= 1:
        return
    ax.plot(
        cols,
        rows,
        color=style.trajectory_color,
        linewidth=1.8,
        alpha=0.82,
        solid_capstyle="round",
        zorder=6,
    )


def _belief_update_rgba(
    before_snapshot: Snapshot,
    after_snapshot: Snapshot,
    canvas: WorldCanvas,
    style: OnlineWorkflowStyle,
) -> tuple[np.ndarray, np.ndarray]:
    before = _project_belief_to_canvas(before_snapshot, canvas)
    after = _project_belief_to_canvas(after_snapshot, canvas)
    rgba = np.asarray(LOCAL_CMAP(LOCAL_NORM(before)), dtype=np.float32)
    new_mask = (before == INVISIBLE) & (after != INVISIBLE)
    new_free = new_mask & (after == EMPTY)
    new_obstacle = new_mask & (after == OBSTACLE)
    rgba[new_free] = np.asarray(to_rgba(style.new_free_color), dtype=np.float32)
    rgba[new_obstacle] = np.asarray(to_rgba(style.new_obstacle_color), dtype=np.float32)
    return rgba, new_mask


def _draw_environment_transition(
    ax,
    *,
    before_snapshot: Snapshot,
    after_snapshot: Snapshot,
    canvas: WorldCanvas,
    chosen_action_index: int,
    style: OnlineWorkflowStyle,
) -> None:
    """Draw the real ACTIONS_8 displacement using old/new robot poses only."""

    before_row, before_col = _agent_world_to_canvas(before_snapshot, canvas)
    after_row, after_col = _agent_world_to_canvas(after_snapshot, canvas)
    expected_dr, expected_dc = ACTIONS_8[int(chosen_action_index)]
    actual = (
        int(after_snapshot.agent_world[0] - before_snapshot.agent_world[0]),
        int(after_snapshot.agent_world[1] - before_snapshot.agent_world[1]),
    )
    if actual != (int(expected_dr), int(expected_dc)):
        raise ValueError(
            f"transition delta {actual} does not match ACTIONS_8[{chosen_action_index}]="
            f"{(expected_dr, expected_dc)}"
        )

    _draw_topdown_robot(
        ax,
        row=before_row,
        col=before_col,
        heading_action=chosen_action_index,
        style=style,
        alpha=0.42,
        zorder=17,
    )
    _draw_topdown_robot(
        ax,
        row=after_row,
        col=after_col,
        heading_action=chosen_action_index,
        style=style,
        alpha=1.0,
        zorder=18,
    )


def _recent_trajectory(
    assets: OnlineWorkflowAssets,
    *,
    recent_steps: int = 10,
) -> np.ndarray:
    source = (
        assets.belief_after_update.trajectory_world
        if assets.trajectory_display_world is None
        else assets.trajectory_display_world
    )
    trajectory = np.asarray(source, dtype=np.int32).reshape((-1, 2))
    max_points = max(1, int(recent_steps) + 1)
    return trajectory[-max_points:].copy()


def _render_local_observation(
    path: Path,
    *,
    assets: OnlineWorkflowAssets,
    sensor: RadarSensor,
    style: OnlineWorkflowStyle,
    dpi: int,
    include_svg: bool,
) -> tuple[Path | None, int]:
    snapshot = assets.local_observation
    fig, ax = _create_axis(snapshot.local_snap.shape, style)
    ax.imshow(snapshot.local_snap, cmap=LOCAL_CMAP, norm=LOCAL_NORM, origin="upper", interpolation="nearest")
    center_r, center_c = float(sensor.center_state[0]), float(sensor.center_state[1])
    ax.add_patch(
        Circle(
            (center_c, center_r),
            radius=float(sensor.scan_r) + 0.12,
            fill=False,
            edgecolor=style.scan_color,
            linewidth=1.0,
            linestyle=(0, (4, 3)),
            alpha=0.34,
            zorder=7,
        )
    )
    _draw_topdown_robot(
        ax,
        row=center_r,
        col=center_c,
        heading_action=assets.action_index,
        style=style,
    )
    _format_clean_axis(ax, snapshot.local_snap.shape)
    return _save_online_figure(fig, path, dpi=dpi, include_svg=include_svg), 0


def _render_action_selection(
    path: Path,
    *,
    assets: OnlineWorkflowAssets,
    sensor: RadarSensor,
    style: OnlineWorkflowStyle,
    dpi: int,
    include_svg: bool,
) -> Path | None:
    snapshot = assets.belief_after_update
    fig, ax = _create_axis(snapshot.local_snap.shape, style)
    ax.imshow(snapshot.local_snap, cmap=LOCAL_CMAP, norm=LOCAL_NORM, origin="upper", interpolation="nearest")
    center_r, center_c = float(sensor.center_state[0]), float(sensor.center_state[1])
    _draw_action_selection(
        ax,
        center_row=center_r,
        center_col=center_c,
        valid_action_indices=assets.valid_action_indices,
        chosen_action_index=assets.action_index,
        style=style,
    )
    _draw_topdown_robot(
        ax,
        row=center_r,
        col=center_c,
        heading_action=assets.action_index,
        style=style,
        zorder=14,
    )
    _format_clean_axis(ax, snapshot.local_snap.shape)
    return _save_online_figure(fig, path, dpi=dpi, include_svg=include_svg)


def _render_belief_update(
    path: Path,
    *,
    assets: OnlineWorkflowAssets,
    canvas: WorldCanvas,
    trajectory_world: np.ndarray,
    style: OnlineWorkflowStyle,
    dpi: int,
    include_svg: bool,
) -> tuple[Path | None, int]:
    panel_width, panel_height = _figure_size(canvas.shape, style)
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(min(8.4, panel_width * 2.35), max(2.2, panel_height * 0.78)),
        gridspec_kw={"width_ratios": (1.0, 0.56, 1.0), "wspace": 0.08},
        frameon=False,
    )
    fig.patch.set_facecolor("white")
    rgba, new_mask = _belief_update_rgba(
        assets.belief_before_update,
        assets.belief_after_update,
        canvas,
        style,
    )
    before = _project_belief_to_canvas(assets.belief_before_update, canvas)
    axes[0].imshow(before, cmap=LOCAL_CMAP, norm=LOCAL_NORM, origin="upper", interpolation="nearest", alpha=0.62)
    _draw_recent_trajectory(axes[0], assets.belief_before_update, canvas, trajectory_world, style)
    before_row, before_col = _agent_world_to_canvas(assets.belief_before_update, canvas)
    _draw_topdown_robot(
        axes[0],
        row=before_row,
        col=before_col,
        heading_action=assets.action_index,
        style=style,
        alpha=0.68,
    )
    _format_clean_axis(axes[0], canvas.shape)

    local = assets.local_observation.local_snap
    axes[1].imshow(local, cmap=LOCAL_CMAP, norm=LOCAL_NORM, origin="upper", interpolation="nearest")
    local_row, local_col = local.shape[0] // 2, local.shape[1] // 2
    _draw_topdown_robot(
        axes[1],
        row=float(local_row),
        col=float(local_col),
        heading_action=assets.action_index,
        style=style,
    )
    _format_clean_axis(axes[1], local.shape)

    axes[2].imshow(rgba, origin="upper", interpolation="nearest")
    _draw_recent_trajectory(axes[2], assets.belief_after_update, canvas, trajectory_world, style)
    after_row, after_col = _agent_world_to_canvas(assets.belief_after_update, canvas)
    _draw_topdown_robot(
        axes[2],
        row=after_row,
        col=after_col,
        heading_action=assets.action_index,
        style=style,
    )
    _format_clean_axis(axes[2], canvas.shape)
    axes[0].set_title(r"$B_{t-1}$", fontsize=10)
    axes[1].set_title(r"$o_t$", fontsize=10)
    axes[2].set_title(r"$B_t$", fontsize=10)
    return (
        _save_online_figure(fig, path, dpi=dpi, include_svg=include_svg),
        int(np.count_nonzero(new_mask)),
    )


def _render_environment_execution(
    path: Path,
    *,
    assets: OnlineWorkflowAssets,
    canvas: WorldCanvas,
    style: OnlineWorkflowStyle,
    dpi: int,
    include_svg: bool,
) -> Path | None:
    fig, ax = _create_axis(canvas.shape, style)
    belief_after = _project_belief_to_canvas(assets.belief_after_update, canvas)
    ax.imshow(belief_after, cmap=LOCAL_CMAP, norm=LOCAL_NORM, origin="upper", interpolation="nearest")
    _draw_environment_transition(
        ax,
        before_snapshot=assets.belief_after_update,
        after_snapshot=assets.environment_after_action,
        canvas=canvas,
        chosen_action_index=assets.action_index,
        style=style,
    )
    _format_clean_axis(ax, canvas.shape)
    return _save_online_figure(fig, path, dpi=dpi, include_svg=include_svg)


def _choose_action(
    *,
    planned_key: str,
    forced_key: str | None,
    valid_actions: tuple[int, ...],
    agent_state: tuple[int, int],
    visit_counts: dict[tuple[int, int], int],
) -> int:
    desired_action = int(KEY_TO_ACTION[forced_key or planned_key])
    if forced_key is not None and desired_action not in valid_actions:
        valid_keys = " ".join(ACTION_TO_KEY[int(idx)] for idx in valid_actions)
        raise RuntimeError(f"forced method action '{forced_key}' is illegal; valid actions: {valid_keys}")
    if desired_action in valid_actions:
        return desired_action
    return _select_fallback_action(
        valid_actions,
        agent_state=agent_state,
        visit_counts=visit_counts,
    )


def _run_online_workflow_rollout(
    config: ExportConfig,
    *,
    target_step: int,
    forced_method_action: str | None,
    trajectory_visual_step: int | None,
) -> tuple[RadarSensor, OnlineWorkflowAssets]:
    """Capture o_t -> B_t -> a_t -> p_(t+1) without fusing o_(t+1)."""

    if int(target_step) < 1:
        raise ValueError("online workflow step must be >= 1")
    forced_key = None if forced_method_action is None else str(forced_method_action).strip().lower()
    if forced_key is not None and forced_key not in KEY_TO_ACTION:
        raise ValueError(f"forced_method_action must be one of: {', '.join(sorted(KEY_TO_ACTION))}")
    if trajectory_visual_step is not None and not (0 <= int(trajectory_visual_step) <= int(target_step)):
        raise ValueError("trajectory_visual_step must be between 0 and target_step")

    _set_global_seed(config.seed)
    generator = RandomMapGenerator(
        rows=int(config.rows),
        cols=int(config.cols),
        obs_size=int(config.obs_size),
        obstacle_ratio=float(config.obstacle_ratio),
    )
    true_grid, start_state = generator.generate_map()
    free_mask = GridTopology.free_mask(true_grid)
    sensor = RadarSensor(scan_radius=int(config.scan_radius))
    obs_model = LocalObservationModel(true_grid, start_state, sensor=sensor)
    agent_state = (int(start_state[0]), int(start_state[1]))
    local_snap = np.asarray(obs_model.local_snap, dtype=np.int8).copy()
    cum_map = CumulativeBeliefMap(true_grid, agent_state, local_snap)
    trajectory_world = [agent_state]
    visit_counts: dict[tuple[int, int], int] = {agent_state: 1}
    trajectory_display_world = (
        np.asarray(trajectory_world, dtype=np.int32).copy()
        if trajectory_visual_step is not None and int(trajectory_visual_step) == 0
        else None
    )

    for step_idx in range(1, int(target_step) + 1):
        incoming_valid = GridTopology.valid_action_indices_fast(free_mask, agent_state)
        if not incoming_valid:
            raise RuntimeError(f"agent has no legal incoming move at step {step_idx}")
        incoming_action = _choose_action(
            planned_key=FIXED_ACTION_PREFERENCES[(step_idx - 1) % len(FIXED_ACTION_PREFERENCES)],
            forced_key=None,
            valid_actions=incoming_valid,
            agent_state=agent_state,
            visit_counts=visit_counts,
        )
        incoming_dr, incoming_dc = ACTIONS_8[incoming_action]
        agent_state = (int(agent_state[0] + incoming_dr), int(agent_state[1] + incoming_dc))
        trajectory_world.append(agent_state)
        visit_counts[agent_state] = int(visit_counts.get(agent_state, 0) + 1)
        local_snap = np.asarray(obs_model.observe_fast(agent_state), dtype=np.int8).copy()
        if trajectory_visual_step is not None and step_idx == int(trajectory_visual_step):
            trajectory_display_world = np.asarray(trajectory_world, dtype=np.int32).copy()

        if step_idx < int(target_step):
            cum_map.update(agent_state, local_snap)
            continue

        belief_before = _capture_snapshot(
            step=step_idx,
            agent_state=agent_state,
            trajectory_world=trajectory_world,
            local_snap=local_snap,
            cum_map=cum_map,
        )
        local_observation = belief_before
        cum_map.update(agent_state, local_snap)
        belief_after = _capture_snapshot(
            step=step_idx,
            agent_state=agent_state,
            trajectory_world=trajectory_world,
            local_snap=local_snap,
            cum_map=cum_map,
        )

        valid_actions = GridTopology.valid_action_indices_fast(free_mask, agent_state)
        if not valid_actions:
            raise RuntimeError(f"agent has no legal decision action at step {step_idx}")
        action_index = _choose_action(
            planned_key=FIXED_ACTION_PREFERENCES[step_idx % len(FIXED_ACTION_PREFERENCES)],
            forced_key=forced_key,
            valid_actions=valid_actions,
            agent_state=agent_state,
            visit_counts=visit_counts,
        )
        dr, dc = ACTIONS_8[action_index]
        next_state = (int(agent_state[0] + dr), int(agent_state[1] + dc))
        environment_trajectory = [*trajectory_world, next_state]
        next_observation = np.asarray(obs_model.observe_fast(next_state), dtype=np.int8).copy()
        environment_after = _capture_snapshot(
            step=step_idx + 1,
            agent_state=next_state,
            trajectory_world=environment_trajectory,
            local_snap=next_observation,
            cum_map=cum_map,
        )
        return sensor, OnlineWorkflowAssets(
            step=step_idx,
            action_index=int(action_index),
            action_key=ACTION_TO_KEY[int(action_index)],
            valid_action_indices=tuple(int(idx) for idx in valid_actions),
            trajectory_display_world=trajectory_display_world,
            local_observation=local_observation,
            belief_before_update=belief_before,
            belief_after_update=belief_after,
            environment_after_action=environment_after,
        )

    raise RuntimeError("online workflow rollout did not capture the requested step")


def export_online_workflow_assets(
    output_dir: Path | str,
    *,
    config: ExportConfig | None = None,
    step: int | None = None,
    forced_method_action: str | None = None,
    trajectory_visual_step: int | None = None,
    include_svg: bool = False,
) -> dict[str, object]:
    """Export four mutually consistent assets from one deterministic rollout transition."""

    output_dir_path = Path(output_dir)
    base = config if config is not None else ExportConfig(output_dir=output_dir_path)
    rollout_config = ExportConfig(
        rows=int(base.rows),
        cols=int(base.cols),
        obstacle_ratio=float(base.obstacle_ratio),
        obs_size=int(base.obs_size),
        scan_radius=int(base.scan_radius),
        seed=int(base.seed),
        step_mid=int(base.step_mid),
        step_late=int(base.step_late),
        dpi=int(base.dpi),
        output_dir=output_dir_path,
    )
    target_step = int(rollout_config.step_late if step is None else step)
    sensor, assets = _run_online_workflow_rollout(
        rollout_config,
        target_step=target_step,
        forced_method_action=forced_method_action,
        trajectory_visual_step=trajectory_visual_step,
    )
    if int(assets.action_index) not in set(int(idx) for idx in assets.valid_action_indices):
        raise RuntimeError("rollout selected an action outside the legal action set")

    expected_delta = tuple(int(v) for v in ACTIONS_8[int(assets.action_index)])
    actual_delta = (
        int(assets.environment_after_action.agent_world[0] - assets.belief_after_update.agent_world[0]),
        int(assets.environment_after_action.agent_world[1] - assets.belief_after_update.agent_world[1]),
    )
    if actual_delta != expected_delta:
        raise RuntimeError(f"rollout transition mismatch: expected {expected_delta}, got {actual_delta}")

    output_dir_path.mkdir(parents=True, exist_ok=True)
    files = {
        "local_observation_with_robot_radar": output_dir_path / "local_observation_with_robot_radar.png",
        "legal_action_selection": output_dir_path / "legal_action_selection.png",
        "belief_map_update_with_robot": output_dir_path / "belief_map_update_with_robot.png",
        "environment_execution": output_dir_path / "environment_execution.png",
    }
    style = OnlineWorkflowStyle()
    canvas = _build_method_world_canvas(
        assets.belief_before_update,
        assets.belief_after_update,
        sensor,
    )
    trajectory_world = _recent_trajectory(assets, recent_steps=10)

    svg_files: dict[str, Path] = {}
    local_svg, radar_ray_count = _render_local_observation(
        files["local_observation_with_robot_radar"],
        assets=assets,
        sensor=sensor,
        style=style,
        dpi=int(rollout_config.dpi),
        include_svg=include_svg,
    )
    if local_svg is not None:
        svg_files["local_observation_with_robot_radar"] = local_svg
    action_svg = _render_action_selection(
        files["legal_action_selection"],
        assets=assets,
        sensor=sensor,
        style=style,
        dpi=int(rollout_config.dpi),
        include_svg=include_svg,
    )
    if action_svg is not None:
        svg_files["legal_action_selection"] = action_svg
    belief_svg, newly_observed_cell_count = _render_belief_update(
        files["belief_map_update_with_robot"],
        assets=assets,
        canvas=canvas,
        trajectory_world=trajectory_world,
        style=style,
        dpi=int(rollout_config.dpi),
        include_svg=include_svg,
    )
    if belief_svg is not None:
        svg_files["belief_map_update_with_robot"] = belief_svg
    environment_svg = _render_environment_execution(
        files["environment_execution"],
        assets=assets,
        canvas=canvas,
        style=style,
        dpi=int(rollout_config.dpi),
        include_svg=include_svg,
    )
    if environment_svg is not None:
        svg_files["environment_execution"] = environment_svg

    manifest = {
        "seed": int(rollout_config.seed),
        "step": int(assets.step),
        "chosen_action_index": int(assets.action_index),
        "chosen_action_key": str(assets.action_key),
        "chosen_action_delta_row_col": list(expected_delta),
        "valid_action_indices": [int(idx) for idx in assets.valid_action_indices],
        "valid_action_keys": [ACTION_TO_KEY[int(idx)] for idx in assets.valid_action_indices],
        "agent_position_at_observation": [int(v) for v in assets.local_observation.agent_world],
        "agent_position_during_belief_update_before": [int(v) for v in assets.belief_before_update.agent_world],
        "agent_position_during_belief_update_after": [int(v) for v in assets.belief_after_update.agent_world],
        "agent_position_before_action": [int(v) for v in assets.belief_after_update.agent_world],
        "agent_position_after_action": [int(v) for v in assets.environment_after_action.agent_world],
        "belief_origin": [int(v) for v in assets.belief_after_update.belief_origin_world],
        "belief_origin_before": [int(v) for v in assets.belief_before_update.belief_origin_world],
        "belief_origin_after": [int(v) for v in assets.belief_after_update.belief_origin_world],
        "belief_canvas_origin": [int(v) for v in canvas.origin_world],
        "belief_canvas_shape": [int(v) for v in canvas.shape],
        "radar_ray_count": int(radar_ray_count),
        "newly_observed_cell_count": int(newly_observed_cell_count),
        "action_arrow_geometry": {
            "start_radius_cells": float(style.action_start_radius_cells),
            "end_radius_cells": float(style.action_start_radius_cells + style.action_length_cells),
            "euclidean_length_cells": float(style.action_length_cells),
            "per_direction_lengths_cells": [
                float(style.action_length_cells) for _ in ACTIONS_8
            ],
        },
        "coordinate_semantics": {
            "world_and_array_order": "row_col",
            "row_direction": "increases_downward",
            "action_order": ["N", "NE", "E", "SE", "S", "SW", "W", "NW"],
            "diagonal_legality": "target and both orthogonal side cells must be free",
        },
        "method_semantics": {
            "local_and_action_assets": "same-time local observation and legal action selection at p_t",
            "belief_update_asset": "o_t fused from B_(t-1) to B_t while the robot remains at p_t",
            "environment_asset": "p_t to p_(t+1) on B_t; o_(t+1) is returned but not fused into B_(t+1)",
            "truth_map_visibility": "not rendered; used only by existing sensor simulation and legal transition",
        },
        "files": {name: _format_output_path(path) for name, path in files.items()},
        "svg_files": {name: _format_output_path(path) for name, path in svg_files.items()},
    }
    manifest_path = output_dir_path / "online_workflow_assets_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {
        "files": files,
        "svg_files": svg_files,
        "manifest_path": manifest_path,
        "manifest": manifest,
    }
