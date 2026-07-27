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

from env.core_radar import RadarSensor
from env.grid_topology import ACTIONS_8, EMPTY, INVISIBLE, OBSTACLE
from tools.export_architecture_pictures import (
    ACTION_TO_KEY,
    ExportConfig,
    MethodFigureAssets,
    Snapshot,
    WorldCanvas,
    _agent_world_to_canvas,
    _build_method_world_canvas,
    _format_clean_axis,
    _format_output_path,
    _project_belief_to_canvas,
    _run_deterministic_rollout_with_method_assets,
    _trajectory_world_to_canvas,
)


LOCAL_CMAP = ListedColormap(("#626b75", "#f5f6f7", "#1c232b"))
LOCAL_NORM = BoundaryNorm((-1.5, -0.5, 0.5, 1.5), LOCAL_CMAP.N)


@dataclass(frozen=True, slots=True)
class OnlineWorkflowStyle:
    """Centralized cell-relative styles shared by all online-workflow assets."""

    max_inches: float = 5.6
    min_inches: float = 3.4
    cell_inches: float = 0.205
    body_width: float = 0.72
    body_length: float = 1.02
    body_color: str = "#f2542d"
    body_edge_color: str = "#7a2f19"
    wheel_color: str = "#252b31"
    lidar_color: str = "#f8fbfd"
    lidar_edge_color: str = "#0f4c5c"
    heading_color: str = "#fff4d6"
    trajectory_color: str = "#2d6a8c"
    scan_color: str = "#277da1"
    legal_color: str = "#2a9d55"
    illegal_color: str = "#d1495b"
    selected_color: str = "#1565c0"
    new_free_color: str = "#9ed9e8"
    new_obstacle_color: str = "#2e7180"


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


def _radar_direction_targets(radius: int) -> tuple[tuple[int, int], ...]:
    diagonal = max(1, int(math.floor(float(radius) / math.sqrt(2.0))))
    return (
        (-int(radius), 0),
        (-diagonal, diagonal),
        (0, int(radius)),
        (diagonal, diagonal),
        (int(radius), 0),
        (diagonal, -diagonal),
        (0, -int(radius)),
        (-diagonal, -diagonal),
    )


def _draw_radar_rays(
    ax,
    *,
    local_snap: np.ndarray,
    sensor: RadarSensor,
    style: OnlineWorkflowStyle,
    zorder: int = 8,
) -> int:
    """Draw eight obstacle-truncated rays using the sensor's own LOS templates."""

    ray_by_target = {
        (int(ray[-1][0]), int(ray[-1][1])): ray
        for ray in sensor.local_ray_templates
        if ray
    }
    center_r, center_c = int(sensor.center_state[0]), int(sensor.center_state[1])
    drawn = 0
    for target in _radar_direction_targets(int(sensor.scan_r)):
        ray = ray_by_target.get(target)
        if ray is None:
            continue
        end_r, end_c = center_r, center_c
        for _, _, local_r, local_c in ray[1:]:
            value = int(local_snap[int(local_r), int(local_c)])
            if value == INVISIBLE:
                break
            end_r, end_c = int(local_r), int(local_c)
            if value == OBSTACLE:
                break
        if end_r == center_r and end_c == center_c:
            continue
        ax.plot(
            (float(center_c), float(end_c)),
            (float(center_r), float(end_r)),
            color=style.scan_color,
            linewidth=0.9,
            alpha=0.62,
            solid_capstyle="round",
            zorder=zorder,
        )
        ax.scatter(
            (float(end_c),),
            (float(end_r),),
            s=8.0,
            c=style.scan_color,
            alpha=0.68,
            linewidths=0.0,
            zorder=zorder + 1,
        )
        drawn += 1
    return drawn


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
        arrow = FancyArrowPatch(
            posA=(
                float(center_col) + 0.48 * unit_c,
                float(center_row) + 0.48 * unit_r,
            ),
            posB=(
                float(center_col) + 1.02 * float(dc),
                float(center_row) + 1.02 * float(dr),
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
    """Draw the real ACTIONS_8 displacement and the old/new robot poses."""

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

    ax.add_patch(
        FancyArrowPatch(
            posA=(before_col, before_row),
            posB=(after_col, after_row),
            arrowstyle="-|>",
            mutation_scale=23.0,
            linewidth=3.8,
            color=style.selected_color,
            capstyle="round",
            joinstyle="round",
            zorder=16,
        )
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
    assets: MethodFigureAssets,
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
    assets: MethodFigureAssets,
    sensor: RadarSensor,
    style: OnlineWorkflowStyle,
    dpi: int,
    include_svg: bool,
) -> tuple[Path | None, int]:
    snapshot = assets.belief_before_update
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
    ray_count = _draw_radar_rays(
        ax,
        local_snap=snapshot.local_snap,
        sensor=sensor,
        style=style,
    )
    _draw_topdown_robot(
        ax,
        row=center_r,
        col=center_c,
        heading_action=assets.action_index,
        style=style,
    )
    _format_clean_axis(ax, snapshot.local_snap.shape)
    return _save_online_figure(fig, path, dpi=dpi, include_svg=include_svg), ray_count


def _render_action_selection(
    path: Path,
    *,
    assets: MethodFigureAssets,
    sensor: RadarSensor,
    style: OnlineWorkflowStyle,
    dpi: int,
    include_svg: bool,
) -> Path | None:
    snapshot = assets.belief_before_update
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
    assets: MethodFigureAssets,
    canvas: WorldCanvas,
    trajectory_world: np.ndarray,
    style: OnlineWorkflowStyle,
    dpi: int,
    include_svg: bool,
) -> tuple[Path | None, int]:
    fig, ax = _create_axis(canvas.shape, style)
    rgba, new_mask = _belief_update_rgba(
        assets.belief_before_update,
        assets.belief_after_update,
        canvas,
        style,
    )
    ax.imshow(rgba, origin="upper", interpolation="nearest")
    _draw_recent_trajectory(ax, assets.belief_after_update, canvas, trajectory_world, style)
    agent_row, agent_col = _agent_world_to_canvas(assets.belief_after_update, canvas)
    _draw_topdown_robot(
        ax,
        row=agent_row,
        col=agent_col,
        heading_action=assets.action_index,
        style=style,
    )
    _format_clean_axis(ax, canvas.shape)
    return (
        _save_online_figure(fig, path, dpi=dpi, include_svg=include_svg),
        int(np.count_nonzero(new_mask)),
    )


def _render_environment_execution(
    path: Path,
    *,
    assets: MethodFigureAssets,
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
        before_snapshot=assets.belief_before_update,
        after_snapshot=assets.belief_after_update,
        canvas=canvas,
        chosen_action_index=assets.action_index,
        style=style,
    )
    _format_clean_axis(ax, canvas.shape)
    return _save_online_figure(fig, path, dpi=dpi, include_svg=include_svg)


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
    sensor, _, _, _, assets = _run_deterministic_rollout_with_method_assets(
        rollout_config,
        method_asset_step=target_step,
        forced_method_action=forced_method_action,
        trajectory_visual_step=trajectory_visual_step,
    )
    if assets is None:
        raise RuntimeError("online workflow export did not capture a transition")
    if int(assets.action_index) not in set(int(idx) for idx in assets.valid_action_indices):
        raise RuntimeError("rollout selected an action outside the legal action set")

    expected_delta = tuple(int(v) for v in ACTIONS_8[int(assets.action_index)])
    actual_delta = (
        int(assets.belief_after_update.agent_world[0] - assets.belief_before_update.agent_world[0]),
        int(assets.belief_after_update.agent_world[1] - assets.belief_before_update.agent_world[1]),
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
        "agent_position_before_action": [int(v) for v in assets.belief_before_update.agent_world],
        "agent_position_after_action": [int(v) for v in assets.belief_after_update.agent_world],
        "belief_origin": [int(v) for v in assets.belief_after_update.belief_origin_world],
        "belief_origin_before": [int(v) for v in assets.belief_before_update.belief_origin_world],
        "belief_origin_after": [int(v) for v in assets.belief_after_update.belief_origin_world],
        "belief_canvas_origin": [int(v) for v in canvas.origin_world],
        "belief_canvas_shape": [int(v) for v in canvas.shape],
        "radar_ray_count": int(radar_ray_count),
        "newly_observed_cell_count": int(newly_observed_cell_count),
        "coordinate_semantics": {
            "world_and_array_order": "row_col",
            "row_direction": "increases_downward",
            "action_order": ["N", "NE", "E", "SE", "S", "SW", "W", "NW"],
            "diagonal_legality": "target and both orthogonal side cells must be free",
        },
        "method_semantics": {
            "local_and_action_assets": "pre-action local observation at the decision state",
            "belief_update_asset": "post-action local observation merged into cumulative belief",
            "environment_asset": "the selected pre-action transition rendered on post-update belief",
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
