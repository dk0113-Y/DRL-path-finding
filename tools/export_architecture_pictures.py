from __future__ import annotations

"""
导出方法示意图所需的静态图片。

运行方式：
    python tools/export_architecture_pictures.py

输出目录：
    run_picture/
"""

import argparse
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

if os.environ.get("DRL_PAPER_FIGURE_INTERACTIVE") != "1":
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib import font_manager
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from env.agent_version import LocalObservationModel
from env.block_random_g import RandomMapGenerator
from env.core_cummap import CumulativeBeliefMap
from env.core_radar import RadarSensor
from env.grid_topology import ACTIONS_8, INVISIBLE, GridTopology


def _configure_matplotlib_chinese_fonts() -> None:
    candidates = [
        "Microsoft YaHei",
        "SimHei",
        "PingFang SC",
        "Noto Sans CJK SC",
        "Source Han Sans SC",
    ]
    available_names = [font.name for font in font_manager.fontManager.ttflist]
    chosen: list[str] = []
    for candidate in candidates:
        candidate_lower = candidate.lower()
        for available in available_names:
            available_lower = available.lower()
            if candidate_lower == available_lower or candidate_lower in available_lower:
                if available not in chosen:
                    chosen.append(available)
                break

    if chosen:
        existing = list(plt.rcParams.get("font.sans-serif", []))
        plt.rcParams["font.family"] = "sans-serif"
        plt.rcParams["font.sans-serif"] = chosen + [name for name in existing if name not in chosen]
    plt.rcParams["axes.unicode_minus"] = False


_configure_matplotlib_chinese_fonts()

BELIEF_CMAP = ListedColormap(
    [
        "#5f6770",  # unknown
        "#f5f6f7",  # free
        "#1c232b",  # obstacle
    ]
)
BELIEF_NORM = BoundaryNorm([-1.5, -0.5, 0.5, 1.5], BELIEF_CMAP.N)

AGENT_COLOR = "#f2542d"
AGENT_MARKER_SIZE = 74
AGENT_EDGE_COLOR = "white"
AGENT_EDGE_WIDTH = 1.1
SCAN_EDGE_COLOR = "#0f4c5c"
TRAJECTORY_COLOR = "#2d6a8c"
TRAJECTORY_LINEWIDTH = 1.8

KEY_TO_ACTION = {
    "w": 0,
    "e": 1,
    "d": 2,
    "c": 3,
    "x": 4,
    "z": 5,
    "a": 6,
    "q": 7,
}
ACTION_TO_KEY = {action_idx: key for key, action_idx in KEY_TO_ACTION.items()}

FIXED_ACTION_PREFERENCES = (
    "q",
    "w",
    "w",
    "e",
    "w",
    "d",
    "d",
    "e",
)


@dataclass(frozen=True)
class ExportConfig:
    rows: int = 40
    cols: int = 60
    obstacle_ratio: float = 0.20
    obs_size: int = 6
    scan_radius: int = 10
    seed: int = 0
    step_mid: int = 4
    step_late: int = 8
    dpi: int = 240
    output_dir: Path = REPO_ROOT / "run_picture"


@dataclass(frozen=True)
class Snapshot:
    step: int
    agent_world: tuple[int, int]
    agent_array: tuple[int, int]
    belief_origin_world: tuple[int, int]
    analysis_box: tuple[int, int, int, int]
    trajectory_world: np.ndarray
    trajectory_array: np.ndarray
    local_snap: np.ndarray
    belief_map: np.ndarray


@dataclass(frozen=True)
class MethodFigureAssets:
    step: int
    action_index: int
    action_key: str
    trajectory_display_world: np.ndarray | None
    local_observation: Snapshot
    belief_before_update: Snapshot
    belief_after_update: Snapshot


@dataclass(frozen=True)
class WorldCanvas:
    origin_world: tuple[int, int]
    shape: tuple[int, int]


@dataclass(frozen=True)
class MethodFigureStyle:
    canvas_size: tuple[float, float] = (6.0, 4.4)
    arrow_canvas_size: tuple[float, float] = (2.1, 2.1)
    margins: tuple[float, float, float, float] = (0.0, 1.0, 1.0, 0.0)
    show_local_scan_circle: bool = True
    show_belief_scan_circle: bool = False
    overlay_known_alpha: float = 0.28
    overlay_new_alpha: float = 0.82
    action_arrow_color: str = "#1f4e79"
    action_arrow_linewidth: float = 2.4


def _set_global_seed(seed: int) -> None:
    np.random.seed(int(seed))
    random.seed(int(seed))


def _ensure_output_dir(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)


def _format_output_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def _clear_old_png_outputs(output_dir: Path) -> None:
    _ensure_output_dir(output_dir)
    for png_path in output_dir.glob("*.png"):
        png_path.unlink()


def _capture_snapshot(
    *,
    step: int,
    agent_state: tuple[int, int],
    trajectory_world: list[tuple[int, int]],
    local_snap: np.ndarray,
    cum_map: CumulativeBeliefMap,
) -> Snapshot:
    trajectory_world_arr = np.asarray(trajectory_world, dtype=np.int32)
    return Snapshot(
        step=int(step),
        agent_world=(int(agent_state[0]), int(agent_state[1])),
        agent_array=tuple(int(v) for v in cum_map.world_to_array(agent_state)),
        belief_origin_world=(int(cum_map.origin_world_rc[0]), int(cum_map.origin_world_rc[1])),
        analysis_box=(
            int(cum_map.analysis_box.r0),
            int(cum_map.analysis_box.r1),
            int(cum_map.analysis_box.c0),
            int(cum_map.analysis_box.c1),
        ),
        trajectory_world=trajectory_world_arr.copy(),
        trajectory_array=np.asarray(
            [tuple(int(v) for v in cum_map.world_to_array(tuple(world_rc))) for world_rc in trajectory_world_arr],
            dtype=np.int32,
        ),
        local_snap=np.asarray(local_snap, dtype=np.int8).copy(),
        belief_map=np.asarray(cum_map.map, dtype=np.int8).copy(),
    )


def _select_fallback_action(
    valid_actions: tuple[int, ...],
    *,
    agent_state: tuple[int, int],
    visit_counts: dict[tuple[int, int], int],
) -> int:
    return min(
        valid_actions,
        key=lambda action_idx: (
            visit_counts.get(
                (
                    int(agent_state[0] + ACTIONS_8[action_idx][0]),
                    int(agent_state[1] + ACTIONS_8[action_idx][1]),
                ),
                0,
            ),
            int(action_idx),
        ),
    )


def _run_deterministic_rollout(
    config: ExportConfig,
) -> tuple[RadarSensor, dict[int, Snapshot], tuple[str, ...], tuple[str, ...], MethodFigureAssets | None]:
    return _run_deterministic_rollout_with_method_assets(config, method_asset_step=None)


def _run_deterministic_rollout_with_method_assets(
    config: ExportConfig,
    *,
    method_asset_step: int | None,
    forced_method_action: str | None = None,
    trajectory_visual_step: int | None = None,
) -> tuple[RadarSensor, dict[int, Snapshot], tuple[str, ...], tuple[str, ...], MethodFigureAssets | None]:
    if method_asset_step is not None and int(method_asset_step) <= 0:
        raise ValueError("method_asset_step must be >= 1")
    forced_method_action_key: str | None = None
    if forced_method_action is not None:
        forced_method_action_key = str(forced_method_action).strip().lower()
        if forced_method_action_key not in KEY_TO_ACTION:
            allowed = ", ".join(sorted(KEY_TO_ACTION))
            raise ValueError(f"forced_method_action must be one of: {allowed}")
        if method_asset_step is None:
            raise ValueError("forced_method_action requires method_asset_step")
    if trajectory_visual_step is not None and int(trajectory_visual_step) < 0:
        raise ValueError("trajectory_visual_step must be >= 0")
    if (
        method_asset_step is not None
        and trajectory_visual_step is not None
        and int(trajectory_visual_step) > int(method_asset_step)
    ):
        raise ValueError("trajectory_visual_step cannot exceed method_asset_step for method asset export")

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
    local_snap = np.asarray(obs_model.local_snap, dtype=np.int8).copy()
    cum_map = CumulativeBeliefMap(true_grid, start_state, local_snap)

    checkpoints = {0, int(config.step_mid), int(config.step_late)}
    rollout_horizon = max(
        int(config.step_mid),
        int(config.step_late),
        int(method_asset_step or 0),
        int(trajectory_visual_step or 0),
    )
    agent_state = (int(start_state[0]), int(start_state[1]))
    trajectory_world = [agent_state]
    trajectory_display_world: np.ndarray | None = None
    if trajectory_visual_step is not None and int(trajectory_visual_step) == 0:
        trajectory_display_world = np.asarray(trajectory_world, dtype=np.int32).copy()
    snapshots: dict[int, Snapshot] = {
        0: _capture_snapshot(
            step=0,
            agent_state=agent_state,
            trajectory_world=trajectory_world,
            local_snap=local_snap,
            cum_map=cum_map,
        )
    }
    visit_counts: dict[tuple[int, int], int] = {agent_state: 1}
    executed_keys: list[str] = []
    method_before_update: Snapshot | None = None
    method_after_update: Snapshot | None = None
    method_action_index: int | None = None
    method_action_key: str | None = None

    for step_idx in range(1, rollout_horizon + 1):
        planned_key = FIXED_ACTION_PREFERENCES[(step_idx - 1) % len(FIXED_ACTION_PREFERENCES)]
        desired_action = int(KEY_TO_ACTION[planned_key])
        valid_actions = GridTopology.valid_action_indices_fast(free_mask, agent_state)
        if not valid_actions:
            raise RuntimeError(f"agent has no legal moves at step {step_idx}")

        if (
            method_asset_step is not None
            and forced_method_action_key is not None
            and step_idx == int(method_asset_step)
        ):
            chosen_action = int(KEY_TO_ACTION[forced_method_action_key])
            if chosen_action not in valid_actions:
                valid_keys = " ".join(ACTION_TO_KEY[int(action_idx)] for action_idx in valid_actions)
                raise RuntimeError(
                    f"forced method action '{forced_method_action_key}' is illegal at step {step_idx}; "
                    f"valid actions: {valid_keys}"
                )
        elif desired_action in valid_actions:
            chosen_action = desired_action
        else:
            chosen_action = _select_fallback_action(
                valid_actions,
                agent_state=agent_state,
                visit_counts=visit_counts,
            )

        if method_asset_step is not None and step_idx == int(method_asset_step):
            method_before_update = _capture_snapshot(
                step=step_idx,
                agent_state=agent_state,
                trajectory_world=trajectory_world,
                local_snap=local_snap,
                cum_map=cum_map,
            )
            method_action_index = int(chosen_action)
            method_action_key = ACTION_TO_KEY[chosen_action]

        dr, dc = ACTIONS_8[chosen_action]
        agent_state = (int(agent_state[0] + dr), int(agent_state[1] + dc))
        visit_counts[agent_state] = int(visit_counts.get(agent_state, 0) + 1)
        trajectory_world.append(agent_state)
        if trajectory_visual_step is not None and step_idx == int(trajectory_visual_step):
            trajectory_display_world = np.asarray(trajectory_world, dtype=np.int32).copy()

        local_snap = np.asarray(obs_model.observe_fast(agent_state), dtype=np.int8).copy()
        cum_map.update(agent_state, local_snap)
        executed_keys.append(ACTION_TO_KEY[chosen_action])
        if method_asset_step is not None and step_idx == int(method_asset_step):
            method_after_update = _capture_snapshot(
                step=step_idx,
                agent_state=agent_state,
                trajectory_world=trajectory_world,
                local_snap=local_snap,
                cum_map=cum_map,
            )

        if step_idx in checkpoints:
            snapshots[step_idx] = _capture_snapshot(
                step=step_idx,
                agent_state=agent_state,
                trajectory_world=trajectory_world,
                local_snap=local_snap,
                cum_map=cum_map,
            )

    missing_steps = [step for step in sorted(checkpoints) if step not in snapshots]
    if missing_steps:
        raise RuntimeError(f"missing rollout checkpoints: {missing_steps}")

    if len(executed_keys) != rollout_horizon:
        raise RuntimeError("rollout did not produce the requested number of effective moves")

    method_assets: MethodFigureAssets | None = None
    if method_asset_step is not None:
        if (
            method_before_update is None
            or method_after_update is None
            or method_action_index is None
            or method_action_key is None
        ):
            raise RuntimeError(f"missing before/after snapshots for method asset step {method_asset_step}")
        if trajectory_visual_step is not None and trajectory_display_world is None:
            raise RuntimeError(f"missing trajectory display source for step {trajectory_visual_step}")
        method_assets = MethodFigureAssets(
            step=int(method_asset_step),
            action_index=int(method_action_index),
            action_key=str(method_action_key),
            trajectory_display_world=trajectory_display_world,
            local_observation=method_after_update,
            belief_before_update=method_before_update,
            belief_after_update=method_after_update,
        )

    return sensor, snapshots, tuple(FIXED_ACTION_PREFERENCES), tuple(executed_keys), method_assets


def _format_clean_axis(ax, shape: tuple[int, int]) -> None:
    ax.set_aspect("equal")
    ax.set_xlim(-0.5, float(shape[1]) - 0.5)
    ax.set_ylim(float(shape[0]) - 0.5, -0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def _draw_scan_circle(ax, *, center_row: float, center_col: float, radius: float, zorder: int = 4) -> None:
    ax.add_patch(
        Circle(
            (float(center_col), float(center_row)),
            radius=float(radius) + 0.15,
            fill=False,
            edgecolor=SCAN_EDGE_COLOR,
            linewidth=1.4,
            linestyle="--",
            alpha=0.90,
            zorder=zorder,
        )
    )


def _draw_agent(ax, *, row: float, col: float, zorder: int = 5) -> None:
    ax.scatter(
        [float(col)],
        [float(row)],
        marker="o",
        s=AGENT_MARKER_SIZE,
        c=AGENT_COLOR,
        edgecolors=AGENT_EDGE_COLOR,
        linewidths=AGENT_EDGE_WIDTH,
        zorder=zorder,
    )


def _draw_trajectory(ax, rows: np.ndarray, cols: np.ndarray, *, zorder: int = 4) -> None:
    if rows.size <= 1 or cols.size <= 1:
        return
    ax.plot(
        cols.astype(np.float32),
        rows.astype(np.float32),
        color=TRAJECTORY_COLOR,
        linewidth=TRAJECTORY_LINEWIDTH,
        alpha=0.96,
        solid_capstyle="round",
        zorder=zorder,
    )


def _render_local_axis(ax, snapshot: Snapshot, sensor: RadarSensor) -> None:
    ax.imshow(snapshot.local_snap, cmap=BELIEF_CMAP, norm=BELIEF_NORM, origin="upper", interpolation="nearest")
    _draw_scan_circle(
        ax,
        center_row=float(sensor.center_state[0]),
        center_col=float(sensor.center_state[1]),
        radius=float(sensor.scan_r),
    )
    _draw_agent(ax, row=float(sensor.center_state[0]), col=float(sensor.center_state[1]))
    _format_clean_axis(ax, snapshot.local_snap.shape)


def _render_belief_axis(ax, snapshot: Snapshot) -> None:
    ax.imshow(snapshot.belief_map, cmap=BELIEF_CMAP, norm=BELIEF_NORM, origin="upper", interpolation="nearest")
    _draw_trajectory(ax, snapshot.trajectory_array[:, 0], snapshot.trajectory_array[:, 1])
    _draw_agent(ax, row=float(snapshot.agent_array[0]), col=float(snapshot.agent_array[1]))
    _format_clean_axis(ax, snapshot.belief_map.shape)


def _grid_figure_size(shape: tuple[int, int], *, height: float = 4.8, min_width: float = 4.0) -> tuple[float, float]:
    rows, cols = int(shape[0]), int(shape[1])
    width = max(float(min_width), float(height) * (float(cols) / max(float(rows), 1.0)))
    return width, float(height)


def _method_figure_size_for_shape(shape: tuple[int, int], *, style: MethodFigureStyle) -> tuple[float, float]:
    rows = max(1, int(shape[0]))
    cols = max(1, int(shape[1]))
    max_width = max(0.1, float(style.canvas_size[0]))
    max_height = max(0.1, float(style.canvas_size[1]))
    scale = min(max_width / float(cols), max_height / float(rows))
    return float(cols) * scale, float(rows) * scale


def _create_method_axis(shape: tuple[int, int], *, style: MethodFigureStyle):
    fig = plt.figure(figsize=_method_figure_size_for_shape(shape, style=style), frameon=False)
    fig.patch.set_facecolor("white")
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.set_facecolor("white")
    return fig, ax


def _trim_external_png_background(path: Path) -> None:
    try:
        from PIL import Image
    except Exception:
        return

    image = Image.open(path).convert("RGBA")
    arr = np.asarray(image, dtype=np.uint8)
    if arr.size <= 0:
        return

    alpha = arr[..., 3]
    rgb = arr[..., :3]
    pure_white = (alpha == 255) & np.all(rgb == 255, axis=2)
    transparent = alpha == 0
    content = ~(pure_white | transparent)
    if not np.any(content):
        return

    rows, cols = np.nonzero(content)
    r0 = int(rows.min())
    r1 = int(rows.max()) + 1
    c0 = int(cols.min())
    c1 = int(cols.max()) + 1
    if r0 == 0 and c0 == 0 and r1 == int(arr.shape[0]) and c1 == int(arr.shape[1]):
        return

    cropped = arr[r0:r1, c0:c1]
    Image.fromarray(cropped, mode="RGBA").save(path)


def _save_figure(fig: plt.Figure, path: Path, *, dpi: int, tight: bool = True, transparent: bool = False) -> None:
    _ensure_output_dir(path.parent)
    facecolor = "none" if transparent else "white"
    if tight:
        fig.savefig(
            path,
            dpi=int(dpi),
            bbox_inches="tight",
            pad_inches=0.0,
            facecolor=facecolor,
            transparent=bool(transparent),
        )
    else:
        fig.savefig(
            path,
            dpi=int(dpi),
            bbox_inches=None,
            pad_inches=0.0,
            facecolor=facecolor,
            transparent=bool(transparent),
        )
    plt.close(fig)
    _trim_external_png_background(path)


def _export_local_radar_observation(path: Path, snapshot: Snapshot, sensor: RadarSensor, *, dpi: int) -> None:
    fig, ax = plt.subplots(figsize=_grid_figure_size(snapshot.local_snap.shape, height=4.4, min_width=4.4))
    _render_local_axis(ax, snapshot, sensor)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)
    _save_figure(fig, path, dpi=dpi)


def _export_belief_map(path: Path, snapshot: Snapshot, *, dpi: int) -> None:
    fig, ax = plt.subplots(figsize=_grid_figure_size(snapshot.belief_map.shape, height=4.9, min_width=4.2))
    _render_belief_axis(ax, snapshot)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)
    _save_figure(fig, path, dpi=dpi)


def _export_belief_growth_montage(
    path: Path,
    *,
    step0: Snapshot,
    step_mid: Snapshot,
    step_late: Snapshot,
    dpi: int,
) -> None:
    panels = ((step0, "初始步"), (step_mid, "第4步"), (step_late, "第8步"))
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(12.4, 4.8),
        gridspec_kw={"width_ratios": [panel.belief_map.shape[1] for panel, _ in panels]},
    )

    for ax, (snapshot, label) in zip(np.ravel(axes), panels):
        _render_belief_axis(ax, snapshot)
        ax.text(
            0.5,
            -0.08,
            label,
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=10,
            color="#243b53",
        )

    fig.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.16, wspace=0.18)
    _save_figure(fig, path, dpi=dpi)


def _export_local_to_belief_pair(
    path: Path,
    *,
    local_snapshot: Snapshot,
    belief_snapshot: Snapshot,
    sensor: RadarSensor,
    dpi: int,
) -> None:
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(9.2, 4.8),
        gridspec_kw={
            "width_ratios": [
                int(local_snapshot.local_snap.shape[1]),
                int(belief_snapshot.belief_map.shape[1]),
            ]
        },
    )

    _render_local_axis(axes[0], local_snapshot, sensor)
    _render_belief_axis(axes[1], belief_snapshot)

    fig.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02, wspace=0.34)
    _save_figure(fig, path, dpi=dpi)


def _snapshot_world_bounds(snapshot: Snapshot) -> tuple[int, int, int, int]:
    world_r0 = int(snapshot.belief_origin_world[0])
    world_c0 = int(snapshot.belief_origin_world[1])
    world_r1 = world_r0 + int(snapshot.belief_map.shape[0])
    world_c1 = world_c0 + int(snapshot.belief_map.shape[1])
    return world_r0, world_r1, world_c0, world_c1


def _local_visible_world(snapshot: Snapshot, sensor: RadarSensor) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    world_rows, world_cols = GridTopology.local_to_global_grid(
        snapshot.agent_world,
        tuple(snapshot.local_snap.shape),
        sensor.center_state,
    )
    visible = np.asarray(snapshot.local_snap != INVISIBLE, dtype=bool)
    return (
        np.asarray(world_rows[visible], dtype=np.int32),
        np.asarray(world_cols[visible], dtype=np.int32),
        np.asarray(snapshot.local_snap[visible], dtype=np.int8),
    )


def _build_method_world_canvas(
    before_snapshot: Snapshot,
    after_snapshot: Snapshot,
    sensor: RadarSensor,
) -> WorldCanvas:
    before_r0, before_r1, before_c0, before_c1 = _snapshot_world_bounds(before_snapshot)
    after_r0, after_r1, after_c0, after_c1 = _snapshot_world_bounds(after_snapshot)

    min_r = min(before_r0, after_r0)
    max_r = max(before_r1, after_r1)
    min_c = min(before_c0, after_c0)
    max_c = max(before_c1, after_c1)

    for snapshot in (before_snapshot, after_snapshot):
        if snapshot.trajectory_world.size > 0:
            min_r = min(min_r, int(np.min(snapshot.trajectory_world[:, 0])))
            max_r = max(max_r, int(np.max(snapshot.trajectory_world[:, 0])) + 1)
            min_c = min(min_c, int(np.min(snapshot.trajectory_world[:, 1])))
            max_c = max(max_c, int(np.max(snapshot.trajectory_world[:, 1])) + 1)
        min_r = min(min_r, int(snapshot.agent_world[0]))
        max_r = max(max_r, int(snapshot.agent_world[0]) + 1)
        min_c = min(min_c, int(snapshot.agent_world[1]))
        max_c = max(max_c, int(snapshot.agent_world[1]) + 1)

    visible_rows, visible_cols, _ = _local_visible_world(after_snapshot, sensor)
    if visible_rows.size > 0 and visible_cols.size > 0:
        min_r = min(min_r, int(np.min(visible_rows)))
        max_r = max(max_r, int(np.max(visible_rows)) + 1)
        min_c = min(min_c, int(np.min(visible_cols)))
        max_c = max(max_c, int(np.max(visible_cols)) + 1)

    return WorldCanvas(
        origin_world=(int(min_r), int(min_c)),
        shape=(int(max_r - min_r), int(max_c - min_c)),
    )


def _world_overlap_slices(
    *,
    src_origin_world: tuple[int, int],
    src_shape: tuple[int, int],
    dst_origin_world: tuple[int, int],
    dst_shape: tuple[int, int],
) -> tuple[slice, slice, slice, slice] | None:
    src_r0 = int(src_origin_world[0])
    src_c0 = int(src_origin_world[1])
    src_r1 = src_r0 + int(src_shape[0])
    src_c1 = src_c0 + int(src_shape[1])

    dst_r0 = int(dst_origin_world[0])
    dst_c0 = int(dst_origin_world[1])
    dst_r1 = dst_r0 + int(dst_shape[0])
    dst_c1 = dst_c0 + int(dst_shape[1])

    overlap_r0 = max(src_r0, dst_r0)
    overlap_r1 = min(src_r1, dst_r1)
    overlap_c0 = max(src_c0, dst_c0)
    overlap_c1 = min(src_c1, dst_c1)
    if overlap_r0 >= overlap_r1 or overlap_c0 >= overlap_c1:
        return None

    src_rows = slice(overlap_r0 - src_r0, overlap_r1 - src_r0)
    src_cols = slice(overlap_c0 - src_c0, overlap_c1 - src_c0)
    dst_rows = slice(overlap_r0 - dst_r0, overlap_r1 - dst_r0)
    dst_cols = slice(overlap_c0 - dst_c0, overlap_c1 - dst_c0)
    return src_rows, src_cols, dst_rows, dst_cols


def _project_belief_to_canvas(snapshot: Snapshot, canvas: WorldCanvas) -> np.ndarray:
    belief_canvas = np.full(canvas.shape, INVISIBLE, dtype=np.int8)
    overlap = _world_overlap_slices(
        src_origin_world=snapshot.belief_origin_world,
        src_shape=snapshot.belief_map.shape,
        dst_origin_world=canvas.origin_world,
        dst_shape=canvas.shape,
    )
    if overlap is None:
        return belief_canvas

    src_rows, src_cols, dst_rows, dst_cols = overlap
    belief_canvas[dst_rows, dst_cols] = snapshot.belief_map[src_rows, src_cols]
    return belief_canvas


def _trajectory_world_to_canvas(
    snapshot: Snapshot,
    canvas: WorldCanvas,
    *,
    trajectory_world: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    display_trajectory = snapshot.trajectory_world if trajectory_world is None else np.asarray(trajectory_world, dtype=np.int32)
    if display_trajectory.size <= 0:
        return np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.float32)
    rows = display_trajectory[:, 0].astype(np.float32) - float(canvas.origin_world[0])
    cols = display_trajectory[:, 1].astype(np.float32) - float(canvas.origin_world[1])
    return rows, cols


def _trajectory_world_to_local(
    snapshot: Snapshot,
    sensor: RadarSensor,
    *,
    trajectory_world: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    display_trajectory = snapshot.trajectory_world if trajectory_world is None else np.asarray(trajectory_world, dtype=np.int32)
    if display_trajectory.size <= 0:
        return np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.float32)
    rows = display_trajectory[:, 0].astype(np.float32) - float(snapshot.agent_world[0]) + float(sensor.center_state[0])
    cols = display_trajectory[:, 1].astype(np.float32) - float(snapshot.agent_world[1]) + float(sensor.center_state[1])
    return rows, cols


def _agent_world_to_canvas(snapshot: Snapshot, canvas: WorldCanvas) -> tuple[float, float]:
    return (
        float(snapshot.agent_world[0]) - float(canvas.origin_world[0]),
        float(snapshot.agent_world[1]) - float(canvas.origin_world[1]),
    )


def _apply_method_layout(fig: plt.Figure, style: MethodFigureStyle) -> None:
    left, right, top, bottom = style.margins
    fig.subplots_adjust(left=left, right=right, top=top, bottom=bottom)


def _render_method_local_axis(
    ax,
    *,
    snapshot: Snapshot,
    sensor: RadarSensor,
    style: MethodFigureStyle,
    trajectory_world: np.ndarray | None = None,
) -> None:
    ax.imshow(snapshot.local_snap, cmap=BELIEF_CMAP, norm=BELIEF_NORM, origin="upper", interpolation="nearest")
    traj_rows, traj_cols = _trajectory_world_to_local(snapshot, sensor, trajectory_world=trajectory_world)
    _draw_trajectory(ax, traj_rows, traj_cols)
    if style.show_local_scan_circle:
        _draw_scan_circle(
            ax,
            center_row=float(sensor.center_state[0]),
            center_col=float(sensor.center_state[1]),
            radius=float(sensor.scan_r),
        )
    _draw_agent(ax, row=float(sensor.center_state[0]), col=float(sensor.center_state[1]))
    _format_clean_axis(ax, snapshot.local_snap.shape)


def _render_method_belief_axis(
    ax,
    *,
    snapshot: Snapshot,
    canvas: WorldCanvas,
    sensor: RadarSensor,
    style: MethodFigureStyle,
    show_analysis_box: bool = False,
    trajectory_world: np.ndarray | None = None,
) -> None:
    belief_canvas = _project_belief_to_canvas(snapshot, canvas)
    ax.imshow(belief_canvas, cmap=BELIEF_CMAP, norm=BELIEF_NORM, origin="upper", interpolation="nearest")
    traj_rows, traj_cols = _trajectory_world_to_canvas(snapshot, canvas, trajectory_world=trajectory_world)
    _draw_trajectory(ax, traj_rows, traj_cols)
    agent_row, agent_col = _agent_world_to_canvas(snapshot, canvas)
    if style.show_belief_scan_circle:
        _draw_scan_circle(ax, center_row=agent_row, center_col=agent_col, radius=float(sensor.scan_r))
    _draw_agent(ax, row=agent_row, col=agent_col)
    if show_analysis_box:
        r0, r1, c0, c1 = snapshot.analysis_box
        world_r0 = int(snapshot.belief_origin_world[0]) + int(r0)
        world_c0 = int(snapshot.belief_origin_world[1]) + int(c0)
        canvas_r0 = world_r0 - int(canvas.origin_world[0])
        canvas_c0 = world_c0 - int(canvas.origin_world[1])
        ax.add_patch(
            Rectangle(
                (float(canvas_c0) - 0.5, float(canvas_r0) - 0.5),
                float(int(c1) - int(c0)),
                float(int(r1) - int(r0)),
                fill=False,
                edgecolor="#0f4c5c",
                linewidth=1.05,
                linestyle="-",
                alpha=0.94,
                zorder=8,
            )
        )
    _format_clean_axis(ax, canvas.shape)


def _render_observation_overlay_axis(
    ax,
    *,
    before_snapshot: Snapshot,
    after_snapshot: Snapshot,
    canvas: WorldCanvas,
    sensor: RadarSensor,
    style: MethodFigureStyle,
    trajectory_world: np.ndarray | None = None,
) -> None:
    belief_before_canvas = _project_belief_to_canvas(before_snapshot, canvas)
    ax.imshow(belief_before_canvas, cmap=BELIEF_CMAP, norm=BELIEF_NORM, origin="upper", interpolation="nearest")

    visible_rows, visible_cols, visible_values = _local_visible_world(after_snapshot, sensor)
    overlay_rgba = np.zeros((*canvas.shape, 4), dtype=np.float32)
    if visible_values.size > 0:
        canvas_rows = visible_rows - int(canvas.origin_world[0])
        canvas_cols = visible_cols - int(canvas.origin_world[1])
        inside = (
            (canvas_rows >= 0)
            & (canvas_rows < int(canvas.shape[0]))
            & (canvas_cols >= 0)
            & (canvas_cols < int(canvas.shape[1]))
        )
        canvas_rows = canvas_rows[inside]
        canvas_cols = canvas_cols[inside]
        visible_values = visible_values[inside]
        if visible_values.size > 0:
            overlay_colors = np.asarray(BELIEF_CMAP(BELIEF_NORM(visible_values)), dtype=np.float32)
            changed = belief_before_canvas[canvas_rows, canvas_cols] != visible_values
            overlay_rgba[canvas_rows, canvas_cols, :3] = overlay_colors[:, :3]
            overlay_rgba[canvas_rows, canvas_cols, 3] = np.where(
                changed,
                np.float32(style.overlay_new_alpha),
                np.float32(style.overlay_known_alpha),
            )

    ax.imshow(overlay_rgba, origin="upper", interpolation="nearest")
    traj_rows, traj_cols = _trajectory_world_to_canvas(after_snapshot, canvas, trajectory_world=trajectory_world)
    _draw_trajectory(ax, traj_rows, traj_cols, zorder=5)
    agent_row, agent_col = _agent_world_to_canvas(after_snapshot, canvas)
    if style.show_belief_scan_circle:
        _draw_scan_circle(ax, center_row=agent_row, center_col=agent_col, radius=float(sensor.scan_r), zorder=6)
    _draw_agent(ax, row=agent_row, col=agent_col, zorder=7)
    _format_clean_axis(ax, canvas.shape)


def _export_method_local_observation(
    path: Path,
    *,
    snapshot: Snapshot,
    sensor: RadarSensor,
    style: MethodFigureStyle,
    dpi: int,
    trajectory_world: np.ndarray | None = None,
) -> None:
    fig, ax = _create_method_axis(snapshot.local_snap.shape, style=style)
    _render_method_local_axis(ax, snapshot=snapshot, sensor=sensor, style=style, trajectory_world=trajectory_world)
    _save_figure(fig, path, dpi=dpi, tight=False)


def _export_method_belief_map(
    path: Path,
    *,
    snapshot: Snapshot,
    canvas: WorldCanvas,
    sensor: RadarSensor,
    style: MethodFigureStyle,
    dpi: int,
    show_analysis_box: bool = False,
    trajectory_world: np.ndarray | None = None,
) -> None:
    fig, ax = _create_method_axis(canvas.shape, style=style)
    _render_method_belief_axis(
        ax,
        snapshot=snapshot,
        canvas=canvas,
        sensor=sensor,
        style=style,
        show_analysis_box=show_analysis_box,
        trajectory_world=trajectory_world,
    )
    _save_figure(fig, path, dpi=dpi, tight=False)


def _export_method_overlay(
    path: Path,
    *,
    before_snapshot: Snapshot,
    after_snapshot: Snapshot,
    canvas: WorldCanvas,
    sensor: RadarSensor,
    style: MethodFigureStyle,
    dpi: int,
    trajectory_world: np.ndarray | None = None,
) -> None:
    fig, ax = _create_method_axis(canvas.shape, style=style)
    _render_observation_overlay_axis(
        ax,
        before_snapshot=before_snapshot,
        after_snapshot=after_snapshot,
        canvas=canvas,
        sensor=sensor,
        style=style,
        trajectory_world=trajectory_world,
    )
    _save_figure(fig, path, dpi=dpi, tight=False)


def _export_executed_action_arrow(
    path: Path,
    *,
    before_snapshot: Snapshot,
    after_snapshot: Snapshot,
    style: MethodFigureStyle,
    dpi: int,
) -> None:
    delta_row = float(after_snapshot.agent_world[0] - before_snapshot.agent_world[0])
    delta_col = float(after_snapshot.agent_world[1] - before_snapshot.agent_world[1])
    max_abs = max(abs(delta_row), abs(delta_col), 1.0)
    unit_row = delta_row / max_abs
    unit_col = delta_col / max_abs

    fig = plt.figure(figsize=style.arrow_canvas_size, frameon=False)
    fig.patch.set_alpha(0.0)
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.set_facecolor("none")
    arrow = FancyArrowPatch(
        posA=(-0.42 * unit_col, -0.42 * unit_row),
        posB=(0.42 * unit_col, 0.42 * unit_row),
        arrowstyle="-|>",
        mutation_scale=18.0,
        linewidth=float(style.action_arrow_linewidth),
        color=style.action_arrow_color,
        capstyle="round",
        joinstyle="round",
    )
    ax.add_patch(arrow)
    ax.set_aspect("equal")
    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(1.0, -1.0)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    _save_figure(fig, path, dpi=dpi, tight=False, transparent=True)


def export_method_figure_assets(
    output_dir: Path | str,
    *,
    config: ExportConfig | None = None,
    step: int | None = None,
    forced_method_action: str | None = None,
    trajectory_visual_step: int | None = None,
    include_observation_overlay: bool = True,
    include_executed_action_arrow: bool = True,
    show_local_scan_circle: bool = True,
    show_belief_scan_circle: bool = False,
) -> dict[str, Path]:
    output_dir_path = Path(output_dir)
    base_config = config if config is not None else ExportConfig(output_dir=output_dir_path)
    rollout_config = ExportConfig(
        rows=int(base_config.rows),
        cols=int(base_config.cols),
        obstacle_ratio=float(base_config.obstacle_ratio),
        obs_size=int(base_config.obs_size),
        scan_radius=int(base_config.scan_radius),
        seed=int(base_config.seed),
        step_mid=int(base_config.step_mid),
        step_late=int(base_config.step_late),
        dpi=int(base_config.dpi),
        output_dir=output_dir_path,
    )
    target_step = int(rollout_config.step_late if step is None else step)
    style = MethodFigureStyle(
        show_local_scan_circle=bool(show_local_scan_circle),
        show_belief_scan_circle=bool(show_belief_scan_circle),
    )

    _ensure_output_dir(output_dir_path)
    sensor, _, _, _, method_assets = _run_deterministic_rollout_with_method_assets(
        rollout_config,
        method_asset_step=target_step,
        forced_method_action=forced_method_action,
        trajectory_visual_step=trajectory_visual_step,
    )
    if method_assets is None:
        raise RuntimeError("method figure asset export did not capture before/after snapshots")

    trajectory_display_world = method_assets.trajectory_display_world
    canvas = _build_method_world_canvas(
        method_assets.belief_before_update,
        method_assets.belief_after_update,
        sensor,
    )

    outputs = {
        "local_lidar_observation": output_dir_path / "local_lidar_observation.png",
        "belief_before_update": output_dir_path / "belief_before_update.png",
        "belief_after_update": output_dir_path / "belief_after_update.png",
    }

    _export_method_local_observation(
        outputs["local_lidar_observation"],
        snapshot=method_assets.local_observation,
        sensor=sensor,
        style=style,
        dpi=rollout_config.dpi,
        trajectory_world=trajectory_display_world,
    )
    _export_method_belief_map(
        outputs["belief_before_update"],
        snapshot=method_assets.belief_before_update,
        canvas=canvas,
        sensor=sensor,
        style=style,
        dpi=rollout_config.dpi,
        trajectory_world=trajectory_display_world,
    )
    _export_method_belief_map(
        outputs["belief_after_update"],
        snapshot=method_assets.belief_after_update,
        canvas=canvas,
        sensor=sensor,
        style=style,
        dpi=rollout_config.dpi,
        show_analysis_box=True,
        trajectory_world=trajectory_display_world,
    )

    if include_observation_overlay:
        outputs["observation_overlay"] = output_dir_path / "observation_overlay.png"
        _export_method_overlay(
            outputs["observation_overlay"],
            before_snapshot=method_assets.belief_before_update,
            after_snapshot=method_assets.belief_after_update,
            canvas=canvas,
            sensor=sensor,
            style=style,
            dpi=rollout_config.dpi,
            trajectory_world=trajectory_display_world,
        )

    if include_executed_action_arrow:
        outputs["executed_action_arrow"] = output_dir_path / "executed_action_arrow.png"
        _export_executed_action_arrow(
            outputs["executed_action_arrow"],
            before_snapshot=method_assets.belief_before_update,
            after_snapshot=method_assets.belief_after_update,
            style=style,
            dpi=rollout_config.dpi,
        )

    return outputs


def main() -> None:
    config = ExportConfig()
    _clear_old_png_outputs(config.output_dir)

    sensor, snapshots, planned_keys, executed_keys, _ = _run_deterministic_rollout(config)

    step0_snapshot = snapshots[0]
    step_mid_snapshot = snapshots[int(config.step_mid)]
    step_late_snapshot = snapshots[int(config.step_late)]

    outputs = [
        config.output_dir / "局部雷达观测.png",
        config.output_dir / "累计认知地图_初始步.png",
        config.output_dir / "累计认知地图_第4步.png",
        config.output_dir / "累计认知地图_第8步.png",
        config.output_dir / "累计认知地图动态增长.png",
        config.output_dir / "局部观测到累计认知地图.png",
    ]

    _export_local_radar_observation(outputs[0], step_late_snapshot, sensor, dpi=config.dpi)
    _export_belief_map(outputs[1], step0_snapshot, dpi=config.dpi)
    _export_belief_map(outputs[2], step_mid_snapshot, dpi=config.dpi)
    _export_belief_map(outputs[3], step_late_snapshot, dpi=config.dpi)
    _export_belief_growth_montage(
        outputs[4],
        step0=step0_snapshot,
        step_mid=step_mid_snapshot,
        step_late=step_late_snapshot,
        dpi=config.dpi,
    )
    _export_local_to_belief_pair(
        outputs[5],
        local_snapshot=step_late_snapshot,
        belief_snapshot=step_late_snapshot,
        sensor=sensor,
        dpi=config.dpi,
    )

    print(f"seed={config.seed}")
    print(f"action_preferences={' '.join(planned_keys)}")
    print(f"executed_actions={' '.join(executed_keys)}")
    print(
        "fallback_rule=当预设动作不合法时，从当前合法动作中选择“下一位置访问次数最少、若并列则动作索引最小”的稳定备选"
    )
    print(f"effective_moves={len(executed_keys)}")
    print(f"checkpoints=0,{config.step_mid},{config.step_late}")
    for output in outputs:
        print(_format_output_path(output))


def export_legacy_architecture_pictures(
    config: ExportConfig,
) -> tuple[list[Path], tuple[str, ...], tuple[str, ...]]:
    _clear_old_png_outputs(config.output_dir)

    sensor, snapshots, planned_keys, executed_keys, _ = _run_deterministic_rollout(config)

    step0_snapshot = snapshots[0]
    step_mid_snapshot = snapshots[int(config.step_mid)]
    step_late_snapshot = snapshots[int(config.step_late)]

    outputs = [
        config.output_dir / "local_radar_observation.png",
        config.output_dir / "belief_map_step0.png",
        config.output_dir / f"belief_map_step{int(config.step_mid)}.png",
        config.output_dir / f"belief_map_step{int(config.step_late)}.png",
        config.output_dir / "belief_growth_montage.png",
        config.output_dir / "local_to_belief_pair.png",
    ]

    _export_local_radar_observation(outputs[0], step_late_snapshot, sensor, dpi=config.dpi)
    _export_belief_map(outputs[1], step0_snapshot, dpi=config.dpi)
    _export_belief_map(outputs[2], step_mid_snapshot, dpi=config.dpi)
    _export_belief_map(outputs[3], step_late_snapshot, dpi=config.dpi)
    _export_belief_growth_montage(
        outputs[4],
        step0=step0_snapshot,
        step_mid=step_mid_snapshot,
        step_late=step_late_snapshot,
        dpi=config.dpi,
    )
    _export_local_to_belief_pair(
        outputs[5],
        local_snapshot=step_late_snapshot,
        belief_snapshot=step_late_snapshot,
        sensor=sensor,
        dpi=config.dpi,
    )

    return outputs, planned_keys, executed_keys


def _parse_action_key_arg(value: str) -> str:
    action_key = str(value).strip().lower()
    if action_key not in KEY_TO_ACTION:
        allowed = ", ".join(sorted(KEY_TO_ACTION))
        raise argparse.ArgumentTypeError(f"must be one of: {allowed}")
    return action_key


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export architecture or method-figure static assets.")
    parser.add_argument(
        "--mode",
        choices=("legacy", "method-assets"),
        default="legacy",
        help="legacy keeps the old multi-picture export; method-assets exports paper-ready before/after belief assets.",
    )
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "run_picture")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rows", type=int, default=40)
    parser.add_argument("--cols", type=int, default=60)
    parser.add_argument("--obstacle-ratio", type=float, default=0.20)
    parser.add_argument("--obs-size", type=int, default=6)
    parser.add_argument("--scan-radius", type=int, default=10)
    parser.add_argument("--step-mid", type=int, default=4)
    parser.add_argument("--step-late", type=int, default=8)
    parser.add_argument(
        "--method-step",
        type=int,
        default=None,
        help="Target step used by --mode method-assets. Defaults to --step-late.",
    )
    parser.add_argument(
        "--forced-method-action",
        type=_parse_action_key_arg,
        default=None,
        help="Force the specified key action only at --method-step for --mode method-assets.",
    )
    parser.add_argument(
        "--trajectory-visual-step",
        type=int,
        default=None,
        help="Truncate only the rendered trajectory to this rollout step for --mode method-assets.",
    )
    parser.add_argument("--dpi", type=int, default=240)
    parser.add_argument(
        "--hide-local-scan-circle",
        dest="show_local_scan_circle",
        action="store_false",
        help="Hide the scan circle on local_lidar_observation.png.",
    )
    parser.add_argument(
        "--show-belief-scan-circle",
        action="store_true",
        help="Show the scan circle on belief_before_update / belief_after_update / observation_overlay.",
    )
    parser.add_argument(
        "--no-observation-overlay",
        dest="include_observation_overlay",
        action="store_false",
        help="Skip exporting observation_overlay.png.",
    )
    parser.add_argument(
        "--no-executed-action-arrow",
        dest="include_executed_action_arrow",
        action="store_false",
        help="Skip exporting executed_action_arrow.png.",
    )
    parser.set_defaults(
        show_local_scan_circle=True,
        include_observation_overlay=True,
        include_executed_action_arrow=True,
    )
    return parser


def cli_main() -> None:
    args = _build_arg_parser().parse_args()
    config = ExportConfig(
        rows=int(args.rows),
        cols=int(args.cols),
        obstacle_ratio=float(args.obstacle_ratio),
        obs_size=int(args.obs_size),
        scan_radius=int(args.scan_radius),
        seed=int(args.seed),
        step_mid=int(args.step_mid),
        step_late=int(args.step_late),
        dpi=int(args.dpi),
        output_dir=Path(args.output_dir),
    )

    if args.mode == "legacy":
        outputs, planned_keys, executed_keys = export_legacy_architecture_pictures(config)
        print(f"mode={args.mode}")
        print(f"seed={config.seed}")
        print(f"action_preferences={' '.join(planned_keys)}")
        print(f"executed_actions={' '.join(executed_keys)}")
        print(f"effective_moves={len(executed_keys)}")
        print(f"checkpoints=0,{config.step_mid},{config.step_late}")
        for output in outputs:
            print(_format_output_path(output))
        return

    method_step = int(config.step_late if args.method_step is None else args.method_step)
    outputs = export_method_figure_assets(
        args.output_dir,
        config=config,
        step=method_step,
        forced_method_action=args.forced_method_action,
        trajectory_visual_step=args.trajectory_visual_step,
        include_observation_overlay=bool(args.include_observation_overlay),
        include_executed_action_arrow=bool(args.include_executed_action_arrow),
        show_local_scan_circle=bool(args.show_local_scan_circle),
        show_belief_scan_circle=bool(args.show_belief_scan_circle),
    )
    print(f"mode={args.mode}")
    print(f"seed={config.seed}")
    print(f"method_asset_step={method_step}")
    print(f"forced_method_action={args.forced_method_action or ''}")
    print(f"trajectory_visual_step={args.trajectory_visual_step if args.trajectory_visual_step is not None else ''}")
    print(f"show_local_scan_circle={bool(args.show_local_scan_circle)}")
    print(f"show_belief_scan_circle={bool(args.show_belief_scan_circle)}")
    for name, output in outputs.items():
        print(f"{name}={_format_output_path(output)}")


if __name__ == "__main__":
    cli_main()
