from __future__ import annotations

"""Export the environment, local radar observation, and 8-neighbor action scene.

This module is intentionally independent from the legacy architecture-asset
exporter. Importing it or requesting ``--help`` never creates output files.
"""

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib

if os.environ.get("DRL_PAPER_FIGURE_INTERACTIVE") != "1":
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch, FancyBboxPatch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from env.agent_version import LocalObservationModel
from env.block_random_g import RandomMapGenerator
from env.core_radar import RadarSensor
from env.grid_topology import ACTIONS_8, EMPTY, INVISIBLE, OBSTACLE, GridTopology


ENVIRONMENT_CMAP = ListedColormap(
    [
        "#99AABB",  # unknown
        "#F7F7F4",  # free
        "#202326",  # obstacle
    ]
)
ENVIRONMENT_NORM = BoundaryNorm([-1.5, -0.5, 0.5, 1.5], ENVIRONMENT_CMAP.N)


@dataclass(frozen=True, slots=True)
class EnvironmentFigureStyle:
    """Centralized, restrained styling for the future paper figure."""

    figure_size: tuple[float, float] = (9.6, 6.4)
    grid_line_color: str = "#D9DEE2"
    grid_line_width: float = 0.28
    grid_line_alpha: float = 0.62
    radar_ray_color: str = "#5185C0"
    radar_ray_width: float = 0.72
    radar_ray_alpha: float = 0.34
    action_color: str = "#C96144"
    action_line_width: float = 2.2
    action_alpha: float = 0.96
    action_mutation_scale: float = 13.0
    action_start_radius: float = 1.10
    action_end_radius: float = 2.25
    robot_body_color: str = "#55966B"
    robot_body_edge_color: str = "#2F5940"
    robot_body_width: float = 1.55
    robot_body_height: float = 1.85
    robot_body_rounding: float = 0.22
    robot_body_line_width: float = 1.15
    robot_wheel_color: str = "#30363B"
    robot_wheel_width: float = 0.34
    robot_wheel_height: float = 0.66
    robot_wheel_row_offset: float = 0.55
    robot_wheel_col_offset: float = 0.89
    robot_wheel_line_width: float = 0.8
    robot_radar_color: str = "#E99D4E"
    robot_radar_radius: float = 0.30
    robot_radar_line_width: float = 1.0
    robot_heading_color: str = "#F7F7F4"
    robot_heading_start_offset: float = 0.12
    robot_heading_end_offset: float = 0.72
    robot_heading_line_width: float = 1.25
    robot_heading_mutation_scale: float = 9.0
    robot_scale: float = 1.0


def _validate_export_arguments(
    *,
    rows: int,
    cols: int,
    obstacle_ratio: float,
    obs_size: int,
    scan_radius: int,
    step: int,
    visual_ray_count: int,
    dpi: int,
) -> None:
    if int(rows) < 6 or int(cols) < 6:
        raise ValueError("rows and cols must be >= 6")
    if not (0.0 <= float(obstacle_ratio) < 0.85):
        raise ValueError("obstacle_ratio must be in [0.0, 0.85)")
    if int(obs_size) < 1:
        raise ValueError("obs_size must be >= 1")
    if int(scan_radius) < 1:
        raise ValueError("scan_radius must be >= 1")
    if int(step) < 0:
        raise ValueError("step must be >= 0")
    if int(visual_ray_count) < 0:
        raise ValueError("visual_ray_count must be >= 0")
    if int(dpi) < 1:
        raise ValueError("dpi must be >= 1")


def _ray_endpoint(ray: Sequence[Sequence[int]]) -> tuple[int, int]:
    if not ray:
        raise ValueError("ray template must not be empty")
    endpoint = ray[-1]
    if len(endpoint) < 2:
        raise ValueError("ray template points must include relative row and column")
    return int(endpoint[0]), int(endpoint[1])


def _normalized_ray_angle(ray: Sequence[Sequence[int]]) -> float:
    rel_row, rel_col = _ray_endpoint(ray)
    return float(np.mod(np.arctan2(float(rel_row), float(rel_col)), 2.0 * np.pi))


def _select_representative_rays(
    ray_templates: Sequence[Sequence[Sequence[int]]],
    visual_ray_count: int = 32,
) -> tuple[Sequence[Sequence[int]], ...]:
    """Select approximately angle-uniform rays without mutating sensor templates.

    The returned rays are references to the existing ``RadarSensor`` LOS
    templates. For duplicate angular directions, the farthest template is used.
    ``visual_ray_count`` is a rendering sample count, not a sensor-channel count.
    """

    requested = max(0, int(visual_ray_count))
    if requested == 0:
        return tuple()

    outermost_by_angle: dict[float, tuple[float, Sequence[Sequence[int]]]] = {}
    for ray in tuple(ray_templates):
        if not ray:
            continue
        rel_row, rel_col = _ray_endpoint(ray)
        if rel_row == 0 and rel_col == 0:
            continue
        angle = _normalized_ray_angle(ray)
        distance_sq = float((rel_row * rel_row) + (rel_col * rel_col))
        angle_key = round(angle, 12)
        previous = outermost_by_angle.get(angle_key)
        if previous is None or distance_sq > previous[0]:
            outermost_by_angle[angle_key] = (distance_sq, ray)

    candidates = [
        (float(angle_key), item[1])
        for angle_key, item in outermost_by_angle.items()
    ]
    candidates.sort(key=lambda item: item[0])
    target_count = min(requested, len(candidates))
    if target_count == 0:
        return tuple()

    unused = set(range(len(candidates)))
    selected: list[tuple[float, Sequence[Sequence[int]]]] = []
    for target_idx in range(target_count):
        target_angle = (2.0 * np.pi * float(target_idx)) / float(target_count)

        def circular_distance(candidate_idx: int) -> tuple[float, float]:
            candidate_angle = candidates[candidate_idx][0]
            delta = abs(candidate_angle - target_angle)
            return min(delta, (2.0 * np.pi) - delta), candidate_angle

        chosen_idx = min(unused, key=circular_distance)
        selected.append(candidates[chosen_idx])
        unused.remove(chosen_idx)

    selected.sort(key=lambda item: item[0])
    return tuple(item[1] for item in selected)


def _clip_ray_to_observation(
    ray: Sequence[Sequence[int]],
    observation: np.ndarray,
    *,
    obstacle_value: int = OBSTACLE,
    invisible_value: int = INVISIBLE,
) -> tuple[tuple[int, int, int, int], ...]:
    """Clip one local LOS template at the first obstacle or unobserved cell."""

    observation_arr = np.asarray(observation)
    if observation_arr.ndim != 2:
        raise ValueError("observation must be a 2D array")

    clipped: list[tuple[int, int, int, int]] = []
    for point_idx, point in enumerate(ray):
        if len(point) < 4:
            raise ValueError("ray template points must have four coordinates")
        rel_row, rel_col, local_row, local_col = (int(point[0]), int(point[1]), int(point[2]), int(point[3]))
        if not (0 <= local_row < observation_arr.shape[0] and 0 <= local_col < observation_arr.shape[1]):
            break

        value = int(observation_arr[local_row, local_col])
        if point_idx > 0 and value == int(invisible_value):
            break
        clipped.append((rel_row, rel_col, local_row, local_col))
        if value == int(obstacle_value):
            break

    return tuple(clipped)


def _draw_radar_rays(
    ax,
    *,
    center_rc: tuple[float, float],
    ray_templates: Sequence[Sequence[Sequence[int]]],
    observation: np.ndarray,
    visual_ray_count: int,
    style: EnvironmentFigureStyle,
) -> tuple[tuple[tuple[int, int, int, int], ...], ...]:
    """Draw angle-sampled LOS rays and return their obstacle-clipped templates."""

    center_row, center_col = float(center_rc[0]), float(center_rc[1])
    clipped_rays: list[tuple[tuple[int, int, int, int], ...]] = []
    for ray in _select_representative_rays(ray_templates, visual_ray_count):
        clipped = _clip_ray_to_observation(ray, observation)
        if len(clipped) <= 1:
            continue
        end_rel_row, end_rel_col, _, _ = clipped[-1]
        ax.plot(
            [center_col, center_col + float(end_rel_col)],
            [center_row, center_row + float(end_rel_row)],
            color=style.radar_ray_color,
            linewidth=float(style.radar_ray_width),
            alpha=float(style.radar_ray_alpha),
            solid_capstyle="round",
            zorder=3,
        )
        clipped_rays.append(clipped)
    return tuple(clipped_rays)


def _draw_eight_neighbor_action_arrows(
    ax,
    *,
    center_rc: tuple[float, float],
    style: EnvironmentFigureStyle,
    actions: Sequence[tuple[int, int]] = ACTIONS_8,
) -> tuple[tuple[int, int], ...]:
    """Draw all candidate 8-neighbor actions with identical visual treatment."""

    center_row, center_col = float(center_rc[0]), float(center_rc[1])
    directions = tuple((int(dr), int(dc)) for dr, dc in actions)
    for delta_row, delta_col in directions:
        norm = float(np.hypot(delta_row, delta_col))
        if norm <= 0.0:
            raise ValueError("action directions must be non-zero")
        unit_row = float(delta_row) / norm
        unit_col = float(delta_col) / norm
        start = (
            center_col + (unit_col * float(style.action_start_radius)),
            center_row + (unit_row * float(style.action_start_radius)),
        )
        end = (
            center_col + (unit_col * float(style.action_end_radius)),
            center_row + (unit_row * float(style.action_end_radius)),
        )
        ax.add_patch(
            FancyArrowPatch(
                posA=start,
                posB=end,
                arrowstyle="-|>",
                mutation_scale=float(style.action_mutation_scale),
                linewidth=float(style.action_line_width),
                color=style.action_color,
                alpha=float(style.action_alpha),
                capstyle="round",
                joinstyle="round",
                zorder=6,
            )
        )
    return directions


def _draw_topdown_robot(
    ax,
    *,
    center_rc: tuple[float, float],
    style: EnvironmentFigureStyle,
) -> tuple[object, ...]:
    """Draw a compact top-down robot using only Matplotlib patches."""

    center_row, center_col = float(center_rc[0]), float(center_rc[1])
    scale = float(style.robot_scale)
    body_width = float(style.robot_body_width) * scale
    body_height = float(style.robot_body_height) * scale
    body = FancyBboxPatch(
        (center_col - (body_width / 2.0), center_row - (body_height / 2.0)),
        body_width,
        body_height,
        boxstyle=f"round,pad=0.02,rounding_size={float(style.robot_body_rounding) * scale}",
        facecolor=style.robot_body_color,
        edgecolor=style.robot_body_edge_color,
        linewidth=float(style.robot_body_line_width),
        zorder=8,
    )
    ax.add_patch(body)

    patches: list[object] = [body]
    wheel_width = float(style.robot_wheel_width) * scale
    wheel_height = float(style.robot_wheel_height) * scale
    wheel_row_offset = float(style.robot_wheel_row_offset) * scale
    wheel_col_offset = float(style.robot_wheel_col_offset) * scale
    for row_offset in (-wheel_row_offset, wheel_row_offset):
        for col_offset in (-wheel_col_offset, wheel_col_offset):
            wheel = Ellipse(
                (center_col + col_offset, center_row + row_offset),
                width=wheel_width,
                height=wheel_height,
                facecolor=style.robot_wheel_color,
                edgecolor=style.robot_wheel_color,
                linewidth=float(style.robot_wheel_line_width),
                zorder=7,
            )
            ax.add_patch(wheel)
            patches.append(wheel)

    radar = Circle(
        (center_col, center_row),
        radius=float(style.robot_radar_radius) * scale,
        facecolor=style.robot_radar_color,
        edgecolor=style.robot_body_edge_color,
        linewidth=float(style.robot_radar_line_width),
        zorder=10,
    )
    ax.add_patch(radar)
    patches.append(radar)

    heading = FancyArrowPatch(
        posA=(center_col, center_row - (float(style.robot_heading_start_offset) * scale)),
        posB=(center_col, center_row - (float(style.robot_heading_end_offset) * scale)),
        arrowstyle="-|>",
        mutation_scale=float(style.robot_heading_mutation_scale),
        linewidth=float(style.robot_heading_line_width),
        color=style.robot_heading_color,
        zorder=11,
    )
    ax.add_patch(heading)
    patches.append(heading)
    return tuple(patches)


def _advance_agent_position(
    grid: np.ndarray,
    start_state: tuple[int, int],
    *,
    step: int,
) -> tuple[int, int]:
    """Advance deterministically without rendering or changing environment state."""

    free = GridTopology.free_mask(grid)
    state = (int(start_state[0]), int(start_state[1]))
    visits: dict[tuple[int, int], int] = {state: 1}
    for step_idx in range(int(step)):
        valid = GridTopology.valid_action_indices_fast(free, state)
        if not valid:
            break
        preferred = int(step_idx % len(ACTIONS_8))
        action_idx = preferred if preferred in valid else min(
            valid,
            key=lambda idx: (
                visits.get(
                    (
                        state[0] + int(ACTIONS_8[idx][0]),
                        state[1] + int(ACTIONS_8[idx][1]),
                    ),
                    0,
                ),
                int(idx),
            ),
        )
        delta_row, delta_col = ACTIONS_8[action_idx]
        state = (state[0] + int(delta_row), state[1] + int(delta_col))
        visits[state] = int(visits.get(state, 0) + 1)
    return state


def _project_local_observation(
    *,
    grid_shape: tuple[int, int],
    agent_state: tuple[int, int],
    observation: np.ndarray,
    center_state: tuple[int, int],
) -> np.ndarray:
    display_grid = np.full(grid_shape, INVISIBLE, dtype=np.int8)
    global_rows, global_cols = GridTopology.local_to_global_grid(
        agent_state,
        tuple(observation.shape),
        center_state,
    )
    inside = (
        (global_rows >= 0)
        & (global_rows < int(grid_shape[0]))
        & (global_cols >= 0)
        & (global_cols < int(grid_shape[1]))
    )
    visible = inside & (np.asarray(observation) != INVISIBLE)
    display_grid[global_rows[visible], global_cols[visible]] = np.asarray(observation, dtype=np.int8)[visible]
    return display_grid


def _configure_publication_style() -> None:
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "Microsoft YaHei", "DejaVu Sans"]
    plt.rcParams["svg.fonttype"] = "none"
    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["legend.frameon"] = False


def export_environment_model_figure(
    output_dir: Path | str,
    *,
    seed: int = 0,
    rows: int = 40,
    cols: int = 60,
    obstacle_ratio: float = 0.20,
    obs_size: int = 6,
    scan_radius: int = 10,
    step: int = 0,
    visual_ray_count: int = 32,
    dpi: int = 300,
    output_format: str = "both",
    style: EnvironmentFigureStyle | None = None,
) -> dict[str, Path]:
    """Export the single-scene environment-model figure as PNG and/or SVG."""

    _validate_export_arguments(
        rows=rows,
        cols=cols,
        obstacle_ratio=obstacle_ratio,
        obs_size=obs_size,
        scan_radius=scan_radius,
        step=step,
        visual_ray_count=visual_ray_count,
        dpi=dpi,
    )
    format_value = str(output_format).strip().lower()
    if format_value not in {"png", "svg", "both"}:
        raise ValueError("output_format must be one of: png, svg, both")

    _configure_publication_style()
    style_use = style if style is not None else EnvironmentFigureStyle()
    generator = RandomMapGenerator(
        rows=int(rows),
        cols=int(cols),
        obs_size=int(obs_size),
        obstacle_ratio=float(obstacle_ratio),
    )
    true_grid, start_state = generator.generate_map(seed=int(seed))
    agent_state = _advance_agent_position(true_grid, start_state, step=int(step))
    sensor = RadarSensor(scan_radius=int(scan_radius))
    observation_model = LocalObservationModel(true_grid, agent_state, sensor=sensor)
    observation = np.asarray(observation_model.local_snap, dtype=np.int8).copy()
    display_grid = _project_local_observation(
        grid_shape=tuple(true_grid.shape),
        agent_state=agent_state,
        observation=observation,
        center_state=sensor.center_state,
    )

    fig, ax = plt.subplots(figsize=style_use.figure_size, frameon=False)
    ax.imshow(
        display_grid,
        cmap=ENVIRONMENT_CMAP,
        norm=ENVIRONMENT_NORM,
        origin="upper",
        interpolation="nearest",
        zorder=1,
    )
    ax.set_xticks(np.arange(-0.5, int(cols), 1.0), minor=True)
    ax.set_yticks(np.arange(-0.5, int(rows), 1.0), minor=True)
    ax.grid(
        which="minor",
        color=style_use.grid_line_color,
        linewidth=float(style_use.grid_line_width),
        alpha=float(style_use.grid_line_alpha),
        zorder=2,
    )
    ax.tick_params(which="both", bottom=False, left=False, labelbottom=False, labelleft=False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    _draw_radar_rays(
        ax,
        center_rc=agent_state,
        ray_templates=sensor.local_ray_templates,
        observation=observation,
        visual_ray_count=int(visual_ray_count),
        style=style_use,
    )
    _draw_eight_neighbor_action_arrows(ax, center_rc=agent_state, style=style_use)
    _draw_topdown_robot(ax, center_rc=agent_state, style=style_use)
    ax.set_aspect("equal")
    ax.set_xlim(-0.5, float(cols) - 0.5)
    ax.set_ylim(float(rows) - 0.5, -0.5)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)

    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    formats = ("png", "svg") if format_value == "both" else (format_value,)
    outputs: dict[str, Path] = {}
    for extension in formats:
        output_path = output_dir_path / f"environment_grid_radar_action_space.{extension}"
        save_kwargs: dict[str, object] = {
            "bbox_inches": "tight",
            "pad_inches": 0.01,
            "facecolor": "white",
        }
        if extension == "png":
            save_kwargs["dpi"] = int(dpi)
        fig.savefig(output_path, format=extension, **save_kwargs)
        outputs[extension] = output_path
    plt.close(fig)
    return outputs


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Export one combined 2D occupancy-grid scene with local radar rays "
            "and the complete 8-neighbor candidate action space."
        )
    )
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "run_picture")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rows", type=int, default=40)
    parser.add_argument("--cols", type=int, default=60)
    parser.add_argument("--obstacle-ratio", type=float, default=0.20)
    parser.add_argument("--obs-size", type=int, default=6)
    parser.add_argument("--scan-radius", type=int, default=10)
    parser.add_argument("--step", type=int, default=0)
    parser.add_argument(
        "--visual-ray-count",
        type=int,
        default=32,
        help="Rendering sample count selected uniformly by angle from existing LOS templates (default: 32).",
    )
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--format", choices=("png", "svg", "both"), default="both")
    return parser


def cli_main() -> None:
    args = _build_arg_parser().parse_args()
    outputs = export_environment_model_figure(
        args.output_dir,
        seed=int(args.seed),
        rows=int(args.rows),
        cols=int(args.cols),
        obstacle_ratio=float(args.obstacle_ratio),
        obs_size=int(args.obs_size),
        scan_radius=int(args.scan_radius),
        step=int(args.step),
        visual_ray_count=int(args.visual_ray_count),
        dpi=int(args.dpi),
        output_format=str(args.format),
    )
    print("mode=environment-model-figure")
    print(f"visual_ray_count={int(args.visual_ray_count)}")
    for extension, path in outputs.items():
        print(f"{extension}={path.resolve()}")


if __name__ == "__main__":
    cli_main()
