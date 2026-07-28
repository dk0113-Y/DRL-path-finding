from __future__ import annotations

"""Export the real 21 x 21 local LOS observation and 8-neighbor action scene."""

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import matplotlib

if os.environ.get("DRL_PAPER_FIGURE_INTERACTIVE") != "1":
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from env.grid_topology import ACTIONS_8, INVISIBLE, OBSTACLE
from tools.export_figure_demo_blueprint import (
    FigureDemoBlueprint,
    blueprint_manifest,
    build_figure_demo_blueprint,
    clip_ray_to_local_snap,
    select_representative_ray_indices,
)
from tools.paper_figure_style import (
    PaperFigureStyle,
    draw_topdown_robot,
    load_paper_figure_style,
    occupancy_colormap,
    robot_envelope_diameter_cells,
)


@dataclass(frozen=True, slots=True)
class EnvironmentFigureStyle:
    """Publication layout backed by the shared JSON style contract."""

    paper: PaperFigureStyle = field(default_factory=load_paper_figure_style)
    figure_size: tuple[float, float] = (7.2, 7.2)
    grid_line_width: float = 0.35
    grid_line_alpha: float = 0.68
    radar_ray_width: float = 0.78
    radar_ray_alpha: float = 0.42
    action_line_width: float = 2.2
    action_alpha: float = 0.96
    action_mutation_scale: float = 13.0
    action_start_radius: float = 0.50
    action_length: float = 1.50


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
    if int(scan_radius) != 10:
        raise ValueError("the paper Figure 2 contract requires scan_radius=10")
    if int(step) < 1:
        raise ValueError("step must be >= 1")
    if int(visual_ray_count) < 0:
        raise ValueError("visual_ray_count must be >= 0")
    if int(dpi) < 1:
        raise ValueError("dpi must be >= 1")


def _ray_endpoint(ray: Sequence[Sequence[int]]) -> tuple[int, int]:
    if not ray:
        raise ValueError("ray template must not be empty")
    return int(ray[-1][0]), int(ray[-1][1])


def _normalized_ray_angle(ray: Sequence[Sequence[int]]) -> float:
    rel_row, rel_col = _ray_endpoint(ray)
    return float(np.mod(np.arctan2(float(rel_row), float(rel_col)), 2.0 * np.pi))


def _select_representative_rays(
    ray_templates: Sequence[Sequence[Sequence[int]]],
    visual_ray_count: int = 32,
) -> tuple[Sequence[Sequence[int]], ...]:
    indices = select_representative_ray_indices(ray_templates, visual_ray_count=int(visual_ray_count))
    return tuple(ray_templates[index] for index in indices)


def _clip_ray_to_observation(
    ray: Sequence[Sequence[int]],
    observation: np.ndarray,
    *,
    obstacle_value: int = OBSTACLE,
    invisible_value: int = INVISIBLE,
) -> tuple[tuple[int, int, int, int], ...]:
    if int(obstacle_value) != OBSTACLE or int(invisible_value) != INVISIBLE:
        observation_copy = np.asarray(observation, dtype=np.int8).copy()
        observation_copy[observation_copy == int(obstacle_value)] = OBSTACLE
        observation_copy[observation_copy == int(invisible_value)] = INVISIBLE
        return clip_ray_to_local_snap(ray, observation_copy)
    return clip_ray_to_local_snap(ray, observation)


def _draw_radar_rays(
    ax,
    *,
    center_rc: tuple[float, float],
    blueprint: FigureDemoBlueprint,
    style: EnvironmentFigureStyle,
) -> tuple[tuple[tuple[int, int, int, int], ...], ...]:
    center_row, center_col = float(center_rc[0]), float(center_rc[1])
    drawn: list[tuple[tuple[int, int, int, int], ...]] = []
    for representative in blueprint.representative_rays:
        clipped = representative.points
        if len(clipped) <= 1:
            continue
        _, _, end_local_row, end_local_col = clipped[-1]
        ax.plot(
            [center_col, float(end_local_col)],
            [center_row, float(end_local_row)],
            color=style.paper.radar_palette["ray"],
            linewidth=float(style.radar_ray_width),
            alpha=float(style.radar_ray_alpha),
            solid_capstyle="round",
            zorder=3,
        )
        drawn.append(clipped)
    return tuple(drawn)


def _draw_eight_neighbor_action_arrows(
    ax,
    *,
    center_rc: tuple[float, float],
    style: EnvironmentFigureStyle,
    actions: Sequence[tuple[int, int]] = ACTIONS_8,
) -> tuple[tuple[int, int], ...]:
    """Draw the complete candidate action space with one uniform treatment."""

    center_row, center_col = float(center_rc[0]), float(center_rc[1])
    directions = tuple((int(delta_row), int(delta_col)) for delta_row, delta_col in actions)
    for delta_row, delta_col in directions:
        norm = float(np.hypot(delta_row, delta_col))
        if norm <= 0.0:
            raise ValueError("action directions must be non-zero")
        unit_row = float(delta_row) / norm
        unit_col = float(delta_col) / norm
        start = (
            center_col + unit_col * float(style.action_start_radius),
            center_row + unit_row * float(style.action_start_radius),
        )
        end = (
            start[0] + unit_col * float(style.action_length),
            start[1] + unit_row * float(style.action_length),
        )
        ax.add_patch(
            FancyArrowPatch(
                posA=start,
                posB=end,
                arrowstyle="-|>",
                mutation_scale=float(style.action_mutation_scale),
                linewidth=float(style.action_line_width),
                color=style.paper.radar_palette["action"],
                alpha=float(style.action_alpha),
                capstyle="round",
                joinstyle="round",
                zorder=6,
            )
        )
    return directions


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
    seed: int = 1,
    rows: int = 40,
    cols: int = 60,
    obstacle_ratio: float = 0.20,
    obs_size: int = 6,
    scan_radius: int = 10,
    step: int = 8,
    visual_ray_count: int = 32,
    dpi: int = 300,
    output_format: str = "both",
    style: EnvironmentFigureStyle | None = None,
) -> dict[str, Path]:
    """Export Figure 2 from the same seed/step blueprint used by Figure 1."""

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
    blueprint = build_figure_demo_blueprint(
        seed=int(seed),
        preferred_step=int(step),
        scan_radius=int(scan_radius),
        rows=int(rows),
        cols=int(cols),
        obstacle_ratio=float(obstacle_ratio),
        obs_size=int(obs_size),
        visual_ray_count=int(visual_ray_count),
    )
    local_snap = np.asarray(blueprint.local_observation.local_snap, dtype=np.int8)
    expected_shape = tuple(int(v) for v in blueprint.sensor.local_shape)
    if expected_shape != (21, 21) or tuple(local_snap.shape) != expected_shape:
        raise RuntimeError(f"Figure 2 requires the real 21x21 local_snap, got {local_snap.shape}")
    center = tuple(float(v) for v in blueprint.sensor.center_state)
    cmap, norm = occupancy_colormap(style_use.paper)

    fig, ax = plt.subplots(figsize=style_use.figure_size, frameon=False)
    ax.imshow(local_snap, cmap=cmap, norm=norm, origin="upper", interpolation="nearest", zorder=1)
    ax.set_xticks(np.arange(-0.5, expected_shape[1], 1.0), minor=True)
    ax.set_yticks(np.arange(-0.5, expected_shape[0], 1.0), minor=True)
    ax.grid(
        which="minor",
        color=style_use.paper.radar_palette["grid_line"],
        linewidth=float(style_use.grid_line_width),
        alpha=float(style_use.grid_line_alpha),
        zorder=2,
    )
    ax.tick_params(which="both", bottom=False, left=False, labelbottom=False, labelleft=False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    drawn_rays = _draw_radar_rays(ax, center_rc=center, blueprint=blueprint, style=style_use)
    ax.add_patch(
        Circle(
            (center[1], center[0]),
            radius=float(blueprint.scan_radius),
            fill=False,
            edgecolor=style_use.paper.radar_palette["nominal_boundary"],
            linewidth=1.05,
            linestyle=(0, (5, 4)),
            alpha=0.55,
            zorder=4,
        )
    )
    _draw_eight_neighbor_action_arrows(ax, center_rc=center, style=style_use)
    draw_topdown_robot(
        ax,
        row=center[0],
        col=center[1],
        heading_action=int(blueprint.selected_action),
        style=style_use.paper,
        zorder=8,
    )
    ax.set_aspect("equal")
    ax.set_xlim(-0.5, float(expected_shape[1]) - 0.5)
    ax.set_ylim(float(expected_shape[0]) - 0.5, -0.5)
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

    manifest = {
        **blueprint_manifest(blueprint),
        "figure": "Figure 2 local grid, obstacle-truncated LOS samples, and full 8-neighbor action space",
        "local_grid_source": "seed-1 FigureDemoBlueprint.local_snap_t",
        "drawn_representative_ray_count": int(len(drawn_rays)),
        "candidate_action_arrow_count": 8,
        "candidate_action_color": str(style_use.paper.radar_palette["action"]),
        "candidate_action_colors_uniform": True,
        "action_length_cells": float(style_use.action_length),
        "robot_style": {
            "contract_path": str(style_use.paper.contract_path),
            "contract_version": str(style_use.paper.version),
            "palette": dict(style_use.paper.robot_palette),
            "geometry_cell_relative": dict(style_use.paper.robot_geometry_cell_relative),
            "envelope_diameter_cells": float(robot_envelope_diameter_cells(style_use.paper)),
        },
        "figure_planning": {
            "claim": "The policy acts on a 21x21 obstacle-occluded local observation whose representative LOS paths terminate at obstacles or invisibility.",
            "anchor_panel_role": "methodological bridge defining the observation and action geometry",
        },
        "outputs": {key: str(value.resolve()) for key, value in outputs.items()},
    }
    manifest_path = output_dir_path / "environment_model_figure_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    outputs["manifest"] = manifest_path
    return outputs


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Export the real seed-1 21x21 local LOS observation with obstacle-truncated "
            "representative rays and the complete 8-neighbor candidate action space."
        )
    )
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "run_picture")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--rows", type=int, default=40)
    parser.add_argument("--cols", type=int, default=60)
    parser.add_argument("--obstacle-ratio", type=float, default=0.20)
    parser.add_argument("--obs-size", type=int, default=6)
    parser.add_argument("--scan-radius", type=int, default=10)
    parser.add_argument("--step", type=int, default=8)
    parser.add_argument(
        "--visual-ray-count",
        type=int,
        default=32,
        help="Visualization samples selected from RadarSensor.local_ray_templates; not sensor channels.",
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
    print(f"seed={int(args.seed)}")
    print(f"step={int(args.step)}")
    print(f"visual_ray_count={int(args.visual_ray_count)}")
    for extension, path in outputs.items():
        print(f"{extension}={path.resolve()}")


if __name__ == "__main__":
    cli_main()
