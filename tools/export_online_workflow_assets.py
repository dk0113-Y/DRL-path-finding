from __future__ import annotations

"""Export seed-aligned paper assets for one online exploration decision cycle."""

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch

from env.grid_topology import ACTIONS_8, OBSTACLE
from tools.export_architecture_pictures import (
    ACTION_TO_KEY,
    ExportConfig,
    Snapshot,
    WorldCanvas,
    _agent_world_to_canvas,
    _format_clean_axis,
    _format_output_path,
    _project_belief_to_canvas,
)
from tools.export_figure_demo_blueprint import (
    FigureDemoBlueprint,
    blueprint_manifest,
    build_figure_demo_blueprint,
)
from tools.paper_figure_style import (
    PaperFigureStyle,
    draw_topdown_robot,
    load_paper_figure_style,
    occupancy_colormap,
    robot_envelope_diameter_cells,
)


@dataclass(frozen=True, slots=True)
class OnlineWorkflowStyle:
    """Figure-1 layout values plus the shared paper style contract."""

    paper: PaperFigureStyle = field(default_factory=load_paper_figure_style)
    max_inches: float = 5.6
    min_inches: float = 3.4
    cell_inches: float = 0.205
    scan_boundary_alpha: float = 0.30
    action_start_radius_cells: float = 0.48
    action_length_cells: float = 1.50


@dataclass(frozen=True, slots=True)
class OnlineWorkflowAssets:
    """One decision cycle with observation fusion separated from motion."""

    step: int
    action_index: int
    action_key: str
    valid_action_indices: tuple[int, ...]
    invalid_action_indices: tuple[int, ...]
    local_observation: Snapshot
    belief_before_update: Snapshot
    belief_after_update: Snapshot
    environment_after_action: Snapshot


@dataclass(frozen=True, slots=True)
class ActionArrowSpec:
    action_index: int
    state: str
    color: str
    start_row: float
    start_col: float
    end_row: float
    end_col: float
    linewidth_pt: float
    mutation_scale: float

    @property
    def length_cells(self) -> float:
        return float(math.hypot(self.end_row - self.start_row, self.end_col - self.start_col))


def _assets_from_blueprint(blueprint: FigureDemoBlueprint) -> OnlineWorkflowAssets:
    return OnlineWorkflowAssets(
        step=int(blueprint.step),
        action_index=int(blueprint.selected_action),
        action_key=ACTION_TO_KEY[int(blueprint.selected_action)],
        valid_action_indices=tuple(int(v) for v in blueprint.valid_action_indices),
        invalid_action_indices=tuple(int(v) for v in blueprint.invalid_action_indices),
        local_observation=blueprint.local_observation,
        belief_before_update=blueprint.belief_before_update,
        belief_after_update=blueprint.belief_after_update,
        environment_after_action=blueprint.environment_after_action,
    )


def _run_online_workflow_rollout(
    config: ExportConfig,
    *,
    target_step: int,
    forced_method_action: str | None,
    trajectory_visual_step: int | None,
):
    """Compatibility wrapper returning the shared seed/step blueprint snapshots."""

    if trajectory_visual_step is not None:
        if not (0 <= int(trajectory_visual_step) <= int(target_step)):
            raise ValueError("trajectory_visual_step must be between 0 and target_step")
    blueprint = build_figure_demo_blueprint(
        seed=int(config.seed),
        preferred_step=int(target_step),
        scan_radius=int(config.scan_radius),
        rows=int(config.rows),
        cols=int(config.cols),
        obstacle_ratio=float(config.obstacle_ratio),
        obs_size=int(config.obs_size),
        visual_ray_count=32,
        forced_method_action=forced_method_action,
    )
    return blueprint.sensor, _assets_from_blueprint(blueprint)


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


def _action_arrow_specs(
    *,
    center_row: float,
    center_col: float,
    valid_action_indices: tuple[int, ...],
    chosen_action_index: int,
    style: OnlineWorkflowStyle,
) -> tuple[ActionArrowSpec, ...]:
    valid = set(int(index) for index in valid_action_indices)
    chosen = int(chosen_action_index)
    if chosen not in valid:
        raise ValueError(f"chosen action {chosen} is not legal: {sorted(valid)}")

    specs: list[ActionArrowSpec] = []
    for action_index, (delta_row, delta_col) in enumerate(ACTIONS_8):
        norm = math.hypot(float(delta_row), float(delta_col))
        unit_row = float(delta_row) / norm
        unit_col = float(delta_col) / norm
        state = "selected" if action_index == chosen else "legal" if action_index in valid else "illegal"
        start_radius = float(style.action_start_radius_cells)
        end_radius = start_radius + float(style.action_length_cells)
        specs.append(
            ActionArrowSpec(
                action_index=int(action_index),
                state=state,
                color=str(style.paper.fig1_action_palette[state]),
                start_row=float(center_row) + start_radius * unit_row,
                start_col=float(center_col) + start_radius * unit_col,
                end_row=float(center_row) + end_radius * unit_row,
                end_col=float(center_col) + end_radius * unit_col,
                linewidth_pt=float(
                    style.paper.rendering[
                        "selected_action_linewidth_pt" if state == "selected" else "normal_action_linewidth_pt"
                    ]
                ),
                mutation_scale=float(
                    style.paper.rendering[
                        "selected_action_arrowhead_scale"
                        if state == "selected"
                        else "normal_action_arrowhead_scale"
                    ]
                ),
            )
        )
    return tuple(specs)


def _draw_action_selection(
    ax,
    *,
    center_row: float,
    center_col: float,
    valid_action_indices: tuple[int, ...],
    chosen_action_index: int,
    style: OnlineWorkflowStyle,
) -> tuple[ActionArrowSpec, ...]:
    specs = _action_arrow_specs(
        center_row=center_row,
        center_col=center_col,
        valid_action_indices=valid_action_indices,
        chosen_action_index=chosen_action_index,
        style=style,
    )
    for spec in specs:
        ax.add_patch(
            FancyArrowPatch(
                posA=(spec.start_col, spec.start_row),
                posB=(spec.end_col, spec.end_row),
                arrowstyle="-|>",
                mutation_scale=spec.mutation_scale,
                linewidth=spec.linewidth_pt,
                color=spec.color,
                alpha=1.0 if spec.state == "selected" else 0.92,
                capstyle="round",
                joinstyle="round",
                zorder=11 if spec.state == "selected" else 9,
            )
        )
    return specs


def _belief_background(assets: OnlineWorkflowAssets, canvas: WorldCanvas) -> np.ndarray:
    return _project_belief_to_canvas(assets.belief_after_update, canvas)


def _draw_environment_transition(
    ax,
    *,
    assets: OnlineWorkflowAssets,
    canvas: WorldCanvas,
    style: OnlineWorkflowStyle,
) -> None:
    before_row, before_col = _agent_world_to_canvas(assets.belief_after_update, canvas)
    after_row, after_col = _agent_world_to_canvas(assets.environment_after_action, canvas)
    expected = tuple(int(v) for v in ACTIONS_8[int(assets.action_index)])
    actual = (
        int(assets.environment_after_action.agent_world[0] - assets.belief_after_update.agent_world[0]),
        int(assets.environment_after_action.agent_world[1] - assets.belief_after_update.agent_world[1]),
    )
    if actual != expected:
        raise ValueError(f"transition delta {actual} does not match ACTIONS_8[{assets.action_index}]={expected}")
    draw_topdown_robot(
        ax,
        row=before_row,
        col=before_col,
        heading_action=assets.action_index,
        style=style.paper,
        alpha=0.42,
        zorder=17,
    )
    draw_topdown_robot(
        ax,
        row=after_row,
        col=after_col,
        heading_action=assets.action_index,
        style=style.paper,
        alpha=1.0,
        zorder=18,
    )


def _render_local_observation(
    path: Path,
    *,
    assets: OnlineWorkflowAssets,
    style: OnlineWorkflowStyle,
    dpi: int,
    include_svg: bool,
) -> tuple[Path | None, int]:
    cmap, norm = occupancy_colormap(style.paper)
    snapshot = assets.local_observation
    fig, ax = _create_axis(tuple(snapshot.local_snap.shape), style)
    ax.imshow(snapshot.local_snap, cmap=cmap, norm=norm, origin="upper", interpolation="nearest")
    center_row = float(snapshot.local_snap.shape[0] // 2)
    center_col = float(snapshot.local_snap.shape[1] // 2)
    ax.add_patch(
        Circle(
            (center_col, center_row),
            radius=float(snapshot.local_snap.shape[0] // 2),
            fill=False,
            edgecolor=style.paper.radar_palette["nominal_boundary"],
            linewidth=1.0,
            linestyle=(0, (4, 3)),
            alpha=float(style.scan_boundary_alpha),
            zorder=7,
        )
    )
    draw_topdown_robot(
        ax,
        row=center_row,
        col=center_col,
        heading_action=assets.action_index,
        style=style.paper,
    )
    _format_clean_axis(ax, tuple(snapshot.local_snap.shape))
    return _save_online_figure(fig, path, dpi=dpi, include_svg=include_svg), 0


def _render_action_selection(
    path: Path,
    *,
    assets: OnlineWorkflowAssets,
    style: OnlineWorkflowStyle,
    dpi: int,
    include_svg: bool,
) -> tuple[Path | None, tuple[ActionArrowSpec, ...]]:
    cmap, norm = occupancy_colormap(style.paper)
    local_snap = assets.local_observation.local_snap
    fig, ax = _create_axis(tuple(local_snap.shape), style)
    ax.imshow(local_snap, cmap=cmap, norm=norm, origin="upper", interpolation="nearest")
    center_row = float(local_snap.shape[0] // 2)
    center_col = float(local_snap.shape[1] // 2)
    specs = _draw_action_selection(
        ax,
        center_row=center_row,
        center_col=center_col,
        valid_action_indices=assets.valid_action_indices,
        chosen_action_index=assets.action_index,
        style=style,
    )
    draw_topdown_robot(
        ax,
        row=center_row,
        col=center_col,
        heading_action=assets.action_index,
        style=style.paper,
        zorder=14,
    )
    _format_clean_axis(ax, tuple(local_snap.shape))
    return _save_online_figure(fig, path, dpi=dpi, include_svg=include_svg), specs


def _render_belief_update(
    path: Path,
    *,
    assets: OnlineWorkflowAssets,
    canvas: WorldCanvas,
    style: OnlineWorkflowStyle,
    dpi: int,
    include_svg: bool,
) -> tuple[Path | None, int]:
    """Render one B_t panel with only the robot at p_t."""

    cmap, norm = occupancy_colormap(style.paper)
    fig, ax = _create_axis(canvas.shape, style)
    background = _belief_background(assets, canvas)
    ax.imshow(background, cmap=cmap, norm=norm, origin="upper", interpolation="nearest")
    robot_row, robot_col = _agent_world_to_canvas(assets.belief_after_update, canvas)
    draw_topdown_robot(
        ax,
        row=robot_row,
        col=robot_col,
        heading_action=assets.action_index,
        style=style.paper,
    )
    _format_clean_axis(ax, canvas.shape)
    before = _project_belief_to_canvas(assets.belief_before_update, canvas)
    new_count = int(np.count_nonzero((before < 0) & (background >= 0)))
    return _save_online_figure(fig, path, dpi=dpi, include_svg=include_svg), new_count


def _render_environment_execution(
    path: Path,
    *,
    assets: OnlineWorkflowAssets,
    canvas: WorldCanvas,
    style: OnlineWorkflowStyle,
    dpi: int,
    include_svg: bool,
) -> Path | None:
    cmap, norm = occupancy_colormap(style.paper)
    fig, ax = _create_axis(canvas.shape, style)
    background = _belief_background(assets, canvas)
    ax.imshow(background, cmap=cmap, norm=norm, origin="upper", interpolation="nearest")
    _draw_environment_transition(ax, assets=assets, canvas=canvas, style=style)
    _format_clean_axis(ax, canvas.shape)
    return _save_online_figure(fig, path, dpi=dpi, include_svg=include_svg)


def _matrix_sha256(matrix: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(np.asarray(matrix, dtype=np.int8))
    return hashlib.sha256(contiguous.tobytes()).hexdigest()


def export_online_workflow_assets(
    output_dir: Path | str,
    *,
    config: ExportConfig | None = None,
    step: int | None = None,
    forced_method_action: str | None = None,
    trajectory_visual_step: int | None = None,
    include_svg: bool = False,
) -> dict[str, object]:
    """Export four mutually consistent assets from the shared deterministic blueprint."""

    output_dir_path = Path(output_dir)
    base = config if config is not None else ExportConfig(output_dir=output_dir_path)
    target_step = int(base.step_late if step is None else step)
    if trajectory_visual_step is not None and not (0 <= int(trajectory_visual_step) <= target_step):
        raise ValueError("trajectory_visual_step must be between 0 and target_step")
    blueprint = build_figure_demo_blueprint(
        seed=int(base.seed),
        preferred_step=target_step,
        scan_radius=int(base.scan_radius),
        rows=int(base.rows),
        cols=int(base.cols),
        obstacle_ratio=float(base.obstacle_ratio),
        obs_size=int(base.obs_size),
        visual_ray_count=32,
        forced_method_action=forced_method_action,
    )
    assets = _assets_from_blueprint(blueprint)
    style = OnlineWorkflowStyle()
    canvas = blueprint.belief_canvas
    if not np.array_equal(_belief_background(assets, canvas), blueprint.belief_display):
        raise RuntimeError("Figure 1 B_t background drifted from the shared blueprint")

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
    svg_files: dict[str, Path] = {}
    local_svg, radar_ray_count = _render_local_observation(
        files["local_observation_with_robot_radar"],
        assets=assets,
        style=style,
        dpi=int(base.dpi),
        include_svg=include_svg,
    )
    if local_svg is not None:
        svg_files["local_observation_with_robot_radar"] = local_svg
    action_svg, action_specs = _render_action_selection(
        files["legal_action_selection"],
        assets=assets,
        style=style,
        dpi=int(base.dpi),
        include_svg=include_svg,
    )
    if action_svg is not None:
        svg_files["legal_action_selection"] = action_svg
    belief_svg, newly_observed_cell_count = _render_belief_update(
        files["belief_map_update_with_robot"],
        assets=assets,
        canvas=canvas,
        style=style,
        dpi=int(base.dpi),
        include_svg=include_svg,
    )
    if belief_svg is not None:
        svg_files["belief_map_update_with_robot"] = belief_svg
    environment_svg = _render_environment_execution(
        files["environment_execution"],
        assets=assets,
        canvas=canvas,
        style=style,
        dpi=int(base.dpi),
        include_svg=include_svg,
    )
    if environment_svg is not None:
        svg_files["environment_execution"] = environment_svg

    background_hash = _matrix_sha256(blueprint.belief_display)
    selected_spec = next(spec for spec in action_specs if spec.state == "selected")
    normal_width = float(style.paper.rendering["normal_action_linewidth_pt"])
    manifest = {
        **blueprint_manifest(blueprint),
        "chosen_action_index": int(assets.action_index),
        "chosen_action_key": str(assets.action_key),
        "chosen_action_delta_row_col": list(expected_delta),
        "agent_position_at_observation": [int(v) for v in assets.local_observation.agent_world],
        "agent_position_before_action": [int(v) for v in assets.belief_after_update.agent_world],
        "agent_position_after_action": [int(v) for v in assets.environment_after_action.agent_world],
        "belief_canvas_origin": [int(v) for v in canvas.origin_world],
        "belief_canvas_shape": [int(v) for v in canvas.shape],
        "belief_update_panel_count": 1,
        "belief_update_semantics": "B_t with one robot at p_t",
        "belief_update_robot_count": 1,
        "belief_update_has_trajectory": False,
        "belief_update_has_new_cell_highlight": False,
        "environment_robot_count": 2,
        "environment_motion_arrow_count": 0,
        "belief_update_background_sha256": background_hash,
        "environment_background_sha256": background_hash,
        "radar_ray_count": int(radar_ray_count),
        "newly_observed_cell_count": int(newly_observed_cell_count),
        "action_arrow_geometry": {
            "start_radius_cells": float(style.action_start_radius_cells),
            "euclidean_length_cells": float(style.action_length_cells),
            "per_direction_lengths_cells": [float(spec.length_cells) for spec in action_specs],
            "normal_linewidth_pt": normal_width,
            "selected_linewidth_pt": float(selected_spec.linewidth_pt),
            "selected_to_normal_linewidth_ratio": float(selected_spec.linewidth_pt / normal_width),
        },
        "action_colors": dict(style.paper.fig1_action_palette),
        "robot_style": {
            "contract_path": str(style.paper.contract_path),
            "contract_version": str(style.paper.version),
            "palette": dict(style.paper.robot_palette),
            "geometry_cell_relative": dict(style.paper.robot_geometry_cell_relative),
            "envelope_diameter_cells": float(robot_envelope_diameter_cells(style.paper)),
        },
        "method_semantics": {
            "local_and_action_assets": "same seed-1 local_snap and legal action selection at p_t",
            "belief_update_asset": "single B_t state with one robot at p_t",
            "environment_asset": "p_t and p_(t+1) over the unchanged B_t background; no motion arrow",
            "truth_map_visibility": "not rendered; used only by the existing sensor and transition semantics",
        },
        "figure_planning": {
            "claim": "One online cycle turns the current observation and cumulative belief into a legal action and feedback without advancing the displayed belief beyond B_t.",
            "anchor_module": "legal action mask and selected action",
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
