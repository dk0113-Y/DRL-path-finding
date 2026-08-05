from __future__ import annotations

"""Export deterministic, data-driven assets for Figure 3.

The Figure 3 input assets share one deterministic cumulative-belief scene.
Occupancy, the complete executed trajectory, robot pose, the four-channel
advantage canvas, and the hierarchical semantic tree are computed through the
active project implementation rather than reconstructed from rendered figures.
"""

import argparse
import hashlib
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.patches import Circle, FancyBboxPatch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from env.advantage_state_builder import (  # noqa: E402
    ADVANTAGE_CANVAS_SCHEMA_FINAL_4CH_NO_FRONTIER_RASTER,
    FINAL_4CH_ADVANTAGE_CANVAS_CHANNELS,
    AdvantageStateBuilder,
)
from env.agent_version import LocalObservationModel  # noqa: E402
from env.block_random_g import RandomMapGenerator  # noqa: E402
from env.core_cummap import CumulativeBeliefMap  # noqa: E402
from env.core_radar import RadarSensor  # noqa: E402
from env.grid_topology import ACTIONS_8, GridTopology  # noqa: E402
from env.shared_semantic_layer import SharedSemanticLayer, SharedSemanticSnapshot  # noqa: E402
from env.value_state_builder import ValueStateBuilder  # noqa: E402
from tools.export_architecture_pictures import (  # noqa: E402
    ACTION_TO_KEY,
    ExportConfig,
    FIXED_ACTION_PREFERENCES,
    KEY_TO_ACTION,
    _select_fallback_action,
)
from tools.export_figure_demo_blueprint import (  # noqa: E402
    FigureDemoBlueprint,
    build_figure_demo_blueprint,
)
from tools.export_online_workflow_assets import (  # noqa: E402
    OnlineWorkflowStyle,
    _figure_size,
)
from tools.paper_figure_style import (  # noqa: E402
    draw_topdown_robot,
    load_paper_figure_style,
    occupancy_colormap,
)


DEFAULT_OUTPUT_DIR = (
    REPO_ROOT.parent
    / "paper_work"
    / "figures"
    / "method_schematics"
    / "fig3_assets"
)
DEFAULT_PAPER_REPO = REPO_ROOT.parent / "paper_work"
DEFAULT_SOURCE_MANIFEST = (
    REPO_ROOT.parent
    / "figure_assets"
    / "fig1_seed1"
    / "online_workflow_assets_manifest.json"
)
MANIFEST_FILENAME = "fig3_overview_assets_manifest.json"
MAX_DISPLAY_BLOCKS = 4
MAX_DISPLAY_ENTRIES_PER_BLOCK = 4

WARM = "#C96144"
WARM_LIGHT = "#F7E8E1"
BLUE = "#5185C0"
BLUE_DARK = "#315F91"
BLUE_LIGHT = "#E8F0F8"
GREEN = "#55966B"
INK = "#233746"
GRID = "#C7D0D8"


@dataclass(frozen=True, slots=True)
class Fig3OverviewScene:
    config: ExportConfig
    requested_step: int
    resolved_step: int
    blueprint: FigureDemoBlueprint
    cum_map: CumulativeBeliefMap
    agent_world: tuple[int, int]
    trajectory_world: tuple[tuple[int, int], ...]
    recent_trajectory_world: tuple[tuple[int, int], ...]
    semantic_snapshot: SharedSemanticSnapshot
    advantage_canvas: np.ndarray
    advantage_meta: dict[str, object]
    value_meta: dict[str, float]


def _configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "svg.fonttype": "none",
            "svg.hashsalt": "msd-hsr-fig3-overview-v1",
            "font.size": 9,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 1.2,
            "legend.frameon": False,
            "savefig.facecolor": "white",
            "savefig.edgecolor": "white",
        }
    )


_configure_matplotlib()


def _git_head(repo: Path) -> str | None:
    """Return a repository commit without making optional repositories mandatory."""

    repo_path = Path(repo)
    if not repo_path.is_dir():
        return None
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_path,
            check=False,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_int8(array: np.ndarray) -> str:
    return hashlib.sha256(
        np.ascontiguousarray(np.asarray(array, dtype=np.int8)).tobytes()
    ).hexdigest()


def _replay_canonical_snapshot(
    config: ExportConfig,
    *,
    resolved_step: int,
) -> tuple[
    CumulativeBeliefMap,
    tuple[int, int],
    tuple[tuple[int, int], ...],
]:
    """Replay the canonical fixed action preferences through project classes."""

    generator = RandomMapGenerator(
        rows=int(config.rows),
        cols=int(config.cols),
        obs_size=int(config.obs_size),
        obstacle_ratio=float(config.obstacle_ratio),
    )
    true_grid, start_state = generator.generate_map(seed=int(config.seed))
    free_mask = GridTopology.free_mask(true_grid)
    sensor = RadarSensor(scan_radius=int(config.scan_radius))
    observation_model = LocalObservationModel(true_grid, start_state, sensor=sensor)
    agent_state = (int(start_state[0]), int(start_state[1]))
    local_snap = np.asarray(observation_model.local_snap, dtype=np.int8).copy()
    cum_map = CumulativeBeliefMap(true_grid, agent_state, local_snap)
    trajectory: list[tuple[int, int]] = [agent_state]
    visit_counts: dict[tuple[int, int], int] = {agent_state: 1}

    for step_index in range(1, int(resolved_step) + 1):
        planned_key = FIXED_ACTION_PREFERENCES[
            (step_index - 1) % len(FIXED_ACTION_PREFERENCES)
        ]
        desired_action = int(KEY_TO_ACTION[planned_key])
        valid_actions = GridTopology.valid_action_indices_fast(
            free_mask, agent_state
        )
        if not valid_actions:
            raise RuntimeError(f"agent has no legal move at step {step_index}")
        chosen_action = (
            desired_action
            if desired_action in valid_actions
            else _select_fallback_action(
                valid_actions,
                agent_state=agent_state,
                visit_counts=visit_counts,
            )
        )
        delta_r, delta_c = ACTIONS_8[chosen_action]
        agent_state = (
            int(agent_state[0] + delta_r),
            int(agent_state[1] + delta_c),
        )
        trajectory.append(agent_state)
        visit_counts[agent_state] = int(visit_counts.get(agent_state, 0) + 1)
        local_snap = np.asarray(
            observation_model.observe_fast(agent_state), dtype=np.int8
        ).copy()
        cum_map.update(agent_state, local_snap)

    return cum_map, agent_state, tuple(trajectory)


def build_fig3_overview_scene(
    *,
    seed: int = 1,
    step: int = 8,
    rows: int = 40,
    cols: int = 60,
    obstacle_ratio: float = 0.20,
    obs_size: int = 6,
    scan_radius: int = 10,
    dpi: int = 300,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
) -> Fig3OverviewScene:
    config = ExportConfig(
        rows=int(rows),
        cols=int(cols),
        obstacle_ratio=float(obstacle_ratio),
        obs_size=int(obs_size),
        scan_radius=int(scan_radius),
        seed=int(seed),
        step_mid=4,
        step_late=int(step),
        dpi=int(dpi),
        output_dir=Path(output_dir),
    )
    blueprint = build_figure_demo_blueprint(
        seed=int(seed),
        preferred_step=int(step),
        scan_radius=int(scan_radius),
        rows=int(rows),
        cols=int(cols),
        obstacle_ratio=float(obstacle_ratio),
        obs_size=int(obs_size),
        visual_ray_count=32,
    )
    resolved_step = int(blueprint.step)
    cum_map, agent_world, trajectory = _replay_canonical_snapshot(
        config, resolved_step=resolved_step
    )

    if agent_world != tuple(blueprint.belief_after_update.agent_world):
        raise RuntimeError("replayed robot position drifted from the shared blueprint")
    if tuple(cum_map.origin_world_rc) != tuple(
        blueprint.belief_after_update.belief_origin_world
    ):
        raise RuntimeError("replayed belief origin drifted from the shared blueprint")
    if not np.array_equal(cum_map.map, blueprint.belief_after_update.belief_map):
        raise RuntimeError("replayed cumulative belief drifted from the shared blueprint")
    if not np.array_equal(
        np.asarray(trajectory, dtype=np.int32),
        blueprint.belief_after_update.trajectory_world,
    ):
        raise RuntimeError("replayed interaction history drifted from the shared blueprint")

    semantic_snapshot = SharedSemanticLayer().analyze(cum_map, agent_world)
    advantage_builder = AdvantageStateBuilder()
    history_steps = int(advantage_builder.config.trajectory_history_steps)
    recent_trajectory = tuple(trajectory[-(history_steps + 1) :])
    if len(recent_trajectory) < 2:
        raise RuntimeError(
            "the requested step does not provide at least two real trajectory points"
        )
    advantage_canvas, advantage_meta = advantage_builder.build(
        cum_map,
        agent_world,
        semantic_snapshot,
        recent_trajectory_positions=recent_trajectory,
    )
    if tuple(advantage_canvas.shape[:1]) != (
        len(FINAL_4CH_ADVANTAGE_CANVAS_CHANNELS),
    ):
        raise RuntimeError(
            f"expected four advantage channels, got {advantage_canvas.shape}"
        )
    if bool(advantage_meta.get("frontier_raster_used", True)):
        raise RuntimeError("the active advantage canvas unexpectedly uses a frontier raster")

    _, _, block_mask, entry_mask, value_meta = ValueStateBuilder().build(
        semantic_snapshot
    )
    total_blocks = len(semantic_snapshot.accessible_blocks)
    total_entries = sum(
        int(block.frontier_cluster_count)
        for block in semantic_snapshot.accessible_blocks
    )
    if int(np.count_nonzero(block_mask)) != total_blocks:
        raise RuntimeError("ValueStateBuilder block packing drifted from semantic snapshot")
    if int(np.count_nonzero(entry_mask)) != total_entries:
        raise RuntimeError("ValueStateBuilder entry packing drifted from semantic snapshot")

    return Fig3OverviewScene(
        config=config,
        requested_step=int(step),
        resolved_step=resolved_step,
        blueprint=blueprint,
        cum_map=cum_map,
        agent_world=agent_world,
        trajectory_world=trajectory,
        recent_trajectory_world=recent_trajectory,
        semantic_snapshot=semantic_snapshot,
        advantage_canvas=np.asarray(advantage_canvas, dtype=np.float32).copy(),
        advantage_meta=dict(advantage_meta),
        value_meta=dict(value_meta),
    )


def _canonical_belief_background(scene: Fig3OverviewScene) -> np.ndarray:
    """Return the exact B_t canvas used by the shared Figure 2 blueprint."""

    background = np.asarray(scene.blueprint.belief_display, dtype=np.int8)
    canvas_shape = tuple(int(v) for v in scene.blueprint.belief_canvas.shape)
    if tuple(background.shape) != canvas_shape:
        raise RuntimeError(
            "shared belief background shape drifted from the blueprint canvas"
        )
    if not np.array_equal(
        scene.cum_map.map,
        scene.blueprint.belief_after_update.belief_map,
    ):
        raise RuntimeError("Figure 3 cumulative belief drifted from Figure 2 B_t")
    return background.copy()


def _trajectory_world_to_shared_canvas(
    scene: Fig3OverviewScene,
) -> tuple[np.ndarray, np.ndarray]:
    """Project the complete executed trajectory onto the shared world canvas."""

    points = np.asarray(scene.trajectory_world, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 2 or points.shape[0] < 2:
        raise RuntimeError("the shared scene needs at least two trajectory points")
    if tuple(int(v) for v in points[-1]) != tuple(scene.agent_world):
        raise RuntimeError("executed trajectory does not terminate at the robot")
    origin_row, origin_col = (
        int(v) for v in scene.blueprint.belief_canvas.origin_world
    )
    rows = points[:, 0] - float(origin_row)
    cols = points[:, 1] - float(origin_col)
    canvas_rows, canvas_cols = (
        int(v) for v in scene.blueprint.belief_canvas.shape
    )
    inside = (
        (rows >= -0.5)
        & (rows <= float(canvas_rows) - 0.5)
        & (cols >= -0.5)
        & (cols <= float(canvas_cols) - 0.5)
    )
    if not bool(np.all(inside)):
        raise RuntimeError("executed trajectory falls outside the shared canvas")
    return rows, cols


def _clean_map_axis(ax, shape: tuple[int, int]) -> None:
    ax.set_xlim(-0.5, float(shape[1]) - 0.5)
    ax.set_ylim(float(shape[0]) - 0.5, -0.5)
    ax.set_aspect("equal")
    ax.set_axis_off()


def _create_shared_map_axis(
    scene: Fig3OverviewScene,
    *,
    transparent: bool,
) -> tuple[plt.Figure, object]:
    """Create the Figure 2-aligned canvas used by all three shared assets."""

    canvas_shape = tuple(int(v) for v in scene.blueprint.belief_canvas.shape)
    workflow_style = OnlineWorkflowStyle()
    facecolor = "none" if transparent else "white"
    fig = plt.figure(
        figsize=_figure_size(canvas_shape, workflow_style),
        frameon=False,
    )
    fig.patch.set_facecolor(facecolor)
    if transparent:
        fig.patch.set_alpha(0.0)
    ax = fig.add_axes((0.0, 0.0, 1.0, 1.0))
    ax.set_facecolor(facecolor)
    _clean_map_axis(ax, canvas_shape)
    return fig, ax


def _draw_shared_trajectory(
    ax,
    *,
    rows: np.ndarray,
    cols: np.ndarray,
    zorder: int,
) -> tuple[object, ...]:
    """Draw the one trajectory layer reused by the composite and overlay PNGs."""

    (line,) = ax.plot(
        np.asarray(cols, dtype=np.float32),
        np.asarray(rows, dtype=np.float32),
        color=BLUE_DARK,
        linewidth=2.0,
        alpha=0.94,
        marker="o",
        markersize=3.2,
        markerfacecolor=BLUE,
        markeredgecolor="white",
        markeredgewidth=0.45,
        solid_capstyle="round",
        solid_joinstyle="round",
        zorder=int(zorder),
    )
    line.set_gid("fig3_executed_trajectory")
    return (line,)


def _save_pair(
    fig: plt.Figure,
    png_path: Path,
    *,
    dpi: int,
    include_svg: bool,
    tight: bool = True,
    transparent: bool = False,
) -> Path | None:
    png_path.parent.mkdir(parents=True, exist_ok=True)
    facecolor = "none" if transparent else "white"
    fig.savefig(
        png_path,
        dpi=int(dpi),
        bbox_inches="tight" if tight else None,
        pad_inches=0.0,
        facecolor=facecolor,
        transparent=bool(transparent),
        metadata={"Software": "DRL-path-finding fig3 deterministic exporter"},
    )
    svg_path = png_path.with_suffix(".svg") if include_svg else None
    if svg_path is not None:
        fig.savefig(
            svg_path,
            bbox_inches="tight" if tight else None,
            pad_inches=0.0,
            facecolor=facecolor,
            transparent=bool(transparent),
            metadata={
                "Date": None,
                "Creator": "DRL-path-finding fig3 deterministic exporter",
            },
        )
    plt.close(fig)
    return svg_path


def _render_dynamic_belief(
    scene: Fig3OverviewScene,
    path: Path,
    *,
    include_svg: bool,
) -> Path | None:
    background = _canonical_belief_background(scene)
    fig, ax = _create_shared_map_axis(scene, transparent=False)
    cmap, norm = occupancy_colormap(load_paper_figure_style())
    image = ax.imshow(
        background,
        cmap=cmap,
        norm=norm,
        origin="upper",
        interpolation="nearest",
        zorder=1,
    )
    image.set_gid("fig3_dynamic_cumulative_belief_raster")
    _clean_map_axis(ax, tuple(background.shape))
    return _save_pair(
        fig,
        path,
        dpi=int(scene.config.dpi),
        include_svg=include_svg,
        tight=False,
    )


def _render_belief_with_robot_and_history(
    scene: Fig3OverviewScene,
    path: Path,
    *,
    include_svg: bool,
) -> Path | None:
    background = _canonical_belief_background(scene)
    trajectory_rows, trajectory_cols = _trajectory_world_to_shared_canvas(scene)
    fig, ax = _create_shared_map_axis(scene, transparent=False)
    style = load_paper_figure_style()
    cmap, norm = occupancy_colormap(style)
    image = ax.imshow(
        background,
        cmap=cmap,
        norm=norm,
        origin="upper",
        interpolation="nearest",
        zorder=1,
    )
    image.set_gid("fig3_shared_cumulative_belief_raster")
    _draw_shared_trajectory(
        ax,
        rows=trajectory_rows,
        cols=trajectory_cols,
        zorder=6,
    )
    origin_row, origin_col = (
        int(v) for v in scene.blueprint.belief_canvas.origin_world
    )
    robot_parts = draw_topdown_robot(
        ax,
        row=float(scene.agent_world[0] - origin_row),
        col=float(scene.agent_world[1] - origin_col),
        heading_action=int(scene.blueprint.selected_action),
        style=style,
        zorder=12,
    )
    for part_index, part in enumerate(robot_parts):
        part.set_gid(f"fig3_topdown_robot_part_{part_index}")
    _clean_map_axis(ax, tuple(background.shape))
    return _save_pair(
        fig,
        path,
        dpi=int(scene.config.dpi),
        include_svg=include_svg,
        tight=False,
    )


def _render_interaction_history(
    scene: Fig3OverviewScene,
    path: Path,
    *,
    include_svg: bool,
) -> Path | None:
    rows, cols = _trajectory_world_to_shared_canvas(scene)
    fig, ax = _create_shared_map_axis(scene, transparent=True)
    _draw_shared_trajectory(
        ax,
        rows=rows,
        cols=cols,
        zorder=6,
    )
    _clean_map_axis(
        ax,
        tuple(int(v) for v in scene.blueprint.belief_canvas.shape),
    )
    return _save_pair(
        fig,
        path,
        dpi=int(scene.config.dpi),
        include_svg=include_svg,
        tight=False,
        transparent=True,
    )


def _render_local_state(
    scene: Fig3OverviewScene,
    path: Path,
    *,
    include_svg: bool,
) -> Path | None:
    fig, axes = plt.subplots(2, 2, figsize=(2.15, 2.15), frameon=False)
    fig.patch.set_facecolor("white")
    channel_cmaps = (
        ListedColormap(["#FFFFFF", "#99C290"]),
        ListedColormap(["#FFFFFF", "#303942"]),
        LinearSegmentedColormap.from_list(
            "fig3_visit", ["#FFFFFF", "#F2CB9F", "#C96144"]
        ),
        LinearSegmentedColormap.from_list(
            "fig3_trajectory", ["#FFFFFF", "#FFD3E0", "#C96144"]
        ),
    )
    for index, (ax, channel, cmap) in enumerate(
        zip(np.ravel(axes), scene.advantage_canvas, channel_cmaps)
    ):
        image = ax.imshow(
            channel,
            cmap=cmap,
            vmin=0.0,
            vmax=1.0,
            origin="upper",
            interpolation="nearest",
        )
        image.set_gid(
            f"fig3_advantage_channel_{index}_{FINAL_4CH_ADVANTAGE_CANVAS_CHANNELS[index]}"
        )
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color(WARM if index >= 2 else "#8A989F")
            spine.set_linewidth(0.8)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01, wspace=0.06, hspace=0.06)
    return _save_pair(
        fig, path, dpi=int(scene.config.dpi), include_svg=include_svg
    )


def _render_global_state(
    scene: Fig3OverviewScene,
    path: Path,
    *,
    include_svg: bool,
) -> Path | None:
    blocks = tuple(scene.semantic_snapshot.accessible_blocks)
    shown_blocks = blocks[:MAX_DISPLAY_BLOCKS]
    fig = plt.figure(figsize=(2.55, 1.7), frameon=False)
    ax = fig.add_axes((0.0, 0.0, 1.0, 1.0))
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")
    if not shown_blocks:
        ax.text(0.5, 0.5, "No frontier-associated\nunknown block", ha="center", va="center", color=BLUE_DARK)
        return _save_pair(
            fig, path, dpi=int(scene.config.dpi), include_svg=include_svg
        )

    total_area = float(max(1, scene.semantic_snapshot.total_accessible_unknown_area))
    centers = (
        np.asarray([0.50], dtype=np.float64)
        if len(shown_blocks) == 1
        else np.linspace(0.22, 0.78, len(shown_blocks))
    )
    for block_slot, (block, center_x) in enumerate(zip(shown_blocks, centers), start=1):
        area_ratio = float(block.block_area) / total_area
        parent_width = 0.22 + (0.18 * min(1.0, area_ratio))
        parent = FancyBboxPatch(
            (float(center_x) - parent_width / 2.0, 0.69),
            parent_width,
            0.20,
            boxstyle="round,pad=0.015,rounding_size=0.035",
            facecolor="#C8D9EB",
            edgecolor=BLUE_DARK,
            linewidth=1.2,
            zorder=4,
        )
        parent.set_gid(f"fig3_semantic_parent_{int(block.block_index)}")
        ax.add_patch(parent)
        clusters = tuple(block.frontier_clusters)
        shown_clusters = clusters[:MAX_DISPLAY_ENTRIES_PER_BLOCK]
        child_x = np.linspace(
            float(center_x) - min(0.20, 0.055 * max(1, len(shown_clusters) - 1)),
            float(center_x) + min(0.20, 0.055 * max(1, len(shown_clusters) - 1)),
            len(shown_clusters),
        )
        for entry_slot, (cluster, entry_x) in enumerate(
            zip(shown_clusters, child_x), start=1
        ):
            line = ax.plot(
                [float(center_x), float(entry_x)],
                [0.69, 0.37],
                color=BLUE,
                linewidth=1.0,
                zorder=2,
            )[0]
            line.set_gid(
                f"fig3_semantic_edge_{int(block.block_index)}_{int(cluster.frontier_index)}"
            )
            child = FancyBboxPatch(
                (float(entry_x) - 0.035, 0.27),
                0.07,
                0.10,
                boxstyle="round,pad=0.006,rounding_size=0.015",
                facecolor=BLUE_LIGHT,
                edgecolor=BLUE,
                linewidth=1.0,
                zorder=4,
            )
            child.set_gid(
                f"fig3_semantic_child_{int(block.block_index)}_{int(cluster.frontier_index)}"
            )
            ax.add_patch(child)
        if len(clusters) > len(shown_clusters):
            ellipsis_x = min(0.96, float(center_x) + 0.27)
            for dot_offset in (-0.025, 0.0, 0.025):
                dot = Circle(
                    (ellipsis_x + dot_offset, 0.31),
                    radius=0.007,
                    facecolor=BLUE_DARK,
                    edgecolor="none",
                    zorder=5,
                )
                dot.set_gid(
                    f"fig3_semantic_entry_ellipsis_{int(block.block_index)}"
                )
                ax.add_patch(dot)
    if len(blocks) > len(shown_blocks):
        for dot_offset in (-0.025, 0.0, 0.025):
            dot = Circle(
                (0.96 + dot_offset, 0.79),
                radius=0.007,
                facecolor=BLUE_DARK,
                edgecolor="none",
                zorder=5,
            )
            dot.set_gid("fig3_semantic_block_ellipsis")
            ax.add_patch(dot)
    return _save_pair(
        fig, path, dpi=int(scene.config.dpi), include_svg=include_svg
    )


def _validate_source_manifest(
    scene: Fig3OverviewScene,
    source_manifest_path: Path | None,
) -> dict[str, object]:
    if source_manifest_path is None or not source_manifest_path.exists():
        return {
            "manifest_path": None
            if source_manifest_path is None
            else str(source_manifest_path.resolve()),
            "manifest_verified": False,
            "validation_status": "skipped",
            "validation_skipped": True,
            "skip_reason": (
                "source_manifest_not_provided"
                if source_manifest_path is None
                else "source_manifest_not_found"
            ),
            "seed_step_verified": False,
            "same_belief_matrix_verified": False,
            "source_png_exists": False,
            "inferred_fields": [
                "seed=1 and step=8 are candidates from current defaults only"
            ],
        }

    payload = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    source_files = dict(payload.get("files", {}))
    source_png = Path(str(source_files.get("belief_map_update_with_robot", "")))
    source_belief = payload.get("belief_t", {})
    source_matrix = np.asarray(source_belief.get("matrix", []), dtype=np.int8)
    source_origin = tuple(int(v) for v in source_belief.get("origin_world_rc", []))
    source_shape = tuple(int(v) for v in source_belief.get("shape", []))
    current_matrix = np.asarray(scene.blueprint.belief_display, dtype=np.int8)
    matrix_match = bool(
        source_matrix.shape == current_matrix.shape
        and np.array_equal(source_matrix, current_matrix)
        and source_origin == tuple(scene.blueprint.belief_canvas.origin_world)
        and source_shape == tuple(scene.blueprint.belief_canvas.shape)
        and str(payload.get("belief_update_background_sha256", ""))
        == _sha256_int8(current_matrix)
    )
    seed_step_match = bool(
        int(payload.get("seed", -1)) == int(scene.config.seed)
        and int(payload.get("step", -1)) == int(scene.resolved_step)
    )
    manifest_verified = bool(
        seed_step_match
        and matrix_match
        and source_png.is_file()
        and source_png.name == "belief_map_update_with_robot.png"
    )
    return {
        "manifest_path": str(source_manifest_path.resolve()),
        "manifest_sha256": _sha256_file(source_manifest_path),
        "manifest_verified": manifest_verified,
        "validation_status": "verified" if manifest_verified else "failed",
        "validation_skipped": False,
        "skip_reason": None,
        "seed_step_verified": seed_step_match,
        "same_belief_matrix_verified": matrix_match,
        "source_png_path": str(source_png.resolve()) if source_png else "",
        "source_png_exists": source_png.is_file(),
        "source_png_sha256": _sha256_file(source_png)
        if source_png.is_file()
        else None,
        "recorded_seed": int(payload.get("seed", -1)),
        "recorded_step": int(payload.get("step", -1)),
        "recorded_scan_radius": int(payload.get("scan_radius", -1)),
        "recorded_belief_canvas_origin": list(source_origin),
        "recorded_belief_canvas_shape": list(source_shape),
        "recorded_agent_position": [
            int(v) for v in payload.get("agent_position_before_action", [])
        ],
        "inferred_fields": [
            "rows=40",
            "cols=60",
            "obstacle_ratio=0.20",
            "obs_size=6",
        ],
        "inference_boundary": (
            "The source manifest records seed, step, scan radius, belief canvas, "
            "and agent position, but not rows, cols, obstacle_ratio, or obs_size; "
            "those four fields are taken from the current deterministic exporter "
            "configuration and are not claimed as manifest-recorded facts."
        ),
    }


def _asset_record(
    name: str,
    png_path: Path,
    svg_path: Path | None,
    *,
    render_layers: Sequence[str],
    background_mode: str,
    expected_transparency: bool | None,
    belief_origin_world: Sequence[int] | None = None,
    belief_canvas_shape: Sequence[int] | None = None,
    trajectory_positions_world: Sequence[Sequence[int]] | None = None,
) -> dict[str, object]:
    with Image.open(png_path) as image:
        png_mode = str(image.mode)
        width_px, height_px = (int(v) for v in image.size)
        rgba = np.asarray(image.convert("RGBA"), dtype=np.uint8)
    alpha = rgba[..., 3]
    transparent_pixel_count = int(np.count_nonzero(alpha == 0))
    nontransparent_pixel_count = int(np.count_nonzero(alpha > 0))
    has_transparency = bool(np.any(alpha < 255))
    if (
        expected_transparency is not None
        and has_transparency != bool(expected_transparency)
    ):
        raise RuntimeError(
            f"{name} transparency={has_transparency} does not match "
            f"expected {bool(expected_transparency)}"
        )
    record: dict[str, object] = {
        "name": name,
        "path": str(png_path.resolve()),
        "png_path": str(png_path.resolve()),
        "svg_path": None if svg_path is None else str(svg_path.resolve()),
        "sha256": _sha256_file(png_path),
        "png_sha256": _sha256_file(png_path),
        "svg_sha256": None if svg_path is None else _sha256_file(svg_path),
        "width_px": width_px,
        "height_px": height_px,
        "png_mode": png_mode,
        "background_mode": str(background_mode),
        "transparent_background": has_transparency,
        "transparent_pixel_count": transparent_pixel_count,
        "nontransparent_pixel_count": nontransparent_pixel_count,
        "render_layers": list(render_layers),
    }
    if belief_origin_world is not None:
        record["belief_origin_world"] = [
            int(v) for v in belief_origin_world
        ]
    if belief_canvas_shape is not None:
        record["belief_canvas_shape"] = [
            int(v) for v in belief_canvas_shape
        ]
        record["coordinate_projection"] = (
            "canvas_rc = world_rc - belief_origin_world; "
            "imshow origin='upper', nearest-neighbor interpolation"
        )
    if trajectory_positions_world is not None:
        record["trajectory_positions_world"] = [
            [int(v) for v in position]
            for position in trajectory_positions_world
        ]
    return record


def export_fig3_overview_assets(
    output_dir: Path | str,
    *,
    seed: int = 1,
    step: int = 8,
    rows: int = 40,
    cols: int = 60,
    obstacle_ratio: float = 0.20,
    obs_size: int = 6,
    scan_radius: int = 10,
    dpi: int = 300,
    include_svg: bool = False,
    paper_repo: Path | str = DEFAULT_PAPER_REPO,
    source_manifest: Path | str | None = DEFAULT_SOURCE_MANIFEST,
) -> dict[str, object]:
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    for legacy_name in ("robot_position.png", "robot_position.svg"):
        legacy_path = output_dir_path / legacy_name
        if legacy_path.is_file():
            legacy_path.unlink()
    scene = build_fig3_overview_scene(
        seed=int(seed),
        step=int(step),
        rows=int(rows),
        cols=int(cols),
        obstacle_ratio=float(obstacle_ratio),
        obs_size=int(obs_size),
        scan_radius=int(scan_radius),
        dpi=int(dpi),
        output_dir=output_dir_path,
    )

    png_paths = {
        "dynamic_cumulative_belief_map": output_dir_path
        / "dynamic_cumulative_belief_map.png",
        "belief_map_with_robot_and_history": output_dir_path
        / "belief_map_with_robot_and_history.png",
        "interaction_history": output_dir_path / "interaction_history.png",
        "local_occupancy_behavior_state": output_dir_path
        / "local_occupancy_behavior_state.png",
        "global_hierarchical_semantic_state": output_dir_path
        / "global_hierarchical_semantic_state.png",
    }
    svg_paths = {
        "dynamic_cumulative_belief_map": _render_dynamic_belief(
            scene,
            png_paths["dynamic_cumulative_belief_map"],
            include_svg=include_svg,
        ),
        "belief_map_with_robot_and_history": (
            _render_belief_with_robot_and_history(
                scene,
                png_paths["belief_map_with_robot_and_history"],
                include_svg=include_svg,
            )
        ),
        "interaction_history": _render_interaction_history(
            scene, png_paths["interaction_history"], include_svg=include_svg
        ),
        "local_occupancy_behavior_state": _render_local_state(
            scene,
            png_paths["local_occupancy_behavior_state"],
            include_svg=include_svg,
        ),
        "global_hierarchical_semantic_state": _render_global_state(
            scene,
            png_paths["global_hierarchical_semantic_state"],
            include_svg=include_svg,
        ),
    }

    blocks = tuple(scene.semantic_snapshot.accessible_blocks)
    displayed_blocks = blocks[:MAX_DISPLAY_BLOCKS]
    displayed_entry_count = int(
        sum(
            min(
                len(block.frontier_clusters),
                MAX_DISPLAY_ENTRIES_PER_BLOCK,
            )
            for block in displayed_blocks
        )
    )
    total_entry_count = int(
        sum(int(block.frontier_cluster_count) for block in blocks)
    )
    source_validation = _validate_source_manifest(
        scene,
        None if source_manifest is None else Path(source_manifest),
    )
    same_snapshot = bool(
        source_validation.get("same_belief_matrix_verified", False)
        and source_validation.get("seed_step_verified", False)
    )
    belief_origin_world = tuple(
        int(v) for v in scene.blueprint.belief_canvas.origin_world
    )
    belief_canvas_shape = tuple(
        int(v) for v in scene.blueprint.belief_canvas.shape
    )
    trajectory_positions_world = tuple(scene.trajectory_world)
    assets = {
        "dynamic_cumulative_belief_map": _asset_record(
            "dynamic_cumulative_belief_map",
            png_paths["dynamic_cumulative_belief_map"],
            svg_paths["dynamic_cumulative_belief_map"],
            render_layers=["cumulative_belief_occupancy"],
            background_mode="opaque_white",
            expected_transparency=False,
            belief_origin_world=belief_origin_world,
            belief_canvas_shape=belief_canvas_shape,
        ),
        "belief_map_with_robot_and_history": _asset_record(
            "belief_map_with_robot_and_history",
            png_paths["belief_map_with_robot_and_history"],
            svg_paths["belief_map_with_robot_and_history"],
            render_layers=[
                "cumulative_belief_occupancy",
                "executed_trajectory",
                "topdown_robot",
            ],
            background_mode="opaque_white",
            expected_transparency=False,
            belief_origin_world=belief_origin_world,
            belief_canvas_shape=belief_canvas_shape,
            trajectory_positions_world=trajectory_positions_world,
        ),
        "interaction_history": _asset_record(
            "interaction_history",
            png_paths["interaction_history"],
            svg_paths["interaction_history"],
            render_layers=["executed_trajectory"],
            background_mode="transparent",
            expected_transparency=True,
            belief_origin_world=belief_origin_world,
            belief_canvas_shape=belief_canvas_shape,
            trajectory_positions_world=trajectory_positions_world,
        ),
        "local_occupancy_behavior_state": _asset_record(
            "local_occupancy_behavior_state",
            png_paths["local_occupancy_behavior_state"],
            svg_paths["local_occupancy_behavior_state"],
            render_layers=list(FINAL_4CH_ADVANTAGE_CANVAS_CHANNELS),
            background_mode="existing_figure_background",
            expected_transparency=None,
        ),
        "global_hierarchical_semantic_state": _asset_record(
            "global_hierarchical_semantic_state",
            png_paths["global_hierarchical_semantic_state"],
            svg_paths["global_hierarchical_semantic_state"],
            render_layers=[
                "frontier_associated_unknown_block_parents",
                "frontier_cluster_children",
                "parent_child_edges",
            ],
            background_mode="existing_figure_background",
            expected_transparency=None,
        ),
    }
    paper_repo_path = Path(paper_repo)
    trajectory_positions_payload = [
        [int(v) for v in position]
        for position in scene.trajectory_world
    ]
    selected_action_index = int(scene.blueprint.selected_action)
    selected_action_key = str(ACTION_TO_KEY[selected_action_index])
    shared_source_scene = {
        "seed": int(scene.config.seed),
        "requested_step": int(scene.requested_step),
        "resolved_step": int(scene.resolved_step),
        "agent_world_position": [int(v) for v in scene.agent_world],
        "selected_action_index": selected_action_index,
        "selected_action_key": selected_action_key,
        "belief_origin_world": [int(v) for v in belief_origin_world],
        "belief_canvas_shape": [int(v) for v in belief_canvas_shape],
        "trajectory_length": int(len(scene.trajectory_world)),
        "trajectory_positions_world": trajectory_positions_payload,
    }
    manifest = {
        "schema_version": 2,
        "code_repo_commit": _git_head(REPO_ROOT),
        "paper_repo_commit_before_change": _git_head(paper_repo_path),
        "paper_repo_path": str(paper_repo_path.resolve()),
        "paper_repo_available": paper_repo_path.is_dir(),
        "dpi": int(scene.config.dpi),
        "include_svg": bool(include_svg),
        "seed": int(scene.config.seed),
        "requested_step": int(scene.requested_step),
        "resolved_step": int(scene.resolved_step),
        "rows": int(scene.config.rows),
        "cols": int(scene.config.cols),
        "obstacle_ratio": float(scene.config.obstacle_ratio),
        "obs_size": int(scene.config.obs_size),
        "scan_radius": int(scene.config.scan_radius),
        "agent_world_position": [int(v) for v in scene.agent_world],
        "selected_action_index": selected_action_index,
        "selected_action_key": selected_action_key,
        "belief_origin_world": [int(v) for v in belief_origin_world],
        "belief_shape": [int(v) for v in belief_canvas_shape],
        "belief_canvas_shape": [int(v) for v in belief_canvas_shape],
        "belief_matrix_sha256": _sha256_int8(
            _canonical_belief_background(scene)
        ),
        "trajectory_length": int(len(scene.trajectory_world)),
        "trajectory_positions_world": trajectory_positions_payload,
        "recent_trajectory_length": int(len(scene.recent_trajectory_world)),
        "recent_trajectory_positions_world": [
            [int(v) for v in position]
            for position in scene.recent_trajectory_world
        ],
        "shared_source_scene": shared_source_scene,
        "shared_source_assets": {
            "dynamic_cumulative_belief_map": (
                "Figure 2-aligned blueprint.belief_display on "
                "blueprint.belief_canvas; occupancy only"
            ),
            "belief_map_with_robot_and_history": (
                "the same B_t canvas plus scene.trajectory_world and "
                "draw_topdown_robot at the current agent pose"
            ),
            "interaction_history": (
                "the same scene.trajectory_world projected onto the same "
                "transparent world canvas"
            ),
        },
        "same_belief_background": True,
        "same_world_canvas": True,
        "same_coordinate_projection": True,
        "same_trajectory_geometry": True,
        "robot_position_asset_generated": False,
        "robot_position_source": (
            "manual crop from belief_map_with_robot_and_history"
        ),
        "advantage_canvas_schema": ADVANTAGE_CANVAS_SCHEMA_FINAL_4CH_NO_FRONTIER_RASTER,
        "advantage_channel_names": list(
            FINAL_4CH_ADVANTAGE_CANVAS_CHANNELS
        ),
        "advantage_tensor_shape": [
            int(v) for v in scene.advantage_canvas.shape
        ],
        "frontier_raster_used": False,
        "total_unknown_block_count": int(len(blocks)),
        "displayed_unknown_block_count": int(len(displayed_blocks)),
        "total_frontier_entrance_count": total_entry_count,
        "displayed_frontier_entrance_count": displayed_entry_count,
        "displayed_block_indices": [
            int(block.block_index) for block in displayed_blocks
        ],
        "displayed_frontier_indices_by_block": {
            str(int(block.block_index)): [
                int(cluster.frontier_index)
                for cluster in block.frontier_clusters[
                    :MAX_DISPLAY_ENTRIES_PER_BLOCK
                ]
            ]
            for block in displayed_blocks
        },
        "same_snapshot_as_source_belief_asset": same_snapshot,
        "figure2_or_belief_map_source": source_validation,
        "truth_map_rendered": False,
        "truth_map_usage_boundary": (
            "Truth is used only by the existing map generator, sensor, and legal "
            "motion semantics during deterministic rollout; no truth-map array or "
            "truth-derived reachability region is rendered."
        ),
        "state_generation_chain": {
            "snapshot": (
                "build_figure_demo_blueprint + deterministic replay through "
                "RandomMapGenerator, LocalObservationModel, CumulativeBeliefMap"
            ),
            "local_state": "AdvantageStateBuilder.build",
            "global_state": "SharedSemanticLayer.analyze",
            "value_packing_validation": "ValueStateBuilder.build",
        },
        "figure_planning": {
            "claim": (
                "The dynamic belief, current robot pose, and interaction "
                "history are synchronized views of one deterministic B_t scene."
            ),
            "anchor_asset": "belief_map_with_robot_and_history",
            "asset_roles": {
                "dynamic_cumulative_belief_map": (
                    "methodological bridge: defines the shared B_t background"
                ),
                "belief_map_with_robot_and_history": (
                    "claim-supporting anchor: shows B_t, executed history, and p_t"
                ),
                "interaction_history": (
                    "claim-supporting overlay: isolates the same executed path"
                ),
                "local_occupancy_behavior_state": (
                    "methodological bridge: unchanged four-channel local state"
                ),
                "global_hierarchical_semantic_state": (
                    "methodological bridge: unchanged hierarchical semantic state"
                ),
            },
        },
        "inference_items": list(
            source_validation.get("inferred_fields", [])
        ),
        "assets": assets,
    }
    manifest_path = output_dir_path / MANIFEST_FILENAME
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return {
        "scene": scene,
        "files": png_paths,
        "svg_files": svg_paths,
        "manifest": manifest,
        "manifest_path": manifest_path,
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export deterministic data-driven assets for Figure 3."
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--step", type=int, default=8)
    parser.add_argument("--rows", type=int, default=40)
    parser.add_argument("--cols", type=int, default=60)
    parser.add_argument("--obstacle-ratio", type=float, default=0.20)
    parser.add_argument("--obs-size", type=int, default=6)
    parser.add_argument("--scan-radius", type=int, default=10)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--include-svg", action="store_true")
    parser.add_argument("--paper-repo", type=Path, default=DEFAULT_PAPER_REPO)
    parser.add_argument(
        "--source-manifest",
        type=Path,
        default=DEFAULT_SOURCE_MANIFEST,
        help="Existing online-workflow manifest used to verify the shared B_t source.",
    )
    return parser


def cli_main() -> None:
    args = _build_arg_parser().parse_args()
    result = export_fig3_overview_assets(
        args.output_dir,
        seed=int(args.seed),
        step=int(args.step),
        rows=int(args.rows),
        cols=int(args.cols),
        obstacle_ratio=float(args.obstacle_ratio),
        obs_size=int(args.obs_size),
        scan_radius=int(args.scan_radius),
        dpi=int(args.dpi),
        include_svg=bool(args.include_svg),
        paper_repo=args.paper_repo,
        source_manifest=args.source_manifest,
    )
    manifest = result["manifest"]
    print("mode=fig3-overview-assets")
    print(f"seed={manifest['seed']}")
    print(f"requested_step={manifest['requested_step']}")
    print(f"resolved_step={manifest['resolved_step']}")
    print(
        "source_manifest_verified="
        f"{manifest['figure2_or_belief_map_source']['manifest_verified']}"
    )
    for name, path in result["files"].items():
        print(f"{name}_png={Path(path).resolve()}")
        svg_path = result["svg_files"].get(name)
        if svg_path is not None:
            print(f"{name}_svg={Path(svg_path).resolve()}")
    print(f"manifest={Path(result['manifest_path']).resolve()}")


if __name__ == "__main__":
    cli_main()
