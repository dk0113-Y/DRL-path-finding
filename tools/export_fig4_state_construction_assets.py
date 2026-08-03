from __future__ import annotations

"""Export deterministic, implementation-traced assets for manuscript Figure 4.

The exporter reuses the Figure 3 seed-1/step-8 scene and builds every raster
from the active cumulative-belief, local-state, and semantic-state code paths.
It does not render the simulator truth map or any network-side computation.
"""

import argparse
import hashlib
import json
import subprocess
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from env.advantage_state_builder import (  # noqa: E402
    FINAL_4CH_ADVANTAGE_CANVAS_CHANNELS,
    AdvantageStateBuilder,
)
from env.grid_topology import INVISIBLE  # noqa: E402
from env.value_state_builder import (  # noqa: E402
    VALUE_BLOCK_FEATURE_COUNT,
    VALUE_ENTRY_FEATURE_COUNT,
    ValueStateBuilder,
)
from tools.export_fig3_overview_assets import (  # noqa: E402
    Fig3OverviewScene,
    _replay_canonical_snapshot,
    build_fig3_overview_scene,
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
    / "fig4_assets"
)
DEFAULT_PAPER_REPO = REPO_ROOT.parent / "paper_work"
MANIFEST_FILENAME = "fig4_state_construction_assets_manifest.json"

INK = "#19324A"
NEUTRAL = "#607487"
NEUTRAL_LIGHT = "#F4F7F9"
WARM = "#C96144"
WARM_MID = "#E99D4E"
WARM_LIGHT = "#FBF0EA"
COOL = "#5185C0"
COOL_DARK = "#315F91"
COOL_MID = "#8EA9D4"
COOL_LIGHT = "#EEF4FA"
GREEN = "#55966B"
GRID = "#C7D0D8"

ASSET_NAMES = (
    "dynamic_cumulative_belief_map",
    "belief_map_with_robot_and_history",
    "robot_centered_local_window",
    "local_channel_free_space",
    "local_channel_obstacle",
    "local_channel_visit_count",
    "local_channel_recent_trajectory",
    "frontier_unknown_region_extraction",
    "compact_hierarchy",
    "compact_packing",
)


@dataclass(frozen=True, slots=True)
class Fig4StateConstructionScene:
    shared: Fig3OverviewScene
    previous_step: int
    previous_origin_world: tuple[int, int]
    previous_shape: tuple[int, int]
    local_sampled_map: np.ndarray
    block_features: np.ndarray
    entry_features: np.ndarray
    block_mask: np.ndarray
    entry_mask: np.ndarray
    value_meta: dict[str, float]


def _configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "svg.fonttype": "none",
            "svg.hashsalt": "msd-hsr-fig4-state-construction-v1",
            "font.size": 8,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 1.0,
            "savefig.facecolor": "white",
            "savefig.edgecolor": "white",
        }
    )


_configure_matplotlib()


def _git_head(repo: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


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


def _sample_local_map(shared: Fig3OverviewScene) -> np.ndarray:
    arr_rows, arr_cols, inside = AdvantageStateBuilder._local_index_arrays(
        shared.cum_map,
        shared.agent_world,
    )
    local_shape = tuple(int(v) for v in shared.cum_map.local_shape)
    sampled = np.full(local_shape, INVISIBLE, dtype=np.int8)
    sampled[inside] = shared.cum_map.map[arr_rows[inside], arr_cols[inside]]
    return sampled


def build_fig4_state_construction_scene(
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
) -> Fig4StateConstructionScene:
    shared = build_fig3_overview_scene(
        seed=int(seed),
        step=int(step),
        rows=int(rows),
        cols=int(cols),
        obstacle_ratio=float(obstacle_ratio),
        obs_size=int(obs_size),
        scan_radius=int(scan_radius),
        dpi=int(dpi),
        output_dir=Path(output_dir),
    )
    if int(shared.resolved_step) <= 0:
        raise RuntimeError("Figure 4 needs a prior step to show map-boundary expansion")
    previous_step = int(shared.resolved_step) - 1
    previous_map, _, _ = _replay_canonical_snapshot(
        shared.config,
        resolved_step=previous_step,
    )
    block_features, entry_features, block_mask, entry_mask, value_meta = (
        ValueStateBuilder().build(shared.semantic_snapshot)
    )

    expected_local_shape = (4, (2 * int(scan_radius)) + 1, (2 * int(scan_radius)) + 1)
    if tuple(int(v) for v in shared.advantage_canvas.shape) != expected_local_shape:
        raise RuntimeError(
            "active local-state shape drifted: "
            f"expected {expected_local_shape}, got {shared.advantage_canvas.shape}"
        )
    expected_global_shapes = ((16, 2), (16, 8, 4), (16,), (16, 8))
    actual_global_shapes = tuple(
        tuple(int(v) for v in array.shape)
        for array in (block_features, entry_features, block_mask, entry_mask)
    )
    if actual_global_shapes != expected_global_shapes:
        raise RuntimeError(
            "active fixed-capacity state shapes drifted: "
            f"expected {expected_global_shapes}, got {actual_global_shapes}"
        )
    if len(shared.semantic_snapshot.accessible_blocks) <= 0:
        raise RuntimeError("the selected deterministic scene has no semantic parent block")
    if int(np.count_nonzero(entry_mask)) <= 0:
        raise RuntimeError("the selected deterministic scene has no frontier entrance")

    previous_origin = tuple(int(v) for v in previous_map.origin_world_rc)
    previous_shape = tuple(int(v) for v in previous_map.map.shape)
    current_origin = tuple(int(v) for v in shared.cum_map.origin_world_rc)
    current_shape = tuple(int(v) for v in shared.cum_map.map.shape)
    previous_world_bounds = (
        previous_origin[0],
        previous_origin[0] + previous_shape[0],
        previous_origin[1],
        previous_origin[1] + previous_shape[1],
    )
    current_world_bounds = (
        current_origin[0],
        current_origin[0] + current_shape[0],
        current_origin[1],
        current_origin[1] + current_shape[1],
    )
    if previous_world_bounds == current_world_bounds:
        raise RuntimeError(
            "the selected step does not show a real cumulative-map storage expansion"
        )

    return Fig4StateConstructionScene(
        shared=shared,
        previous_step=previous_step,
        previous_origin_world=previous_origin,
        previous_shape=previous_shape,
        local_sampled_map=_sample_local_map(shared),
        block_features=np.asarray(block_features, dtype=np.float32).copy(),
        entry_features=np.asarray(entry_features, dtype=np.float32).copy(),
        block_mask=np.asarray(block_mask, dtype=bool).copy(),
        entry_mask=np.asarray(entry_mask, dtype=bool).copy(),
        value_meta=dict(value_meta),
    )


def _clean_map_axis(ax, shape: tuple[int, int]) -> None:
    ax.set_xlim(-0.5, float(shape[1]) - 0.5)
    ax.set_ylim(float(shape[0]) - 0.5, -0.5)
    ax.set_aspect("equal")
    ax.set_axis_off()


def _draw_occupancy(ax, array: np.ndarray, *, gid: str) -> None:
    cmap, norm = occupancy_colormap(load_paper_figure_style())
    image = ax.imshow(
        np.asarray(array, dtype=np.int8),
        cmap=cmap,
        norm=norm,
        origin="upper",
        interpolation="nearest",
        zorder=1,
    )
    image.set_gid(gid)


def _current_array_rc(scene: Fig4StateConstructionScene, world_rc: Sequence[int]) -> tuple[float, float]:
    origin_r, origin_c = scene.shared.cum_map.origin_world_rc
    return float(int(world_rc[0]) - int(origin_r)), float(int(world_rc[1]) - int(origin_c))


def _draw_storage_boundaries(ax, scene: Fig4StateConstructionScene) -> None:
    current_shape = tuple(int(v) for v in scene.shared.cum_map.map.shape)
    expanded = Rectangle(
        (-0.46, -0.46),
        float(current_shape[1]) - 0.08,
        float(current_shape[0]) - 0.08,
        fill=False,
        edgecolor=COOL_DARK,
        linewidth=1.6,
        zorder=8,
    )
    expanded.set_gid("fig4_expanded_storage_boundary")
    ax.add_patch(expanded)

    previous_top, previous_left = _current_array_rc(
        scene,
        scene.previous_origin_world,
    )
    previous = Rectangle(
        (previous_left - 0.46, previous_top - 0.46),
        float(scene.previous_shape[1]) - 0.08,
        float(scene.previous_shape[0]) - 0.08,
        fill=False,
        edgecolor=NEUTRAL,
        linewidth=1.25,
        linestyle=(0, (4, 2.5)),
        zorder=9,
    )
    previous.set_gid("fig4_previous_storage_boundary")
    ax.add_patch(previous)


def _draw_history_and_robot(ax, scene: Fig4StateConstructionScene) -> None:
    points = np.asarray(
        [_current_array_rc(scene, rc) for rc in scene.shared.trajectory_world],
        dtype=np.float32,
    )
    rows = points[:, 0]
    cols = points[:, 1]
    weights = np.linspace(0.30, 1.0, len(points), dtype=np.float32)
    line = ax.plot(
        cols,
        rows,
        color=WARM,
        linewidth=1.8,
        alpha=0.88,
        solid_capstyle="round",
        zorder=10,
    )[0]
    line.set_gid("fig4_executed_trajectory")
    markers = ax.scatter(
        cols,
        rows,
        s=10.0 + (18.0 * weights),
        c=np.repeat(np.asarray([[201 / 255, 97 / 255, 68 / 255, 1.0]]), len(points), axis=0),
        alpha=weights,
        edgecolors="white",
        linewidths=0.35,
        zorder=11,
    )
    markers.set_gid("fig4_executed_trajectory_time_markers")

    agent_r, agent_c = _current_array_rc(scene, scene.shared.agent_world)
    robot_parts = draw_topdown_robot(
        ax,
        row=agent_r,
        col=agent_c,
        heading_action=int(scene.shared.blueprint.selected_action),
        style=load_paper_figure_style(),
        zorder=14,
    )
    for index, part in enumerate(robot_parts):
        part.set_gid(f"fig4_robot_part_{index}")


def _draw_local_window(ax, scene: Fig4StateConstructionScene) -> None:
    agent_r, agent_c = _current_array_rc(scene, scene.shared.agent_world)
    h, w = (int(v) for v in scene.shared.cum_map.local_shape)
    window = Rectangle(
        (agent_c - (w / 2.0), agent_r - (h / 2.0)),
        float(w),
        float(h),
        fill=False,
        edgecolor=WARM,
        linewidth=1.5,
        linestyle=(0, (3.5, 2.0)),
        zorder=13,
    )
    window.set_gid("fig4_robot_centered_local_window")
    ax.add_patch(window)


def _normalize_svg_whitespace(svg_path: Path) -> None:
    """Remove generator-only trailing spaces without changing SVG geometry."""
    text = svg_path.read_text(encoding="utf-8")
    normalized = "\n".join(line.rstrip() for line in text.splitlines()) + "\n"
    svg_path.write_text(normalized, encoding="utf-8", newline="\n")


def _save_pair(
    fig: plt.Figure,
    png_path: Path,
    *,
    dpi: int,
    include_svg: bool,
    tight: bool = True,
) -> Path | None:
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        png_path,
        dpi=int(dpi),
        bbox_inches="tight" if tight else None,
        pad_inches=0.0,
        facecolor="white",
        metadata={"Software": "DRL-path-finding fig4 deterministic exporter"},
    )
    svg_path = png_path.with_suffix(".svg") if include_svg else None
    if svg_path is not None:
        fig.savefig(
            svg_path,
            bbox_inches="tight" if tight else None,
            pad_inches=0.0,
            facecolor="white",
            metadata={
                "Date": None,
                "Creator": "DRL-path-finding fig4 deterministic exporter",
            },
        )
        _normalize_svg_whitespace(svg_path)
        ET.parse(svg_path)
    plt.close(fig)
    return svg_path


def _map_figure(array: np.ndarray) -> tuple[plt.Figure, object]:
    rows, cols = (int(v) for v in array.shape)
    fig = plt.figure(figsize=(2.70, 2.70 * rows / max(1, cols)), frameon=False)
    ax = fig.add_axes((0.0, 0.0, 1.0, 1.0))
    _clean_map_axis(ax, (rows, cols))
    return fig, ax


def _render_dynamic_map(
    scene: Fig4StateConstructionScene,
    path: Path,
    *,
    include_svg: bool,
) -> Path | None:
    belief = np.asarray(scene.shared.cum_map.map, dtype=np.int8)
    fig, ax = _map_figure(belief)
    _draw_occupancy(ax, belief, gid="fig4_dynamic_cumulative_belief")
    _draw_storage_boundaries(ax, scene)
    _clean_map_axis(ax, tuple(belief.shape))
    return _save_pair(
        fig,
        path,
        dpi=int(scene.shared.config.dpi),
        include_svg=include_svg,
        tight=False,
    )


def _render_belief_with_robot_and_history(
    scene: Fig4StateConstructionScene,
    path: Path,
    *,
    include_svg: bool,
) -> Path | None:
    belief = np.asarray(scene.shared.cum_map.map, dtype=np.int8)
    fig, ax = _map_figure(belief)
    _draw_occupancy(ax, belief, gid="fig4_belief_with_robot_and_history")
    _draw_storage_boundaries(ax, scene)
    _draw_history_and_robot(ax, scene)
    _draw_local_window(ax, scene)
    _clean_map_axis(ax, tuple(belief.shape))
    return _save_pair(
        fig,
        path,
        dpi=int(scene.shared.config.dpi),
        include_svg=include_svg,
        tight=False,
    )


def _render_local_window(
    scene: Fig4StateConstructionScene,
    path: Path,
    *,
    include_svg: bool,
) -> Path | None:
    fig, ax = _map_figure(scene.local_sampled_map)
    _draw_occupancy(ax, scene.local_sampled_map, gid="fig4_local_window_occupancy")
    trajectory = list(scene.shared.recent_trajectory_world[:-1])
    center_r = int(scene.local_sampled_map.shape[0] // 2)
    center_c = int(scene.local_sampled_map.shape[1] // 2)
    if trajectory:
        agent_r, agent_c = scene.shared.agent_world
        rows = np.asarray([int(rc[0]) - int(agent_r) + center_r for rc in trajectory])
        cols = np.asarray([int(rc[1]) - int(agent_c) + center_c for rc in trajectory])
        weights = np.linspace(1.0 / len(trajectory), 1.0, len(trajectory))
        ax.plot(cols, rows, color=WARM, linewidth=1.35, alpha=0.82, zorder=8)
        ax.scatter(
            cols,
            rows,
            s=8.0 + (16.0 * weights),
            c=WARM,
            alpha=weights,
            edgecolors="white",
            linewidths=0.3,
            zorder=9,
        )
    robot = Circle(
        (float(center_c), float(center_r)),
        radius=0.58,
        facecolor=COOL_DARK,
        edgecolor="white",
        linewidth=0.6,
        zorder=12,
    )
    robot.set_gid("fig4_local_window_robot_center")
    ax.add_patch(robot)
    _clean_map_axis(ax, tuple(scene.local_sampled_map.shape))
    return _save_pair(
        fig,
        path,
        dpi=int(scene.shared.config.dpi),
        include_svg=include_svg,
        tight=False,
    )


def _render_local_channel(
    scene: Fig4StateConstructionScene,
    path: Path,
    *,
    channel_index: int,
    include_svg: bool,
) -> Path | None:
    cmaps = (
        ListedColormap(["#FFFFFF", "#E99D4E"]),
        ListedColormap(["#FFFFFF", "#303942"]),
        LinearSegmentedColormap.from_list(
            "fig4_visit_count", ["#FFFFFF", "#F2CB9F", WARM]
        ),
        LinearSegmentedColormap.from_list(
            "fig4_recent_trajectory", ["#FFFFFF", "#F2CB9F", WARM]
        ),
    )
    channel = np.asarray(scene.shared.advantage_canvas[channel_index], dtype=np.float32)
    fig = plt.figure(figsize=(1.05, 1.05), frameon=False)
    ax = fig.add_axes((0.0, 0.0, 1.0, 1.0))
    image = ax.imshow(
        channel,
        cmap=cmaps[channel_index],
        vmin=0.0,
        vmax=1.0,
        origin="upper",
        interpolation="nearest",
    )
    image.set_gid(
        f"fig4_local_channel_{channel_index}_{FINAL_4CH_ADVANTAGE_CANVAS_CHANNELS[channel_index]}"
    )
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(WARM if channel_index >= 2 else NEUTRAL)
        spine.set_linewidth(0.9)
    return _save_pair(
        fig,
        path,
        dpi=int(scene.shared.config.dpi),
        include_svg=include_svg,
    )


def _render_frontier_unknown_regions(
    scene: Fig4StateConstructionScene,
    path: Path,
    *,
    include_svg: bool,
) -> Path | None:
    belief = np.asarray(scene.shared.cum_map.map, dtype=np.int8)
    fig, ax = _map_figure(belief)
    _draw_occupancy(ax, belief, gid="fig4_frontier_overlay_belief")
    rgba = np.zeros((*belief.shape, 4), dtype=np.float32)
    blocks = tuple(scene.shared.semantic_snapshot.accessible_blocks)
    block_colors = (
        np.asarray([81, 133, 192, 255], dtype=np.float32) / 255.0,
        np.asarray([142, 169, 212, 255], dtype=np.float32) / 255.0,
    )
    for block_slot, block in enumerate(blocks):
        color = block_colors[block_slot % len(block_colors)].copy()
        color[3] = 0.48
        rgba[block.rows, block.cols] = color
        for cluster in block.frontier_clusters:
            rgba[cluster.rows, cluster.cols] = np.asarray(
                [49 / 255, 95 / 255, 145 / 255, 0.98],
                dtype=np.float32,
            )
    overlay = ax.imshow(rgba, origin="upper", interpolation="nearest", zorder=7)
    overlay.set_gid("fig4_unknown_regions_and_frontier_clusters")
    agent_r, agent_c = _current_array_rc(scene, scene.shared.agent_world)
    ax.scatter(
        [agent_c],
        [agent_r],
        s=28,
        c=WARM,
        edgecolors="white",
        linewidths=0.6,
        zorder=10,
    )
    box = scene.shared.semantic_snapshot.analysis_box
    analysis_outline = Rectangle(
        (float(box.c0) - 0.45, float(box.r0) - 0.45),
        float(box.c1 - box.c0) - 0.1,
        float(box.r1 - box.r0) - 0.1,
        fill=False,
        edgecolor=COOL,
        linewidth=0.9,
        linestyle=(0, (3, 2)),
        zorder=9,
    )
    analysis_outline.set_gid("fig4_semantic_analysis_box")
    ax.add_patch(analysis_outline)
    _clean_map_axis(ax, tuple(belief.shape))
    return _save_pair(
        fig,
        path,
        dpi=int(scene.shared.config.dpi),
        include_svg=include_svg,
        tight=False,
    )


def _render_compact_hierarchy(
    scene: Fig4StateConstructionScene,
    path: Path,
    *,
    include_svg: bool,
) -> Path | None:
    blocks = tuple(scene.shared.semantic_snapshot.accessible_blocks[:2])
    fig = plt.figure(figsize=(2.25, 1.15), frameon=False)
    ax = fig.add_axes((0.0, 0.0, 1.0, 1.0))
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")
    centers = np.asarray([0.50]) if len(blocks) == 1 else np.linspace(0.28, 0.72, len(blocks))
    for block_slot, (block, center_x) in enumerate(zip(blocks, centers), start=1):
        parent = FancyBboxPatch(
            (float(center_x) - 0.13, 0.66),
            0.26,
            0.20,
            boxstyle="round,pad=0.012,rounding_size=0.035",
            facecolor="#C8D9EB",
            edgecolor=COOL_DARK,
            linewidth=1.2,
            zorder=5,
        )
        parent.set_gid(f"fig4_hierarchy_parent_{block_slot}")
        ax.add_patch(parent)
        ax.text(
            float(center_x),
            0.76,
            f"B{block_slot}",
            ha="center",
            va="center",
            fontsize=7.5,
            color=INK,
            weight="bold",
            zorder=6,
        )
        children = tuple(block.frontier_clusters[:5])
        child_xs = np.linspace(float(center_x) - 0.20, float(center_x) + 0.20, len(children))
        for child_slot, (cluster, child_x) in enumerate(zip(children, child_xs), start=1):
            edge = ax.plot(
                [float(center_x), float(child_x)],
                [0.66, 0.34],
                color=COOL,
                linewidth=1.0,
                zorder=2,
            )[0]
            edge.set_gid(f"fig4_hierarchy_edge_{block_slot}_{child_slot}")
            child = FancyBboxPatch(
                (float(child_x) - 0.045, 0.22),
                0.09,
                0.12,
                boxstyle="round,pad=0.006,rounding_size=0.014",
                facecolor=COOL_LIGHT,
                edgecolor=COOL,
                linewidth=0.9,
                zorder=5,
            )
            child.set_gid(f"fig4_hierarchy_child_{block_slot}_{child_slot}")
            ax.add_patch(child)
            ax.text(
                float(child_x),
                0.28,
                f"E{child_slot}",
                ha="center",
                va="center",
                fontsize=5.3,
                color=INK,
                zorder=6,
            )
            _ = cluster
    return _save_pair(
        fig,
        path,
        dpi=int(scene.shared.config.dpi),
        include_svg=include_svg,
    )


def _render_compact_packing(
    scene: Fig4StateConstructionScene,
    path: Path,
    *,
    include_svg: bool,
) -> Path | None:
    fig, axes = plt.subplots(
        1,
        4,
        figsize=(3.2, 0.95),
        gridspec_kw={"width_ratios": [0.8, 1.8, 0.35, 0.9]},
        frameon=False,
    )
    arrays = (
        np.clip(scene.block_features, 0.0, 1.0),
        np.max(np.abs(scene.entry_features), axis=2),
        scene.block_mask[:, None].astype(np.float32),
        scene.entry_mask.astype(np.float32),
    )
    cmaps = (
        LinearSegmentedColormap.from_list("fig4_block_features", ["#FFFFFF", COOL]),
        LinearSegmentedColormap.from_list("fig4_entry_features", ["#FFFFFF", COOL_DARK]),
        ListedColormap(["#FFFFFF", GREEN]),
        ListedColormap(["#FFFFFF", GREEN]),
    )
    labels = ("B", "E", "M_B", "M_E")
    for ax, array, cmap, label in zip(axes, arrays, cmaps, labels):
        image = ax.imshow(
            array,
            cmap=cmap,
            vmin=0.0,
            vmax=max(1.0, float(np.max(array))) if array.size else 1.0,
            origin="upper",
            interpolation="nearest",
            aspect="auto",
        )
        image.set_gid(f"fig4_packing_{label}")
        ax.set_title(label, fontsize=6.2, color=INK, pad=2)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color(COOL if label in {"B", "E"} else GREEN)
            spine.set_linewidth(0.7)
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.03, top=0.82, wspace=0.20)
    return _save_pair(
        fig,
        path,
        dpi=int(scene.shared.config.dpi),
        include_svg=include_svg,
        tight=False,
    )


def _asset_record(name: str, png_path: Path, svg_path: Path | None) -> dict[str, object]:
    with Image.open(png_path) as image:
        image.verify()
    with Image.open(png_path) as image:
        width, height = (int(v) for v in image.size)
        mode = str(image.mode)
    if svg_path is not None:
        ET.parse(svg_path)
    return {
        "name": name,
        "png_path": str(png_path.resolve()),
        "svg_path": None if svg_path is None else str(svg_path.resolve()),
        "png_sha256": _sha256_file(png_path),
        "svg_sha256": None if svg_path is None else _sha256_file(svg_path),
        "width_px": width,
        "height_px": height,
        "png_mode": mode,
    }


def _world_bounds(origin: Sequence[int], shape: Sequence[int]) -> list[int]:
    return [
        int(origin[0]),
        int(origin[0]) + int(shape[0]),
        int(origin[1]),
        int(origin[1]) + int(shape[1]),
    ]


def export_fig4_state_construction_assets(
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
) -> dict[str, object]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    scene = build_fig4_state_construction_scene(
        seed=int(seed),
        step=int(step),
        rows=int(rows),
        cols=int(cols),
        obstacle_ratio=float(obstacle_ratio),
        obs_size=int(obs_size),
        scan_radius=int(scan_radius),
        dpi=int(dpi),
        output_dir=output_path,
    )

    renderers: dict[str, Callable[[Fig4StateConstructionScene, Path], Path | None]] = {
        "dynamic_cumulative_belief_map": lambda s, p: _render_dynamic_map(s, p, include_svg=include_svg),
        "belief_map_with_robot_and_history": lambda s, p: _render_belief_with_robot_and_history(s, p, include_svg=include_svg),
        "robot_centered_local_window": lambda s, p: _render_local_window(s, p, include_svg=include_svg),
        "local_channel_free_space": lambda s, p: _render_local_channel(s, p, channel_index=0, include_svg=include_svg),
        "local_channel_obstacle": lambda s, p: _render_local_channel(s, p, channel_index=1, include_svg=include_svg),
        "local_channel_visit_count": lambda s, p: _render_local_channel(s, p, channel_index=2, include_svg=include_svg),
        "local_channel_recent_trajectory": lambda s, p: _render_local_channel(s, p, channel_index=3, include_svg=include_svg),
        "frontier_unknown_region_extraction": lambda s, p: _render_frontier_unknown_regions(s, p, include_svg=include_svg),
        "compact_hierarchy": lambda s, p: _render_compact_hierarchy(s, p, include_svg=include_svg),
        "compact_packing": lambda s, p: _render_compact_packing(s, p, include_svg=include_svg),
    }
    png_paths: dict[str, Path] = {}
    svg_paths: dict[str, Path | None] = {}
    for name in ASSET_NAMES:
        png_path = output_path / f"{name}.png"
        png_paths[name] = png_path
        svg_paths[name] = renderers[name](scene, png_path)

    shared = scene.shared
    current_origin = tuple(int(v) for v in shared.cum_map.origin_world_rc)
    current_shape = tuple(int(v) for v in shared.cum_map.map.shape)
    local_h, local_w = (int(v) for v in shared.cum_map.local_shape)
    agent_r, agent_c = (int(v) for v in shared.agent_world)
    local_world_bounds = [
        agent_r - (local_h // 2),
        agent_r + (local_h // 2) + 1,
        agent_c - (local_w // 2),
        agent_c + (local_w // 2) + 1,
    ]
    blocks = tuple(shared.semantic_snapshot.accessible_blocks)
    total_entries = int(sum(int(block.frontier_cluster_count) for block in blocks))
    active_blocks = int(np.count_nonzero(scene.block_mask))
    active_entries = int(np.count_nonzero(scene.entry_mask))
    current_bounds = _world_bounds(current_origin, current_shape)
    previous_bounds = _world_bounds(scene.previous_origin_world, scene.previous_shape)
    expansion = {
        "top": max(0, previous_bounds[0] - current_bounds[0]),
        "bottom": max(0, current_bounds[1] - previous_bounds[1]),
        "left": max(0, previous_bounds[2] - current_bounds[2]),
        "right": max(0, current_bounds[3] - previous_bounds[3]),
    }
    trajectory_history = tuple(shared.recent_trajectory_world[:-1])[-10:]
    trajectory_weights = [
        float((index + 1) / max(1, len(trajectory_history)))
        for index in range(len(trajectory_history))
    ]
    asset_records = {
        name: _asset_record(name, png_paths[name], svg_paths[name])
        for name in ASSET_NAMES
    }
    manifest = {
        "schema_version": 1,
        "code_repo_commit": _git_head(REPO_ROOT),
        "paper_repo_commit_before_change": _git_head(Path(paper_repo)),
        "scene_source": "tools.export_fig3_overview_assets.build_fig3_overview_scene",
        "seed": int(shared.config.seed),
        "requested_step": int(shared.requested_step),
        "resolved_step": int(shared.resolved_step),
        "previous_step": int(scene.previous_step),
        "scan_radius": int(shared.config.scan_radius),
        "belief_matrix_sha256": _sha256_int8(shared.cum_map.map),
        "previous_storage_origin_world": list(scene.previous_origin_world),
        "previous_storage_shape": list(scene.previous_shape),
        "previous_storage_bounds_world_half_open": previous_bounds,
        "expanded_storage_origin_world": list(current_origin),
        "expanded_storage_shape": list(current_shape),
        "expanded_storage_bounds_world_half_open": current_bounds,
        "storage_expansion_cells": expansion,
        "robot_world_position": list(shared.agent_world),
        "executed_trajectory_world": [list(rc) for rc in shared.trajectory_world],
        "robot_centered_local_window_world_half_open": local_world_bounds,
        "local_window_shape": [local_h, local_w],
        "local_channel_names": list(FINAL_4CH_ADVANTAGE_CANVAS_CHANNELS),
        "local_state_shape": [int(v) for v in shared.advantage_canvas.shape],
        "local_unknown_and_outside_encoding": (
            "free=0 and obstacle=0 for unknown cells and positions outside current belief storage"
        ),
        "recent_trajectory_semantics": {
            "history_positions_world": [list(rc) for rc in trajectory_history],
            "weights_old_to_new": trajectory_weights,
            "weight_formula": "(index + 1) / history_length",
            "current_robot_cell_repainted": False,
        },
        "global_semantic_state": {
            "frontier_definition": "known free cells with an orthogonally adjacent unknown cell",
            "frontier_cluster_connectivity": 8,
            "unknown_region_connectivity": 4,
            "region_frontier_association": "orthogonal adjacency in the current belief map",
            "frontier_associated_unknown_region_block_count": len(blocks),
            "frontier_entrance_count": total_entries,
            "analysis_box_shape": list(shared.semantic_snapshot.analysis_box.shape),
        },
        "feature_semantics": {
            "block_features": ["relative area", "number of entrances"],
            "entrance_features": [
                "delta_r / H",
                "delta_c / W",
                "frontier-cluster scale",
                "obstacle density",
            ],
            "frontier_cluster_scale_implementation": (
                "frontier-cluster cell count divided by H + W; legacy source fields are not figure labels"
            ),
        },
        "fixed_capacity_packing": {
            "max_blocks": 16,
            "max_entrances_per_block": 8,
            "block_feature_count": int(VALUE_BLOCK_FEATURE_COUNT),
            "entrance_feature_count": int(VALUE_ENTRY_FEATURE_COUNT),
            "block_features_shape": list(scene.block_features.shape),
            "entrance_features_shape": list(scene.entry_features.shape),
            "block_mask_shape": list(scene.block_mask.shape),
            "entrance_mask_shape": list(scene.entry_mask.shape),
            "active_block_count": active_blocks,
            "active_entrance_count": active_entries,
            "active_block_features": scene.block_features[scene.block_mask].tolist(),
            "active_entrance_features_by_block": [
                scene.entry_features[index, scene.entry_mask[index]].tolist()
                for index in range(active_blocks)
            ],
            "block_mask": scene.block_mask.astype(np.uint8).tolist(),
            "entrance_mask": scene.entry_mask.astype(np.uint8).tolist(),
            "diagnostics": scene.value_meta,
        },
        "five_policy_inputs": {
            "advantage_canvas": [4, 21, 21],
            "value_block_features": [16, 2],
            "value_entry_features": [16, 8, 4],
            "value_block_mask": [16],
            "value_entry_mask": [16, 8],
        },
        "truth_map_rendered": False,
        "claim_boundaries": {
            "reachability_asserted": False,
            "network_structure_rendered": False,
            "figure_stops_at_fixed_dimensional_state_interface": True,
        },
        "assets": asset_records,
    }
    manifest_path = output_path / MANIFEST_FILENAME
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
        description="Export deterministic implementation-traced Figure 4 assets."
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
    return parser


def cli_main() -> None:
    args = _build_arg_parser().parse_args()
    result = export_fig4_state_construction_assets(
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
    )
    manifest = result["manifest"]
    print("mode=fig4-state-construction-assets")
    print(f"seed={manifest['seed']}")
    print(f"requested_step={manifest['requested_step']}")
    print(f"resolved_step={manifest['resolved_step']}")
    print(f"previous_step={manifest['previous_step']}")
    print(f"local_state_shape={manifest['local_state_shape']}")
    print(f"five_policy_inputs={manifest['five_policy_inputs']}")
    for name, path in result["files"].items():
        print(f"{name}_png={Path(path).resolve()}")
        svg_path = result["svg_files"][name]
        if svg_path is not None:
            print(f"{name}_svg={Path(svg_path).resolve()}")
    print(f"manifest={Path(result['manifest_path']).resolve()}")


if __name__ == "__main__":
    cli_main()
