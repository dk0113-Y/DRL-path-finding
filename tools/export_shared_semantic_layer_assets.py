from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

if os.environ.get("DRL_PAPER_FIGURE_INTERACTIVE") != "1":
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap, to_rgba
from matplotlib.patches import FancyBboxPatch, Rectangle

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from env.agent_version import LocalObservationModel
from env.block_random_g import RandomMapGenerator
from env.core_cummap import AnalysisBox, CumulativeBeliefMap
from env.core_radar import RadarSensor
from env.grid_topology import ACTIONS_8, INVISIBLE, GridTopology
from env.shared_semantic_layer import FrontierCluster, SharedSemanticLayer, SharedSemanticSnapshot, UnknownBlock
from tools.export_architecture_pictures import (
    ExportConfig,
    FIXED_ACTION_PREFERENCES,
    KEY_TO_ACTION,
    Snapshot,
    _capture_snapshot,
    _clear_old_png_outputs,
    _draw_agent,
    _draw_trajectory,
    _format_output_path,
    _select_fallback_action,
    _set_global_seed,
)

BELIEF_CMAP = ListedColormap(
    [
        "#5f6770",
        "#f5f6f7",
        "#1c232b",
    ]
)
BELIEF_NORM = BoundaryNorm([-1.5, -0.5, 0.5, 1.5], BELIEF_CMAP.N)

DEFAULT_OUTPUT_DIR = REPO_ROOT / "run_picture" / "shared_semantic_layer_assets"

BLOCK_PALETTE = (
    "#7aa6c2",
    "#8bbf9f",
    "#c2a46f",
    "#a88fbf",
    "#6fa3a7",
)
CLUSTER_PALETTE = (
    "#f4b942",
    "#ef8354",
    "#7fb069",
    "#5b8e7d",
    "#d17b88",
)
RAW_FRONTIER_COLOR = CLUSTER_PALETTE[0]
UNKNOWN_BLOCK_BOX_COLOR = "#2f6f7e"
DEFAULT_SEMANTIC_TRAJECTORY_LENGTH = 10

ANALYSIS_BOX_COLOR = "#0f4c5c"
SUMMARY_BOX_COLOR = "#2f6f7e"
SUMMARY_CARD_FILL = "#f9fbfc"
SUMMARY_CARD_TEXT = "#243b53"


@dataclass(frozen=True, slots=True)
class CropBounds:
    r0: int
    r1: int
    c0: int
    c1: int

    @property
    def shape(self) -> tuple[int, int]:
        return int(self.r1 - self.r0), int(self.c1 - self.c0)


@dataclass(frozen=True, slots=True)
class SemanticExportScene:
    seed: int
    requested_step: int
    resolved_step: int
    snapshot: Snapshot
    frontier_mask: np.ndarray
    semantic_snapshot: SharedSemanticSnapshot
    focus_block: UnknownBlock
    focus_cluster: FrontierCluster


@dataclass(frozen=True, slots=True)
class SharedSemanticAssetStyle:
    dpi: int = 260
    max_inches: float = 5.2
    min_inches: float = 2.4
    cell_inches: float = 0.16
    parsing_block_alpha: float = 0.24
    unknown_block_alpha: float = 0.56
    frontier_alpha: float = 0.96
    support_fill_alpha: float = 0.10
    summary_font_size: float = 8.0
    label_font_size: float = 8.2


def _belief_rgba(belief_map: np.ndarray) -> np.ndarray:
    return np.asarray(BELIEF_CMAP(BELIEF_NORM(np.asarray(belief_map, dtype=np.int8))), dtype=np.float32)


def _figure_size_for_shape(shape: tuple[int, int], *, style: SharedSemanticAssetStyle) -> tuple[float, float]:
    rows = max(1, int(shape[0]))
    cols = max(1, int(shape[1]))
    width = float(cols) * float(style.cell_inches)
    height = float(rows) * float(style.cell_inches)
    max_side = max(width, height)
    if max_side > float(style.max_inches):
        scale_down = float(style.max_inches) / max_side
        width *= scale_down
        height *= scale_down
    min_side = min(width, height)
    if min_side < float(style.min_inches):
        scale_up = float(style.min_inches) / max(min_side, 1e-6)
        width *= scale_up
        height *= scale_up
    max_side = max(width, height)
    if max_side > float(style.max_inches):
        scale_down = float(style.max_inches) / max_side
        width *= scale_down
        height *= scale_down
    return width, height


def _create_asset_axes(shape: tuple[int, int], *, style: SharedSemanticAssetStyle, facecolor: str = "white"):
    fig = plt.figure(figsize=_figure_size_for_shape(shape, style=style), frameon=False)
    fig.patch.set_facecolor(facecolor)
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.set_facecolor(facecolor)
    ax.set_aspect("equal")
    ax.set_xlim(-0.5, float(shape[1]) - 0.5)
    ax.set_ylim(float(shape[0]) - 0.5, -0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
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


def _save_asset_figure(fig: plt.Figure, path: Path, *, dpi: int, facecolor: str = "white") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=int(dpi), pad_inches=0.0, bbox_inches=None, facecolor=facecolor)
    plt.close(fig)
    _trim_external_png_background(path)


def _crop_from_box(box: AnalysisBox) -> CropBounds:
    return CropBounds(r0=int(box.r0), r1=int(box.r1), c0=int(box.c0), c1=int(box.c1))


def _coords_bounds(rows: np.ndarray, cols: np.ndarray) -> CropBounds:
    if rows.size <= 0 or cols.size <= 0:
        raise ValueError("cannot build bounds from empty coordinates")
    return CropBounds(
        r0=int(np.min(rows)),
        r1=int(np.max(rows)) + 1,
        c0=int(np.min(cols)),
        c1=int(np.max(cols)) + 1,
    )


def _union_bounds(bounds: Iterable[CropBounds]) -> CropBounds:
    bounds_list = list(bounds)
    if not bounds_list:
        raise ValueError("cannot union empty bounds list")
    return CropBounds(
        r0=min(int(item.r0) for item in bounds_list),
        r1=max(int(item.r1) for item in bounds_list),
        c0=min(int(item.c0) for item in bounds_list),
        c1=max(int(item.c1) for item in bounds_list),
    )


def _pad_bounds(bounds: CropBounds, shape: tuple[int, int], *, pad: int) -> CropBounds:
    pad_use = max(0, int(pad))
    return CropBounds(
        r0=max(0, int(bounds.r0) - pad_use),
        r1=min(int(shape[0]), int(bounds.r1) + pad_use),
        c0=max(0, int(bounds.c0) - pad_use),
        c1=min(int(shape[1]), int(bounds.c1) + pad_use),
    )


def _bounds_for_block(block: UnknownBlock) -> CropBounds:
    return _coords_bounds(np.asarray(block.rows, dtype=np.int32), np.asarray(block.cols, dtype=np.int32))


def _bounds_for_cluster(cluster: FrontierCluster) -> CropBounds:
    return _coords_bounds(np.asarray(cluster.rows, dtype=np.int32), np.asarray(cluster.cols, dtype=np.int32))


def _bounds_for_support(cluster: FrontierCluster) -> CropBounds:
    return CropBounds(
        r0=int(cluster.support_geometry.local_box_r0),
        r1=int(cluster.support_geometry.local_box_r1),
        c0=int(cluster.support_geometry.local_box_c0),
        c1=int(cluster.support_geometry.local_box_c1),
    )


def _crop_belief(snapshot: Snapshot, crop: CropBounds) -> np.ndarray:
    return np.asarray(snapshot.belief_map[crop.r0:crop.r1, crop.c0:crop.c1], dtype=np.int8)


def _mask_from_coords(rows: np.ndarray, cols: np.ndarray, crop: CropBounds) -> np.ndarray:
    mask = np.zeros(crop.shape, dtype=bool)
    if rows.size <= 0 or cols.size <= 0:
        return mask
    local_rows = np.asarray(rows, dtype=np.int32) - int(crop.r0)
    local_cols = np.asarray(cols, dtype=np.int32) - int(crop.c0)
    inside = (
        (local_rows >= 0)
        & (local_rows < int(crop.shape[0]))
        & (local_cols >= 0)
        & (local_cols < int(crop.shape[1]))
    )
    if np.any(inside):
        mask[local_rows[inside], local_cols[inside]] = True
    return mask


def _cluster_local_anchor(cluster: FrontierCluster, crop: CropBounds) -> tuple[float, float]:
    return (
        float(int(cluster.frontier_anchor_rc[0]) - int(crop.r0)),
        float(int(cluster.frontier_anchor_rc[1]) - int(crop.c0)),
    )


def _render_base_map(ax, belief_crop: np.ndarray) -> None:
    ax.imshow(_belief_rgba(belief_crop), origin="upper", interpolation="nearest")


def _overlay_mask(ax, mask: np.ndarray, *, color: str, alpha: float) -> None:
    if not np.any(mask):
        return
    rgba = np.zeros((*mask.shape, 4), dtype=np.float32)
    rgba[..., :] = np.asarray(to_rgba(color), dtype=np.float32)
    rgba[..., 3] = 0.0
    rgba[mask, 3] = float(alpha)
    ax.imshow(rgba, origin="upper", interpolation="nearest")


def _recent_trajectory_window(trajectory_world: np.ndarray, length: int = DEFAULT_SEMANTIC_TRAJECTORY_LENGTH) -> np.ndarray:
    trajectory = np.asarray(trajectory_world, dtype=np.int32)
    if trajectory.ndim != 2 or trajectory.shape[1] != 2:
        trajectory = trajectory.reshape((-1, 2))
    max_points = max(0, int(length)) + 1
    if trajectory.shape[0] <= max_points:
        return trajectory.copy()
    return trajectory[-max_points:].copy()


def _draw_cropped_trajectory_and_agent(
    ax,
    snapshot: Snapshot,
    crop: CropBounds,
    *,
    trajectory_world: np.ndarray | None = None,
) -> None:
    trajectory = (
        _recent_trajectory_window(snapshot.trajectory_world)
        if trajectory_world is None
        else np.asarray(trajectory_world, dtype=np.int32).reshape((-1, 2))
    )
    if trajectory.size > 0:
        origin_r, origin_c = int(snapshot.belief_origin_world[0]), int(snapshot.belief_origin_world[1])
        rows = trajectory[:, 0].astype(np.float32) - float(origin_r) - float(crop.r0)
        cols = trajectory[:, 1].astype(np.float32) - float(origin_c) - float(crop.c0)
        inside = (
            (rows >= -0.5)
            & (rows <= float(crop.shape[0]) - 0.5)
            & (cols >= -0.5)
            & (cols <= float(crop.shape[1]) - 0.5)
        )
        start = 0
        for idx in range(int(inside.size) + 1):
            if idx < int(inside.size) and bool(inside[idx]):
                continue
            if idx - start > 1:
                _draw_trajectory(ax, rows[start:idx], cols[start:idx], zorder=6)
            start = idx + 1

    agent_r = float(int(snapshot.agent_array[0]) - int(crop.r0))
    agent_c = float(int(snapshot.agent_array[1]) - int(crop.c0))
    if -0.5 <= agent_r <= float(crop.shape[0]) - 0.5 and -0.5 <= agent_c <= float(crop.shape[1]) - 0.5:
        _draw_agent(ax, row=agent_r, col=agent_c, zorder=7)


def _add_local_bounds_box(
    ax,
    bounds: CropBounds,
    crop: CropBounds,
    *,
    edge_color: str = UNKNOWN_BLOCK_BOX_COLOR,
    linewidth: float = 1.15,
) -> None:
    local_r0 = float(int(bounds.r0) - int(crop.r0)) - 0.5
    local_c0 = float(int(bounds.c0) - int(crop.c0)) - 0.5
    width = float(int(bounds.c1) - int(bounds.c0))
    height = float(int(bounds.r1) - int(bounds.r0))
    if width <= 0.0 or height <= 0.0:
        return
    ax.add_patch(
        Rectangle(
            (local_c0, local_r0),
            width,
            height,
            fill=False,
            edgecolor=edge_color,
            linewidth=linewidth,
            linestyle="-",
            alpha=0.92,
            zorder=5,
        )
    )


def _frontier_crop(scene: SemanticExportScene, crop: CropBounds) -> np.ndarray:
    return np.asarray(scene.frontier_mask[crop.r0 : crop.r1, crop.c0 : crop.c1], dtype=bool)


def _add_summary_card(
    ax,
    *,
    anchor_xy: tuple[float, float],
    lines: list[str],
    edge_color: str,
    style: SharedSemanticAssetStyle,
) -> None:
    anchor_x = float(anchor_xy[0])
    anchor_y = float(anchor_xy[1])
    line_height = 0.92
    box_height = max(1.8, 0.68 + (len(lines) * line_height))
    text_width = max(4.6, max((len(line) for line in lines), default=0) * 0.18)
    box_width = float(text_width)
    patch = FancyBboxPatch(
        (anchor_x, anchor_y),
        box_width,
        box_height,
        boxstyle="round,pad=0.18,rounding_size=0.24",
        linewidth=1.0,
        edgecolor=edge_color,
        facecolor=SUMMARY_CARD_FILL,
        alpha=0.95,
        zorder=9,
    )
    ax.add_patch(patch)
    for line_idx, line in enumerate(lines):
        ax.text(
            anchor_x + 0.34,
            anchor_y + 0.64 + (line_idx * line_height),
            line,
            fontsize=float(style.summary_font_size),
            color=SUMMARY_CARD_TEXT,
            ha="left",
            va="center",
            zorder=10,
        )


def _focus_block(blocks: tuple[UnknownBlock, ...]) -> UnknownBlock:
    if not blocks:
        raise RuntimeError("shared semantic scene has no accessible unknown block")
    return max(
        blocks,
        key=lambda block: (
            int(block.frontier_cluster_count > 1),
            int(block.frontier_cluster_count),
            int(block.block_area),
            -int(block.block_index),
        ),
    )


def _focus_cluster(block: UnknownBlock) -> FrontierCluster:
    if not block.frontier_clusters:
        raise RuntimeError("focus block has no frontier cluster")
    return max(
        block.frontier_clusters,
        key=lambda cluster: (
            float(cluster.entry_width),
            -float(cluster.anchor_distance),
            -int(cluster.frontier_index),
        ),
    )


def _scene_score(semantic_snapshot: SharedSemanticSnapshot) -> tuple[int, int, int, int]:
    blocks = tuple(semantic_snapshot.accessible_blocks)
    total_clusters = int(sum(block.frontier_cluster_count for block in blocks))
    focus_clusters = max((int(block.frontier_cluster_count) for block in blocks), default=0)
    return (
        int(bool(blocks and total_clusters > 0)),
        int(focus_clusters),
        int(total_clusters),
        int(semantic_snapshot.total_accessible_unknown_area),
    )


def _build_scene(
    *,
    seed: int,
    requested_step: int,
    resolved_step: int,
    snapshot: Snapshot,
    frontier_mask: np.ndarray,
    semantic_snapshot: SharedSemanticSnapshot,
) -> SemanticExportScene:
    focus_block = _focus_block(tuple(semantic_snapshot.accessible_blocks))
    focus_cluster = _focus_cluster(focus_block)
    return SemanticExportScene(
        seed=int(seed),
        requested_step=int(requested_step),
        resolved_step=int(resolved_step),
        snapshot=snapshot,
        frontier_mask=np.asarray(frontier_mask, dtype=bool).copy(),
        semantic_snapshot=semantic_snapshot,
        focus_block=focus_block,
        focus_cluster=focus_cluster,
    )


def _collect_semantic_scene(config: ExportConfig, *, step: int | None = None) -> SemanticExportScene:
    target_step = int(config.step_mid if step is None else step)
    if target_step < 0:
        raise ValueError("step must be >= 0")

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
    layer = SharedSemanticLayer()

    agent_state = (int(start_state[0]), int(start_state[1]))
    trajectory_world = [agent_state]
    visit_counts: dict[tuple[int, int], int] = {agent_state: 1}
    rollout_horizon = max(int(config.step_late), int(target_step))

    chosen_scene: SemanticExportScene | None = None
    fallback_scene: SemanticExportScene | None = None
    fallback_score: tuple[int, int, int, int] | None = None

    for step_idx in range(0, rollout_horizon + 1):
        semantic_snapshot = layer.analyze(cum_map, agent_state)
        if semantic_snapshot.accessible_blocks:
            scene = _build_scene(
                seed=int(config.seed),
                requested_step=target_step,
                resolved_step=step_idx,
                snapshot=_capture_snapshot(
                    step=step_idx,
                    agent_state=agent_state,
                    trajectory_world=trajectory_world,
                    local_snap=local_snap,
                    cum_map=cum_map,
                ),
                frontier_mask=cum_map.compute_analysis_box_frontier_bool(),
                semantic_snapshot=semantic_snapshot,
            )
            scene_score = _scene_score(semantic_snapshot)
            if step_idx == target_step:
                chosen_scene = scene
            if fallback_score is None or scene_score > fallback_score:
                fallback_scene = scene
                fallback_score = scene_score

        if step_idx >= rollout_horizon:
            break

        planned_key = FIXED_ACTION_PREFERENCES[step_idx % len(FIXED_ACTION_PREFERENCES)]
        desired_action = int(KEY_TO_ACTION[planned_key])
        valid_actions = GridTopology.valid_action_indices_fast(free_mask, agent_state)
        if not valid_actions:
            raise RuntimeError(f"agent has no legal moves at step {step_idx + 1}")
        if desired_action in valid_actions:
            chosen_action = desired_action
        else:
            chosen_action = _select_fallback_action(
                valid_actions,
                agent_state=agent_state,
                visit_counts=visit_counts,
            )

        dr, dc = ACTIONS_8[chosen_action]
        agent_state = (int(agent_state[0] + dr), int(agent_state[1] + dc))
        visit_counts[agent_state] = int(visit_counts.get(agent_state, 0) + 1)
        trajectory_world.append(agent_state)

        local_snap = np.asarray(obs_model.observe_fast(agent_state), dtype=np.int8).copy()
        cum_map.update(agent_state, local_snap)

    if chosen_scene is not None:
        return chosen_scene
    if fallback_scene is not None:
        return fallback_scene
    raise RuntimeError("unable to find a rollout step with shared semantic content")


def _cluster_count(scene: SemanticExportScene) -> int:
    return int(sum(block.frontier_cluster_count for block in scene.semantic_snapshot.accessible_blocks))


def _export_semantic_input_belief_map(
    path: Path,
    scene: SemanticExportScene,
    *,
    style: SharedSemanticAssetStyle,
    trajectory_world: np.ndarray | None = None,
) -> None:
    crop = _crop_from_box(scene.semantic_snapshot.analysis_box)
    belief_crop = _crop_belief(scene.snapshot, crop)
    fig, ax = _create_asset_axes(crop.shape, style=style)
    _render_base_map(ax, belief_crop)
    _overlay_mask(ax, _frontier_crop(scene, crop), color=RAW_FRONTIER_COLOR, alpha=float(style.frontier_alpha))
    _draw_cropped_trajectory_and_agent(ax, scene.snapshot, crop, trajectory_world=trajectory_world)
    _save_asset_figure(fig, path, dpi=style.dpi)


def _export_frontier_parsing_overlay(
    path: Path,
    scene: SemanticExportScene,
    *,
    style: SharedSemanticAssetStyle,
) -> None:
    crop = _crop_from_box(scene.semantic_snapshot.analysis_box)
    belief_crop = _crop_belief(scene.snapshot, crop)
    fig, ax = _create_asset_axes(crop.shape, style=style)
    _render_base_map(ax, belief_crop)

    for block in scene.semantic_snapshot.accessible_blocks:
        _add_local_bounds_box(ax, _bounds_for_block(block), crop)
    _overlay_mask(ax, _frontier_crop(scene, crop), color=RAW_FRONTIER_COLOR, alpha=float(style.frontier_alpha))

    _save_asset_figure(fig, path, dpi=style.dpi)


def _export_unknown_block(
    path: Path,
    scene: SemanticExportScene,
    *,
    style: SharedSemanticAssetStyle,
) -> None:
    focus_block = scene.focus_block
    crop = _pad_bounds(
        _union_bounds([_bounds_for_block(focus_block)] + [_bounds_for_cluster(cluster) for cluster in focus_block.frontier_clusters]),
        scene.snapshot.belief_map.shape,
        pad=2,
    )
    belief_crop = _crop_belief(scene.snapshot, crop)
    fig, ax = _create_asset_axes(crop.shape, style=style)
    _render_base_map(ax, belief_crop)

    block_color = BLOCK_PALETTE[0]
    _overlay_mask(
        ax,
        _mask_from_coords(np.asarray(focus_block.rows), np.asarray(focus_block.cols), crop),
        color=block_color,
        alpha=float(style.unknown_block_alpha),
    )
    for cluster_idx, cluster in enumerate(focus_block.frontier_clusters):
        frontier_color = CLUSTER_PALETTE[int(cluster_idx) % len(CLUSTER_PALETTE)]
        _overlay_mask(
            ax,
            _mask_from_coords(np.asarray(cluster.rows), np.asarray(cluster.cols), crop),
            color=frontier_color,
            alpha=float(style.frontier_alpha),
        )

    _save_asset_figure(fig, path, dpi=style.dpi)


def _export_frontier_cluster(
    path: Path,
    scene: SemanticExportScene,
    *,
    style: SharedSemanticAssetStyle,
) -> None:
    focus_cluster = scene.focus_cluster
    crop = _pad_bounds(_bounds_for_cluster(focus_cluster), scene.snapshot.belief_map.shape, pad=3)
    belief_crop = _crop_belief(scene.snapshot, crop)
    fig, ax = _create_asset_axes(crop.shape, style=style)
    _render_base_map(ax, belief_crop)

    cluster_color = CLUSTER_PALETTE[0]
    _overlay_mask(
        ax,
        _mask_from_coords(np.asarray(focus_cluster.rows), np.asarray(focus_cluster.cols), crop),
        color=cluster_color,
        alpha=float(style.frontier_alpha),
    )
    anchor_r, anchor_c = _cluster_local_anchor(focus_cluster, crop)
    ax.scatter(
        [anchor_c],
        [anchor_r],
        marker="o",
        s=28,
        c="#fff7d6",
        edgecolors=cluster_color,
        linewidths=0.9,
        zorder=9,
    )
    _save_asset_figure(fig, path, dpi=style.dpi)


def _summary_anchor(crop: CropBounds, focus_box: CropBounds) -> tuple[float, float]:
    local_box_right = float(focus_box.c1 - crop.c0)
    local_box_top = float(focus_box.r0 - crop.r0)
    x = min(float(crop.shape[1]) - 5.4, local_box_right + 0.35)
    y = max(0.35, local_box_top - 0.15)
    return max(0.35, x), min(float(crop.shape[0]) - 2.4, y)


def _export_local_attribute_summary(
    path: Path,
    scene: SemanticExportScene,
    *,
    style: SharedSemanticAssetStyle,
) -> None:
    focus_cluster = scene.focus_cluster
    support_bounds = _bounds_for_support(focus_cluster)
    crop = _pad_bounds(
        _union_bounds([_bounds_for_cluster(focus_cluster), support_bounds]),
        scene.snapshot.belief_map.shape,
        pad=2,
    )
    belief_crop = _crop_belief(scene.snapshot, crop)
    fig, ax = _create_asset_axes(crop.shape, style=style)
    _render_base_map(ax, belief_crop)

    cluster_color = CLUSTER_PALETTE[0]
    _overlay_mask(
        ax,
        _mask_from_coords(np.asarray(focus_cluster.rows), np.asarray(focus_cluster.cols), crop),
        color=cluster_color,
        alpha=float(style.frontier_alpha),
    )

    support_mask = _mask_from_coords(
        np.asarray(focus_cluster.support_rows, dtype=np.int32),
        np.asarray(focus_cluster.support_cols, dtype=np.int32),
        crop,
    )
    _overlay_mask(
        ax,
        support_mask,
        color=SUMMARY_BOX_COLOR,
        alpha=float(style.support_fill_alpha),
    )

    local_box = CropBounds(
        r0=int(support_bounds.r0 - crop.r0),
        r1=int(support_bounds.r1 - crop.r0),
        c0=int(support_bounds.c0 - crop.c0),
        c1=int(support_bounds.c1 - crop.c0),
    )
    ax.add_patch(
        Rectangle(
            (float(local_box.c0) - 0.5, float(local_box.r0) - 0.5),
            float(local_box.c1 - local_box.c0),
            float(local_box.r1 - local_box.r0),
            fill=False,
            edgecolor=SUMMARY_BOX_COLOR,
            linewidth=1.3,
            linestyle="-",
            alpha=0.92,
            zorder=8,
        )
    )

    summary_lines = [
        f"entry width  {int(round(float(focus_cluster.entry_width)))}",
        f"obs density  {float(focus_cluster.support_obstacle_density):.2f}",
    ]
    card_x, card_y = _summary_anchor(crop, support_bounds)
    _add_summary_card(
        ax,
        anchor_xy=(card_x, card_y),
        lines=summary_lines,
        edge_color=SUMMARY_BOX_COLOR,
        style=style,
    )

    support_center_c = float(local_box.c0 + local_box.c1 - 1) / 2.0
    support_center_r = float(local_box.r0 + local_box.r1 - 1) / 2.0
    ax.plot(
        [card_x, support_center_c],
        [card_y + 1.2, support_center_r],
        color=SUMMARY_BOX_COLOR,
        linewidth=1.0,
        alpha=0.76,
        zorder=8,
    )
    _save_asset_figure(fig, path, dpi=style.dpi)


def _export_shared_semantic_states(
    path: Path,
    scene: SemanticExportScene,
    *,
    style: SharedSemanticAssetStyle,
) -> None:
    focus_cluster = scene.focus_cluster
    focus_block = scene.focus_block
    support_bounds = _bounds_for_support(focus_cluster)
    crop = _pad_bounds(
        _union_bounds([_bounds_for_block(focus_block), support_bounds]),
        scene.snapshot.belief_map.shape,
        pad=2,
    )
    belief_crop = _crop_belief(scene.snapshot, crop)
    fig, ax = _create_asset_axes(crop.shape, style=style)
    _render_base_map(ax, belief_crop)

    _overlay_mask(
        ax,
        _mask_from_coords(np.asarray(focus_block.rows), np.asarray(focus_block.cols), crop),
        color=BLOCK_PALETTE[0],
        alpha=0.24,
    )
    _overlay_mask(
        ax,
        _mask_from_coords(np.asarray(focus_cluster.rows), np.asarray(focus_cluster.cols), crop),
        color=CLUSTER_PALETTE[0],
        alpha=float(style.frontier_alpha),
    )

    metrics = scene.semantic_snapshot.metrics()
    state_lines = [
        f"blocks    {int(metrics['accessible_block_count'])}",
        f"clusters  {_cluster_count(scene)}",
        f"unknown   {int(metrics['total_accessible_unknown_area'])}",
    ]
    _add_summary_card(
        ax,
        anchor_xy=(0.45, 0.45),
        lines=state_lines,
        edge_color=ANALYSIS_BOX_COLOR,
        style=style,
    )
    _save_asset_figure(fig, path, dpi=style.dpi)


def _build_manifest(outputs: dict[str, Path], scene: SemanticExportScene) -> dict[str, object]:
    return {
        "scene": {
            "seed": int(scene.seed),
            "requested_step": int(scene.requested_step),
            "resolved_step": int(scene.resolved_step),
            "analysis_box": {
                "r0": int(scene.semantic_snapshot.analysis_box.r0),
                "r1": int(scene.semantic_snapshot.analysis_box.r1),
                "c0": int(scene.semantic_snapshot.analysis_box.c0),
                "c1": int(scene.semantic_snapshot.analysis_box.c1),
                "margin": int(scene.semantic_snapshot.analysis_box.margin),
            },
            "focus_block_index": int(scene.focus_block.block_index),
            "focus_frontier_index": int(scene.focus_cluster.frontier_index),
            "focus_block_frontier_cluster_count": int(scene.focus_block.frontier_cluster_count),
            "focus_block_area": int(scene.focus_block.block_area),
        },
        "files": {
            "semantic_input_belief_map.png": {
                "path": _format_output_path(outputs["semantic_input_belief_map"]),
                "paper_node": "Dynamic Cumulative Belief Map + Raw Frontier",
                "render_source": "real_code_data",
                "data_basis": ["belief_map", "analysis_box", "raw_frontier", "recent_trajectory", "agent_state"],
            },
            "frontier_parsing_overlay.png": {
                "path": _format_output_path(outputs["frontier_parsing_overlay"]),
                "paper_node": "Frontier-first Semantic Parsing",
                "render_source": "real_code_data",
                "data_basis": ["belief_map", "analysis_box", "raw_frontier", "unknown_blocks"],
                "note": "Raw frontier is shown before clustered semantic outputs; unknown blocks are framed with uniform local boxes, and debug-style center links are not rendered.",
            },
            "unknown_block.png": {
                "path": _format_output_path(outputs["unknown_block"]),
                "paper_node": "Unknown Block",
                "render_source": "real_code_data",
                "data_basis": ["focus_unknown_block", "frontier_clusters"],
            },
            "frontier_cluster.png": {
                "path": _format_output_path(outputs["frontier_cluster"]),
                "paper_node": "Frontier Cluster",
                "render_source": "real_code_data",
                "data_basis": ["focus_frontier_cluster"],
            },
            "local_attribute_summary.png": {
                "path": _format_output_path(outputs["local_attribute_summary"]),
                "paper_node": "Local Attribute Summary",
                "render_source": "real_code_data_with_summary_card",
                "data_basis": ["focus_frontier_cluster", "support_local_box", "support_obstacle_density", "entry_width"],
                "note": "The card is a compact visualization of real local summary attributes rather than a support dilation region.",
            },
        },
    }


def export_shared_semantic_layer_assets(
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    *,
    config: ExportConfig | None = None,
    step: int | None = None,
    include_shared_semantic_states: bool = True,
) -> dict[str, object]:
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    _clear_old_png_outputs(output_dir_path)

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
    scene = _collect_semantic_scene(rollout_config, step=step)
    style = SharedSemanticAssetStyle(dpi=int(rollout_config.dpi))

    outputs: dict[str, Path] = {
        "semantic_input_belief_map": output_dir_path / "semantic_input_belief_map.png",
        "frontier_parsing_overlay": output_dir_path / "frontier_parsing_overlay.png",
        "unknown_block": output_dir_path / "unknown_block.png",
        "frontier_cluster": output_dir_path / "frontier_cluster.png",
        "local_attribute_summary": output_dir_path / "local_attribute_summary.png",
    }

    _export_semantic_input_belief_map(outputs["semantic_input_belief_map"], scene, style=style)
    _export_frontier_parsing_overlay(outputs["frontier_parsing_overlay"], scene, style=style)
    _export_unknown_block(outputs["unknown_block"], scene, style=style)
    _export_frontier_cluster(outputs["frontier_cluster"], scene, style=style)
    _export_local_attribute_summary(outputs["local_attribute_summary"], scene, style=style)

    if include_shared_semantic_states:
        outputs["shared_semantic_states"] = output_dir_path / "shared_semantic_states.png"
        _export_shared_semantic_states(outputs["shared_semantic_states"], scene, style=style)

    manifest = _build_manifest(outputs, scene)
    if include_shared_semantic_states and "shared_semantic_states" in outputs:
        manifest["files"]["shared_semantic_states.png"] = {
            "path": _format_output_path(outputs["shared_semantic_states"]),
            "paper_node": "Shared Semantic States",
            "render_source": "real_code_data_with_compact_state_card",
            "data_basis": ["focus_unknown_block", "focus_frontier_cluster", "semantic_metrics"],
        }

    manifest_path = output_dir_path / "shared_semantic_layer_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return {
        "scene": manifest["scene"],
        "files": outputs,
        "manifest_path": manifest_path,
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export paper-ready shared semantic layer PNG assets.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rows", type=int, default=40)
    parser.add_argument("--cols", type=int, default=60)
    parser.add_argument("--obstacle-ratio", type=float, default=0.20)
    parser.add_argument("--obs-size", type=int, default=6)
    parser.add_argument("--scan-radius", type=int, default=10)
    parser.add_argument("--step-mid", type=int, default=4)
    parser.add_argument("--step-late", type=int, default=8)
    parser.add_argument(
        "--step",
        type=int,
        default=None,
        help="Target rollout step for semantic export. Defaults to --step-mid. If that step has no semantic content, the exporter falls back to the best available step.",
    )
    parser.add_argument("--dpi", type=int, default=260)
    parser.add_argument(
        "--no-shared-semantic-states",
        dest="include_shared_semantic_states",
        action="store_false",
        help="Skip exporting shared_semantic_states.png.",
    )
    parser.set_defaults(include_shared_semantic_states=True)
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
    result = export_shared_semantic_layer_assets(
        args.output_dir,
        config=config,
        step=args.step,
        include_shared_semantic_states=bool(args.include_shared_semantic_states),
    )
    scene = result["scene"]
    print("mode=shared-semantic")
    print(f"seed={scene['seed']}")
    print(f"requested_step={scene['requested_step']}")
    print(f"resolved_step={scene['resolved_step']}")
    for name, path in result["files"].items():
        print(f"{name}={_format_output_path(path)}")
    print(f"manifest={_format_output_path(result['manifest_path'])}")


if __name__ == "__main__":
    cli_main()
