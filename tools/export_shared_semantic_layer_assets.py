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
    MethodFigureStyle,
    Snapshot,
    _capture_snapshot,
    _clear_old_png_outputs,
    _create_method_axis,
    _draw_agent,
    _draw_scan_circle,
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
TRAJECTORY_DECAY_COLOR = "#2d6a8c"
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
    belief_frontier_mask: np.ndarray
    semantic_snapshot: SharedSemanticSnapshot
    focus_block: UnknownBlock
    focus_cluster: FrontierCluster


@dataclass(frozen=True, slots=True)
class SemanticFullCanvas:
    belief_map: np.ndarray
    raw_frontier_mask: np.ndarray
    trajectory_array: np.ndarray
    analysis_crop: CropBounds


@dataclass(frozen=True, slots=True)
class SemanticAnalysisWindow:
    crop: CropBounds
    belief_crop: np.ndarray
    raw_frontier_crop: np.ndarray
    trajectory_rows: np.ndarray
    trajectory_cols: np.ndarray


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


def _format_asset_axis(ax, shape: tuple[int, int]) -> None:
    ax.set_aspect("equal")
    ax.set_xlim(-0.5, float(shape[1]) - 0.5)
    ax.set_ylim(float(shape[0]) - 0.5, -0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def _create_local_asset_axes(shape: tuple[int, int]):
    fig, ax = _create_method_axis(shape, style=MethodFigureStyle())
    _format_asset_axis(ax, shape)
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


def _save_asset_figure(
    fig: plt.Figure,
    path: Path,
    *,
    dpi: int,
    facecolor: str = "white",
    trim: bool = True,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=int(dpi), pad_inches=0.0, bbox_inches=None, facecolor=facecolor)
    plt.close(fig)
    if trim:
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


def _fixed_centered_crop(
    array: np.ndarray,
    *,
    center_rc: tuple[int, int],
    radius: int,
    fill_value,
) -> np.ndarray:
    source = np.asarray(array)
    radius_use = max(0, int(radius))
    size = (2 * radius_use) + 1
    out = np.full((size, size), fill_value, dtype=source.dtype)

    center_r, center_c = int(center_rc[0]), int(center_rc[1])
    desired_r0 = center_r - radius_use
    desired_r1 = center_r + radius_use + 1
    desired_c0 = center_c - radius_use
    desired_c1 = center_c + radius_use + 1

    src_r0 = max(0, desired_r0)
    src_r1 = min(int(source.shape[0]), desired_r1)
    src_c0 = max(0, desired_c0)
    src_c1 = min(int(source.shape[1]), desired_c1)
    if src_r0 >= src_r1 or src_c0 >= src_c1:
        return out

    dst_r0 = src_r0 - desired_r0
    dst_r1 = dst_r0 + (src_r1 - src_r0)
    dst_c0 = src_c0 - desired_c0
    dst_c1 = dst_c0 + (src_c1 - src_c0)
    out[dst_r0:dst_r1, dst_c0:dst_c1] = source[src_r0:src_r1, src_c0:src_c1]
    return out


def _fixed_centered_crop_shape(
    array: np.ndarray,
    *,
    center_rc: tuple[int, int],
    shape: tuple[int, int],
    fill_value,
) -> np.ndarray:
    source = np.asarray(array)
    rows = max(1, int(shape[0]))
    cols = max(1, int(shape[1]))
    out = np.full((rows, cols), fill_value, dtype=source.dtype)

    center_r, center_c = int(center_rc[0]), int(center_rc[1])
    half_before_r = rows // 2
    half_before_c = cols // 2
    desired_r0 = center_r - half_before_r
    desired_r1 = desired_r0 + rows
    desired_c0 = center_c - half_before_c
    desired_c1 = desired_c0 + cols

    src_r0 = max(0, desired_r0)
    src_r1 = min(int(source.shape[0]), desired_r1)
    src_c0 = max(0, desired_c0)
    src_c1 = min(int(source.shape[1]), desired_c1)
    if src_r0 >= src_r1 or src_c0 >= src_c1:
        return out

    dst_r0 = src_r0 - desired_r0
    dst_r1 = dst_r0 + (src_r1 - src_r0)
    dst_c0 = src_c0 - desired_c0
    dst_c1 = dst_c0 + (src_c1 - src_c0)
    out[dst_r0:dst_r1, dst_c0:dst_c1] = source[src_r0:src_r1, src_c0:src_c1]
    return out


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
    show_agent: bool = True,
    show_trajectory: bool = True,
) -> None:
    trajectory = (
        _recent_trajectory_window(snapshot.trajectory_world)
        if trajectory_world is None
        else np.asarray(trajectory_world, dtype=np.int32).reshape((-1, 2))
    )
    if show_trajectory and trajectory.size > 0:
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
    if show_agent and -0.5 <= agent_r <= float(crop.shape[0]) - 0.5 and -0.5 <= agent_c <= float(crop.shape[1]) - 0.5:
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


def _iter_frontier_clusters(scene: SemanticExportScene) -> list[FrontierCluster]:
    clusters: list[FrontierCluster] = []
    for block in scene.semantic_snapshot.accessible_blocks:
        clusters.extend(list(block.frontier_clusters))
    return clusters


def _render_frontier_clusters(
    ax,
    scene: SemanticExportScene,
    crop: CropBounds,
    *,
    style: SharedSemanticAssetStyle,
    alpha: float | None = None,
) -> None:
    alpha_use = float(style.frontier_alpha if alpha is None else alpha)
    for cluster_idx, cluster in enumerate(_iter_frontier_clusters(scene)):
        cluster_color = CLUSTER_PALETTE[int(cluster_idx) % len(CLUSTER_PALETTE)]
        _overlay_mask(
            ax,
            _mask_from_coords(np.asarray(cluster.rows, dtype=np.int32), np.asarray(cluster.cols, dtype=np.int32), crop),
            color=cluster_color,
            alpha=alpha_use,
        )


def _render_cluster_analysis_boxes(
    ax,
    scene: SemanticExportScene,
    crop: CropBounds,
    *,
    style: SharedSemanticAssetStyle,
) -> None:
    _render_frontier_clusters(ax, scene, crop, style=style, alpha=0.68)
    for cluster in _iter_frontier_clusters(scene):
        _add_local_bounds_box(
            ax,
            _bounds_for_support(cluster),
            crop,
            edge_color=UNKNOWN_BLOCK_BOX_COLOR,
            linewidth=1.05,
        )


def _frontier_crop(scene: SemanticExportScene, crop: CropBounds) -> np.ndarray:
    return np.asarray(scene.frontier_mask[crop.r0 : crop.r1, crop.c0 : crop.c1], dtype=bool)


def _belief_frontier_mask(scene: SemanticExportScene) -> np.ndarray:
    return np.asarray(getattr(scene, "belief_frontier_mask", scene.frontier_mask), dtype=bool)


def _snapshot_trajectory_array(snapshot: Snapshot, trajectory_world: np.ndarray) -> np.ndarray:
    trajectory = np.asarray(trajectory_world, dtype=np.int32)
    if trajectory.ndim != 2 or trajectory.shape[1] != 2:
        trajectory = trajectory.reshape((-1, 2))
    if trajectory.size <= 0:
        return np.zeros((0, 2), dtype=np.float32)
    origin_r, origin_c = int(snapshot.belief_origin_world[0]), int(snapshot.belief_origin_world[1])
    rows = trajectory[:, 0].astype(np.float32) - float(origin_r)
    cols = trajectory[:, 1].astype(np.float32) - float(origin_c)
    return np.stack([rows, cols], axis=1)


def _build_semantic_full_canvas(
    scene: SemanticExportScene,
    *,
    trajectory_world: np.ndarray | None = None,
    trajectory_length: int = DEFAULT_SEMANTIC_TRAJECTORY_LENGTH,
    show_trajectory: bool = True,
) -> SemanticFullCanvas:
    belief_map = np.asarray(scene.snapshot.belief_map, dtype=np.int8).copy()
    raw_frontier_mask = _belief_frontier_mask(scene).copy()
    if tuple(raw_frontier_mask.shape) != tuple(belief_map.shape):
        raise ValueError(
            f"raw frontier mask shape mismatch: expected {belief_map.shape}, got {raw_frontier_mask.shape}"
        )

    trajectory_source = (
        _recent_trajectory_window(scene.snapshot.trajectory_world, length=int(trajectory_length))
        if trajectory_world is None
        else np.asarray(trajectory_world, dtype=np.int32).reshape((-1, 2))
    )
    trajectory_array = (
        _snapshot_trajectory_array(scene.snapshot, trajectory_source)
        if show_trajectory
        else np.zeros((0, 2), dtype=np.float32)
    )
    crop = _crop_from_box(scene.semantic_snapshot.analysis_box)
    rows, cols = int(belief_map.shape[0]), int(belief_map.shape[1])
    if int(crop.r0) < 0 or int(crop.c0) < 0 or int(crop.r1) > rows or int(crop.c1) > cols:
        raise ValueError(f"analysis crop {crop} is outside belief map shape {belief_map.shape}")

    return SemanticFullCanvas(
        belief_map=belief_map,
        raw_frontier_mask=raw_frontier_mask,
        trajectory_array=np.asarray(trajectory_array, dtype=np.float32).copy(),
        analysis_crop=crop,
    )


def _analysis_window_from_full_canvas(full_canvas: SemanticFullCanvas) -> SemanticAnalysisWindow:
    crop = full_canvas.analysis_crop
    trajectory_array = np.asarray(full_canvas.trajectory_array, dtype=np.float32)
    if trajectory_array.shape[0] > 0:
        trajectory_rows = trajectory_array[:, 0].astype(np.float32, copy=True) - float(crop.r0)
        trajectory_cols = trajectory_array[:, 1].astype(np.float32, copy=True) - float(crop.c0)
    else:
        trajectory_rows = np.zeros((0,), dtype=np.float32)
        trajectory_cols = np.zeros((0,), dtype=np.float32)

    return SemanticAnalysisWindow(
        crop=crop,
        belief_crop=np.asarray(full_canvas.belief_map[crop.r0 : crop.r1, crop.c0 : crop.c1], dtype=np.int8).copy(),
        raw_frontier_crop=np.asarray(
            full_canvas.raw_frontier_mask[crop.r0 : crop.r1, crop.c0 : crop.c1],
            dtype=bool,
        ).copy(),
        trajectory_rows=trajectory_rows,
        trajectory_cols=trajectory_cols,
    )


def _build_semantic_analysis_window(
    scene: SemanticExportScene,
    *,
    trajectory_world: np.ndarray | None = None,
    trajectory_length: int = DEFAULT_SEMANTIC_TRAJECTORY_LENGTH,
    show_trajectory: bool = True,
) -> SemanticAnalysisWindow:
    full_canvas = _build_semantic_full_canvas(
        scene,
        trajectory_world=trajectory_world,
        trajectory_length=int(trajectory_length),
        show_trajectory=show_trajectory,
    )
    return _analysis_window_from_full_canvas(full_canvas)


def _draw_window_trajectory(
    ax,
    rows: np.ndarray,
    cols: np.ndarray,
    shape: tuple[int, int],
    *,
    zorder: int = 6,
) -> None:
    rows_arr = np.asarray(rows, dtype=np.float32)
    cols_arr = np.asarray(cols, dtype=np.float32)
    if rows_arr.size <= 0 or cols_arr.size <= 0:
        return
    inside = (
        (rows_arr >= -0.5)
        & (rows_arr <= float(int(shape[0])) - 0.5)
        & (cols_arr >= -0.5)
        & (cols_arr <= float(int(shape[1])) - 0.5)
    )
    start = 0
    for idx in range(int(inside.size) + 1):
        if idx < int(inside.size) and bool(inside[idx]):
            continue
        if idx - start > 1:
            _draw_trajectory(ax, rows_arr[start:idx], cols_arr[start:idx], zorder=zorder)
        start = idx + 1


def _centered_local_trajectory(
    snapshot: Snapshot,
    trajectory_world: np.ndarray,
    crop_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    trajectory_array = _snapshot_trajectory_array(snapshot, trajectory_world)
    if trajectory_array.shape[0] <= 0:
        return np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.float32)
    center_r = float(int(crop_shape[0]) // 2)
    center_c = float(int(crop_shape[1]) // 2)
    rows = trajectory_array[:, 0] - float(int(snapshot.agent_array[0])) + center_r
    cols = trajectory_array[:, 1] - float(int(snapshot.agent_array[1])) + center_c
    return rows.astype(np.float32), cols.astype(np.float32)


def _draw_centered_trajectory(
    ax,
    snapshot: Snapshot,
    trajectory_world: np.ndarray,
    crop_shape: tuple[int, int],
    *,
    zorder: int = 6,
) -> None:
    rows, cols = _centered_local_trajectory(snapshot, trajectory_world, crop_shape)
    if rows.size <= 1 or cols.size <= 1:
        return
    _draw_trajectory(ax, rows, cols, zorder=zorder)


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
    belief_frontier_mask: np.ndarray,
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
        belief_frontier_mask=np.asarray(belief_frontier_mask, dtype=bool).copy(),
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
                frontier_mask=np.asarray(cum_map.get_frontier_u8(refresh=False), dtype=np.uint8) > 0,
                belief_frontier_mask=np.asarray(cum_map.frontier_bool, dtype=bool),
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


def _export_belief_after_update_with_frontier(
    path: Path,
    scene: SemanticExportScene,
    *,
    style: SharedSemanticAssetStyle,
    full_canvas: SemanticFullCanvas | None = None,
    trajectory_world: np.ndarray | None = None,
    trajectory_length: int = DEFAULT_SEMANTIC_TRAJECTORY_LENGTH,
    show_agent: bool = False,
    show_scan_circle: bool = False,
    show_trajectory: bool = True,
    scan_radius: int | None = None,
) -> None:
    canvas = full_canvas or _build_semantic_full_canvas(
        scene,
        trajectory_world=trajectory_world,
        trajectory_length=int(trajectory_length),
        show_trajectory=show_trajectory,
    )
    fig, ax = _create_asset_axes(canvas.belief_map.shape, style=style)
    _render_base_map(ax, canvas.belief_map)
    _overlay_mask(ax, canvas.raw_frontier_mask, color=RAW_FRONTIER_COLOR, alpha=float(style.frontier_alpha))

    if show_trajectory and canvas.trajectory_array.shape[0] > 0:
        _draw_trajectory(ax, canvas.trajectory_array[:, 0], canvas.trajectory_array[:, 1], zorder=6)

    if show_scan_circle and scan_radius is not None:
        _draw_scan_circle(
            ax,
            center_row=float(scene.snapshot.agent_array[0]),
            center_col=float(scene.snapshot.agent_array[1]),
            radius=float(scan_radius),
            zorder=7,
        )
    if show_agent:
        _draw_agent(
            ax,
            row=float(scene.snapshot.agent_array[0]),
            col=float(scene.snapshot.agent_array[1]),
            zorder=8,
        )
    _save_asset_figure(fig, path, dpi=style.dpi)


def _export_semantic_input_analysis_crop(
    path: Path,
    scene: SemanticExportScene,
    *,
    style: SharedSemanticAssetStyle,
    analysis_window: SemanticAnalysisWindow | None = None,
    trajectory_world: np.ndarray | None = None,
    trajectory_length: int = DEFAULT_SEMANTIC_TRAJECTORY_LENGTH,
    show_agent: bool = False,
    show_scan_circle: bool = False,
    show_trajectory: bool = True,
    scan_radius: int | None = None,
) -> None:
    window = analysis_window or _build_semantic_analysis_window(
        scene,
        trajectory_world=trajectory_world,
        trajectory_length=int(trajectory_length),
        show_trajectory=show_trajectory,
    )
    crop = window.crop
    fig, ax = _create_asset_axes(crop.shape, style=style)
    _render_base_map(ax, window.belief_crop)
    _overlay_mask(ax, window.raw_frontier_crop, color=RAW_FRONTIER_COLOR, alpha=float(style.frontier_alpha))
    if show_trajectory:
        _draw_window_trajectory(ax, window.trajectory_rows, window.trajectory_cols, crop.shape, zorder=6)
    if show_agent:
        agent_r = float(int(scene.snapshot.agent_array[0]) - int(crop.r0))
        agent_c = float(int(scene.snapshot.agent_array[1]) - int(crop.c0))
        if -0.5 <= agent_r <= float(crop.shape[0]) - 0.5 and -0.5 <= agent_c <= float(crop.shape[1]) - 0.5:
            _draw_agent(ax, row=agent_r, col=agent_c, zorder=7)
    if show_scan_circle and scan_radius is not None:
        _draw_scan_circle(
            ax,
            center_row=float(int(scene.snapshot.agent_array[0]) - int(crop.r0)),
            center_col=float(int(scene.snapshot.agent_array[1]) - int(crop.c0)),
            radius=float(scan_radius),
            zorder=7,
        )
    _save_asset_figure(fig, path, dpi=style.dpi)


def _export_semantic_input_belief_map(
    path: Path,
    scene: SemanticExportScene,
    *,
    style: SharedSemanticAssetStyle,
    trajectory_world: np.ndarray | None = None,
) -> None:
    _export_semantic_input_analysis_crop(
        path,
        scene,
        style=style,
        trajectory_world=trajectory_world,
        show_agent=False,
        show_scan_circle=False,
        show_trajectory=False,
    )


def _export_local_semantic_crop(
    path: Path,
    scene: SemanticExportScene,
    *,
    style: SharedSemanticAssetStyle,
    crop_shape: tuple[int, int] | None = None,
    crop_radius: int | None = None,
    trajectory_world: np.ndarray | None = None,
    trajectory_length: int = DEFAULT_SEMANTIC_TRAJECTORY_LENGTH,
    show_agent: bool = False,
    show_scan_circle: bool = False,
    show_trajectory: bool = True,
    scan_radius: int | None = None,
) -> None:
    center = (int(scene.snapshot.agent_array[0]), int(scene.snapshot.agent_array[1]))
    shape = (
        (2 * max(0, int(crop_radius)) + 1, 2 * max(0, int(crop_radius)) + 1)
        if crop_shape is None and crop_radius is not None
        else tuple(int(v) for v in (scene.snapshot.local_snap.shape if crop_shape is None else crop_shape))
    )
    belief_crop = _fixed_centered_crop_shape(
        scene.snapshot.belief_map,
        center_rc=center,
        shape=shape,
        fill_value=np.int8(INVISIBLE),
    )
    frontier_crop = _fixed_centered_crop_shape(
        _belief_frontier_mask(scene),
        center_rc=center,
        shape=shape,
        fill_value=False,
    )
    fig, ax = _create_local_asset_axes(belief_crop.shape)
    _render_base_map(ax, belief_crop)
    _overlay_mask(ax, frontier_crop, color=RAW_FRONTIER_COLOR, alpha=float(style.frontier_alpha))
    trajectory = (
        _recent_trajectory_window(scene.snapshot.trajectory_world, length=int(trajectory_length))
        if trajectory_world is None
        else np.asarray(trajectory_world, dtype=np.int32).reshape((-1, 2))
    )
    if show_trajectory:
        _draw_centered_trajectory(ax, scene.snapshot, trajectory, belief_crop.shape, zorder=6)
    center_row = float(int(belief_crop.shape[0]) // 2)
    center_col = float(int(belief_crop.shape[1]) // 2)
    if show_scan_circle and scan_radius is not None:
        _draw_scan_circle(ax, center_row=center_row, center_col=center_col, radius=float(scan_radius), zorder=7)
    if show_agent:
        _draw_agent(ax, row=center_row, col=center_col, zorder=8)
    _save_asset_figure(fig, path, dpi=style.dpi)


def _export_trajectory_decay_10step_local(
    path: Path,
    snapshot: Snapshot,
    *,
    style: SharedSemanticAssetStyle,
    crop_shape: tuple[int, int],
    trajectory_world: np.ndarray | None = None,
    length: int = DEFAULT_SEMANTIC_TRAJECTORY_LENGTH,
) -> None:
    trajectory_source = snapshot.trajectory_world if trajectory_world is None else trajectory_world
    trajectory = _recent_trajectory_window(np.asarray(trajectory_source, dtype=np.int32), length=length)
    shape = tuple(int(v) for v in crop_shape)
    fig, ax = _create_local_asset_axes(shape)
    local_rows, local_cols = _centered_local_trajectory(snapshot, trajectory, shape)

    segment_count = max(0, int(local_rows.shape[0]) - 1)
    if segment_count > 0:
        for segment_idx in range(segment_count):
            progress = float(segment_idx + 1) / float(segment_count)
            alpha = 0.22 + (0.74 * progress)
            linewidth = 1.25 + (0.65 * progress)
            ax.plot(
                local_cols[segment_idx : segment_idx + 2],
                local_rows[segment_idx : segment_idx + 2],
                color=TRAJECTORY_DECAY_COLOR,
                linewidth=linewidth,
                alpha=alpha,
                solid_capstyle="round",
                zorder=4 + segment_idx,
            )
    _save_asset_figure(fig, path, dpi=style.dpi, facecolor="white", trim=False)


def _export_trajectory_decay(
    path: Path,
    snapshot: Snapshot,
    *,
    style: SharedSemanticAssetStyle,
    trajectory_world: np.ndarray | None = None,
    length: int = DEFAULT_SEMANTIC_TRAJECTORY_LENGTH,
) -> None:
    _export_trajectory_decay_10step_local(
        path,
        snapshot,
        style=style,
        crop_shape=snapshot.local_snap.shape,
        trajectory_world=trajectory_world,
        length=length,
    )


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
    _render_cluster_analysis_boxes(ax, scene, crop, style=style)
    _save_asset_figure(fig, path, dpi=style.dpi)


def _export_frontier_cluster_overlay(
    path: Path,
    scene: SemanticExportScene,
    *,
    style: SharedSemanticAssetStyle,
    analysis_window: SemanticAnalysisWindow | None = None,
    trajectory_world: np.ndarray | None = None,
    trajectory_length: int = DEFAULT_SEMANTIC_TRAJECTORY_LENGTH,
    show_trajectory: bool = False,
) -> None:
    window = analysis_window or _build_semantic_analysis_window(
        scene,
        trajectory_world=trajectory_world,
        trajectory_length=int(trajectory_length),
        show_trajectory=show_trajectory,
    )
    crop = window.crop
    fig, ax = _create_asset_axes(crop.shape, style=style)
    _render_base_map(ax, window.belief_crop)
    _render_frontier_clusters(ax, scene, crop, style=style)
    if show_trajectory:
        _draw_window_trajectory(ax, window.trajectory_rows, window.trajectory_cols, crop.shape, zorder=6)
    _save_asset_figure(fig, path, dpi=style.dpi)


def _export_cluster_analysis_boxes(
    path: Path,
    scene: SemanticExportScene,
    *,
    style: SharedSemanticAssetStyle,
) -> None:
    crop = _crop_from_box(scene.semantic_snapshot.analysis_box)
    belief_crop = _crop_belief(scene.snapshot, crop)
    fig, ax = _create_asset_axes(crop.shape, style=style)
    _render_base_map(ax, belief_crop)
    _render_cluster_analysis_boxes(ax, scene, crop, style=style)
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


def _build_manifest(
    outputs: dict[str, Path],
    scene: SemanticExportScene,
    *,
    trajectory_decay_length: int = DEFAULT_SEMANTIC_TRAJECTORY_LENGTH,
) -> dict[str, object]:
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
            "belief_after_update_with_frontier.png": {
                "path": _format_output_path(outputs["belief_after_update_with_frontier"]),
                "paper_node": "Shared Semantic Layer Full-Map Frontier Input",
                "render_source": "real_code_data",
                "data_basis": ["belief_map", "cumulative_frontier", "recent_trajectory"],
            },
            "local_semantic_crop.png": {
                "path": _format_output_path(outputs["local_semantic_crop"]),
                "paper_node": "Shared Semantic Layer Local Crop",
                "render_source": "real_code_data",
                "data_basis": ["belief_map", "agent_centered_local_crop", "cumulative_frontier", "recent_trajectory"],
            },
            "semantic_input_analysis_crop.png": {
                "path": _format_output_path(outputs["semantic_input_analysis_crop"]),
                "paper_node": "Shared Semantic Layer Analysis Crop",
                "render_source": "real_code_data",
                "data_basis": ["shared_full_canvas_crop", "analysis_box", "raw_frontier", "recent_trajectory"],
            },
            "frontier_cluster_overlay.png": {
                "path": _format_output_path(outputs["frontier_cluster_overlay"]),
                "paper_node": "Frontier Cluster Overlay",
                "render_source": "real_code_data",
                "data_basis": ["shared_analysis_window", "analysis_box", "frontier_clusters"],
            },
            "trajectory_decay_10step_local.png": {
                "path": _format_output_path(outputs["trajectory_decay_10step_local"]),
                "paper_node": "Local Recent Trajectory Decay",
                "render_source": "real_code_data",
                "data_basis": ["trajectory_world", "local_observation_shape", "recent_trajectory_length"],
                "trajectory_decay_length": int(trajectory_decay_length),
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
    local_semantic_crop_radius: int | None = None,
    trajectory_decay_length: int = DEFAULT_SEMANTIC_TRAJECTORY_LENGTH,
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
    local_crop_shape = (
        scene.snapshot.local_snap.shape
        if local_semantic_crop_radius is None
        else (2 * max(0, int(local_semantic_crop_radius)) + 1, 2 * max(0, int(local_semantic_crop_radius)) + 1)
    )
    semantic_full_canvas = _build_semantic_full_canvas(
        scene,
        trajectory_length=int(trajectory_decay_length),
        show_trajectory=True,
    )
    semantic_analysis_window = _analysis_window_from_full_canvas(semantic_full_canvas)

    outputs: dict[str, Path] = {
        "belief_after_update_with_frontier": output_dir_path / "belief_after_update_with_frontier.png",
        "local_semantic_crop": output_dir_path / "local_semantic_crop.png",
        "semantic_input_analysis_crop": output_dir_path / "semantic_input_analysis_crop.png",
        "frontier_cluster_overlay": output_dir_path / "frontier_cluster_overlay.png",
        "trajectory_decay_10step_local": output_dir_path / "trajectory_decay_10step_local.png",
        "unknown_block": output_dir_path / "unknown_block.png",
        "frontier_cluster": output_dir_path / "frontier_cluster.png",
        "local_attribute_summary": output_dir_path / "local_attribute_summary.png",
    }

    _export_belief_after_update_with_frontier(
        outputs["belief_after_update_with_frontier"],
        scene,
        style=style,
        full_canvas=semantic_full_canvas,
        trajectory_length=int(trajectory_decay_length),
        show_agent=False,
        show_scan_circle=False,
        show_trajectory=True,
    )
    _export_local_semantic_crop(
        outputs["local_semantic_crop"],
        scene,
        style=style,
        crop_shape=local_crop_shape,
        trajectory_length=int(trajectory_decay_length),
        show_agent=False,
        show_scan_circle=False,
        show_trajectory=True,
    )
    _export_semantic_input_analysis_crop(
        outputs["semantic_input_analysis_crop"],
        scene,
        style=style,
        analysis_window=semantic_analysis_window,
        trajectory_length=int(trajectory_decay_length),
        show_agent=False,
        show_scan_circle=False,
        show_trajectory=True,
    )
    _export_frontier_cluster_overlay(
        outputs["frontier_cluster_overlay"],
        scene,
        style=style,
        analysis_window=semantic_analysis_window,
        show_trajectory=False,
    )
    _export_trajectory_decay_10step_local(
        outputs["trajectory_decay_10step_local"],
        scene.snapshot,
        style=style,
        crop_shape=local_crop_shape,
        length=int(trajectory_decay_length),
    )
    _export_unknown_block(outputs["unknown_block"], scene, style=style)
    _export_frontier_cluster(outputs["frontier_cluster"], scene, style=style)
    _export_local_attribute_summary(outputs["local_attribute_summary"], scene, style=style)

    if include_shared_semantic_states:
        outputs["shared_semantic_states"] = output_dir_path / "shared_semantic_states.png"
        _export_shared_semantic_states(outputs["shared_semantic_states"], scene, style=style)

    manifest = _build_manifest(outputs, scene, trajectory_decay_length=int(trajectory_decay_length))
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
        "--local-semantic-crop-radius",
        type=int,
        default=None,
        help="Optional agent-centered cumulative-belief crop radius for local_semantic_crop.png. Omit to match the Local LiDAR Observation shape.",
    )
    parser.add_argument(
        "--trajectory-decay-length",
        type=int,
        default=DEFAULT_SEMANTIC_TRAJECTORY_LENGTH,
        help="Recent trajectory window length for trajectory_decay_10step_local.png. Default: 10.",
    )
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
        local_semantic_crop_radius=args.local_semantic_crop_radius,
        trajectory_decay_length=int(args.trajectory_decay_length),
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
