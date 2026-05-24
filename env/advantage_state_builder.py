from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from env.grid_topology import EMPTY, INVISIBLE, OBSTACLE
from env.shared_semantic_layer import SharedSemanticSnapshot


ADVANTAGE_CANVAS_CHANNELS = (
    "free",
    "obstacle",
    "frontier_block_area_map",
    "visit_count_log_norm",
    "recent_trajectory_decay",
)
ADVANTAGE_CANVAS_CHANNEL_COUNT = len(ADVANTAGE_CANVAS_CHANNELS)
FRONTIER_CHANNEL_MODE_SEMANTIC_BLOCK_AREA_RASTER = "semantic_block_area_raster"
FRONTIER_CHANNEL_MODE_LOCAL_BINARY = "local_binary"
FRONTIER_CHANNEL_MODE_LOCAL_GLOBAL_AREA = "local_global_area"
FRONTIER_CHANNEL_MODES = (
    FRONTIER_CHANNEL_MODE_SEMANTIC_BLOCK_AREA_RASTER,
    FRONTIER_CHANNEL_MODE_LOCAL_BINARY,
    FRONTIER_CHANNEL_MODE_LOCAL_GLOBAL_AREA,
)


@dataclass(frozen=True)
class AdvantageStateConfig:
    enable_timing: bool = False
    visit_count_log_saturation: float = 8.0
    trajectory_history_steps: int = 10
    frontier_channel_mode: str = FRONTIER_CHANNEL_MODE_SEMANTIC_BLOCK_AREA_RASTER

    def __post_init__(self) -> None:
        mode = str(self.frontier_channel_mode or "").strip().lower()
        if mode not in FRONTIER_CHANNEL_MODES:
            available = ", ".join(FRONTIER_CHANNEL_MODES)
            raise ValueError(
                f"Unsupported frontier_channel_mode {self.frontier_channel_mode!r}; "
                f"expected one of: {available}"
            )
        object.__setattr__(self, "frontier_channel_mode", mode)


class AdvantageStateBuilder:
    """
    Build the local decision canvas consumed by the advantage branch.

    Canvas size is tied directly to the radar observation window (`cum_map.local_shape`),
    so the advantage branch always reasons over the same local scale that the
    agent can currently observe. The canvas exposes every locally visible
    frontier cluster plus a projected block-area attribute on those frontier
    cells, plus a cumulative revisit-pressure channel derived from the
    cumulative belief map visit counters. This revisit signal is deliberately
    cumulative rather than recent-recency based, so the advantage branch can
    distinguish pushing into fresh space from circling over established old
    routes without introducing a short-horizon recency heuristic. A separate
    recent-trajectory channel paints the last few occupied cells with a linear
    time decay, so the advantage branch can also see the short-horizon motion
    geometry that led into the current local situation.
    """

    def __init__(self, config: Optional[AdvantageStateConfig] = None):
        self.config = config if config is not None else AdvantageStateConfig()
        self._timing_enabled = bool(self.config.enable_timing)
        self.build_time = 0.0
        self._canvas_cache: dict[tuple[int, int], np.ndarray] = {}

    @staticmethod
    def _local_index_arrays(
        cum_map,
        agent_state: tuple[int, int],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        h, w = int(cum_map.local_shape[0]), int(cum_map.local_shape[1])
        center_r = h // 2
        center_c = w // 2
        local_rows = np.arange(h, dtype=np.int32) - center_r
        local_cols = np.arange(w, dtype=np.int32) - center_c
        row_offsets, col_offsets = np.meshgrid(local_rows, local_cols, indexing="ij")
        agent_arr_r, agent_arr_c = cum_map.world_to_array(agent_state)
        arr_rows = row_offsets + int(agent_arr_r)
        arr_cols = col_offsets + int(agent_arr_c)
        inside = (
            (arr_rows >= 0) & (arr_rows < int(cum_map.map.shape[0])) &
            (arr_cols >= 0) & (arr_cols < int(cum_map.map.shape[1]))
        )
        return arr_rows, arr_cols, inside

    def _canvas_buffer(self, shape: tuple[int, int]) -> np.ndarray:
        cached = self._canvas_cache.get(shape)
        if cached is None:
            cached = np.zeros((ADVANTAGE_CANVAS_CHANNEL_COUNT, int(shape[0]), int(shape[1])), dtype=np.float32)
            self._canvas_cache[shape] = cached
        cached.fill(0.0)
        return cached

    @staticmethod
    def _local_frontier_mask(
        cum_map,
        *,
        arr_rows: np.ndarray,
        arr_cols: np.ndarray,
        inside: np.ndarray,
        local_shape: tuple[int, int],
    ) -> np.ndarray:
        frontier_u8 = np.asarray(cum_map.get_frontier_u8(refresh=False), dtype=np.uint8)
        if frontier_u8.shape != cum_map.map.shape:
            raise ValueError(
                f"frontier_u8 shape mismatch: expected {cum_map.map.shape}, got {frontier_u8.shape}"
            )
        local_frontier = np.zeros(local_shape, dtype=bool)
        if np.any(inside):
            local_frontier[inside] = frontier_u8[arr_rows[inside], arr_cols[inside]] > 0
        return local_frontier

    @staticmethod
    def _semantic_frontier_area_map(
        cum_map,
        semantic_snapshot: SharedSemanticSnapshot,
    ) -> np.ndarray:
        area_map = np.zeros(cum_map.map.shape, dtype=np.float32)
        total_unknown_area = float(max(1, semantic_snapshot.total_accessible_unknown_area))
        map_h = int(cum_map.map.shape[0])
        map_w = int(cum_map.map.shape[1])
        for block in semantic_snapshot.accessible_blocks:
            block_area_ratio = np.float32(float(block.block_area) / total_unknown_area)
            for frontier_cluster in block.frontier_clusters:
                rows = np.asarray(frontier_cluster.frontier_geometry.rows, dtype=np.int32)
                cols = np.asarray(frontier_cluster.frontier_geometry.cols, dtype=np.int32)
                if rows.size <= 0 or cols.size <= 0:
                    continue
                in_bounds = (
                    (rows >= 0) & (rows < map_h) &
                    (cols >= 0) & (cols < map_w)
                )
                if not np.any(in_bounds):
                    continue
                area_map[rows[in_bounds], cols[in_bounds]] = block_area_ratio
        return area_map

    def _paint_semantic_block_area_raster(
        self,
        canvas_channel: np.ndarray,
        *,
        semantic_snapshot: SharedSemanticSnapshot,
        agent_arr: tuple[int, int],
        local_shape: tuple[int, int],
    ) -> None:
        total_unknown_area = float(max(1, semantic_snapshot.total_accessible_unknown_area))
        for block in semantic_snapshot.accessible_blocks:
            block_area_ratio = float(block.block_area) / total_unknown_area
            for frontier_cluster in block.frontier_clusters:
                self._paint_geometry_value_to_local_canvas(
                    frontier_cluster.frontier_geometry,
                    canvas_channel,
                    value=block_area_ratio,
                    agent_arr=agent_arr,
                    local_shape=local_shape,
                )

    def _paint_local_binary_frontier(
        self,
        canvas_channel: np.ndarray,
        *,
        local_frontier_mask: np.ndarray,
    ) -> dict[str, float | str]:
        canvas_channel[local_frontier_mask] = np.float32(1.0)
        local_positive = int(np.count_nonzero(local_frontier_mask))
        window_area = float(max(1, local_frontier_mask.size))
        return {
            "frontier_channel_mode": FRONTIER_CHANNEL_MODE_LOCAL_BINARY,
            "local_frontier_coverage": float(local_positive) / window_area,
            "local_frontier_positive_count": float(local_positive),
            "local_frontier_block_area_mean": 0.0,
        }

    def _paint_local_global_area_frontier(
        self,
        canvas_channel: np.ndarray,
        *,
        cum_map,
        arr_rows: np.ndarray,
        arr_cols: np.ndarray,
        inside: np.ndarray,
        local_frontier_mask: np.ndarray,
        semantic_snapshot: SharedSemanticSnapshot,
        local_shape: tuple[int, int],
    ) -> dict[str, float | str]:
        semantic_area_map = self._semantic_frontier_area_map(cum_map, semantic_snapshot)
        local_area_values = np.zeros(local_shape, dtype=np.float32)
        if np.any(inside):
            local_area_values[inside] = semantic_area_map[arr_rows[inside], arr_cols[inside]]
        canvas_channel[local_frontier_mask] = local_area_values[local_frontier_mask]

        local_positive = int(np.count_nonzero(local_frontier_mask))
        assigned_mask = local_frontier_mask & (canvas_channel > 0.0)
        assigned_positive = int(np.count_nonzero(assigned_mask))
        unmatched = max(0, local_positive - assigned_positive)
        window_area = float(max(1, local_frontier_mask.size))
        return {
            "frontier_channel_mode": FRONTIER_CHANNEL_MODE_LOCAL_GLOBAL_AREA,
            "local_frontier_coverage": float(local_positive) / window_area,
            "local_frontier_positive_count": float(local_positive),
            "local_frontier_global_area_positive_count": float(assigned_positive),
            "local_frontier_unmatched_count": float(unmatched),
            "local_frontier_block_area_mean": (
                float(canvas_channel[assigned_mask].mean()) if assigned_positive > 0 else 0.0
            ),
        }

    @staticmethod
    def _paint_geometry_value_to_local_canvas(
        geometry,
        canvas_channel: np.ndarray,
        *,
        value: float,
        agent_arr: tuple[int, int],
        local_shape: tuple[int, int],
    ) -> None:
        rows = np.asarray(geometry.rows, dtype=np.int32)
        cols = np.asarray(geometry.cols, dtype=np.int32)
        if rows.size <= 0 or cols.size <= 0:
            return
        h = int(local_shape[0])
        w = int(local_shape[1])
        center_r = h // 2
        center_c = w // 2
        local_rows = rows - int(agent_arr[0]) + center_r
        local_cols = cols - int(agent_arr[1]) + center_c
        inside = (
            (local_rows >= 0) & (local_rows < h) &
            (local_cols >= 0) & (local_cols < w)
        )
        if not np.any(inside):
            return
        canvas_channel[local_rows[inside], local_cols[inside]] = np.float32(value)

    @staticmethod
    def _paint_recent_trajectory_to_local_canvas(
        trajectory_arr_positions: Sequence[tuple[int, int]],
        canvas_channel: np.ndarray,
        *,
        current_agent_arr: tuple[int, int],
        local_shape: tuple[int, int],
    ) -> None:
        if len(trajectory_arr_positions) <= 0:
            return

        h = int(local_shape[0])
        w = int(local_shape[1])
        center_r = h // 2
        center_c = w // 2
        history_len = len(trajectory_arr_positions)
        denom = float(max(1, history_len))
        for idx, arr_rc in enumerate(trajectory_arr_positions):
            traj_arr_r, traj_arr_c = int(arr_rc[0]), int(arr_rc[1])
            local_r = int(traj_arr_r - int(current_agent_arr[0]) + center_r)
            local_c = int(traj_arr_c - int(current_agent_arr[1]) + center_c)
            if not (0 <= local_r < h and 0 <= local_c < w):
                continue
            weight = np.float32((idx + 1) / denom)
            canvas_channel[local_r, local_c] = np.maximum(canvas_channel[local_r, local_c], weight)

    def build(
        self,
        cum_map,
        agent_state: tuple[int, int],
        semantic_snapshot: SharedSemanticSnapshot,
        recent_trajectory_positions: Optional[Sequence[tuple[int, int]]] = None,
    ) -> tuple[np.ndarray, dict[str, float]]:
        t0 = time.perf_counter() if self._timing_enabled else 0.0
        local_shape = (int(cum_map.local_shape[0]), int(cum_map.local_shape[1]))
        canvas = self._canvas_buffer(local_shape)
        arr_rows, arr_cols, inside = self._local_index_arrays(cum_map, agent_state)
        sampled_map = np.full(local_shape, INVISIBLE, dtype=np.int8)
        sampled_visit = np.zeros(local_shape, dtype=np.float32)
        if np.any(inside):
            sampled_map[inside] = cum_map.map[arr_rows[inside], arr_cols[inside]]
            sampled_visit[inside] = cum_map.visit_count[arr_rows[inside], arr_cols[inside]].astype(np.float32)

        canvas[0] = (sampled_map == EMPTY)
        canvas[1] = (sampled_map == OBSTACLE)

        agent_arr = cum_map.world_to_array(agent_state)
        frontier_channel_mode = str(self.config.frontier_channel_mode)
        local_frontier_mask = None
        frontier_meta: dict[str, float | str]
        if frontier_channel_mode == FRONTIER_CHANNEL_MODE_SEMANTIC_BLOCK_AREA_RASTER:
            self._paint_semantic_block_area_raster(
                canvas[2],
                semantic_snapshot=semantic_snapshot,
                agent_arr=agent_arr,
                local_shape=local_shape,
            )
            frontier_visible_mask = canvas[2] > 0.0
            frontier_visible = int(np.count_nonzero(frontier_visible_mask))
            window_area = float(max(1, local_shape[0] * local_shape[1]))
            frontier_meta = {
                "frontier_channel_mode": FRONTIER_CHANNEL_MODE_SEMANTIC_BLOCK_AREA_RASTER,
                "local_frontier_coverage": float(frontier_visible) / window_area,
                "local_frontier_positive_count": float(frontier_visible),
                "local_frontier_block_area_mean": (
                    float(canvas[2][frontier_visible_mask].mean()) if frontier_visible > 0 else 0.0
                ),
            }
        else:
            local_frontier_mask = self._local_frontier_mask(
                cum_map,
                arr_rows=arr_rows,
                arr_cols=arr_cols,
                inside=inside,
                local_shape=local_shape,
            )
            if frontier_channel_mode == FRONTIER_CHANNEL_MODE_LOCAL_BINARY:
                frontier_meta = self._paint_local_binary_frontier(
                    canvas[2],
                    local_frontier_mask=local_frontier_mask,
                )
            elif frontier_channel_mode == FRONTIER_CHANNEL_MODE_LOCAL_GLOBAL_AREA:
                frontier_meta = self._paint_local_global_area_frontier(
                    canvas[2],
                    cum_map=cum_map,
                    arr_rows=arr_rows,
                    arr_cols=arr_cols,
                    inside=inside,
                    local_frontier_mask=local_frontier_mask,
                    semantic_snapshot=semantic_snapshot,
                    local_shape=local_shape,
                )
            else:
                raise ValueError(f"Unsupported frontier_channel_mode: {frontier_channel_mode!r}")

        revisit_count = np.maximum(sampled_visit - 1.0, 0.0).astype(np.float32, copy=False)
        visit_log_denominator = float(np.log1p(max(1e-6, float(self.config.visit_count_log_saturation))))
        visit_count_log_norm = np.log1p(revisit_count).astype(np.float32, copy=False) / max(1e-6, visit_log_denominator)
        canvas[3] = np.clip(visit_count_log_norm, 0.0, 1.0).astype(np.float32, copy=False)

        history_limit = max(1, int(self.config.trajectory_history_steps))
        raw_history = list(recent_trajectory_positions or ())
        # The current agent cell is already implicit as the canvas center, so the
        # trajectory channel paints only the recent path that led into the current
        # state rather than re-marking the center every step.
        decayed_history = raw_history[:-1] if len(raw_history) > 1 else []
        if len(decayed_history) > history_limit:
            decayed_history = decayed_history[-history_limit:]
        decayed_history_arr = [cum_map.world_to_array(world_rc) for world_rc in decayed_history]
        self._paint_recent_trajectory_to_local_canvas(
            decayed_history_arr,
            canvas[4],
            current_agent_arr=agent_arr,
            local_shape=local_shape,
        )

        meta = dict(frontier_meta)
        if self._timing_enabled:
            self.build_time += time.perf_counter() - t0
        return canvas.copy(), meta

    def get_timing_stats(self) -> dict[str, float]:
        return {"build_time": float(self.build_time)}
