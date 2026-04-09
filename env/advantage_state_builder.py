from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from env.grid_topology import EMPTY, INVISIBLE, OBSTACLE
from env.shared_semantic_layer import SharedSemanticSnapshot


ADVANTAGE_CANVAS_CHANNELS = (
    "unknown",
    "free",
    "obstacle",
    "frontier_mask",
    "frontier_block_area_map",
    "visit_count_log_norm",
    "recent_trajectory_decay",
)
ADVANTAGE_CANVAS_CHANNEL_COUNT = len(ADVANTAGE_CANVAS_CHANNELS)


@dataclass(frozen=True)
class AdvantageStateConfig:
    enable_timing: bool = False
    visit_count_log_saturation: float = 8.0
    trajectory_history_steps: int = 10


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

        canvas[0] = (sampled_map == INVISIBLE)
        canvas[1] = (sampled_map == EMPTY)
        canvas[2] = (sampled_map == OBSTACLE)

        agent_arr = cum_map.world_to_array(agent_state)
        total_unknown_area = float(max(1, semantic_snapshot.total_accessible_unknown_area))
        for block in semantic_snapshot.accessible_blocks:
            block_area_ratio = float(block.block_area) / total_unknown_area
            for frontier_cluster in block.frontier_clusters:
                frontier_cluster.paint_to_local_canvas(
                    canvas[3],
                    agent_arr=agent_arr,
                    local_shape=local_shape,
                )
                self._paint_geometry_value_to_local_canvas(
                    frontier_cluster.frontier_geometry,
                    canvas[4],
                    value=block_area_ratio,
                    agent_arr=agent_arr,
                    local_shape=local_shape,
                )

        revisit_count = np.maximum(sampled_visit - 1.0, 0.0).astype(np.float32, copy=False)
        visit_log_denominator = float(np.log1p(max(1e-6, float(self.config.visit_count_log_saturation))))
        visit_count_log_norm = np.log1p(revisit_count).astype(np.float32, copy=False) / max(1e-6, visit_log_denominator)
        canvas[5] = np.clip(visit_count_log_norm, 0.0, 1.0).astype(np.float32, copy=False)

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
            canvas[6],
            current_agent_arr=agent_arr,
            local_shape=local_shape,
        )

        window_area = float(max(1, local_shape[0] * local_shape[1]))
        frontier_visible = np.count_nonzero(canvas[3])
        meta = {
            "local_frontier_coverage": float(frontier_visible) / window_area,
            "local_frontier_block_area_mean": float(canvas[4][canvas[3] > 0.0].mean()) if frontier_visible > 0 else 0.0,
        }
        if self._timing_enabled:
            self.build_time += time.perf_counter() - t0
        return canvas.copy(), meta

    def get_timing_stats(self) -> dict[str, float]:
        return {"build_time": float(self.build_time)}
