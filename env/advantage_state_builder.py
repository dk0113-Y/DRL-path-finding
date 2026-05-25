from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from env.grid_topology import EMPTY, INVISIBLE, OBSTACLE
from env.shared_semantic_layer import SharedSemanticSnapshot


ADVANTAGE_CANVAS_SCHEMA_FINAL_4CH_NO_FRONTIER_RASTER = "final_4ch_no_frontier_raster"
FINAL_4CH_ADVANTAGE_CANVAS_CHANNELS = (
    "free",
    "obstacle",
    "visit_count_log_norm",
    "recent_trajectory_decay",
)
ADVANTAGE_CANVAS_CHANNELS_BY_SCHEMA = {
    ADVANTAGE_CANVAS_SCHEMA_FINAL_4CH_NO_FRONTIER_RASTER: FINAL_4CH_ADVANTAGE_CANVAS_CHANNELS,
}
ADVANTAGE_CANVAS_SCHEMAS = tuple(ADVANTAGE_CANVAS_CHANNELS_BY_SCHEMA.keys())
ADVANTAGE_CANVAS_CHANNELS = FINAL_4CH_ADVANTAGE_CANVAS_CHANNELS
ADVANTAGE_CANVAS_CHANNEL_COUNT = len(ADVANTAGE_CANVAS_CHANNELS)


def normalize_advantage_canvas_schema(schema: str | None) -> str:
    normalized = str(schema or ADVANTAGE_CANVAS_SCHEMA_FINAL_4CH_NO_FRONTIER_RASTER).strip().lower()
    if normalized not in ADVANTAGE_CANVAS_CHANNELS_BY_SCHEMA:
        available = ", ".join(ADVANTAGE_CANVAS_SCHEMAS)
        raise ValueError(
            f"Unsupported advantage_canvas_schema {schema!r}; expected one of: {available}"
        )
    return normalized


def advantage_canvas_channels_for_schema(schema: str | None) -> tuple[str, ...]:
    return ADVANTAGE_CANVAS_CHANNELS_BY_SCHEMA[normalize_advantage_canvas_schema(schema)]


def advantage_canvas_channel_count_for_schema(schema: str | None) -> int:
    return len(advantage_canvas_channels_for_schema(schema))


def advantage_canvas_uses_frontier_raster(schema: str | None) -> bool:
    normalize_advantage_canvas_schema(schema)
    return False


frontier_raster_used_for_schema = advantage_canvas_uses_frontier_raster


@dataclass(frozen=True)
class AdvantageStateConfig:
    enable_timing: bool = False
    advantage_canvas_schema: str = ADVANTAGE_CANVAS_SCHEMA_FINAL_4CH_NO_FRONTIER_RASTER
    visit_count_log_saturation: float = 8.0
    trajectory_history_steps: int = 10

    def __post_init__(self) -> None:
        schema = normalize_advantage_canvas_schema(self.advantage_canvas_schema)
        object.__setattr__(self, "advantage_canvas_schema", schema)

    @property
    def advantage_canvas_channels(self) -> tuple[str, ...]:
        return advantage_canvas_channels_for_schema(self.advantage_canvas_schema)

    @property
    def advantage_canvas_channel_count(self) -> int:
        return len(self.advantage_canvas_channels)

    @property
    def frontier_raster_used(self) -> bool:
        return advantage_canvas_uses_frontier_raster(self.advantage_canvas_schema)


class AdvantageStateBuilder:
    """
    Build the local decision canvas consumed by the advantage branch.

    The final A_new schema is a 4-channel local spatial representation tied to
    the radar observation window (`cum_map.local_shape`). Its channels are
    free-space occupancy, obstacle occupancy, cumulative revisit pressure, and
    recent trajectory decay. The final schema does not include a frontier raster
    channel in the local advantage canvas.

    Frontier and unknown-block semantics remain represented by the structured
    value-tree branch through `SharedSemanticSnapshot`; they are not painted
    into the local advantage canvas. The historical 5-channel frontier-raster
    diagnostics were archived before this cleanup on `legacy/pre-a-new-cleanup`
    and tag `legacy-pre-a-new-cleanup-20260525`.
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
            channel_count = int(self.config.advantage_canvas_channel_count)
            cached = np.zeros((channel_count, int(shape[0]), int(shape[1])), dtype=np.float32)
            self._canvas_cache[shape] = cached
        cached.fill(0.0)
        return cached

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
        _ = semantic_snapshot
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
        canvas_schema = str(self.config.advantage_canvas_schema)
        visit_channel_index = 2
        trajectory_channel_index = 3

        revisit_count = np.maximum(sampled_visit - 1.0, 0.0).astype(np.float32, copy=False)
        visit_log_denominator = float(np.log1p(max(1e-6, float(self.config.visit_count_log_saturation))))
        visit_count_log_norm = np.log1p(revisit_count).astype(np.float32, copy=False) / max(1e-6, visit_log_denominator)
        canvas[visit_channel_index] = np.clip(visit_count_log_norm, 0.0, 1.0).astype(np.float32, copy=False)

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
            canvas[trajectory_channel_index],
            current_agent_arr=agent_arr,
            local_shape=local_shape,
        )

        meta = {
            "advantage_canvas_schema": canvas_schema,
            "advantage_canvas_channel_count": float(canvas.shape[0]),
            "frontier_raster_used": False,
            "local_frontier_coverage": 0.0,
            "local_frontier_positive_count": 0.0,
            "local_frontier_block_area_mean": 0.0,
        }
        if self._timing_enabled:
            self.build_time += time.perf_counter() - t0
        return canvas.copy(), meta

    def get_timing_stats(self) -> dict[str, float]:
        return {"build_time": float(self.build_time)}
