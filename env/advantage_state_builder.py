from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

from env.grid_topology import EMPTY, INVISIBLE, OBSTACLE
from env.shared_semantic_layer import SharedSemanticSnapshot


ADVANTAGE_CANVAS_CHANNELS = (
    "unknown",
    "free",
    "obstacle",
    "revisit_recency",
    "main_entry_mask",
    "nonmain_entry_mask",
    "main_block_fragment_mask",
)
ADVANTAGE_CANVAS_CHANNEL_COUNT = len(ADVANTAGE_CANVAS_CHANNELS)


@dataclass(frozen=True)
class AdvantageStateConfig:
    enable_timing: bool = False


class AdvantageStateBuilder:
    """
    Build the local decision canvas consumed by the advantage branch.

    Canvas size is tied directly to the radar observation window (`cum_map.local_shape`),
    so the advantage branch always reasons over the same local scale that the
    agent can currently observe.
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

    def build(
        self,
        cum_map,
        agent_state: tuple[int, int],
        semantic_snapshot: SharedSemanticSnapshot,
    ) -> tuple[np.ndarray, dict[str, float]]:
        t0 = time.perf_counter() if self._timing_enabled else 0.0
        local_shape = (int(cum_map.local_shape[0]), int(cum_map.local_shape[1]))
        canvas = self._canvas_buffer(local_shape)
        arr_rows, arr_cols, inside = self._local_index_arrays(cum_map, agent_state)
        if np.any(inside):
            sampled_map = np.full(local_shape, INVISIBLE, dtype=np.int8)
            sampled_map[inside] = cum_map.map[arr_rows[inside], arr_cols[inside]]
            revisit_map = np.zeros(local_shape, dtype=np.float32)
            full_revisit = cum_map.get_revisit_recency_map(refresh=False)
            revisit_map[inside] = full_revisit[arr_rows[inside], arr_cols[inside]]
        else:
            sampled_map = np.full(local_shape, INVISIBLE, dtype=np.int8)
            revisit_map = np.zeros(local_shape, dtype=np.float32)

        canvas[0] = (sampled_map == INVISIBLE)
        canvas[1] = (sampled_map == EMPTY)
        canvas[2] = (sampled_map == OBSTACLE)
        canvas[3] = revisit_map

        main_block = semantic_snapshot.main_block()
        agent_arr = cum_map.world_to_array(agent_state)
        if main_block is not None:
            main_block.paint_to_local_canvas(
                canvas[6],
                agent_arr=agent_arr,
                local_shape=local_shape,
            )

        for block in semantic_snapshot.accessible_blocks:
            target_channel = 4 if (main_block is not None and block.block_index == main_block.block_index) else 5
            for entry in block.entries:
                entry.paint_to_local_canvas(
                    canvas[target_channel],
                    agent_arr=agent_arr,
                    local_shape=local_shape,
                )

        window_area = float(max(1, local_shape[0] * local_shape[1]))
        meta = {
            "local_main_entry_coverage": float(np.count_nonzero(canvas[4])) / window_area,
            "local_nonmain_entry_coverage": float(np.count_nonzero(canvas[5])) / window_area,
            "local_revisit_pressure": float(np.mean(canvas[3])),
        }
        if self._timing_enabled:
            self.build_time += time.perf_counter() - t0
        return canvas.copy(), meta

    def get_timing_stats(self) -> dict[str, float]:
        return {"build_time": float(self.build_time)}
