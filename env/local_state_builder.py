from __future__ import annotations

"""Legacy local-state builder kept only as historical reference."""

import time
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from env.grid_topology import EMPTY, INVISIBLE, OBSTACLE


LOCAL_STATE_CHANNELS = (
    "unknown",
    "obstacle",
    "free",
    "frontier",
    "trajectory_recency",
    "global_row_norm",
    "global_col_norm",
)
LOCAL_STATE_CHANNEL_COUNT = len(LOCAL_STATE_CHANNELS)


@dataclass(frozen=True)
class LocalStateConfig:
    # Fixed network-input local window shape (H, W), independent of sensor scan radius.
    local_window_shape: Tuple[int, int] = (21, 21)
    trajectory_decay_steps: float = 20.0
    # Profiling only; does not change local-state semantics or outputs.
    enable_timing: bool = False


class LocalStateBuilder:
    """
    Build fixed-size agent-centered local tensor from cumulative belief/state.

    Output layout:
      local_tensor: [C_local, H_local, W_local]
      channel order fixed by LOCAL_STATE_CHANNELS

    Semantics:
    - local window size is controlled by LocalStateConfig.local_window_shape;
      it is not inferred from sensor local_snap or scan_radius.
    - near = local geometry + local frontier cue + short-term recency + global normalized position context.
    - all channels come from cumulative belief/state (belief map, visit history, agent global position).
    - the frontier channel is a dense local mask sampled from the current belief-map frontier.
    """

    def __init__(self, config: Optional[LocalStateConfig] = None):
        self.config = config if config is not None else LocalStateConfig()
        self.local_window_shape = self._validate_local_window_shape(self.config.local_window_shape)
        h, w = self.local_window_shape
        center_r = h // 2
        center_c = w // 2
        local_rows = np.arange(h, dtype=np.int32) - center_r
        local_cols = np.arange(w, dtype=np.int32) - center_c
        self._local_row_offsets, self._local_col_offsets = np.meshgrid(
            local_rows,
            local_cols,
            indexing="ij",
        )

        self._sampled_map_buf = np.empty((h, w), dtype=np.int8)
        self._frontier_buf = np.empty((h, w), dtype=np.float32)
        self._trajectory_buf = np.empty((h, w), dtype=np.float32)
        self._local_tensor_buf = np.empty((LOCAL_STATE_CHANNEL_COUNT, h, w), dtype=np.float32)

        self._trajectory_decay_cache_decay: Optional[float] = None
        self._trajectory_decay_cache = np.empty(0, dtype=np.float32)

        self._timing_enabled = bool(self.config.enable_timing)
        self.sampling_time = 0.0
        self.context_channel_time = 0.0
        self.trajectory_time = 0.0
        self.build_total_time = 0.0

    @staticmethod
    def _validate_local_window_shape(local_window_shape: Tuple[int, int]) -> Tuple[int, int]:
        if len(local_window_shape) != 2:
            raise ValueError(f"local_window_shape must be (H, W), got {local_window_shape}")

        h, w = int(local_window_shape[0]), int(local_window_shape[1])
        if h <= 0 or w <= 0:
            raise ValueError(f"local_window_shape must be positive, got {(h, w)}")
        if (h % 2 == 0) or (w % 2 == 0):
            raise ValueError(f"local_window_shape must be odd-sized for centering, got {(h, w)}")
        return h, w

    def _sample_world_to_local(
        self,
        cum_map,
        agent_state: Tuple[int, int],
    ):
        ar = int(agent_state[0]) + self._local_row_offsets - int(cum_map.origin_world_rc[0])
        ac = int(agent_state[1]) + self._local_col_offsets - int(cum_map.origin_world_rc[1])
        inside = (
            (ar >= 0) & (ar < cum_map.map.shape[0]) &
            (ac >= 0) & (ac < cum_map.map.shape[1])
        )
        return ar, ac, inside

    @staticmethod
    def _shared_artifact_value(shared_artifacts, key: str):
        if shared_artifacts is None:
            return None
        if isinstance(shared_artifacts, dict):
            return shared_artifacts.get(key)
        return getattr(shared_artifacts, key, None)

    def _resolve_frontier_bool(
        self,
        cum_map,
        *,
        frontier_u8: Optional[np.ndarray] = None,
        frontier_stats=None,
        shared_artifacts=None,
    ) -> np.ndarray:
        frontier_stats_use = frontier_stats
        if frontier_stats_use is None:
            frontier_stats_use = self._shared_artifact_value(shared_artifacts, "frontier_stats")

        frontier_u8_use = frontier_u8
        if frontier_u8_use is None:
            frontier_u8_use = self._shared_artifact_value(shared_artifacts, "frontier_u8")

        if frontier_stats_use is None:
            frontier_stats_use = cum_map.get_frontier_derived_stats(refresh=False, frontier_u8=frontier_u8_use)

        frontier_bool = np.asarray(frontier_stats_use.frontier_bool, dtype=bool)
        if frontier_bool.shape != cum_map.map.shape:
            raise ValueError(
                f"frontier shape mismatch: expected {cum_map.map.shape}, got {frontier_bool.shape}"
            )
        return frontier_bool

    def _ensure_trajectory_decay_cache(self, max_delta: int) -> None:
        decay = float(max(1e-6, self.config.trajectory_decay_steps))
        if self._trajectory_decay_cache_decay != decay:
            self._trajectory_decay_cache_decay = decay
            self._trajectory_decay_cache = np.empty(0, dtype=np.float32)

        needed = int(max_delta) + 1
        current = int(self._trajectory_decay_cache.shape[0])
        if current >= needed:
            return

        delta = np.arange(current, needed, dtype=np.float64)
        values = np.exp(-delta / decay).astype(np.float32)
        if current == 0:
            self._trajectory_decay_cache = values
        else:
            self._trajectory_decay_cache = np.concatenate([self._trajectory_decay_cache, values], axis=0)

    def get_timing_stats(self) -> dict[str, float]:
        return {
            "sampling_time": float(self.sampling_time),
            "context_channel_time": float(self.context_channel_time),
            "trajectory_time": float(self.trajectory_time),
            "build_total_time": float(self.build_total_time),
        }

    def build(
        self,
        cum_map,
        agent_state: Tuple[int, int],
        frontier_u8: Optional[np.ndarray] = None,
        frontier_stats=None,
        shared_artifacts=None,
    ) -> np.ndarray:
        t_build = time.perf_counter() if self._timing_enabled else 0.0
        _ = frontier_u8, frontier_stats, shared_artifacts
        h, w = self.local_window_shape
        t_sampling = time.perf_counter() if self._timing_enabled else 0.0
        ar, ac, inside = self._sample_world_to_local(cum_map, agent_state)
        inside_any = bool(np.any(inside))

        sampled_map = self._sampled_map_buf
        sampled_map.fill(INVISIBLE)
        if inside_any:
            ir = ar[inside]
            ic = ac[inside]
            sampled_map[inside] = cum_map.map[ir, ic]
        if self._timing_enabled:
            self.sampling_time += time.perf_counter() - t_sampling

        frontier_local = self._frontier_buf
        frontier_local.fill(0.0)
        if inside_any:
            frontier_bool = self._resolve_frontier_bool(
                cum_map,
                frontier_u8=frontier_u8,
                frontier_stats=frontier_stats,
                shared_artifacts=shared_artifacts,
            )
            frontier_local[inside] = frontier_bool[ir, ic]

        t_trajectory = time.perf_counter() if self._timing_enabled else 0.0
        trajectory_recency = self._trajectory_buf
        trajectory_recency.fill(0.0)
        if inside_any:
            last_step = cum_map.last_visit_step[ir, ic]
            valid = (last_step >= 0)
            if np.any(valid):
                delta = np.asarray(np.maximum(0, int(cum_map.step_count) - last_step[valid]), dtype=np.int32)
                self._ensure_trajectory_decay_cache(int(delta.max()))
                inside_flat = np.flatnonzero(inside)
                trajectory_recency.reshape(-1)[inside_flat[valid]] = self._trajectory_decay_cache[delta]
        if self._timing_enabled:
            self.trajectory_time += time.perf_counter() - t_trajectory

        t_context = time.perf_counter() if self._timing_enabled else 0.0
        map_h, map_w = int(cum_map.map.shape[0]), int(cum_map.map.shape[1])
        agent_arr_r = int(agent_state[0]) - int(cum_map.origin_world_rc[0])
        agent_arr_c = int(agent_state[1]) - int(cum_map.origin_world_rc[1])
        row_norm = float(np.clip(agent_arr_r / float(max(1, map_h - 1)), 0.0, 1.0))
        col_norm = float(np.clip(agent_arr_c / float(max(1, map_w - 1)), 0.0, 1.0))
        if self._timing_enabled:
            self.context_channel_time += time.perf_counter() - t_context

        local_tensor = self._local_tensor_buf
        local_tensor[0] = (sampled_map == INVISIBLE)
        local_tensor[1] = (sampled_map == OBSTACLE)
        local_tensor[2] = (sampled_map == EMPTY)
        local_tensor[3] = frontier_local
        local_tensor[4] = trajectory_recency
        local_tensor[5].fill(np.float32(row_norm))
        local_tensor[6].fill(np.float32(col_norm))
        if local_tensor.shape[0] != LOCAL_STATE_CHANNEL_COUNT:
            raise RuntimeError(
                f"local channel count mismatch: expected {LOCAL_STATE_CHANNEL_COUNT}, got {local_tensor.shape[0]}"
            )

        out = local_tensor.copy()
        if self._timing_enabled:
            self.build_total_time += time.perf_counter() - t_build
        return out


def _smoke_test() -> None:
    from env.block_random_g import RandomMapGenerator
    from env.agent_version import LocalObservationModel
    from env.core_cummap import CumulativeBeliefMap

    g, s = RandomMapGenerator(30, 40, 5, 0.2).generate_map()
    obs = LocalObservationModel(g, s)
    snap, _ = obs.observe(s)

    cm = CumulativeBeliefMap(g, s, snap)
    cfg = LocalStateConfig(local_window_shape=(25, 25))
    builder = LocalStateBuilder(cfg)
    local = builder.build(cm, s)

    assert LOCAL_STATE_CHANNELS == (
        "unknown",
        "obstacle",
        "free",
        "frontier",
        "trajectory_recency",
        "global_row_norm",
        "global_col_norm",
    )
    assert local.shape[0] == LOCAL_STATE_CHANNEL_COUNT
    assert local.shape[1:] == cfg.local_window_shape
    print("LocalStateBuilder smoke test passed", local.shape)


if __name__ == "__main__":
    _smoke_test()
