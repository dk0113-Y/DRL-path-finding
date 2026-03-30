from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from env.grid_topology import EMPTY, INVISIBLE, OBSTACLE, GridTopology


MID_MAP_CHANNELS = (
    "unknown_density",
    "known_obstacle_density",
    "coarse_visit_density",
    "frontier_density",
)
MID_MAP_CHANNEL_COUNT = len(MID_MAP_CHANNELS)
FRONTIER_MIN_UNKNOWN_NEIGHBORS = 1
FRONTIER_NEIGHBOR_CONNECTIVITY = 4


@dataclass(frozen=True)
class MidMapConfig:
    """Agent-centered fixed-window mid map config, not a far/full-world map config."""

    mid_map_shape: Tuple[int, int] = (32, 32)         # fixed output size (H_m, W_m)
    world_window_shape: Tuple[int, int] = (128, 128)  # agent-centered world window (H_w, W_w)


@dataclass(frozen=True)
class FrontierDerivedStats:
    """Shared frontier snapshot derived from the incrementally maintained frontier cache."""

    frontier_u8: np.ndarray
    frontier_bool: np.ndarray
    frontier_source_uid: int
    frontier_revision: int


@dataclass(frozen=True)
class FrontierConsistencyStats:
    """Debug-only comparison summary for incremental frontier cache validation."""

    consistent: bool
    mismatch_count: int
    frontier_source_uid: int
    frontier_revision: int
    map_shape: tuple[int, int]


@dataclass(frozen=True)
class DirtyRect:
    """Half-open array-space dirty rectangle [r0:r1, c0:c1)."""

    r0: int
    r1: int
    c0: int
    c1: int


@dataclass(frozen=True)
class WorldBoundsExpansion:
    """Map growth metadata needed by frontier seam maintenance."""

    pad_top: int
    pad_left: int
    pad_bottom: int
    pad_right: int
    seam_dirty_rects: tuple[DirtyRect, ...]


@dataclass(frozen=True)
class AnalysisBox:
    """
    Array-space analysis window used by the shared semantic layer.

    The box tracks the currently known region plus a margin tied to the
    advantage local decision canvas half-span. This intentionally limits
    semantic parsing to unknown structure that is immediately relevant to
    exploration, instead of treating the unbounded outside-of-map ocean as a
    real decision object.
    """

    r0: int
    r1: int
    c0: int
    c1: int
    margin: int
    known_r0: int
    known_r1: int
    known_c0: int
    known_c1: int

    @property
    def shape(self) -> tuple[int, int]:
        return int(self.r1 - self.r0), int(self.c1 - self.c0)


class CumulativeBeliefMap:
    """
    Dynamic-size belief map for agent-side knowledge.

    Key semantics:
    - Agent state/observation is built only from cumulative belief (`self.map`, visits, frontier).
    - Effective coverage is a simulator-side metric/termination statistic.
    - Agent does not receive true-map total area or reachable-area priors as observation channels.
    """

    _aggregate_edge_cache: dict[tuple[int, int, int, int], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    _instance_uid_counter: int = 0

    def __init__(
        self,
        true_grid: np.ndarray,
        start_state: Tuple[int, int],
        first_local_snap: np.ndarray,
        enable_timing: bool = False,
    ):
        self.true_grid = np.asarray(true_grid, dtype=np.int8)
        if self.true_grid.ndim != 2:
            raise ValueError("true_grid must be a 2D array")
        self.enable_timing = bool(enable_timing)
        type(self)._instance_uid_counter += 1
        self.frontier_source_uid = int(type(self)._instance_uid_counter)

        # Observation-side local_snap geometry for belief update projection only.
        # This is not the policy-network local window size.
        self.local_shape = tuple(first_local_snap.shape)
        self.local_center = (self.local_shape[0] // 2, self.local_shape[1] // 2)
        # Bind semantic analysis extent to the radar-driven local decision
        # canvas scale; this avoids introducing a separate hand-tuned margin.
        self.analysis_margin = int(max(self.local_center))
        self.revisit_recency_decay_steps = float(max(1, max(self.local_shape)))

        # Light-weight growth buffer to avoid frequent near-edge reallocations.
        self._growth_margin = int(max(4, max(self.local_shape) // 2))

        # Effective coverage denominator from simulator-known truth map.
        self._reachable_free_mask = self._compute_effective_reachable_free_mask(start_state)
        self.tpm_count = int(self._reachable_free_mask.sum())

        self.kpm_count = 0
        self.coverage_rate = 0.0

        # Dynamic belief map storage and world origin mapping.
        self.map = np.full((1, 1), INVISIBLE, dtype=np.int8)
        self.visit_count = np.zeros((1, 1), dtype=np.int32)
        self.last_visit_step = np.full((1, 1), -1, dtype=np.int32)
        self.step_count = 0
        self.origin_world_rc = (int(start_state[0]), int(start_state[1]))

        self.frontier_bool = np.zeros((1, 1), dtype=bool)
        self.frontier_u8 = np.zeros((1, 1), dtype=np.uint8)
        self.frontier_revision = 0
        self._latest_frontier_stats: Optional[FrontierDerivedStats] = None
        self._cached_visit_log_map: Optional[np.ndarray] = None
        self._cached_visit_log_max: float = 0.0
        self._cached_obstacle_integral: Optional[np.ndarray] = None
        self._cached_revisit_recency_map: Optional[np.ndarray] = None
        self._visit_cache_step: int = -1
        self._domain_buffer_cache: dict[tuple[int, int, bool, str], tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]] = {}
        self._mid_input_channels_cache: dict[tuple[int, int], np.ndarray] = {}
        self._mid_output_cache: dict[tuple[int, int], np.ndarray] = {}
        self._mid_integral_cache: dict[tuple[int, int], np.ndarray] = {}
        self._mid_map_cache_key = None
        self._mid_map_cache_value: Optional[np.ndarray] = None

        self.frontier_stats_time = 0.0
        self.mid_map_time = 0.0
        self.domain_extract_time = 0.0
        self.aggregate_time = 0.0
        self.analysis_box = AnalysisBox(
            r0=0,
            r1=1,
            c0=0,
            c1=1,
            margin=int(self.analysis_margin),
            known_r0=0,
            known_r1=1,
            known_c0=0,
            known_c1=1,
        )

        self._init_from_first_snap(start_state, first_local_snap)

    def _compute_effective_reachable_free_mask(self, start_state: Tuple[int, int]) -> np.ndarray:
        """
        Simulator-side effective coverage domain.

        Denominator semantics:
        reachable free cells from episode start under current movement kinematics.
        """
        free = GridTopology.free_mask(self.true_grid)
        reachable = GridTopology.bfs_reachable(free, start_state)
        return reachable & free

    def _project_local_world(self, agent_state: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
        return GridTopology.local_to_global_grid(agent_state, self.local_shape, self.local_center)

    def _init_from_first_snap(self, agent_state: Tuple[int, int], local_snap: np.ndarray) -> None:
        snap = np.asarray(local_snap, dtype=np.int8)
        if snap.shape != self.local_shape:
            raise ValueError(f"local_snap shape mismatch: expected {self.local_shape}, got {snap.shape}")

        gr, gc = self._project_local_world(agent_state)
        visible = (snap != INVISIBLE)

        if not np.any(visible):
            ar, ac = int(agent_state[0]), int(agent_state[1])
            self.map = np.full((1, 1), INVISIBLE, dtype=np.int8)
            self.visit_count = np.zeros((1, 1), dtype=np.int32)
            self.last_visit_step = np.full((1, 1), -1, dtype=np.int32)
            self.origin_world_rc = (ar, ac)
            self.frontier_bool = np.zeros_like(self.map, dtype=bool)
            self.frontier_u8 = np.zeros_like(self.map, dtype=np.uint8)
            self._record_visit_in_bounds(agent_state)
            self._refresh_coverage()
            self._update_analysis_box()
            return

        wr = gr[visible]
        wc = gc[visible]
        min_r, max_r = int(wr.min()), int(wr.max())
        min_c, max_c = int(wc.min()), int(wc.max())

        h = max_r - min_r + 1
        w = max_c - min_c + 1
        self.map = np.full((h, w), INVISIBLE, dtype=np.int8)
        self.visit_count = np.zeros((h, w), dtype=np.int32)
        self.last_visit_step = np.full((h, w), -1, dtype=np.int32)
        self.origin_world_rc = (min_r, min_c)
        self.frontier_bool = np.zeros_like(self.map, dtype=bool)
        self.frontier_u8 = np.zeros_like(self.map, dtype=np.uint8)

        self.update(agent_state, snap)

    def world_to_array(self, world_rc: Tuple[int, int]) -> Tuple[int, int]:
        r, c = int(world_rc[0]), int(world_rc[1])
        orr, orc = self.origin_world_rc
        return r - orr, c - orc

    def array_to_world(self, array_rc: Tuple[int, int]) -> Tuple[int, int]:
        r, c = int(array_rc[0]), int(array_rc[1])
        orr, orc = self.origin_world_rc
        return r + orr, c + orc

    @staticmethod
    def _shared_artifact_value(shared_artifacts, key: str):
        if shared_artifacts is None:
            return None
        if isinstance(shared_artifacts, dict):
            return shared_artifacts.get(key)
        return getattr(shared_artifacts, key, None)

    def _invalidate_map_build_caches(self) -> None:
        self._mid_map_cache_key = None
        self._mid_map_cache_value = None

    def _invalidate_frontier_stats_cache(self) -> None:
        self._latest_frontier_stats = None

    def _invalidate_obstacle_cache(self) -> None:
        self._cached_obstacle_integral = None

    def _invalidate_map_state_caches(self) -> None:
        self._invalidate_frontier_stats_cache()
        self._invalidate_obstacle_cache()
        self._invalidate_map_build_caches()

    def _invalidate_visit_cache(self) -> None:
        self._cached_visit_log_map = None
        self._cached_visit_log_max = 0.0
        self._cached_revisit_recency_map = None
        self._visit_cache_step = -1
        self._invalidate_map_build_caches()

    def _update_analysis_box(self) -> None:
        known = (self.map != INVISIBLE)
        if np.any(known):
            rows, cols = np.nonzero(known)
            known_r0 = int(rows.min())
            known_r1 = int(rows.max()) + 1
            known_c0 = int(cols.min())
            known_c1 = int(cols.max()) + 1
        else:
            known_r0 = 0
            known_r1 = int(self.map.shape[0])
            known_c0 = 0
            known_c1 = int(self.map.shape[1])

        margin = int(self.analysis_margin)
        self.analysis_box = AnalysisBox(
            r0=max(0, known_r0 - margin),
            r1=min(int(self.map.shape[0]), known_r1 + margin),
            c0=max(0, known_c0 - margin),
            c1=min(int(self.map.shape[1]), known_c1 + margin),
            margin=margin,
            known_r0=known_r0,
            known_r1=known_r1,
            known_c0=known_c0,
            known_c1=known_c1,
        )

    def _get_domain_buffers(
        self,
        domain_h: int,
        domain_w: int,
        include_visit: bool,
        visit_dtype: np.dtype | type[np.generic] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        visit_dtype_use = np.dtype(np.int32 if visit_dtype is None else visit_dtype)
        key = (int(domain_h), int(domain_w), bool(include_visit), visit_dtype_use.str)
        cached = self._domain_buffer_cache.get(key)
        if cached is None:
            sampled_map = np.empty((domain_h, domain_w), dtype=np.int8)
            sampled_frontier = np.empty((domain_h, domain_w), dtype=bool)
            sampled_visit = np.empty((domain_h, domain_w), dtype=visit_dtype_use) if include_visit else None
            cached = (sampled_map, sampled_frontier, sampled_visit)
            self._domain_buffer_cache[key] = cached
        return cached

    def _make_mid_map_cache_key(
        self,
        agent_state: Tuple[int, int],
        cfg: MidMapConfig,
    ) -> tuple[int, int, int, int, int, int, int, int, int, int, int]:
        return (
            int(self.step_count),
            int(agent_state[0]),
            int(agent_state[1]),
            int(cfg.mid_map_shape[0]),
            int(cfg.mid_map_shape[1]),
            int(cfg.world_window_shape[0]),
            int(cfg.world_window_shape[1]),
            int(self.map.shape[0]),
            int(self.map.shape[1]),
            int(self.origin_world_rc[0]),
            int(self.origin_world_rc[1]),
        )

    @staticmethod
    def _aggregate_density(flat_bin: np.ndarray, n_bins: int, values_2d: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
        denom = np.bincount(flat_bin, minlength=n_bins).astype(np.float32)
        denom = np.maximum(denom, 1.0)
        sums = np.bincount(flat_bin, weights=values_2d.reshape(-1).astype(np.float32), minlength=n_bins)
        return (sums / denom).reshape(out_h, out_w)

    @staticmethod
    def _grid_bin_edges(size: int, bins: int) -> np.ndarray:
        idx = np.arange(int(bins) + 1, dtype=np.int32)
        return ((idx * int(size)) // int(bins)).astype(np.int32)

    @staticmethod
    def _integral_channels(channels: np.ndarray) -> np.ndarray:
        channels_f = np.asarray(channels, dtype=np.float32)
        padded = np.pad(channels_f, ((0, 0), (1, 0), (1, 0)), mode="constant", constant_values=0.0)
        return padded.cumsum(axis=1, dtype=np.float32).cumsum(axis=2, dtype=np.float32)

    @staticmethod
    def _aggregate_edge_data(src_h: int, src_w: int, out_h: int, out_w: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        key = (int(src_h), int(src_w), int(out_h), int(out_w))
        cached = CumulativeBeliefMap._aggregate_edge_cache.get(key)
        if cached is None:
            row_edges = CumulativeBeliefMap._grid_bin_edges(src_h, out_h)
            col_edges = CumulativeBeliefMap._grid_bin_edges(src_w, out_w)
            denom = np.maximum(
                1,
                np.diff(row_edges).astype(np.int32)[:, None] * np.diff(col_edges).astype(np.int32)[None, :],
            ).astype(np.float32)
            cached = (row_edges, col_edges, denom)
            CumulativeBeliefMap._aggregate_edge_cache[key] = cached
        return cached

    @staticmethod
    def _aggregate_channels_to_grid(channels: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
        out = np.empty((int(channels.shape[0]), int(out_h), int(out_w)), dtype=np.float32)
        prefix = np.empty((int(channels.shape[0]), int(channels.shape[1]) + 1, int(channels.shape[2]) + 1), dtype=np.float32)
        CumulativeBeliefMap._aggregate_channels_to_grid_into(channels, out, prefix)
        return out

    @staticmethod
    def _aggregate_channels_to_grid_into(
        channels: np.ndarray,
        out: np.ndarray,
        prefix: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        ch = int(channels.shape[0])
        src_h = int(channels.shape[1])
        src_w = int(channels.shape[2])
        out_h = int(out.shape[1])
        out_w = int(out.shape[2])
        if out.shape != (ch, out_h, out_w):
            raise ValueError(f"aggregate output shape mismatch: got {out.shape}")

        row_edges, col_edges, denom = CumulativeBeliefMap._aggregate_edge_data(src_h, src_w, out_h, out_w)
        prefix_use = prefix
        if prefix_use is None:
            prefix_use = np.empty((ch, src_h + 1, src_w + 1), dtype=np.float32)
        elif prefix_use.shape != (ch, src_h + 1, src_w + 1):
            raise ValueError(
                "aggregate prefix shape mismatch: "
                f"expected {(ch, src_h + 1, src_w + 1)}, got {prefix_use.shape}"
            )

        prefix_use[:, 0, :] = 0.0
        prefix_use[:, :, 0] = 0.0
        prefix_use[:, 1:, 1:] = np.asarray(channels, dtype=np.float32)
        np.cumsum(prefix_use[:, 1:, 1:], axis=1, dtype=np.float32, out=prefix_use[:, 1:, 1:])
        np.cumsum(prefix_use[:, 1:, 1:], axis=2, dtype=np.float32, out=prefix_use[:, 1:, 1:])

        for channel_idx in range(ch):
            corners = prefix_use[channel_idx][row_edges[:, None], col_edges[None, :]]
            sums = corners[1:, 1:] - corners[:-1, 1:] - corners[1:, :-1] + corners[:-1, :-1]
            np.divide(sums, denom, out=out[channel_idx], casting="unsafe")
        return out

    def _get_mid_input_channels_buffer(self, world_h: int, world_w: int) -> np.ndarray:
        key = (int(world_h), int(world_w))
        cached = self._mid_input_channels_cache.get(key)
        if cached is None:
            cached = np.empty((MID_MAP_CHANNEL_COUNT, int(world_h), int(world_w)), dtype=np.float32)
            self._mid_input_channels_cache[key] = cached
        return cached

    def _get_mid_output_buffer(self, mid_h: int, mid_w: int) -> np.ndarray:
        key = (int(mid_h), int(mid_w))
        cached = self._mid_output_cache.get(key)
        if cached is None:
            cached = np.empty((MID_MAP_CHANNEL_COUNT, int(mid_h), int(mid_w)), dtype=np.float32)
            self._mid_output_cache[key] = cached
        return cached

    def _get_mid_integral_buffer(self, world_h: int, world_w: int) -> np.ndarray:
        key = (int(world_h), int(world_w))
        cached = self._mid_integral_cache.get(key)
        if cached is None:
            cached = np.empty((MID_MAP_CHANNEL_COUNT, int(world_h) + 1, int(world_w) + 1), dtype=np.float32)
            self._mid_integral_cache[key] = cached
        return cached

        prefix = CumulativeBeliefMap._integral_channels(channels)
        corners = prefix[:, row_edges[:, None], col_edges[None, :]]
        sums = corners[:, 1:, 1:] - corners[:, :-1, 1:] - corners[:, 1:, :-1] + corners[:, :-1, :-1]
        return (sums / denom[None, :, :]).astype(np.float32, copy=False)

    @staticmethod
    def _clip_dirty_rect(rect: DirtyRect, shape: tuple[int, int]) -> Optional[DirtyRect]:
        h, w = int(shape[0]), int(shape[1])
        r0 = max(0, min(h, int(rect.r0)))
        r1 = max(0, min(h, int(rect.r1)))
        c0 = max(0, min(w, int(rect.c0)))
        c1 = max(0, min(w, int(rect.c1)))
        if r0 >= r1 or c0 >= c1:
            return None
        return DirtyRect(r0=r0, r1=r1, c0=c0, c1=c1)

    @staticmethod
    def _dirty_rect_from_points(rows: np.ndarray, cols: np.ndarray) -> Optional[DirtyRect]:
        if rows.size <= 0 or cols.size <= 0:
            return None
        return DirtyRect(
            r0=int(rows.min()),
            r1=int(rows.max()) + 1,
            c0=int(cols.min()),
            c1=int(cols.max()) + 1,
        )

    @staticmethod
    def _normalize_dirty_rects(rects, shape: tuple[int, int]) -> list[DirtyRect]:
        out: list[DirtyRect] = []
        seen: set[tuple[int, int, int, int]] = set()
        for rect in rects:
            if rect is None:
                continue
            clipped = CumulativeBeliefMap._clip_dirty_rect(rect, shape)
            if clipped is None:
                continue
            key = (clipped.r0, clipped.r1, clipped.c0, clipped.c1)
            if key in seen:
                continue
            seen.add(key)
            out.append(clipped)
        return out

    @staticmethod
    def _expand_dirty_rect(rect: Optional[DirtyRect], radius: int) -> Optional[DirtyRect]:
        if rect is None:
            return None
        pad = int(max(0, radius))
        return DirtyRect(
            r0=int(rect.r0) - pad,
            r1=int(rect.r1) + pad,
            c0=int(rect.c0) - pad,
            c1=int(rect.c1) + pad,
        )

    def _recompute_frontier_full_bool(self) -> np.ndarray:
        return GridTopology.frontier_mask(
            self.map,
            min_unknown_neighbors=FRONTIER_MIN_UNKNOWN_NEIGHBORS,
            connectivity=FRONTIER_NEIGHBOR_CONNECTIVITY,
        )

    def _recompute_frontier_full_u8(self) -> np.ndarray:
        return self._recompute_frontier_full_bool().astype(np.uint8) * 255

    def debug_frontier_consistency_stats(self) -> FrontierConsistencyStats:
        full_bool = self._recompute_frontier_full_bool()
        mismatch = int(np.count_nonzero(self.frontier_bool != full_bool))
        return FrontierConsistencyStats(
            consistent=bool(mismatch == 0),
            mismatch_count=mismatch,
            frontier_source_uid=int(self.frontier_source_uid),
            frontier_revision=int(self.frontier_revision),
            map_shape=(int(self.map.shape[0]), int(self.map.shape[1])),
        )

    def debug_compare_frontier_with_full_recompute(self, *, assert_on_mismatch: bool = True) -> bool:
        stats = self.debug_frontier_consistency_stats()
        if (not stats.consistent) and assert_on_mismatch:
            raise AssertionError(
                "incremental frontier cache mismatch against full recompute: "
                f"{stats.mismatch_count} cells differ"
            )
        return bool(stats.consistent)

    def _update_frontier_dirty_rects(self, dirty_rects) -> None:
        rects = self._normalize_dirty_rects(dirty_rects, tuple(self.map.shape))
        if len(rects) <= 0:
            return

        for rect in rects:
            patch_r0 = max(0, int(rect.r0) - 1)
            patch_r1 = min(int(self.map.shape[0]), int(rect.r1) + 1)
            patch_c0 = max(0, int(rect.c0) - 1)
            patch_c1 = min(int(self.map.shape[1]), int(rect.c1) + 1)

            patch_frontier = GridTopology.frontier_mask(
                self.map[patch_r0:patch_r1, patch_c0:patch_c1],
                min_unknown_neighbors=FRONTIER_MIN_UNKNOWN_NEIGHBORS,
                connectivity=FRONTIER_NEIGHBOR_CONNECTIVITY,
            )
            inner = patch_frontier[
                int(rect.r0) - patch_r0:int(rect.r1) - patch_r0,
                int(rect.c0) - patch_c0:int(rect.c1) - patch_c0,
            ]
            self.frontier_bool[int(rect.r0):int(rect.r1), int(rect.c0):int(rect.c1)] = inner
            self.frontier_u8[int(rect.r0):int(rect.r1), int(rect.c0):int(rect.c1)] = inner.astype(np.uint8) * 255

        self.frontier_revision += 1
        self._invalidate_frontier_stats_cache()

    def _full_frontier_rebuild(self, *, bump_revision: bool = False) -> None:
        full_bool = self._recompute_frontier_full_bool()
        if self.frontier_bool.shape != full_bool.shape:
            self.frontier_bool = np.zeros_like(full_bool, dtype=bool)
            self.frontier_u8 = np.zeros_like(full_bool, dtype=np.uint8)
        self.frontier_bool[:, :] = full_bool
        self.frontier_u8[:, :] = full_bool.astype(np.uint8) * 255
        if bump_revision:
            self.frontier_revision += 1
        self._invalidate_frontier_stats_cache()

    def _build_seam_dirty_rects(
        self,
        *,
        pad_top: int,
        pad_left: int,
        pad_bottom: int,
        pad_right: int,
        old_h: int,
        old_w: int,
    ) -> tuple[DirtyRect, ...]:
        if old_h <= 0 or old_w <= 0:
            return tuple()

        rects: list[DirtyRect] = []
        if pad_top > 0:
            rects.append(DirtyRect(r0=pad_top, r1=pad_top + 1, c0=pad_left, c1=pad_left + old_w))
        if pad_bottom > 0:
            rects.append(DirtyRect(r0=pad_top + old_h - 1, r1=pad_top + old_h, c0=pad_left, c1=pad_left + old_w))
        if pad_left > 0:
            rects.append(DirtyRect(r0=pad_top, r1=pad_top + old_h, c0=pad_left, c1=pad_left + 1))
        if pad_right > 0:
            rects.append(
                DirtyRect(r0=pad_top, r1=pad_top + old_h, c0=pad_left + old_w - 1, c1=pad_left + old_w)
            )
        return tuple(rects)

    def _ensure_world_bounds(self, min_r: int, max_r: int, min_c: int, max_c: int) -> Optional[WorldBoundsExpansion]:
        cur_min_r, cur_min_c = self.origin_world_rc
        cur_max_r = cur_min_r + self.map.shape[0] - 1
        cur_max_c = cur_min_c + self.map.shape[1] - 1

        need_top = max(0, cur_min_r - min_r)
        need_left = max(0, cur_min_c - min_c)
        need_bottom = max(0, max_r - cur_max_r)
        need_right = max(0, max_c - cur_max_c)

        margin = int(self._growth_margin)
        pad_top = need_top + (margin if need_top > 0 else 0)
        pad_left = need_left + (margin if need_left > 0 else 0)
        pad_bottom = need_bottom + (margin if need_bottom > 0 else 0)
        pad_right = need_right + (margin if need_right > 0 else 0)

        if pad_top == 0 and pad_left == 0 and pad_bottom == 0 and pad_right == 0:
            return None

        old_map = self.map
        old_visit = self.visit_count
        old_last_step = self.last_visit_step
        old_frontier_bool = self.frontier_bool
        old_frontier_u8 = self.frontier_u8
        old_h = int(old_map.shape[0])
        old_w = int(old_map.shape[1])

        new_h = old_map.shape[0] + pad_top + pad_bottom
        new_w = old_map.shape[1] + pad_left + pad_right

        new_map = np.full((new_h, new_w), INVISIBLE, dtype=np.int8)
        new_visit = np.zeros((new_h, new_w), dtype=np.int32)
        new_last_step = np.full((new_h, new_w), -1, dtype=np.int32)
        new_frontier_bool = np.zeros((new_h, new_w), dtype=bool)
        new_frontier_u8 = np.zeros((new_h, new_w), dtype=np.uint8)

        r0, c0 = pad_top, pad_left
        new_map[r0:r0 + old_map.shape[0], c0:c0 + old_map.shape[1]] = old_map
        new_visit[r0:r0 + old_visit.shape[0], c0:c0 + old_visit.shape[1]] = old_visit
        new_last_step[r0:r0 + old_last_step.shape[0], c0:c0 + old_last_step.shape[1]] = old_last_step
        new_frontier_bool[r0:r0 + old_frontier_bool.shape[0], c0:c0 + old_frontier_bool.shape[1]] = old_frontier_bool
        new_frontier_u8[r0:r0 + old_frontier_u8.shape[0], c0:c0 + old_frontier_u8.shape[1]] = old_frontier_u8

        self.map = new_map
        self.visit_count = new_visit
        self.last_visit_step = new_last_step
        self.frontier_bool = new_frontier_bool
        self.frontier_u8 = new_frontier_u8
        self.origin_world_rc = (cur_min_r - pad_top, cur_min_c - pad_left)

        self._invalidate_visit_cache()
        self._invalidate_map_state_caches()
        return WorldBoundsExpansion(
            pad_top=pad_top,
            pad_left=pad_left,
            pad_bottom=pad_bottom,
            pad_right=pad_right,
            seam_dirty_rects=self._build_seam_dirty_rects(
                pad_top=pad_top,
                pad_left=pad_left,
                pad_bottom=pad_bottom,
                pad_right=pad_right,
                old_h=old_h,
                old_w=old_w,
            ),
        )

    def _record_visit_in_bounds(self, agent_state: Tuple[int, int]) -> None:
        ar, ac = int(agent_state[0]), int(agent_state[1])
        ir, ic = self.world_to_array((ar, ac))
        self.visit_count[ir, ic] += 1
        self.last_visit_step[ir, ic] = int(self.step_count)
        self._invalidate_visit_cache()

    def _count_reachable_known_free(self) -> int:
        free_known = (self.map == EMPTY)
        if not np.any(free_known):
            return 0

        rr, cc = np.where(free_known)
        wr = rr + int(self.origin_world_rc[0])
        wc = cc + int(self.origin_world_rc[1])

        inside = (
            (wr >= 0) & (wr < self.true_grid.shape[0]) &
            (wc >= 0) & (wc < self.true_grid.shape[1])
        )
        if not np.any(inside):
            return 0

        return int(self._reachable_free_mask[wr[inside], wc[inside]].sum())

    def _refresh_coverage(self) -> None:
        # Effective coverage numerator is reachable-known-free count in current belief.
        self.kpm_count = self._count_reachable_known_free()
        if self.tpm_count <= 0:
            self.coverage_rate = 0.0
        else:
            self.coverage_rate = float(min(1.0, max(0.0, round(float(self.kpm_count) / float(self.tpm_count), 4))))

    def update(self, agent_state: Tuple[int, int], local_snap: np.ndarray) -> Tuple[int, int, int]:
        snap = np.asarray(local_snap, dtype=np.int8)
        if snap.shape != self.local_shape:
            raise ValueError(f"local_snap shape mismatch: expected {self.local_shape}, got {snap.shape}")

        self.step_count += 1

        gr, gc = self._project_local_world(agent_state)
        visible = (snap != INVISIBLE)
        dirty_rects: list[DirtyRect] = []

        ar, ac = int(agent_state[0]), int(agent_state[1])
        if np.any(visible):
            wr = gr[visible]
            wc = gc[visible]
            min_r = min(ar, int(wr.min()))
            max_r = max(ar, int(wr.max()))
            min_c = min(ac, int(wc.min()))
            max_c = max(ac, int(wc.max()))
        else:
            wr = np.empty((0,), dtype=np.int32)
            wc = np.empty((0,), dtype=np.int32)
            min_r = ar
            max_r = ar
            min_c = ac
            max_c = ac

        expansion = self._ensure_world_bounds(min_r, max_r, min_c, max_c)
        if expansion is not None:
            dirty_rects.extend(expansion.seam_dirty_rects)

        self._record_visit_in_bounds(agent_state)

        if not np.any(visible):
            self._update_frontier_dirty_rects(dirty_rects)
            self._refresh_coverage()
            self._update_analysis_box()
            return 0, 0, 0

        vv = snap[visible]

        ir = wr - int(self.origin_world_rc[0])
        ic = wc - int(self.origin_world_rc[1])

        unseen = (self.map[ir, ic] == INVISIBLE)
        if not np.any(unseen):
            self._update_frontier_dirty_rects(dirty_rects)
            self._refresh_coverage()
            self._update_analysis_box()
            return 0, 0, 0

        wir = ir[unseen]
        wic = ic[unseen]
        wvv = vv[unseen]
        self.map[wir, wic] = wvv
        self._invalidate_map_state_caches()
        reveal_dirty = self._expand_dirty_rect(self._dirty_rect_from_points(wir, wic), radius=1)
        if reveal_dirty is not None:
            dirty_rects.append(reveal_dirty)
        self._update_frontier_dirty_rects(dirty_rects)

        updated = int(wvv.size)
        delta_empty = int((wvv == EMPTY).sum())
        delta_obstacle = updated - delta_empty

        self._refresh_coverage()
        self._update_analysis_box()
        return updated, delta_empty, delta_obstacle

    def get_frontier_u8(self, refresh: bool = False) -> np.ndarray:
        """
        Canonical frontier getter.

        frontier := known_free cells adjacent to orthogonally connected unknown
        cells in current belief map.
        Frontier semantics are rule-based and independent of truth-side
        reachability. Higher-level value assessment is handled downstream by the
        frontier token builder.

        Normal path returns the incrementally maintained frontier cache.
        refresh=True is a debug/full-recompute path and is not used by the
        standard training or inference pipeline.
        """
        if refresh:
            return self._recompute_frontier_full_u8()
        return self.frontier_u8

    def get_frontier_derived_stats(
        self,
        refresh: bool = False,
        frontier_u8: Optional[np.ndarray] = None,
    ) -> FrontierDerivedStats:
        """
        Return frontier-related reusable statistics with caching.

        These are summary auxiliaries for map/token channels and do not alter
        canonical frontier membership.
        """
        if (
            frontier_u8 is None
            and (not refresh)
            and (self._latest_frontier_stats is not None)
            and (self._latest_frontier_stats.frontier_revision == int(self.frontier_revision))
        ):
            return self._latest_frontier_stats

        t0 = time.perf_counter() if self.enable_timing else 0.0

        if frontier_u8 is None:
            frontier_u8 = self.get_frontier_u8(refresh=refresh)

        frontier_arr = np.asarray(frontier_u8, dtype=np.uint8)
        if frontier_arr.shape != self.map.shape:
            raise ValueError(
                f"frontier_u8 shape mismatch: expected {self.map.shape}, got {frontier_arr.shape}"
            )

        frontier_bool = (frontier_arr > 0)

        out = FrontierDerivedStats(
            frontier_u8=(frontier_bool.astype(np.uint8) * 255),
            frontier_bool=frontier_bool,
            frontier_source_uid=int(self.frontier_source_uid),
            frontier_revision=int(self.frontier_revision),
        )

        if not refresh:
            self._latest_frontier_stats = out
        if self.enable_timing:
            self.frontier_stats_time += time.perf_counter() - t0

        return out

    def get_visit_log_map(self, refresh: bool = False) -> Tuple[np.ndarray, float]:
        """Cached visit log map used by frontier cluster statistics."""
        if (
            (not refresh)
            and (self._cached_visit_log_map is not None)
            and (self._visit_cache_step == int(self.step_count))
        ):
            return self._cached_visit_log_map, self._cached_visit_log_max

        visit_log = np.log1p(self.visit_count.astype(np.float32))
        visit_log_max = float(np.max(visit_log)) if visit_log.size > 0 else 0.0

        self._cached_visit_log_map = visit_log
        self._cached_visit_log_max = visit_log_max
        self._visit_cache_step = int(self.step_count)
        return visit_log, visit_log_max

    def get_revisit_recency_map(self, refresh: bool = False) -> np.ndarray:
        """
        Canonical revisit/recency map reused by semantic parsing and local state.

        The map is derived from the existing visit counters and last-visit step,
        so the project keeps one coherent notion of revisit pressure instead of
        maintaining multiple overlapping tracker grids.
        """
        if (
            (not refresh)
            and (self._cached_revisit_recency_map is not None)
            and (self._visit_cache_step == int(self.step_count))
        ):
            return self._cached_revisit_recency_map

        visit_over = np.maximum(self.visit_count.astype(np.float32) - 1.0, 0.0)
        visit_over_max = float(np.max(visit_over)) if visit_over.size > 0 else 0.0
        if visit_over_max > 0.0:
            revisit = np.log1p(visit_over) / np.log1p(visit_over_max + 1.0)
        else:
            revisit = np.zeros_like(visit_over, dtype=np.float32)

        recency = np.zeros_like(revisit, dtype=np.float32)
        visited = (self.last_visit_step >= 0)
        if np.any(visited):
            delta = np.maximum(0, int(self.step_count) - self.last_visit_step[visited]).astype(np.float32)
            recency[visited] = np.exp(-delta / float(self.revisit_recency_decay_steps))

        revisit_recency = np.clip((0.6 * recency) + (0.4 * revisit), 0.0, 1.0).astype(np.float32, copy=False)
        self._cached_revisit_recency_map = revisit_recency
        self._visit_cache_step = int(self.step_count)
        return revisit_recency

    def get_obstacle_integral(self, refresh: bool = False) -> np.ndarray:
        """Cached obstacle integral image used by frontier-region obstacle-density queries."""
        if (not refresh) and (self._cached_obstacle_integral is not None):
            return self._cached_obstacle_integral

        h, w = int(self.map.shape[0]), int(self.map.shape[1])
        prefix = np.empty((h + 1, w + 1), dtype=np.int32)
        prefix[0, :] = 0
        prefix[:, 0] = 0
        prefix[1:, 1:] = (self.map == OBSTACLE)
        np.cumsum(prefix[1:, 1:], axis=0, dtype=np.int32, out=prefix[1:, 1:])
        np.cumsum(prefix[1:, 1:], axis=1, dtype=np.int32, out=prefix[1:, 1:])

        self._cached_obstacle_integral = prefix
        return prefix

    def _world_overlap_slices(
        self,
        domain_min_r: int,
        domain_max_r: int,
        domain_min_c: int,
        domain_max_c: int,
    ) -> Optional[Tuple[slice, slice, slice, slice]]:
        src_min_r = int(self.origin_world_rc[0])
        src_min_c = int(self.origin_world_rc[1])
        src_max_r = src_min_r + int(self.map.shape[0]) - 1
        src_max_c = src_min_c + int(self.map.shape[1]) - 1

        ov_min_r = max(int(domain_min_r), src_min_r)
        ov_max_r = min(int(domain_max_r), src_max_r)
        ov_min_c = max(int(domain_min_c), src_min_c)
        ov_max_c = min(int(domain_max_c), src_max_c)
        if ov_min_r > ov_max_r or ov_min_c > ov_max_c:
            return None

        src_rows = slice(ov_min_r - src_min_r, ov_max_r - src_min_r + 1)
        src_cols = slice(ov_min_c - src_min_c, ov_max_c - src_min_c + 1)
        dst_rows = slice(ov_min_r - int(domain_min_r), ov_max_r - int(domain_min_r) + 1)
        dst_cols = slice(ov_min_c - int(domain_min_c), ov_max_c - int(domain_min_c) + 1)
        return src_rows, src_cols, dst_rows, dst_cols

    def _extract_domain_arrays(
        self,
        domain_min_r: int,
        domain_max_r: int,
        domain_min_c: int,
        domain_max_c: int,
        frontier_map: Optional[np.ndarray],
        include_visit: bool,
        visit_source: Optional[np.ndarray] = None,
        sampled_map_out: Optional[np.ndarray] = None,
        sampled_frontier_out: Optional[np.ndarray] = None,
        sampled_visit_out: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        t0 = time.perf_counter() if self.enable_timing else 0.0
        domain_h = int(domain_max_r - domain_min_r + 1)
        domain_w = int(domain_max_c - domain_min_c + 1)

        sampled_map = sampled_map_out
        if sampled_map is None:
            sampled_map = np.empty((domain_h, domain_w), dtype=np.int8)
        elif sampled_map.shape != (domain_h, domain_w):
            raise ValueError(
                f"sampled_map_out shape mismatch: expected {(domain_h, domain_w)}, got {sampled_map.shape}"
            )
        sampled_map.fill(INVISIBLE)

        sampled_frontier: Optional[np.ndarray] = sampled_frontier_out
        if frontier_map is not None or sampled_frontier is not None:
            if sampled_frontier is None:
                sampled_frontier = np.empty((domain_h, domain_w), dtype=bool)
            elif sampled_frontier.shape != (domain_h, domain_w):
                raise ValueError(
                    "sampled_frontier_out shape mismatch: "
                    f"expected {(domain_h, domain_w)}, got {sampled_frontier.shape}"
                )
            sampled_frontier.fill(False)

        sampled_visit: Optional[np.ndarray]
        visit_source_use = self.visit_count if visit_source is None else visit_source
        if include_visit:
            sampled_visit = sampled_visit_out
            if sampled_visit is None:
                sampled_visit = np.empty((domain_h, domain_w), dtype=visit_source_use.dtype)
            elif sampled_visit.shape != (domain_h, domain_w):
                raise ValueError(
                    f"sampled_visit_out shape mismatch: expected {(domain_h, domain_w)}, got {sampled_visit.shape}"
                )
            sampled_visit.fill(0)
        else:
            sampled_visit = None

        overlap = self._world_overlap_slices(domain_min_r, domain_max_r, domain_min_c, domain_max_c)
        if overlap is not None:
            src_rows, src_cols, dst_rows, dst_cols = overlap
            sampled_map[dst_rows, dst_cols] = self.map[src_rows, src_cols]
            if frontier_map is not None and sampled_frontier is not None:
                sampled_frontier[dst_rows, dst_cols] = frontier_map[src_rows, src_cols]
            if sampled_visit is not None:
                sampled_visit[dst_rows, dst_cols] = visit_source_use[src_rows, src_cols]

        if self.enable_timing:
            self.domain_extract_time += time.perf_counter() - t0

        return sampled_map, sampled_frontier, sampled_visit

    def build_mid_map(
        self,
        agent_state: Tuple[int, int],
        config: Optional[MidMapConfig] = None,
        frontier_u8: Optional[np.ndarray] = None,
        frontier_stats: Optional[FrontierDerivedStats] = None,
        shared_artifacts=None,
    ) -> np.ndarray:
        """
        Build fixed-size agent-centered mid map with channel-first layout:
            shape = [4, H_m, W_m]

        Fixed channel order is MID_MAP_CHANNELS:
            unknown_density
            known_obstacle_density
            coarse_visit_density
            frontier_density

        This remains a fixed-range mid map:
            cumulative belief -> fixed world window -> coarse aggregation -> fixed mid map
        It is not a token replacement: frontier_density provides a coarse
        regional entrance-distribution background, while the token branch keeps
        the sparse frontier-candidate summary.
        """
        t0 = time.perf_counter() if self.enable_timing else 0.0
        cfg = config if config is not None else MidMapConfig()
        hm, wm = int(cfg.mid_map_shape[0]), int(cfg.mid_map_shape[1])
        wh, ww = int(cfg.world_window_shape[0]), int(cfg.world_window_shape[1])

        if hm <= 0 or wm <= 0 or wh <= 0 or ww <= 0:
            raise ValueError("mid_map_shape and world_window_shape must be positive")

        ar, ac = int(agent_state[0]), int(agent_state[1])
        r_min = ar - (wh // 2)
        c_min = ac - (ww // 2)
        r_max = r_min + wh - 1
        c_max = c_min + ww - 1

        cache_key = self._make_mid_map_cache_key(agent_state, cfg)
        if cache_key == self._mid_map_cache_key and self._mid_map_cache_value is not None:
            return self._mid_map_cache_value

        visit_log_map, visit_log_max = self.get_visit_log_map(refresh=False)
        frontier_stats_use = frontier_stats
        if frontier_stats_use is None:
            frontier_stats_use = self._shared_artifact_value(shared_artifacts, "frontier_stats")

        frontier_u8_use = frontier_u8
        if frontier_u8_use is None:
            frontier_u8_use = self._shared_artifact_value(shared_artifacts, "frontier_u8")

        if frontier_stats_use is None:
            frontier_stats_use = self.get_frontier_derived_stats(refresh=False, frontier_u8=frontier_u8_use)

        sampled_map_buf, sampled_frontier_buf, sampled_visit_log_buf = self._get_domain_buffers(
            wh,
            ww,
            include_visit=True,
            visit_dtype=np.float32,
        )
        sampled_map, sampled_frontier, sampled_visit_log = self._extract_domain_arrays(
            r_min,
            r_max,
            c_min,
            c_max,
            frontier_map=frontier_stats_use.frontier_bool,
            include_visit=True,
            visit_source=visit_log_map,
            sampled_map_out=sampled_map_buf,
            sampled_frontier_out=sampled_frontier_buf,
            sampled_visit_out=sampled_visit_log_buf,
        )

        input_channels = self._get_mid_input_channels_buffer(wh, ww)
        input_channels[0, :, :] = (sampled_map == INVISIBLE)
        input_channels[1, :, :] = (sampled_map == OBSTACLE)
        if visit_log_max > 0.0:
            np.divide(sampled_visit_log, np.float32(visit_log_max), out=input_channels[2, :, :], casting="unsafe")
        else:
            input_channels[2, :, :].fill(0.0)
        input_channels[3, :, :] = np.asarray(sampled_frontier, dtype=np.float32)

        agg_t0 = time.perf_counter() if self.enable_timing else 0.0
        aggregated_scratch = self._get_mid_output_buffer(hm, wm)
        aggregated = self._aggregate_channels_to_grid_into(
            input_channels,
            aggregated_scratch,
            prefix=self._get_mid_integral_buffer(wh, ww),
        )
        if self.enable_timing:
            self.aggregate_time += time.perf_counter() - agg_t0

        # Method split:
        # near = local geometry + short-term recency + global normalized position context
        # mid = regional unknown/obstacle structure + coarse visitation memory
        #     + coarse frontier-density background
        # token = sparse frontier candidate representation
        mid_map = aggregated.copy()

        self._mid_map_cache_key = cache_key
        self._mid_map_cache_value = mid_map
        if self.enable_timing:
            self.mid_map_time += time.perf_counter() - t0
        return mid_map

    def get_timing_stats(self) -> dict[str, float]:
        return {
            "frontier_stats_time": float(self.frontier_stats_time),
            "mid_map_time": float(self.mid_map_time),
            "domain_extract_time": float(self.domain_extract_time),
            "aggregate_time": float(self.aggregate_time),
        }
