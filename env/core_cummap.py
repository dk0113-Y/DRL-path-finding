from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from env.grid_topology import EMPTY, INVISIBLE, OBSTACLE, GridTopology


FRONTIER_MIN_UNKNOWN_NEIGHBORS = 1
FRONTIER_NEIGHBOR_CONNECTIVITY = 4


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

    The box is a strict known-region bounding box.
    It should not extend into extra unknown area.
    This avoids merging otherwise separable frontier-adjacent unknown
    structures through heuristic margin extension.
    In the current project, unknown topology beyond the known boundary should
    not be guessed at the representation layer.
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
        # Keep semantic analysis strictly inside the currently known-region
        # bounding box. Do not extend into extra unknown area via heuristic
        # margin, because unknown topology beyond the known boundary should not
        # be guessed at the representation layer.
        self.analysis_margin = 0

        # Light-weight growth buffer to avoid frequent near-edge reallocations.
        self._growth_margin = int(max(4, max(self.local_shape) // 2))

        # Effective coverage denominator from simulator-known truth map:
        # 4-neighborhood reachable free cells plus their orthogonally adjacent obstacle boundary.
        self._coverage_domain_mask = self._compute_coverage_domain_mask(start_state)
        self.tpm_count = int(self._coverage_domain_mask.sum())

        self.kpm_count = 0
        self.coverage_rate = 0.0

        # Dynamic belief map storage and world origin mapping.
        self.map = np.full((1, 1), INVISIBLE, dtype=np.int8)
        self.visit_count = np.zeros((1, 1), dtype=np.int32)
        self.step_count = 0
        self.origin_world_rc = (int(start_state[0]), int(start_state[1]))

        self.frontier_bool = np.zeros((1, 1), dtype=bool)
        self.frontier_u8 = np.zeros((1, 1), dtype=np.uint8)
        self.frontier_revision = 0
        self._latest_frontier_stats: Optional[FrontierDerivedStats] = None
        self._cached_visit_log_map: Optional[np.ndarray] = None
        self._cached_visit_log_max: float = 0.0
        self._cached_obstacle_integral: Optional[np.ndarray] = None
        self._visit_cache_step: int = -1

        self.update_time = 0.0
        self.local_projection_time = 0.0
        self.local_observation_merge_time = 0.0
        self.bounds_expand_time = 0.0
        self.visit_update_time = 0.0
        self.map_merge_time = 0.0
        self.frontier_dirty_update_time = 0.0
        self.frontier_full_rebuild_time = 0.0
        self.frontier_fetch_time = 0.0
        self.frontier_cache_invalidation_time = 0.0
        self.coverage_update_time = 0.0
        self.analysis_box_time = 0.0
        self.frontier_stats_time = 0.0
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

    def _timing_start(self) -> float:
        return time.perf_counter() if self.enable_timing else 0.0

    def _add_timing(self, attr: str, start: float) -> None:
        if self.enable_timing:
            setattr(self, attr, float(getattr(self, attr)) + (time.perf_counter() - start))

    def _compute_coverage_domain_mask(self, start_state: Tuple[int, int]) -> np.ndarray:
        """Truth-side coverage domain used by episode completion statistics."""
        free = GridTopology.free_mask(self.true_grid)
        reachable_free = GridTopology.bfs_reachable_4(free, start_state) & free
        if not np.any(reachable_free):
            return reachable_free

        padded = np.pad(reachable_free.astype(np.uint8), 1, mode="constant", constant_values=0)
        adjacent_reachable = (
            (padded[:-2, 1:-1] > 0)
            | (padded[2:, 1:-1] > 0)
            | (padded[1:-1, :-2] > 0)
            | (padded[1:-1, 2:] > 0)
        )
        obstacle_boundary = (self.true_grid == OBSTACLE) & adjacent_reachable
        return reachable_free | obstacle_boundary

    def _count_coverage_hits(self, world_rows: np.ndarray, world_cols: np.ndarray) -> int:
        if world_rows.size <= 0 or world_cols.size <= 0:
            return 0
        inside = (
            (world_rows >= 0)
            & (world_rows < int(self.true_grid.shape[0]))
            & (world_cols >= 0)
            & (world_cols < int(self.true_grid.shape[1]))
        )
        if not np.any(inside):
            return 0
        return int(self._coverage_domain_mask[world_rows[inside], world_cols[inside]].sum())

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

    def _invalidate_frontier_stats_cache(self) -> None:
        t0 = self._timing_start()
        self._latest_frontier_stats = None
        self._add_timing("frontier_cache_invalidation_time", t0)

    def _invalidate_obstacle_cache(self) -> None:
        self._cached_obstacle_integral = None

    def _invalidate_map_state_caches(self) -> None:
        self._invalidate_frontier_stats_cache()
        self._invalidate_obstacle_cache()

    def _invalidate_visit_cache(self) -> None:
        self._cached_visit_log_map = None
        self._cached_visit_log_max = 0.0
        self._visit_cache_step = -1

    def _update_analysis_box(self) -> None:
        t0 = self._timing_start()
        try:
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

            # analysis_box is the strict known-region bounding box. It should not
            # extend into extra unknown area, so the box equals the known bbox.
            margin = 0
            self.analysis_box = AnalysisBox(
                r0=known_r0,
                r1=known_r1,
                c0=known_c0,
                c1=known_c1,
                margin=margin,
                known_r0=known_r0,
                known_r1=known_r1,
                known_c0=known_c0,
                known_c1=known_c1,
            )
        finally:
            self._add_timing("analysis_box_time", t0)

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
        t0 = self._timing_start()
        try:
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
        finally:
            self._add_timing("frontier_dirty_update_time", t0)

    def _full_frontier_rebuild(self, *, bump_revision: bool = False) -> None:
        t0 = self._timing_start()
        try:
            full_bool = self._recompute_frontier_full_bool()
            if self.frontier_bool.shape != full_bool.shape:
                self.frontier_bool = np.zeros_like(full_bool, dtype=bool)
                self.frontier_u8 = np.zeros_like(full_bool, dtype=np.uint8)
            self.frontier_bool[:, :] = full_bool
            self.frontier_u8[:, :] = full_bool.astype(np.uint8) * 255
            if bump_revision:
                self.frontier_revision += 1
            self._invalidate_frontier_stats_cache()
        finally:
            self._add_timing("frontier_full_rebuild_time", t0)

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
        old_frontier_bool = self.frontier_bool
        old_frontier_u8 = self.frontier_u8
        old_h = int(old_map.shape[0])
        old_w = int(old_map.shape[1])

        new_h = old_map.shape[0] + pad_top + pad_bottom
        new_w = old_map.shape[1] + pad_left + pad_right

        new_map = np.full((new_h, new_w), INVISIBLE, dtype=np.int8)
        new_visit = np.zeros((new_h, new_w), dtype=np.int32)
        new_frontier_bool = np.zeros((new_h, new_w), dtype=bool)
        new_frontier_u8 = np.zeros((new_h, new_w), dtype=np.uint8)

        r0, c0 = pad_top, pad_left
        new_map[r0:r0 + old_map.shape[0], c0:c0 + old_map.shape[1]] = old_map
        new_visit[r0:r0 + old_visit.shape[0], c0:c0 + old_visit.shape[1]] = old_visit
        new_frontier_bool[r0:r0 + old_frontier_bool.shape[0], c0:c0 + old_frontier_bool.shape[1]] = old_frontier_bool
        new_frontier_u8[r0:r0 + old_frontier_u8.shape[0], c0:c0 + old_frontier_u8.shape[1]] = old_frontier_u8

        self.map = new_map
        self.visit_count = new_visit
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
        t0 = self._timing_start()
        try:
            ar, ac = int(agent_state[0]), int(agent_state[1])
            ir, ic = self.world_to_array((ar, ac))
            self.visit_count[ir, ic] += 1
            self._invalidate_visit_cache()
        finally:
            self._add_timing("visit_update_time", t0)

    def _refresh_coverage(self) -> None:
        t0 = self._timing_start()
        try:
            if self.tpm_count <= 0:
                self.coverage_rate = 0.0
            else:
                self.coverage_rate = float(min(1.0, max(0.0, round(float(self.kpm_count) / float(self.tpm_count), 4))))
        finally:
            self._add_timing("coverage_update_time", t0)

    def update(self, agent_state: Tuple[int, int], local_snap: np.ndarray) -> Tuple[int, int, int]:
        t_update = self._timing_start()
        try:
            snap = np.asarray(local_snap, dtype=np.int8)
            if snap.shape != self.local_shape:
                raise ValueError(f"local_snap shape mismatch: expected {self.local_shape}, got {snap.shape}")

            self.step_count += 1

            t0 = self._timing_start()
            gr, gc = self._project_local_world(agent_state)
            visible = (snap != INVISIBLE)
            has_visible = bool(np.any(visible))
            self._add_timing("local_projection_time", t0)

            dirty_rects: list[DirtyRect] = []

            t0 = self._timing_start()
            ar, ac = int(agent_state[0]), int(agent_state[1])
            if has_visible:
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
            self._add_timing("local_observation_merge_time", t0)

            t0 = self._timing_start()
            expansion = self._ensure_world_bounds(min_r, max_r, min_c, max_c)
            self._add_timing("bounds_expand_time", t0)
            if expansion is not None:
                dirty_rects.extend(expansion.seam_dirty_rects)

            self._record_visit_in_bounds(agent_state)

            if not has_visible:
                self._update_frontier_dirty_rects(dirty_rects)
                self._refresh_coverage()
                self._update_analysis_box()
                return 0, 0, 0

            t0 = self._timing_start()
            vv = snap[visible]

            ir = wr - int(self.origin_world_rc[0])
            ic = wc - int(self.origin_world_rc[1])

            unseen = (self.map[ir, ic] == INVISIBLE)
            has_unseen = bool(np.any(unseen))
            self._add_timing("map_merge_time", t0)
            if not has_unseen:
                self._update_frontier_dirty_rects(dirty_rects)
                self._refresh_coverage()
                self._update_analysis_box()
                return 0, 0, 0

            t0 = self._timing_start()
            wir = ir[unseen]
            wic = ic[unseen]
            wvv = vv[unseen]
            self.map[wir, wic] = wvv
            self.kpm_count += self._count_coverage_hits(wr[unseen], wc[unseen])
            self._invalidate_map_state_caches()
            reveal_dirty = self._expand_dirty_rect(self._dirty_rect_from_points(wir, wic), radius=1)
            if reveal_dirty is not None:
                dirty_rects.append(reveal_dirty)
            updated = int(wvv.size)
            delta_empty = int((wvv == EMPTY).sum())
            delta_obstacle = updated - delta_empty
            self._add_timing("map_merge_time", t0)

            self._update_frontier_dirty_rects(dirty_rects)

            self._refresh_coverage()
            self._update_analysis_box()
            return updated, delta_empty, delta_obstacle
        finally:
            self._add_timing("update_time", t_update)

    def get_frontier_u8(self, refresh: bool = False) -> np.ndarray:
        """
        Canonical frontier getter.

        frontier := known_free cells adjacent to orthogonally connected unknown
        cells in current belief map.
        Frontier semantics are rule-based and independent of truth-side
        reachability. Higher-level semantic organization is handled downstream
        by the shared semantic layer.

        Normal path returns the incrementally maintained frontier cache.
        refresh=True is a debug/full-recompute path and is not used by the
        standard training or inference pipeline.
        """
        t0 = self._timing_start()
        try:
            if refresh:
                t_rebuild = self._timing_start()
                out = self._recompute_frontier_full_u8()
                self._add_timing("frontier_full_rebuild_time", t_rebuild)
                return out
            return self.frontier_u8
        finally:
            self._add_timing("frontier_fetch_time", t0)

    def get_frontier_derived_stats(
        self,
        refresh: bool = False,
        frontier_u8: Optional[np.ndarray] = None,
    ) -> FrontierDerivedStats:
        """
        Return frontier-related reusable statistics with caching.

        These are summary auxiliaries for shared semantic consumers and do not alter
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

    def get_timing_stats(self) -> dict[str, float]:
        total_time_sec = float(self.update_time + self.frontier_fetch_time + self.frontier_stats_time)
        return {
            "total_time_sec": total_time_sec,
            "update_time": float(self.update_time),
            "local_projection_time": float(self.local_projection_time),
            "local_observation_merge_time": float(self.local_observation_merge_time),
            "bounds_expand_time": float(self.bounds_expand_time),
            "visit_update_time": float(self.visit_update_time),
            "map_merge_time": float(self.map_merge_time),
            "frontier_dirty_update_time": float(self.frontier_dirty_update_time),
            "frontier_full_rebuild_time": float(self.frontier_full_rebuild_time),
            "frontier_fetch_time": float(self.frontier_fetch_time),
            "frontier_cache_invalidation_time": float(self.frontier_cache_invalidation_time),
            "coverage_update_time": float(self.coverage_update_time),
            "analysis_box_time": float(self.analysis_box_time),
            "frontier_stats_time": float(self.frontier_stats_time),
        }
