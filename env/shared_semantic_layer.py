from __future__ import annotations

import heapq
import math
import time
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

try:
    from scipy import ndimage as _scipy_ndimage
except Exception:
    _scipy_ndimage = None

from env.core_cummap import AnalysisBox
from env.grid_topology import ACTIONS_8, EMPTY, INVISIBLE, OBSTACLE

_FOUR_CONNECTED_STRUCTURE = np.array(
    [[0, 1, 0],
     [1, 1, 1],
     [0, 1, 0]],
    dtype=np.uint8,
)
_MOVE_DR = np.asarray([int(dr) for dr, _ in ACTIONS_8], dtype=np.int32)
_MOVE_DC = np.asarray([int(dc) for _, dc in ACTIONS_8], dtype=np.int32)


@dataclass(frozen=True)
class SharedSemanticConfig:
    enable_timing: bool = False


@dataclass(frozen=True, slots=True)
class SparseMaskGeometry:
    r0: int
    c0: int
    mask: np.ndarray
    count: int
    _rows_cache: Optional[np.ndarray] = field(default=None, init=False, repr=False, compare=False)
    _cols_cache: Optional[np.ndarray] = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        mask_bool = np.ascontiguousarray(np.asarray(self.mask, dtype=bool))
        object.__setattr__(self, "mask", mask_bool)
        object.__setattr__(self, "count", int(self.count))

    @classmethod
    def empty(cls, *, r0: int = 0, c0: int = 0) -> "SparseMaskGeometry":
        return cls(r0=int(r0), c0=int(c0), mask=np.zeros((0, 0), dtype=bool), count=0)

    @property
    def shape(self) -> tuple[int, int]:
        return int(self.mask.shape[0]), int(self.mask.shape[1])

    def _cached_coords(self) -> tuple[np.ndarray, np.ndarray]:
        rows = self._rows_cache
        cols = self._cols_cache
        if rows is not None and cols is not None:
            return rows, cols
        if self.count <= 0:
            rows = np.zeros((0,), dtype=np.int32)
            cols = np.zeros((0,), dtype=np.int32)
        else:
            rows_local, cols_local = np.nonzero(self.mask)
            rows = rows_local.astype(np.int32, copy=False) + int(self.r0)
            cols = cols_local.astype(np.int32, copy=False) + int(self.c0)
        object.__setattr__(self, "_rows_cache", rows)
        object.__setattr__(self, "_cols_cache", cols)
        return rows, cols

    @property
    def rows(self) -> np.ndarray:
        return self._cached_coords()[0]

    @property
    def cols(self) -> np.ndarray:
        return self._cached_coords()[1]

    def paint_to_local_canvas(
        self,
        canvas_mask: np.ndarray,
        *,
        agent_arr: tuple[int, int],
        local_shape: tuple[int, int],
    ) -> None:
        if self.count <= 0:
            return
        h = int(local_shape[0])
        w = int(local_shape[1])
        center_r = h // 2
        center_c = w // 2
        local_arr_r0 = int(agent_arr[0]) - center_r
        local_arr_c0 = int(agent_arr[1]) - center_c
        local_arr_r1 = local_arr_r0 + h
        local_arr_c1 = local_arr_c0 + w

        geom_r0 = int(self.r0)
        geom_c0 = int(self.c0)
        geom_r1 = geom_r0 + int(self.mask.shape[0])
        geom_c1 = geom_c0 + int(self.mask.shape[1])
        overlap_r0 = max(geom_r0, local_arr_r0)
        overlap_c0 = max(geom_c0, local_arr_c0)
        overlap_r1 = min(geom_r1, local_arr_r1)
        overlap_c1 = min(geom_c1, local_arr_c1)
        if overlap_r0 >= overlap_r1 or overlap_c0 >= overlap_c1:
            return

        src_r0 = overlap_r0 - geom_r0
        src_r1 = overlap_r1 - geom_r0
        src_c0 = overlap_c0 - geom_c0
        src_c1 = overlap_c1 - geom_c0
        dst_r0 = overlap_r0 - local_arr_r0
        dst_r1 = overlap_r1 - local_arr_r0
        dst_c0 = overlap_c0 - local_arr_c0
        dst_c1 = overlap_c1 - local_arr_c0
        src_mask = self.mask[src_r0:src_r1, src_c0:src_c1]
        dst = canvas_mask[dst_r0:dst_r1, dst_c0:dst_c1]
        dst[src_mask] = 1.0


@dataclass(frozen=True, slots=True)
class FrontierEntryCluster:
    entry_index: int
    block_index: int
    support_geometry: SparseMaskGeometry
    boundary_geometry: SparseMaskGeometry
    entry_dir: tuple[float, float]
    entry_dist: float
    entry_width: float
    entry_clearance: float
    entry_local_revisit_pressure: float
    support_area: int

    @property
    def rows(self) -> np.ndarray:
        return self.support_geometry.rows

    @property
    def cols(self) -> np.ndarray:
        return self.support_geometry.cols

    @property
    def boundary_rows(self) -> np.ndarray:
        return self.boundary_geometry.rows

    @property
    def boundary_cols(self) -> np.ndarray:
        return self.boundary_geometry.cols

    def paint_to_local_canvas(
        self,
        canvas_mask: np.ndarray,
        *,
        agent_arr: tuple[int, int],
        local_shape: tuple[int, int],
    ) -> None:
        self.support_geometry.paint_to_local_canvas(
            canvas_mask,
            agent_arr=agent_arr,
            local_shape=local_shape,
        )


@dataclass(frozen=True, slots=True)
class AccessibleUnknownBlock:
    block_index: int
    block_geometry: SparseMaskGeometry
    block_area: int
    block_bbox_shape: tuple[float, float, float]
    entry_count: int
    entries: tuple[FrontierEntryCluster, ...]
    nearest_entry_dist: float
    opportunity_score: float

    @property
    def rows(self) -> np.ndarray:
        return self.block_geometry.rows

    @property
    def cols(self) -> np.ndarray:
        return self.block_geometry.cols

    def paint_to_local_canvas(
        self,
        canvas_mask: np.ndarray,
        *,
        agent_arr: tuple[int, int],
        local_shape: tuple[int, int],
    ) -> None:
        self.block_geometry.paint_to_local_canvas(
            canvas_mask,
            agent_arr=agent_arr,
            local_shape=local_shape,
        )


@dataclass(frozen=True)
class SharedSemanticSnapshot:
    analysis_box: AnalysisBox
    accessible_blocks: tuple[AccessibleUnknownBlock, ...]
    main_block_index: Optional[int]
    total_accessible_unknown_area: int
    top1_block_area_ratio: float
    scene_orderliness: float
    main_block_area: float
    main_block_entry_count: float
    nearest_main_entry_dist: float

    def main_block(self) -> Optional[AccessibleUnknownBlock]:
        if self.main_block_index is None:
            return None
        for block in self.accessible_blocks:
            if int(block.block_index) == int(self.main_block_index):
                return block
        return None

    def metrics(self) -> dict[str, float]:
        return {
            "accessible_block_count": float(len(self.accessible_blocks)),
            "total_accessible_unknown_area": float(self.total_accessible_unknown_area),
            "top1_block_area_ratio": float(self.top1_block_area_ratio),
            "scene_orderliness": float(self.scene_orderliness),
            "main_block_area": float(self.main_block_area),
            "main_block_entry_count": float(self.main_block_entry_count),
            "nearest_main_entry_dist": float(self.nearest_main_entry_dist),
        }

def _label_components_2d(mask: np.ndarray) -> tuple[np.ndarray, int]:
    mask_bool = np.asarray(mask, dtype=bool)
    if not np.any(mask_bool):
        return np.zeros(mask_bool.shape, dtype=np.int32), 0

    if _scipy_ndimage is not None:
        labels, count = _scipy_ndimage.label(mask_bool, structure=_FOUR_CONNECTED_STRUCTURE)
        return np.asarray(labels, dtype=np.int32), int(count)

    h, w = mask_bool.shape
    labels = np.zeros((h, w), dtype=np.int32)
    next_label = 1
    for start_r, start_c in zip(*np.nonzero(mask_bool)):
        if labels[start_r, start_c] > 0:
            continue
        queue = [(int(start_r), int(start_c))]
        labels[start_r, start_c] = next_label
        while queue:
            r, c = queue.pop()
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr = r + dr
                nc = c + dc
                if nr < 0 or nr >= h or nc < 0 or nc >= w:
                    continue
                if (not mask_bool[nr, nc]) or labels[nr, nc] > 0:
                    continue
                labels[nr, nc] = next_label
                queue.append((nr, nc))
        next_label += 1
    return labels, int(next_label - 1)


def _find_objects(labels: np.ndarray, count: int):
    if count <= 0:
        return []
    if _scipy_ndimage is not None:
        return list(_scipy_ndimage.find_objects(labels, max_label=int(count)))

    objects = []
    for label_id in range(1, int(count) + 1):
        coords = np.argwhere(labels == label_id)
        if coords.size <= 0:
            objects.append(None)
            continue
        mins = coords.min(axis=0)
        maxs = coords.max(axis=0) + 1
        objects.append(tuple(slice(int(mn), int(mx)) for mn, mx in zip(mins.tolist(), maxs.tolist())))
    return objects


def _orth_dilate(mask: np.ndarray, *, steps: int, restrict_mask: Optional[np.ndarray] = None) -> np.ndarray:
    out = np.asarray(mask, dtype=bool).copy()
    if int(steps) <= 0:
        return out

    for _ in range(int(steps)):
        p = np.pad(out, 1, mode="constant", constant_values=False)
        expanded = (
            out |
            p[:-2, 1:-1] |
            p[2:, 1:-1] |
            p[1:-1, :-2] |
            p[1:-1, 2:]
        )
        if restrict_mask is not None:
            expanded &= np.asarray(restrict_mask, dtype=bool)
        out = expanded
    return out


def _mask_entropy(values: Sequence[int]) -> float:
    if len(values) <= 1:
        return 0.0
    arr = np.asarray(values, dtype=np.float32)
    total = float(arr.sum())
    if total <= 0.0:
        return 0.0
    probs = arr / total
    probs = probs[probs > 0.0]
    return float(-(probs * np.log(probs)).sum())


def _build_move_valid_stack(free_box: np.ndarray) -> np.ndarray:
    free = np.asarray(free_box, dtype=bool)
    h, w = free.shape
    move_valid = np.zeros((len(ACTIONS_8), h, w), dtype=bool)

    move_valid[0, 1:, :] = free[1:, :] & free[:-1, :]
    move_valid[2, :, :-1] = free[:, :-1] & free[:, 1:]
    move_valid[4, :-1, :] = free[:-1, :] & free[1:, :]
    move_valid[6, :, 1:] = free[:, 1:] & free[:, :-1]

    move_valid[1, 1:, :-1] = free[1:, :-1] & free[:-1, 1:] & free[:-1, :-1] & free[1:, 1:]
    move_valid[3, :-1, :-1] = free[:-1, :-1] & free[1:, 1:] & free[1:, :-1] & free[:-1, 1:]
    move_valid[5, :-1, 1:] = free[:-1, 1:] & free[1:, :-1] & free[1:, 1:] & free[:-1, :-1]
    move_valid[7, 1:, 1:] = free[1:, 1:] & free[:-1, :-1] & free[:-1, 1:] & free[1:, :-1]
    return move_valid


def _chebyshev_distance_to_bbox(
    r: int,
    c: int,
    *,
    r0: int,
    r1: int,
    c0: int,
    c1: int,
) -> int:
    if r < int(r0):
        dr = int(r0) - int(r)
    elif r >= int(r1):
        dr = int(r) - int(r1) + 1
    else:
        dr = 0

    if c < int(c0):
        dc = int(c0) - int(c)
    elif c >= int(c1):
        dc = int(c) - int(c1) + 1
    else:
        dc = 0
    return max(dr, dc)


def _astar_exact_distance_to_target(
    *,
    move_valid: np.ndarray,
    start: tuple[int, int],
    target_r0: int,
    target_c0: int,
    target_mask: np.ndarray,
    seen_stamp: np.ndarray,
    g_score: np.ndarray,
    search_token: int,
) -> tuple[Optional[int], Optional[int], float, int]:
    target_mask_bool = np.asarray(target_mask, dtype=bool)
    if not np.any(target_mask_bool):
        return None, None, float("inf"), int(search_token)

    sr, sc = int(start[0]), int(start[1])
    h = int(move_valid.shape[1])
    w = int(move_valid.shape[2])
    if sr < 0 or sr >= h or sc < 0 or sc >= w:
        return None, None, float("inf"), int(search_token)

    target_h, target_w = target_mask_bool.shape
    target_r1 = int(target_r0) + int(target_h)
    target_c1 = int(target_c0) + int(target_w)

    def _target_contains(r: int, c: int) -> bool:
        if r < int(target_r0) or r >= target_r1 or c < int(target_c0) or c >= target_c1:
            return False
        return bool(target_mask_bool[r - int(target_r0), c - int(target_c0)])

    if _target_contains(sr, sc):
        return sr, sc, 0.0, int(search_token)

    token = int(search_token) + 1
    if token >= np.iinfo(seen_stamp.dtype).max:
        seen_stamp.fill(0)
        token = 1

    seen_stamp[sr, sc] = token
    g_score[sr, sc] = 0
    heap: list[tuple[int, int, int, int]] = [
        (
            _chebyshev_distance_to_bbox(sr, sc, r0=target_r0, r1=target_r1, c0=target_c0, c1=target_c1),
            0,
            sr,
            sc,
        )
    ]

    while heap:
        _, cur_g, r, c = heapq.heappop(heap)
        if seen_stamp[r, c] != token or int(g_score[r, c]) != int(cur_g):
            continue
        if _target_contains(r, c):
            return int(r), int(c), float(cur_g), token

        valid_moves = move_valid[:, r, c]
        for action_idx in range(int(valid_moves.shape[0])):
            if not bool(valid_moves[action_idx]):
                continue
            nr = int(r + _MOVE_DR[action_idx])
            nc = int(c + _MOVE_DC[action_idx])
            ng = int(cur_g) + 1
            if seen_stamp[nr, nc] == token and ng >= int(g_score[nr, nc]):
                continue
            seen_stamp[nr, nc] = token
            g_score[nr, nc] = ng
            heapq.heappush(
                heap,
                (
                    ng + _chebyshev_distance_to_bbox(
                        nr,
                        nc,
                        r0=target_r0,
                        r1=target_r1,
                        c0=target_c0,
                        c1=target_c1,
                    ),
                    ng,
                    nr,
                    nc,
                ),
            )

    return None, None, float("inf"), token


class SharedSemanticLayer:
    """
    Shared semantic layer built on top of the cumulative belief map.

    It first turns the belief map into exploration-semantic objects:
      Accessible Unknown Block (mother)
      Frontier Entry Cluster (child)

    Then the advantage/value state builders reuse the same snapshot instead of
    re-deriving incompatible local/global views.
    """

    def __init__(self, config: Optional[SharedSemanticConfig] = None):
        self.config = config if config is not None else SharedSemanticConfig()
        self._timing_enabled = bool(self.config.enable_timing)
        self.analysis_time = 0.0

    @staticmethod
    def _analysis_arrays(cum_map) -> tuple[AnalysisBox, np.ndarray, np.ndarray, np.ndarray]:
        box = cum_map.analysis_box
        map_view = np.asarray(cum_map.map, dtype=np.int8)
        revisit_map = np.asarray(cum_map.get_revisit_recency_map(refresh=False), dtype=np.float32)
        box_map = map_view[box.r0:box.r1, box.c0:box.c1]
        revisit_box = revisit_map[box.r0:box.r1, box.c0:box.c1]
        free_box = (box_map == EMPTY)
        return box, box_map, revisit_box, free_box

    @staticmethod
    def _neighbor_label_stack(unknown_labels: np.ndarray) -> np.ndarray:
        padded = np.pad(unknown_labels, 1, mode="constant", constant_values=0)
        return np.stack(
            (
                padded[:-2, 1:-1],
                padded[2:, 1:-1],
                padded[1:-1, :-2],
                padded[1:-1, 2:],
            ),
            axis=0,
        )

    @staticmethod
    def _entry_seed_assignments(
        unknown_labels: np.ndarray,
        block_areas: np.ndarray,
        free_mask: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        neighbor_labels = SharedSemanticLayer._neighbor_label_stack(unknown_labels)
        seed_mask = np.asarray(free_mask, dtype=bool) & np.any(neighbor_labels > 0, axis=0)
        assignments = np.full(unknown_labels.shape, -1, dtype=np.int32)
        if not np.any(seed_mask):
            return seed_mask, assignments

        candidates = neighbor_labels[:, seed_mask]
        valid = candidates > 0
        safe_block_idx = np.clip(candidates - 1, 0, max(0, int(block_areas.shape[0]) - 1))
        counts = np.sum(
            (candidates[:, None, :] == candidates[None, :, :]) &
            valid[:, None, :] &
            valid[None, :, :],
            axis=1,
            dtype=np.int16,
        )
        areas = np.where(valid, block_areas[safe_block_idx], -1)

        best_labels = candidates[0].copy()
        best_counts = counts[0].copy()
        best_areas = areas[0].copy()
        best_valid = valid[0].copy()
        for cand_idx in range(1, 4):
            cand_valid = valid[cand_idx]
            better = cand_valid & (
                (~best_valid) |
                (counts[cand_idx] > best_counts) |
                ((counts[cand_idx] == best_counts) & (areas[cand_idx] > best_areas)) |
                (
                    (counts[cand_idx] == best_counts) &
                    (areas[cand_idx] == best_areas) &
                    (candidates[cand_idx] < best_labels)
                )
            )
            best_labels = np.where(better, candidates[cand_idx], best_labels)
            best_counts = np.where(better, counts[cand_idx], best_counts)
            best_areas = np.where(better, areas[cand_idx], best_areas)
            best_valid = best_valid | cand_valid

        assignments[seed_mask] = best_labels.astype(np.int32, copy=False) - 1
        return seed_mask, assignments

    @staticmethod
    def _entry_direction(agent_arr: tuple[int, int], target_arr: tuple[int, int]) -> tuple[float, float]:
        dr = float(int(target_arr[0]) - int(agent_arr[0]))
        dc = float(int(target_arr[1]) - int(agent_arr[1]))
        norm = math.hypot(dr, dc)
        if norm <= 1e-6:
            return 0.0, 0.0
        return dr / norm, dc / norm

    @staticmethod
    def _bbox_shape_from_mask(mask_geometry: SparseMaskGeometry) -> tuple[float, float, float]:
        height = float(int(mask_geometry.shape[0]))
        width = float(int(mask_geometry.shape[1]))
        aspect = width / max(1.0, height)
        return height, width, aspect

    @staticmethod
    def _block_priority_score(
        *,
        block_area: int,
        nearest_entry_dist: float,
        best_entry_clearance: float,
        lowest_entry_revisit_pressure: float,
        analysis_diagonal: float,
    ) -> float:
        dist_norm = float(np.clip(nearest_entry_dist / max(1.0, analysis_diagonal), 0.0, 1.0))
        return float(
            math.log1p(float(block_area))
            + (0.25 * float(best_entry_clearance))
            - (0.20 * float(lowest_entry_revisit_pressure))
            - (0.35 * dist_norm)
        )

    @staticmethod
    def _entry_clearance(
        box_map: np.ndarray,
        support_mask: np.ndarray,
        *,
        pad_row0: int,
        pad_col0: int,
    ) -> float:
        local_h, local_w = support_mask.shape
        pad_r0 = max(0, int(pad_row0) - 1)
        pad_r1 = min(int(box_map.shape[0]), int(pad_row0) + int(local_h) + 1)
        pad_c0 = max(0, int(pad_col0) - 1)
        pad_c1 = min(int(box_map.shape[1]), int(pad_col0) + int(local_w) + 1)
        support_padded = np.zeros((int(pad_r1 - pad_r0), int(pad_c1 - pad_c0)), dtype=bool)
        dst_r0 = int(pad_row0) - int(pad_r0)
        dst_c0 = int(pad_col0) - int(pad_c0)
        support_padded[dst_r0:dst_r0 + int(local_h), dst_c0:dst_c0 + int(local_w)] = np.asarray(
            support_mask,
            dtype=bool,
        )
        clearance_region = _orth_dilate(support_padded, steps=1)
        area = int(np.count_nonzero(clearance_region))
        if area <= 0:
            return 0.0
        obstacle_ratio = float(
            np.count_nonzero((box_map[pad_r0:pad_r1, pad_c0:pad_c1] == OBSTACLE) & clearance_region)
        ) / float(area)
        return float(np.clip(1.0 - obstacle_ratio, 0.0, 1.0))

    @staticmethod
    def _empty_snapshot(analysis_box: AnalysisBox) -> SharedSemanticSnapshot:
        return SharedSemanticSnapshot(
            analysis_box=analysis_box,
            accessible_blocks=tuple(),
            main_block_index=None,
            total_accessible_unknown_area=0,
            top1_block_area_ratio=0.0,
            scene_orderliness=1.0,
            main_block_area=0.0,
            main_block_entry_count=0.0,
            nearest_main_entry_dist=float("nan"),
        )

    def analyze(
        self,
        cum_map,
        agent_state: tuple[int, int],
    ) -> SharedSemanticSnapshot:
        t0 = time.perf_counter() if self._timing_enabled else 0.0
        analysis_box, box_map, revisit_box, free_box = self._analysis_arrays(cum_map)
        unknown_labels, block_count = _label_components_2d(box_map == INVISIBLE)
        if block_count <= 0:
            snapshot = self._empty_snapshot(analysis_box)
            if self._timing_enabled:
                self.analysis_time += time.perf_counter() - t0
            return snapshot

        block_areas = np.bincount(unknown_labels[unknown_labels > 0], minlength=int(block_count) + 1)[1:].astype(
            np.int32,
            copy=False,
        )
        seed_mask, seed_assignments = self._entry_seed_assignments(unknown_labels, block_areas, free_box)
        if not np.any(seed_mask):
            snapshot = self._empty_snapshot(analysis_box)
            if self._timing_enabled:
                self.analysis_time += time.perf_counter() - t0
            return snapshot

        seed_rows, seed_cols = np.nonzero(seed_mask)
        seed_block_ids = seed_assignments[seed_rows, seed_cols]
        active_block_ids = np.unique(seed_block_ids)
        active_block_ids = active_block_ids[active_block_ids >= 0]
        if active_block_ids.size <= 0:
            snapshot = self._empty_snapshot(analysis_box)
            if self._timing_enabled:
                self.analysis_time += time.perf_counter() - t0
            return snapshot

        box_h, box_w = free_box.shape
        seed_row_min = np.full(int(block_count), int(box_h), dtype=np.int32)
        seed_row_max = np.full(int(block_count), -1, dtype=np.int32)
        seed_col_min = np.full(int(block_count), int(box_w), dtype=np.int32)
        seed_col_max = np.full(int(block_count), -1, dtype=np.int32)
        np.minimum.at(seed_row_min, seed_block_ids, seed_rows.astype(np.int32, copy=False))
        np.maximum.at(seed_row_max, seed_block_ids, seed_rows.astype(np.int32, copy=False))
        np.minimum.at(seed_col_min, seed_block_ids, seed_cols.astype(np.int32, copy=False))
        np.maximum.at(seed_col_max, seed_block_ids, seed_cols.astype(np.int32, copy=False))

        move_valid = _build_move_valid_stack(free_box)
        search_seen = np.zeros((int(box_h), int(box_w)), dtype=np.int32)
        search_gscore = np.zeros((int(box_h), int(box_w)), dtype=np.int32)
        search_token = 0
        agent_arr = cum_map.world_to_array(agent_state)
        agent_box_rc = (int(agent_arr[0]) - int(analysis_box.r0), int(agent_arr[1]) - int(analysis_box.c0))

        block_objects = _find_objects(unknown_labels, block_count)
        accessible_blocks: list[AccessibleUnknownBlock] = []
        analysis_diagonal = math.hypot(float(max(1, analysis_box.shape[0])), float(max(1, analysis_box.shape[1])))
        next_entry_index = 0

        for block_id in active_block_ids.tolist():
            local_r0 = max(0, int(seed_row_min[block_id]) - 2)
            local_r1 = min(int(box_h), int(seed_row_max[block_id]) + 3)
            local_c0 = max(0, int(seed_col_min[block_id]) - 2)
            local_c1 = min(int(box_w), int(seed_col_max[block_id]) + 3)
            if local_r0 >= local_r1 or local_c0 >= local_c1:
                continue

            local_seed_mask = (seed_assignments[local_r0:local_r1, local_c0:local_c1] == int(block_id))
            if not np.any(local_seed_mask):
                continue
            local_support_mask = _orth_dilate(
                local_seed_mask,
                steps=2,
                restrict_mask=free_box[local_r0:local_r1, local_c0:local_c1],
            )
            entry_labels, entry_count = _label_components_2d(local_support_mask)
            if entry_count <= 0:
                continue

            entry_objects = _find_objects(entry_labels, entry_count)
            entry_areas = np.bincount(entry_labels[entry_labels > 0], minlength=int(entry_count) + 1)[1:].astype(
                np.int32,
                copy=False,
            )
            boundary_counts = np.bincount(
                entry_labels[local_seed_mask],
                minlength=int(entry_count) + 1,
            )[1:].astype(np.int32, copy=False)
            if _scipy_ndimage is not None:
                revisit_means = np.asarray(
                    _scipy_ndimage.mean(
                        revisit_box[local_r0:local_r1, local_c0:local_c1],
                        labels=entry_labels,
                        index=np.arange(1, int(entry_count) + 1, dtype=np.int32),
                    ),
                    dtype=np.float32,
                )
            else:
                revisit_means = np.zeros((int(entry_count),), dtype=np.float32)
            entries: list[FrontierEntryCluster] = []
            for entry_label in range(1, int(entry_count) + 1):
                entry_obj = entry_objects[entry_label - 1]
                if entry_obj is None:
                    continue
                support_local = (entry_labels[entry_obj] == int(entry_label))
                if not np.any(support_local):
                    continue

                boundary_count = int(boundary_counts[entry_label - 1])
                if boundary_count <= 0:
                    continue
                boundary_local = support_local & local_seed_mask[entry_obj]
                support_r0 = int(local_r0) + int(entry_obj[0].start)
                support_c0 = int(local_c0) + int(entry_obj[1].start)
                best_r_local, best_c_local, entry_dist, search_token = _astar_exact_distance_to_target(
                    move_valid=move_valid,
                    start=agent_box_rc,
                    target_r0=support_r0,
                    target_c0=support_c0,
                    target_mask=support_local,
                    seen_stamp=search_seen,
                    g_score=search_gscore,
                    search_token=search_token,
                )
                if best_r_local is None or best_c_local is None or not np.isfinite(entry_dist):
                    continue

                support_geometry = SparseMaskGeometry(
                    r0=int(analysis_box.r0) + support_r0,
                    c0=int(analysis_box.c0) + support_c0,
                    mask=support_local,
                    count=int(entry_areas[entry_label - 1]),
                )
                boundary_geometry = SparseMaskGeometry(
                    r0=int(analysis_box.r0) + support_r0,
                    c0=int(analysis_box.c0) + support_c0,
                    mask=boundary_local,
                    count=boundary_count,
                )
                entries.append(
                    FrontierEntryCluster(
                        entry_index=next_entry_index,
                        block_index=int(block_id),
                        support_geometry=support_geometry,
                        boundary_geometry=boundary_geometry,
                        entry_dir=self._entry_direction(
                            agent_arr,
                            (
                                int(analysis_box.r0) + best_r_local,
                                int(analysis_box.c0) + best_c_local,
                            ),
                        ),
                        entry_dist=float(entry_dist),
                        entry_width=float(boundary_count),
                        entry_clearance=self._entry_clearance(
                            box_map,
                            support_local,
                            pad_row0=support_r0,
                            pad_col0=support_c0,
                        ),
                        entry_local_revisit_pressure=float(revisit_means[entry_label - 1]),
                        support_area=int(entry_areas[entry_label - 1]),
                    )
                )
                next_entry_index += 1

            if len(entries) <= 0:
                continue

            block_obj = block_objects[int(block_id)]
            if block_obj is None:
                continue
            block_mask = (unknown_labels[block_obj] == int(block_id) + 1)
            block_geometry = SparseMaskGeometry(
                r0=int(analysis_box.r0) + int(block_obj[0].start),
                c0=int(analysis_box.c0) + int(block_obj[1].start),
                mask=block_mask,
                count=int(block_areas[int(block_id)]),
            )
            nearest_entry_dist = min(float(entry.entry_dist) for entry in entries)
            best_entry_clearance = max(float(entry.entry_clearance) for entry in entries)
            lowest_entry_revisit_pressure = min(float(entry.entry_local_revisit_pressure) for entry in entries)
            accessible_blocks.append(
                AccessibleUnknownBlock(
                    block_index=int(block_id),
                    block_geometry=block_geometry,
                    block_area=int(block_areas[int(block_id)]),
                    block_bbox_shape=self._bbox_shape_from_mask(block_geometry),
                    entry_count=int(len(entries)),
                    entries=tuple(
                        sorted(
                            entries,
                            key=lambda entry: (
                                float(entry.entry_dist),
                                -float(entry.entry_clearance),
                                float(entry.entry_local_revisit_pressure),
                                int(entry.entry_index),
                            ),
                        )
                    ),
                    nearest_entry_dist=float(nearest_entry_dist),
                    opportunity_score=self._block_priority_score(
                        block_area=int(block_areas[int(block_id)]),
                        nearest_entry_dist=float(nearest_entry_dist),
                        best_entry_clearance=float(best_entry_clearance),
                        lowest_entry_revisit_pressure=float(lowest_entry_revisit_pressure),
                        analysis_diagonal=float(analysis_diagonal),
                    ),
                )
            )

        accessible_blocks.sort(
            key=lambda block: (
                -float(block.opportunity_score),
                -int(block.block_area),
                float(block.nearest_entry_dist),
                int(block.block_index),
            )
        )

        total_accessible_unknown_area = int(sum(block.block_area for block in accessible_blocks))
        if total_accessible_unknown_area > 0 and len(accessible_blocks) > 0:
            top1_block_area_ratio = float(accessible_blocks[0].block_area) / float(total_accessible_unknown_area)
            area_entropy = _mask_entropy([int(block.block_area) for block in accessible_blocks])
            max_entropy = math.log(float(len(accessible_blocks))) if len(accessible_blocks) > 1 else 1.0
            scene_orderliness = 1.0 if len(accessible_blocks) <= 1 else float(1.0 - (area_entropy / max_entropy))
            main_block = accessible_blocks[0]
            main_block_index = int(main_block.block_index)
            main_block_area = float(main_block.block_area)
            main_block_entry_count = float(main_block.entry_count)
            nearest_main_entry_dist = float(main_block.nearest_entry_dist)
        else:
            top1_block_area_ratio = 0.0
            scene_orderliness = 1.0
            main_block_index = None
            main_block_area = 0.0
            main_block_entry_count = 0.0
            nearest_main_entry_dist = float("nan")

        snapshot = SharedSemanticSnapshot(
            analysis_box=analysis_box,
            accessible_blocks=tuple(accessible_blocks),
            main_block_index=main_block_index,
            total_accessible_unknown_area=total_accessible_unknown_area,
            top1_block_area_ratio=float(top1_block_area_ratio),
            scene_orderliness=float(scene_orderliness),
            main_block_area=float(main_block_area),
            main_block_entry_count=float(main_block_entry_count),
            nearest_main_entry_dist=float(nearest_main_entry_dist),
        )
        if self._timing_enabled:
            self.analysis_time += time.perf_counter() - t0
        return snapshot

    def get_timing_stats(self) -> dict[str, float]:
        return {"analysis_time": float(self.analysis_time)}


def build_semantic_visualization_payload(snapshot: SharedSemanticSnapshot) -> dict[str, object]:
    return {
        "analysis_box": {
            "r0": int(snapshot.analysis_box.r0),
            "r1": int(snapshot.analysis_box.r1),
            "c0": int(snapshot.analysis_box.c0),
            "c1": int(snapshot.analysis_box.c1),
            "margin": int(snapshot.analysis_box.margin),
        },
        "main_block_index": (
            None if snapshot.main_block_index is None else int(snapshot.main_block_index)
        ),
        "blocks": [
            {
                "block_index": int(block.block_index),
                "rows": np.asarray(block.rows, dtype=np.int32).copy(),
                "cols": np.asarray(block.cols, dtype=np.int32).copy(),
                "block_area": int(block.block_area),
                "entry_count": int(block.entry_count),
                "nearest_entry_dist": float(block.nearest_entry_dist),
                "opportunity_score": float(block.opportunity_score),
                "entries": [
                    {
                        "entry_index": int(entry.entry_index),
                        "rows": np.asarray(entry.rows, dtype=np.int32).copy(),
                        "cols": np.asarray(entry.cols, dtype=np.int32).copy(),
                        "boundary_rows": np.asarray(entry.boundary_rows, dtype=np.int32).copy(),
                        "boundary_cols": np.asarray(entry.boundary_cols, dtype=np.int32).copy(),
                        "entry_dist": float(entry.entry_dist),
                        "entry_width": float(entry.entry_width),
                        "entry_clearance": float(entry.entry_clearance),
                        "entry_local_revisit_pressure": float(entry.entry_local_revisit_pressure),
                    }
                    for entry in block.entries
                ],
            }
            for block in snapshot.accessible_blocks
        ],
    }
