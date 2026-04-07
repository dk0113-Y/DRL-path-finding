from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

try:
    from scipy import ndimage as _scipy_ndimage
except Exception:
    _scipy_ndimage = None

from env.core_cummap import AnalysisBox
from env.grid_topology import EMPTY, INVISIBLE, OBSTACLE

_FOUR_CONNECTED_STRUCTURE = np.array(
    [[0, 1, 0],
     [1, 1, 1],
     [0, 1, 0]],
    dtype=np.uint8,
)
_EIGHT_CONNECTED_STRUCTURE = np.ones((3, 3), dtype=np.uint8)
_SUPPORT_LOCAL_BOX_PADDING = 2


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
class SupportGeometry:
    """
    Local known-side support descriptor attached to a frontier cluster.

    The support local box is only a sampling window around the frontier cluster.
    The learning-facing summary is reduced to obstacle density over known cells
    inside that box. Free-cell geometry is kept only for visualization/debug.
    """

    local_box_bounds: tuple[int, int, int, int]
    support_free_geometry: SparseMaskGeometry
    support_obstacle_density: float

    @property
    def rows(self) -> np.ndarray:
        return self.support_free_geometry.rows

    @property
    def cols(self) -> np.ndarray:
        return self.support_free_geometry.cols

    @property
    def local_box_r0(self) -> int:
        return int(self.local_box_bounds[0])

    @property
    def local_box_r1(self) -> int:
        return int(self.local_box_bounds[1])

    @property
    def local_box_c0(self) -> int:
        return int(self.local_box_bounds[2])

    @property
    def local_box_c1(self) -> int:
        return int(self.local_box_bounds[3])

    def paint_to_local_canvas(
        self,
        canvas_mask: np.ndarray,
        *,
        agent_arr: tuple[int, int],
        local_shape: tuple[int, int],
    ) -> None:
        self.support_free_geometry.paint_to_local_canvas(
            canvas_mask,
            agent_arr=agent_arr,
            local_shape=local_shape,
        )


@dataclass(frozen=True, slots=True)
class FrontierCluster:
    frontier_index: int
    block_index: int
    frontier_geometry: SparseMaskGeometry
    support_geometry: SupportGeometry
    frontier_anchor_rc: tuple[int, int]
    delta_r: float
    delta_c: float
    entry_width: float

    @property
    def rows(self) -> np.ndarray:
        return self.frontier_geometry.rows

    @property
    def cols(self) -> np.ndarray:
        return self.frontier_geometry.cols

    @property
    def support_rows(self) -> np.ndarray:
        return self.support_geometry.rows

    @property
    def support_cols(self) -> np.ndarray:
        return self.support_geometry.cols

    @property
    def support_obstacle_density(self) -> float:
        return float(self.support_geometry.support_obstacle_density)

    @property
    def anchor_distance(self) -> float:
        return float(math.hypot(float(self.delta_r), float(self.delta_c)))

    def paint_to_local_canvas(
        self,
        canvas_mask: np.ndarray,
        *,
        agent_arr: tuple[int, int],
        local_shape: tuple[int, int],
    ) -> None:
        self.frontier_geometry.paint_to_local_canvas(
            canvas_mask,
            agent_arr=agent_arr,
            local_shape=local_shape,
        )


@dataclass(frozen=True, slots=True)
class UnknownBlock:
    block_index: int
    unknown_geometry: SparseMaskGeometry
    frontier_clusters: tuple[FrontierCluster, ...]
    block_area: int
    block_bbox_shape: tuple[float, float, float]
    frontier_cluster_count: int
    nearest_frontier_dist: float

    @property
    def rows(self) -> np.ndarray:
        return self.unknown_geometry.rows

    @property
    def cols(self) -> np.ndarray:
        return self.unknown_geometry.cols

    def paint_to_local_canvas(
        self,
        canvas_mask: np.ndarray,
        *,
        agent_arr: tuple[int, int],
        local_shape: tuple[int, int],
    ) -> None:
        self.unknown_geometry.paint_to_local_canvas(
            canvas_mask,
            agent_arr=agent_arr,
            local_shape=local_shape,
        )


@dataclass(frozen=True)
class SharedSemanticSnapshot:
    analysis_box: AnalysisBox
    accessible_blocks: tuple[UnknownBlock, ...]
    total_accessible_unknown_area: int

    def metrics(self) -> dict[str, float]:
        accessible_block_count = float(len(self.accessible_blocks))
        total_unknown_area = float(self.total_accessible_unknown_area)
        total_frontier_cluster_count = float(sum(int(block.frontier_cluster_count) for block in self.accessible_blocks))
        mean_block_area = (
            total_unknown_area / accessible_block_count
            if accessible_block_count > 0.0
            else 0.0
        )
        return {
            "accessible_block_count": accessible_block_count,
            "total_accessible_unknown_area": total_unknown_area,
            "total_frontier_cluster_count": total_frontier_cluster_count,
            "mean_block_area": float(mean_block_area),
        }


@dataclass
class _UnionFind:
    size: int

    def __post_init__(self) -> None:
        self.parent = np.arange(int(self.size) + 1, dtype=np.int32)

    def find(self, value: int) -> int:
        node = int(value)
        parent = self.parent
        while int(parent[node]) != node:
            parent[node] = parent[int(parent[node])]
            node = int(parent[node])
        return node

    def union(self, lhs: int, rhs: int) -> int:
        left_root = self.find(int(lhs))
        right_root = self.find(int(rhs))
        if left_root == right_root:
            return left_root
        if left_root < right_root:
            self.parent[right_root] = left_root
            return left_root
        self.parent[left_root] = right_root
        return right_root


def _label_components_2d(mask: np.ndarray, *, connectivity: int = 4) -> tuple[np.ndarray, int]:
    mask_bool = np.asarray(mask, dtype=bool)
    if not np.any(mask_bool):
        return np.zeros(mask_bool.shape, dtype=np.int32), 0

    if int(connectivity) == 4:
        structure = _FOUR_CONNECTED_STRUCTURE
        neighbor_offsets = ((-1, 0), (1, 0), (0, -1), (0, 1))
    elif int(connectivity) == 8:
        structure = _EIGHT_CONNECTED_STRUCTURE
        neighbor_offsets = (
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),            (0, 1),
            (1, -1),  (1, 0),   (1, 1),
        )
    else:
        raise ValueError(f"unsupported connectivity: {connectivity}")

    if _scipy_ndimage is not None:
        labels, count = _scipy_ndimage.label(mask_bool, structure=structure)
        return np.asarray(labels, dtype=np.int32), int(count)

    h, w = mask_bool.shape
    labels = np.zeros((h, w), dtype=np.int32)
    next_label = 1
    for start_r, start_c in zip(*np.nonzero(mask_bool)):
        if labels[start_r, start_c] > 0:
            continue
        queue = deque([(int(start_r), int(start_c))])
        labels[start_r, start_c] = next_label
        while queue:
            r, c = queue.popleft()
            for dr, dc in neighbor_offsets:
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


def _sparse_geometry_from_local_mask(
    local_mask: np.ndarray,
    *,
    offset_r0: int,
    offset_c0: int,
) -> SparseMaskGeometry:
    mask_bool = np.asarray(local_mask, dtype=bool)
    if not np.any(mask_bool):
        return SparseMaskGeometry.empty(r0=int(offset_r0), c0=int(offset_c0))
    rows, cols = np.nonzero(mask_bool)
    r0 = int(rows.min())
    r1 = int(rows.max()) + 1
    c0 = int(cols.min())
    c1 = int(cols.max()) + 1
    cropped = mask_bool[r0:r1, c0:c1]
    return SparseMaskGeometry(
        r0=int(offset_r0) + r0,
        c0=int(offset_c0) + c0,
        mask=cropped,
        count=int(np.count_nonzero(cropped)),
    )


class SharedSemanticLayer:
    """
    Shared semantic layer built on top of the cumulative belief map.

    The semantic tree is frontier-first:
      UnknownBlock
        -> FrontierCluster
            -> SupportGeometry

    UnknownBlock groups one or more frontier clusters through frontier-first
    unknown-side grouping and only carries lightweight parent-level summaries.

    FrontierCluster is an 8-connected pure frontier-cell cluster and carries the
    local entry geometry seen by the model.

    SupportGeometry is not a frontier dilation result. It is reduced to a local
    obstacle-density descriptor derived from the frontier cluster's local
    support box on the known side.
    """

    def __init__(self, config: Optional[SharedSemanticConfig] = None):
        self.config = config if config is not None else SharedSemanticConfig()
        self._timing_enabled = bool(self.config.enable_timing)
        self.analysis_time = 0.0

    @staticmethod
    def _analysis_arrays(cum_map) -> tuple[AnalysisBox, np.ndarray, np.ndarray, np.ndarray]:
        box = cum_map.analysis_box
        map_view = np.asarray(cum_map.map, dtype=np.int8)
        box_map = map_view[box.r0:box.r1, box.c0:box.c1]
        unknown_box = (box_map == INVISIBLE)
        frontier_view = np.asarray(cum_map.get_frontier_u8(refresh=False), dtype=np.uint8) > 0
        frontier_box = frontier_view[box.r0:box.r1, box.c0:box.c1]
        return box, box_map, unknown_box, frontier_box

    @staticmethod
    def _neighbor_label_stack(labels: np.ndarray) -> np.ndarray:
        padded = np.pad(np.asarray(labels, dtype=np.int32), 1, mode="constant", constant_values=0)
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
    def _frontier_anchor_rc(frontier_geometry: SparseMaskGeometry) -> tuple[int, int]:
        rows = np.asarray(frontier_geometry.rows, dtype=np.int32)
        cols = np.asarray(frontier_geometry.cols, dtype=np.int32)
        if rows.size <= 0 or cols.size <= 0:
            return int(frontier_geometry.r0), int(frontier_geometry.c0)

        centroid_r = float(np.mean(rows))
        centroid_c = float(np.mean(cols))
        dist2 = ((rows.astype(np.float32) - centroid_r) ** 2) + ((cols.astype(np.float32) - centroid_c) ** 2)
        min_dist2 = float(np.min(dist2))
        candidates = np.flatnonzero(np.isclose(dist2, min_dist2))
        if candidates.size <= 1:
            best_idx = int(candidates[0]) if candidates.size == 1 else int(np.argmin(dist2))
            return int(rows[best_idx]), int(cols[best_idx])

        order = np.lexsort((cols[candidates], rows[candidates]))
        best_idx = int(candidates[int(order[0])])
        return int(rows[best_idx]), int(cols[best_idx])

    @staticmethod
    def _bbox_shape_from_mask(mask_geometry: SparseMaskGeometry) -> tuple[float, float, float]:
        height = float(int(mask_geometry.shape[0]))
        width = float(int(mask_geometry.shape[1]))
        aspect = width / max(1.0, height)
        return height, width, aspect

    @staticmethod
    def _support_local_box_bounds(
        frontier_obj,
        *,
        box_shape: tuple[int, int],
    ) -> tuple[int, int, int, int]:
        pad = int(_SUPPORT_LOCAL_BOX_PADDING)
        local_r0 = max(0, int(frontier_obj[0].start) - pad)
        local_r1 = min(int(box_shape[0]), int(frontier_obj[0].stop) + pad)
        local_c0 = max(0, int(frontier_obj[1].start) - pad)
        local_c1 = min(int(box_shape[1]), int(frontier_obj[1].stop) + pad)
        return local_r0, local_r1, local_c0, local_c1

    @staticmethod
    def _support_stats(
        local_box_map: np.ndarray,
    ) -> float:
        local_box = np.asarray(local_box_map, dtype=np.int8)
        known_mask = (local_box != INVISIBLE)
        known_count = int(np.count_nonzero(known_mask))
        if known_count <= 0:
            return 0.0
        obstacle_density = float(np.count_nonzero((local_box == OBSTACLE) & known_mask)) / float(known_count)
        return float(np.clip(obstacle_density, 0.0, 1.0))

    @staticmethod
    def _empty_snapshot(analysis_box: AnalysisBox) -> SharedSemanticSnapshot:
        return SharedSemanticSnapshot(
            analysis_box=analysis_box,
            accessible_blocks=tuple(),
            total_accessible_unknown_area=0,
        )

    @staticmethod
    def _group_unknown_from_frontiers(
        frontier_labels: np.ndarray,
        unknown_box: np.ndarray,
    ) -> tuple[np.ndarray, dict[int, list[int]]]:
        frontier_count = int(np.max(frontier_labels)) if frontier_labels.size > 0 else 0
        if frontier_count <= 0 or not np.any(unknown_box):
            return np.zeros_like(frontier_labels, dtype=np.int32), {}

        neighbor_frontier_labels = SharedSemanticLayer._neighbor_label_stack(frontier_labels)
        seed_unknown_mask = np.asarray(unknown_box, dtype=bool) & np.any(neighbor_frontier_labels > 0, axis=0)
        if not np.any(seed_unknown_mask):
            return np.zeros_like(frontier_labels, dtype=np.int32), {}

        union_find = _UnionFind(frontier_count)
        owner = np.zeros_like(frontier_labels, dtype=np.int32)
        queue: deque[tuple[int, int]] = deque()
        active_frontier_labels: set[int] = set()

        # Unified frontier-first unknown grouping:
        # frontier labels seed adjacent unknown cells, then a single multi-source
        # propagation expands through unknown space and unions frontier labels
        # whenever their propagation waves meet.
        seed_rows, seed_cols = np.nonzero(seed_unknown_mask)
        for seed_r, seed_c in zip(seed_rows.tolist(), seed_cols.tolist()):
            labels = np.asarray(neighbor_frontier_labels[:, seed_r, seed_c], dtype=np.int32)
            labels = labels[labels > 0]
            if labels.size <= 0:
                continue
            unique_labels = np.unique(labels)
            root = int(unique_labels[0])
            active_frontier_labels.add(root)
            for label in unique_labels[1:].tolist():
                active_frontier_labels.add(int(label))
                root = union_find.union(root, int(label))
            root = union_find.find(root)
            if owner[seed_r, seed_c] == 0:
                owner[seed_r, seed_c] = root
                queue.append((int(seed_r), int(seed_c)))
            else:
                owner[seed_r, seed_c] = union_find.union(int(owner[seed_r, seed_c]), root)

        while queue:
            r, c = queue.popleft()
            current_root = union_find.find(int(owner[r, c]))
            owner[r, c] = current_root
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr = r + dr
                nc = c + dc
                if nr < 0 or nr >= int(unknown_box.shape[0]) or nc < 0 or nc >= int(unknown_box.shape[1]):
                    continue
                if not bool(unknown_box[nr, nc]):
                    continue
                neighbor_owner = int(owner[nr, nc])
                if neighbor_owner <= 0:
                    owner[nr, nc] = current_root
                    queue.append((int(nr), int(nc)))
                    continue
                neighbor_root = union_find.find(neighbor_owner)
                if neighbor_root != current_root:
                    current_root = union_find.union(current_root, neighbor_root)
                    owner[r, c] = current_root

        visited = (owner > 0)
        if not np.any(visited):
            return owner, {}

        lut = np.arange(int(frontier_count) + 1, dtype=np.int32)
        for label in active_frontier_labels:
            lut[int(label)] = int(union_find.find(int(label)))
        owner[visited] = lut[owner[visited]]

        root_to_frontier_labels: dict[int, list[int]] = {}
        for label in sorted(active_frontier_labels):
            root = int(lut[int(label)])
            if root <= 0:
                continue
            root_to_frontier_labels.setdefault(root, []).append(int(label))
        return owner, root_to_frontier_labels

    def analyze(
        self,
        cum_map,
        agent_state: tuple[int, int],
    ) -> SharedSemanticSnapshot:
        t0 = time.perf_counter() if self._timing_enabled else 0.0
        analysis_box, box_map, unknown_box, frontier_box = self._analysis_arrays(cum_map)
        # FrontierCluster is an 8-connected pure frontier-cell group. This
        # keeps diagonally continuous frontier segments and corner-generated
        # frontier fragments in the same cluster when appropriate.
        frontier_labels, frontier_count = _label_components_2d(frontier_box, connectivity=8)
        if frontier_count <= 0 or not np.any(unknown_box):
            snapshot = self._empty_snapshot(analysis_box)
            if self._timing_enabled:
                self.analysis_time += time.perf_counter() - t0
            return snapshot

        unknown_owner, root_to_frontier_labels = self._group_unknown_from_frontiers(frontier_labels, unknown_box)
        if not root_to_frontier_labels or not np.any(unknown_owner > 0):
            snapshot = self._empty_snapshot(analysis_box)
            if self._timing_enabled:
                self.analysis_time += time.perf_counter() - t0
            return snapshot

        frontier_objects = _find_objects(frontier_labels, frontier_count)
        label_to_root = {
            int(label): int(root)
            for root, labels in root_to_frontier_labels.items()
            for label in labels
        }
        agent_arr = cum_map.world_to_array(agent_state)

        frontier_records: dict[int, dict[str, object]] = {}
        for frontier_label in range(1, int(frontier_count) + 1):
            assigned_root = label_to_root.get(int(frontier_label))
            if assigned_root is None:
                continue

            frontier_obj = frontier_objects[int(frontier_label) - 1]
            if frontier_obj is None:
                continue
            local_r0 = int(frontier_obj[0].start)
            local_r1 = int(frontier_obj[0].stop)
            local_c0 = int(frontier_obj[1].start)
            local_c1 = int(frontier_obj[1].stop)
            frontier_local = (frontier_labels[local_r0:local_r1, local_c0:local_c1] == int(frontier_label))
            if not np.any(frontier_local):
                continue

            frontier_geometry = _sparse_geometry_from_local_mask(
                frontier_local,
                offset_r0=int(analysis_box.r0) + int(local_r0),
                offset_c0=int(analysis_box.c0) + int(local_c0),
            )

            # SupportGeometry is sampled from a symmetric local analysis box
            # around the frontier cluster bbox. It is not a frontier dilation
            # result and only summarizes the known-side content inside that box.
            support_r0, support_r1, support_c0, support_c1 = self._support_local_box_bounds(
                frontier_obj,
                box_shape=tuple(box_map.shape),
            )
            local_support_box = box_map[support_r0:support_r1, support_c0:support_c1]
            support_free_local = (local_support_box == EMPTY)
            support_free_geometry = _sparse_geometry_from_local_mask(
                support_free_local,
                offset_r0=int(analysis_box.r0) + int(support_r0),
                offset_c0=int(analysis_box.c0) + int(support_c0),
            )
            support_obstacle_density = self._support_stats(local_support_box)
            support_geometry = SupportGeometry(
                local_box_bounds=(
                    int(analysis_box.r0) + int(support_r0),
                    int(analysis_box.r0) + int(support_r1),
                    int(analysis_box.c0) + int(support_c0),
                    int(analysis_box.c0) + int(support_c1),
                ),
                support_free_geometry=support_free_geometry,
                support_obstacle_density=float(support_obstacle_density),
            )

            frontier_anchor_rc = self._frontier_anchor_rc(frontier_geometry)
            delta_r = float(int(frontier_anchor_rc[0]) - int(agent_arr[0]))
            delta_c = float(int(frontier_anchor_rc[1]) - int(agent_arr[1]))
            frontier_records[int(frontier_label)] = {
                "frontier_index": int(frontier_label) - 1,
                "frontier_geometry": frontier_geometry,
                "support_geometry": support_geometry,
                "frontier_anchor_rc": tuple(int(v) for v in frontier_anchor_rc),
                "delta_r": float(delta_r),
                "delta_c": float(delta_c),
                "entry_width": float(frontier_geometry.count),
            }

        accessible_blocks: list[UnknownBlock] = []
        for root in sorted(int(v) for v in np.unique(unknown_owner[unknown_owner > 0]).tolist()):
            frontier_labels_for_root = [
                int(label)
                for label in root_to_frontier_labels.get(int(root), [])
                if int(label) in frontier_records
            ]
            if len(frontier_labels_for_root) <= 0:
                continue

            block_local_mask = (unknown_owner == int(root))
            if not np.any(block_local_mask):
                continue

            block_geometry = _sparse_geometry_from_local_mask(
                block_local_mask,
                offset_r0=int(analysis_box.r0),
                offset_c0=int(analysis_box.c0),
            )
            block_index = min(int(frontier_records[label]["frontier_index"]) for label in frontier_labels_for_root)
            frontier_clusters = tuple(
                sorted(
                    (
                        FrontierCluster(
                            frontier_index=int(frontier_records[label]["frontier_index"]),
                            block_index=int(block_index),
                            frontier_geometry=frontier_records[label]["frontier_geometry"],
                            support_geometry=frontier_records[label]["support_geometry"],
                            frontier_anchor_rc=frontier_records[label]["frontier_anchor_rc"],
                            delta_r=float(frontier_records[label]["delta_r"]),
                            delta_c=float(frontier_records[label]["delta_c"]),
                            entry_width=float(frontier_records[label]["entry_width"]),
                        )
                        for label in frontier_labels_for_root
                    ),
                    key=lambda cluster: (
                        int(cluster.frontier_index),
                        int(cluster.frontier_anchor_rc[0]),
                        int(cluster.frontier_anchor_rc[1]),
                    ),
                )
            )
            nearest_frontier_dist = min(float(cluster.anchor_distance) for cluster in frontier_clusters)
            accessible_blocks.append(
                UnknownBlock(
                    block_index=int(block_index),
                    unknown_geometry=block_geometry,
                    frontier_clusters=frontier_clusters,
                    block_area=int(block_geometry.count),
                    block_bbox_shape=self._bbox_shape_from_mask(block_geometry),
                    frontier_cluster_count=int(len(frontier_clusters)),
                    nearest_frontier_dist=float(nearest_frontier_dist),
                )
            )

        # Block ordering is only for stable tensor packing. It must not encode
        # an expert preference over which block is more worthwhile to explore.
        accessible_blocks.sort(
            key=lambda block: (
                int(block.block_index),
                int(block.rows[0]) if block.rows.size > 0 else int(analysis_box.r0),
                int(block.cols[0]) if block.cols.size > 0 else int(analysis_box.c0),
            )
        )

        total_accessible_unknown_area = int(sum(block.block_area for block in accessible_blocks))
        snapshot = SharedSemanticSnapshot(
            analysis_box=analysis_box,
            accessible_blocks=tuple(accessible_blocks),
            total_accessible_unknown_area=total_accessible_unknown_area,
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
        "blocks": [
            {
                "block_index": int(block.block_index),
                "rows": np.asarray(block.rows, dtype=np.int32).copy(),
                "cols": np.asarray(block.cols, dtype=np.int32).copy(),
                "block_area": int(block.block_area),
                "block_bbox_shape": tuple(float(v) for v in block.block_bbox_shape),
                "frontier_cluster_count": int(block.frontier_cluster_count),
                "nearest_frontier_dist": float(block.nearest_frontier_dist),
                "frontier_clusters": [
                    {
                        "frontier_index": int(cluster.frontier_index),
                        "frontier_rows": np.asarray(cluster.rows, dtype=np.int32).copy(),
                        "frontier_cols": np.asarray(cluster.cols, dtype=np.int32).copy(),
                        "frontier_anchor_rc": (
                            int(cluster.frontier_anchor_rc[0]),
                            int(cluster.frontier_anchor_rc[1]),
                        ),
                        "delta_r": float(cluster.delta_r),
                        "delta_c": float(cluster.delta_c),
                        "anchor_distance": float(cluster.anchor_distance),
                        "entry_width": float(cluster.entry_width),
                        "support": {
                            "local_box": {
                                "r0": int(cluster.support_geometry.local_box_r0),
                                "r1": int(cluster.support_geometry.local_box_r1),
                                "c0": int(cluster.support_geometry.local_box_c0),
                                "c1": int(cluster.support_geometry.local_box_c1),
                            },
                            "free_rows": np.asarray(cluster.support_rows, dtype=np.int32).copy(),
                            "free_cols": np.asarray(cluster.support_cols, dtype=np.int32).copy(),
                            "support_obstacle_density": float(cluster.support_obstacle_density),
                        },
                    }
                    for cluster in block.frontier_clusters
                ],
            }
            for block in snapshot.accessible_blocks
        ],
    }
