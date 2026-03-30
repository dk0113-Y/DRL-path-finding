from __future__ import annotations

"""Legacy frontier-token builder kept only as historical reference."""

import time
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

try:
    from scipy import ndimage as _scipy_ndimage
except Exception:
    _scipy_ndimage = None

from env.grid_topology import INVISIBLE, OBSTACLE

_FOUR_CONNECTED_STRUCTURE = np.array(
    [[0, 1, 0],
     [1, 1, 1],
     [0, 1, 0]],
    dtype=np.uint8,
)
_EIGHT_CONNECTED_STRUCTURE = np.ones((3, 3), dtype=np.uint8)

FRONTIER_REGION_TOKEN_FIELDS = (
    "dx",
    "dy",
    "local_potential_gain",
    "frontier_contact_span",
    "obstacle_density",
)
FRONTIER_REGION_TOKEN_DIM = len(FRONTIER_REGION_TOKEN_FIELDS)
FRONTIER_REGION_TOKEN_FIELD_COUNT = FRONTIER_REGION_TOKEN_DIM


@dataclass(frozen=True)
class FrontierRegionTokenConfig:
    top_k: int = 12
    neighborhood_radius: int = 2
    min_cluster_size: int = 1
    min_local_gain_area: int = 2
    gain_expand_steps: int = 2
    merge_expand_steps: int = 1
    frontier_connectivity: int = 4
    enable_timing: bool = False


@dataclass(frozen=True)
class _ClusterScanResult:
    label_map: np.ndarray
    rows_sorted: np.ndarray
    cols_sorted: np.ndarray
    sorted_cluster_ids: np.ndarray
    offsets: np.ndarray
    sizes: np.ndarray
    sum_rows: np.ndarray
    sum_cols: np.ndarray
    min_rows: np.ndarray
    max_rows: np.ndarray
    min_cols: np.ndarray
    max_cols: np.ndarray

    @property
    def cluster_count(self) -> int:
        return int(self.sizes.shape[0])


@dataclass(frozen=True)
class _FrontierFeatureCache:
    frontier_source_uid: int
    frontier_revision: int
    primitive_cluster_count_raw: int
    primitive_cluster_count: int
    cluster_count: int
    label_map: np.ndarray
    rows_sorted: np.ndarray
    cols_sorted: np.ndarray
    offsets: np.ndarray
    sizes: np.ndarray
    log_sizes: np.ndarray
    centroid_rows_world: np.ndarray
    centroid_cols_world: np.ndarray
    obstacle_density: np.ndarray


@dataclass(frozen=True)
class _FrontierTokenCandidate:
    cluster_id: int
    dx: float
    dy: float
    local_potential_gain: float
    frontier_contact_span: float
    obstacle_density: float
    geom_dist_norm: float
    priority: float


class FrontierRegionTokenBuilder:
    """
    Semantic frontier-cluster-centered local potential-gain token representation.

    A token now represents:
      - a semantic frontier cluster / frontier entry family
      - the local potential gain visible from that semantic entry

    Construction is two-stage:
      1. primitive frontier clusters are extracted geometrically with 4-connectivity
      2. semantic frontier clusters are built by merging primitive clusters whose
         semantic support regions overlap

    support_region = primitive_cluster_mask U merge_unknown_mask

    The merge_unknown_mask is only used for semantic merging. Local potential
    gain keeps its own frontier-driven unknown expansion and is intentionally
    decoupled from semantic merge strength.
    """

    def __init__(self, config: Optional[FrontierRegionTokenConfig] = None):
        self.config = config if config is not None else FrontierRegionTokenConfig()
        self._timing_enabled = bool(self.config.enable_timing)
        self.cluster_extract_time = 0.0
        self.build_total_time = 0.0
        self.fast_path_hits = 0
        self.fallback_hits = 0
        self._feature_cache: Optional[_FrontierFeatureCache] = None

    @staticmethod
    def _shared_artifact_value(shared_artifacts, key: str):
        if shared_artifacts is None:
            return None
        if isinstance(shared_artifacts, dict):
            return shared_artifacts.get(key)
        return getattr(shared_artifacts, key, None)

    @staticmethod
    def _structure_for_connectivity(connectivity: int) -> np.ndarray:
        connectivity_use = int(connectivity)
        if connectivity_use == 4:
            return _FOUR_CONNECTED_STRUCTURE
        if connectivity_use == 8:
            return _EIGHT_CONNECTED_STRUCTURE
        raise ValueError(f"component connectivity must be 4 or 8, got {connectivity}")

    @staticmethod
    def _scan_neighbor_offsets(connectivity: int) -> tuple[tuple[int, int], ...]:
        connectivity_use = int(connectivity)
        if connectivity_use == 4:
            return ((-1, 0), (0, -1))
        if connectivity_use == 8:
            return ((-1, -1), (-1, 0), (-1, 1), (0, -1))
        raise ValueError(f"component connectivity must be 4 or 8, got {connectivity}")

    @staticmethod
    def _empty_cluster_scan(shape: tuple[int, int]) -> _ClusterScanResult:
        empty_i32 = np.zeros((0,), dtype=np.int32)
        empty_f32 = np.zeros((0,), dtype=np.float32)
        return _ClusterScanResult(
            label_map=np.full(shape, -1, dtype=np.int32),
            rows_sorted=empty_i32,
            cols_sorted=empty_i32,
            sorted_cluster_ids=empty_i32,
            offsets=np.zeros((1,), dtype=np.int32),
            sizes=empty_i32,
            sum_rows=empty_f32,
            sum_cols=empty_f32,
            min_rows=empty_i32,
            max_rows=empty_i32,
            min_cols=empty_i32,
            max_cols=empty_i32,
        )

    def _resolve_frontier_stats(
        self,
        cum_map,
        *,
        frontier_u8: Optional[np.ndarray] = None,
        frontier_stats=None,
        shared_artifacts=None,
    ):
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
        return frontier_stats_use, frontier_bool

    @staticmethod
    def _build_cluster_scan_result(
        rows: np.ndarray,
        cols: np.ndarray,
        cluster_ids: np.ndarray,
        counts: np.ndarray,
        shape: tuple[int, int],
    ) -> _ClusterScanResult:
        component_count = int(counts.shape[0])
        rows_i32 = rows.astype(np.int32, copy=False)
        cols_i32 = cols.astype(np.int32, copy=False)
        cluster_ids_i32 = cluster_ids.astype(np.int32, copy=False)

        order = np.argsort(cluster_ids_i32, kind="stable")
        rows_sorted = rows_i32[order]
        cols_sorted = cols_i32[order]
        sorted_cluster_ids = cluster_ids_i32[order]

        offsets = np.empty((component_count + 1,), dtype=np.int32)
        offsets[0] = 0
        offsets[1:] = np.cumsum(counts.astype(np.int32, copy=False), dtype=np.int32)

        row_weights = rows_i32.astype(np.float32, copy=False)
        col_weights = cols_i32.astype(np.float32, copy=False)
        sum_rows = np.bincount(cluster_ids_i32, weights=row_weights, minlength=component_count).astype(np.float32)
        sum_cols = np.bincount(cluster_ids_i32, weights=col_weights, minlength=component_count).astype(np.float32)

        min_rows = np.full((component_count,), np.iinfo(np.int32).max, dtype=np.int32)
        max_rows = np.full((component_count,), np.iinfo(np.int32).min, dtype=np.int32)
        min_cols = np.full((component_count,), np.iinfo(np.int32).max, dtype=np.int32)
        max_cols = np.full((component_count,), np.iinfo(np.int32).min, dtype=np.int32)
        np.minimum.at(min_rows, cluster_ids_i32, rows_i32)
        np.maximum.at(max_rows, cluster_ids_i32, rows_i32)
        np.minimum.at(min_cols, cluster_ids_i32, cols_i32)
        np.maximum.at(max_cols, cluster_ids_i32, cols_i32)

        label_map = np.full(shape, -1, dtype=np.int32)
        if rows_i32.size > 0:
            label_map[rows_i32, cols_i32] = cluster_ids_i32

        return _ClusterScanResult(
            label_map=label_map,
            rows_sorted=rows_sorted,
            cols_sorted=cols_sorted,
            sorted_cluster_ids=sorted_cluster_ids,
            offsets=offsets,
            sizes=counts.astype(np.int32, copy=False),
            sum_rows=sum_rows,
            sum_cols=sum_cols,
            min_rows=min_rows,
            max_rows=max_rows,
            min_cols=min_cols,
            max_cols=max_cols,
        )

    def _extract_components_fast(
        self,
        component_mask: np.ndarray,
        *,
        connectivity: int,
    ) -> Optional[_ClusterScanResult]:
        if _scipy_ndimage is None:
            return None

        rows, cols = np.nonzero(component_mask)
        pixel_count = int(rows.shape[0])
        if pixel_count <= 0:
            return self._empty_cluster_scan(tuple(component_mask.shape))

        labels, component_count = _scipy_ndimage.label(
            component_mask,
            structure=self._structure_for_connectivity(connectivity),
        )
        if int(component_count) <= 0:
            return self._empty_cluster_scan(tuple(component_mask.shape))

        cluster_ids = np.asarray(labels[rows, cols], dtype=np.int32)
        cluster_ids -= 1
        counts = np.bincount(cluster_ids, minlength=int(component_count))
        return self._build_cluster_scan_result(rows, cols, cluster_ids, counts, tuple(component_mask.shape))

    def _extract_components_fallback(
        self,
        component_mask: np.ndarray,
        *,
        connectivity: int,
    ) -> _ClusterScanResult:
        rows, cols = np.nonzero(component_mask)
        pixel_count = int(rows.shape[0])
        if pixel_count <= 0:
            return self._empty_cluster_scan(tuple(component_mask.shape))

        h, w = component_mask.shape
        label_map = np.full((h, w), -1, dtype=np.int32)
        parents: list[int] = []
        neighbor_offsets = self._scan_neighbor_offsets(connectivity)

        def _uf_find(label: int) -> int:
            root = label
            while parents[root] != root:
                root = parents[root]
            while parents[label] != label:
                parent = parents[label]
                parents[label] = root
                label = parent
            return root

        def _uf_union(a: int, b: int) -> int:
            ra = _uf_find(a)
            rb = _uf_find(b)
            if ra == rb:
                return ra
            if ra < rb:
                parents[rb] = ra
                return ra
            parents[ra] = rb
            return rb

        for idx in range(pixel_count):
            r = int(rows[idx])
            c = int(cols[idx])
            neighbor_labels: list[int] = []

            for dr, dc in neighbor_offsets:
                nr = r + int(dr)
                nc = c + int(dc)
                if nr < 0 or nr >= h or nc < 0 or nc >= w:
                    continue
                lab = int(label_map[nr, nc])
                if lab >= 0:
                    neighbor_labels.append(lab)

            if len(neighbor_labels) <= 0:
                label = len(parents)
                parents.append(label)
            else:
                label = min(neighbor_labels)
                for other in neighbor_labels:
                    label = _uf_union(label, other)

            label_map[r, c] = int(label)

        raw_labels = label_map[rows, cols]
        roots = np.fromiter((_uf_find(int(lbl)) for lbl in raw_labels), dtype=np.int32, count=pixel_count)
        _, inverse, counts = np.unique(roots, return_inverse=True, return_counts=True)
        return self._build_cluster_scan_result(rows, cols, inverse, counts, tuple(component_mask.shape))

    def _extract_components(
        self,
        component_mask: np.ndarray,
        *,
        connectivity: int,
    ) -> _ClusterScanResult:
        fast = self._extract_components_fast(component_mask, connectivity=connectivity)
        if fast is not None:
            self.fast_path_hits += 1
            return fast
        self.fallback_hits += 1
        return self._extract_components_fallback(component_mask, connectivity=connectivity)

    def _extract_frontier_clusters(self, frontier_bool: np.ndarray) -> _ClusterScanResult:
        return self._extract_components(
            frontier_bool,
            connectivity=self.config.frontier_connectivity,
        )

    @staticmethod
    def _orth_neighbor_mask(mask: np.ndarray) -> np.ndarray:
        p = np.pad(np.asarray(mask, dtype=bool), 1, mode="constant", constant_values=False)
        return (
            p[:-2, 1:-1] |
            p[1:-1, :-2] |
            p[1:-1, 2:] |
            p[2:, 1:-1]
        )

    def _expand_unknown_from_frontier(
        self,
        frontier_bool: np.ndarray,
        unknown_bool: np.ndarray,
        *,
        total_layers: int,
    ) -> np.ndarray:
        layers = int(max(0, total_layers))
        if layers <= 0:
            return np.zeros_like(unknown_bool, dtype=bool)

        frontier_like = np.asarray(frontier_bool, dtype=bool)
        expanded = np.zeros_like(unknown_bool, dtype=bool)
        for _ in range(layers):
            next_layer = unknown_bool & self._orth_neighbor_mask(frontier_like) & (~expanded)
            if not np.any(next_layer):
                break
            expanded |= next_layer
            frontier_like = next_layer
        return expanded

    def _build_local_gain_mask(
        self,
        frontier_bool: np.ndarray,
        unknown_bool: np.ndarray,
    ) -> np.ndarray:
        return self._expand_unknown_from_frontier(
            frontier_bool,
            unknown_bool,
            total_layers=int(max(0, self.config.gain_expand_steps)),
        )

    def _build_merge_unknown_mask(
        self,
        frontier_bool: np.ndarray,
        unknown_bool: np.ndarray,
    ) -> np.ndarray:
        total_merge_layers = 1 + int(max(0, self.config.merge_expand_steps))
        return self._expand_unknown_from_frontier(
            frontier_bool,
            unknown_bool,
            total_layers=total_merge_layers,
        )

    def _cluster_semantic_support_coords(
        self,
        unknown_bool: np.ndarray,
        cluster_rows: np.ndarray,
        cluster_cols: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        if cluster_rows.size <= 0:
            empty = np.zeros((0,), dtype=np.int32)
            return empty, empty

        pad = 1 + int(max(0, self.config.merge_expand_steps))
        r_min = max(0, int(cluster_rows.min()) - pad)
        r_max = min(int(unknown_bool.shape[0]), int(cluster_rows.max()) + pad + 1)
        c_min = max(0, int(cluster_cols.min()) - pad)
        c_max = min(int(unknown_bool.shape[1]), int(cluster_cols.max()) + pad + 1)

        local_unknown = unknown_bool[r_min:r_max, c_min:c_max]
        local_frontier = np.zeros_like(local_unknown, dtype=bool)
        local_frontier[cluster_rows - r_min, cluster_cols - c_min] = True

        merge_unknown_mask = self._build_merge_unknown_mask(local_frontier, local_unknown)
        support_region = local_frontier | merge_unknown_mask
        rr_local, cc_local = np.nonzero(support_region)
        return (
            rr_local.astype(np.int32, copy=False) + np.int32(r_min),
            cc_local.astype(np.int32, copy=False) + np.int32(c_min),
        )

    def _cluster_local_gain_area(
        self,
        unknown_bool: np.ndarray,
        cluster_rows: np.ndarray,
        cluster_cols: np.ndarray,
    ) -> int:
        if cluster_rows.size <= 0:
            return 0

        # Local potential gain uses its own configurable frontier-driven unknown
        # expansion and stays decoupled from semantic merge support expansion.
        pad = int(max(0, self.config.gain_expand_steps))
        r_min = max(0, int(cluster_rows.min()) - pad)
        r_max = min(int(unknown_bool.shape[0]), int(cluster_rows.max()) + pad + 1)
        c_min = max(0, int(cluster_cols.min()) - pad)
        c_max = min(int(unknown_bool.shape[1]), int(cluster_cols.max()) + pad + 1)

        local_unknown = unknown_bool[r_min:r_max, c_min:c_max]
        local_frontier = np.zeros_like(local_unknown, dtype=bool)
        local_frontier[cluster_rows - r_min, cluster_cols - c_min] = True

        local_gain_mask = self._build_local_gain_mask(local_frontier, local_unknown)
        return int(np.count_nonzero(local_gain_mask))

    def _cluster_local_gain_coords(
        self,
        unknown_bool: np.ndarray,
        cluster_rows: np.ndarray,
        cluster_cols: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        if cluster_rows.size <= 0:
            empty = np.zeros((0,), dtype=np.int32)
            return empty, empty

        pad = int(max(0, self.config.gain_expand_steps))
        r_min = max(0, int(cluster_rows.min()) - pad)
        r_max = min(int(unknown_bool.shape[0]), int(cluster_rows.max()) + pad + 1)
        c_min = max(0, int(cluster_cols.min()) - pad)
        c_max = min(int(unknown_bool.shape[1]), int(cluster_cols.max()) + pad + 1)

        local_unknown = unknown_bool[r_min:r_max, c_min:c_max]
        local_frontier = np.zeros_like(local_unknown, dtype=bool)
        local_frontier[cluster_rows - r_min, cluster_cols - c_min] = True

        local_gain_mask = self._build_local_gain_mask(local_frontier, local_unknown)
        gain_rows_local, gain_cols_local = np.nonzero(local_gain_mask)
        return (
            gain_rows_local.astype(np.int32, copy=False) + np.int32(r_min),
            gain_cols_local.astype(np.int32, copy=False) + np.int32(c_min),
        )

    @staticmethod
    def _uf_find(parents: np.ndarray, idx: int) -> int:
        root = int(idx)
        while int(parents[root]) != root:
            root = int(parents[root])
        while int(parents[idx]) != idx:
            parent = int(parents[idx])
            parents[idx] = np.int32(root)
            idx = parent
        return root

    @classmethod
    def _uf_union(cls, parents: np.ndarray, a: int, b: int) -> int:
        ra = cls._uf_find(parents, int(a))
        rb = cls._uf_find(parents, int(b))
        if ra == rb:
            return ra
        if ra < rb:
            parents[rb] = np.int32(ra)
            return ra
        parents[ra] = np.int32(rb)
        return rb

    def _merge_primitive_clusters_to_semantic(
        self,
        primitive_clusters: _ClusterScanResult,
        unknown_bool: np.ndarray,
    ) -> _ClusterScanResult:
        primitive_count = int(primitive_clusters.cluster_count)
        if primitive_count <= 1:
            return primitive_clusters

        parents = np.arange(primitive_count, dtype=np.int32)
        support_owner = np.full(unknown_bool.shape, -1, dtype=np.int32)

        for primitive_id in range(primitive_count):
            start = int(primitive_clusters.offsets[primitive_id])
            end = int(primitive_clusters.offsets[primitive_id + 1])
            cluster_rows = primitive_clusters.rows_sorted[start:end]
            cluster_cols = primitive_clusters.cols_sorted[start:end]

            support_rows, support_cols = self._cluster_semantic_support_coords(
                unknown_bool,
                cluster_rows,
                cluster_cols,
            )
            if support_rows.size <= 0:
                continue

            owners = support_owner[support_rows, support_cols]
            overlap_owners = np.unique(owners[owners >= 0])
            for other in overlap_owners:
                self._uf_union(parents, primitive_id, int(other))

            unassigned = (owners < 0)
            if np.any(unassigned):
                support_owner[support_rows[unassigned], support_cols[unassigned]] = np.int32(primitive_id)

        primitive_ids = np.arange(primitive_count, dtype=np.int32)
        roots = np.fromiter(
            (self._uf_find(parents, int(cluster_id)) for cluster_id in primitive_ids),
            dtype=np.int32,
            count=primitive_count,
        )
        _, primitive_to_semantic, _ = np.unique(roots, return_inverse=True, return_counts=True)

        if int(np.max(primitive_to_semantic, initial=-1)) + 1 == primitive_count:
            return primitive_clusters

        pixel_semantic_ids = primitive_to_semantic[primitive_clusters.sorted_cluster_ids]
        semantic_count = int(np.max(pixel_semantic_ids, initial=-1)) + 1
        semantic_sizes = np.bincount(pixel_semantic_ids, minlength=semantic_count).astype(np.int32, copy=False)
        return self._build_cluster_scan_result(
            primitive_clusters.rows_sorted,
            primitive_clusters.cols_sorted,
            pixel_semantic_ids.astype(np.int32, copy=False),
            semantic_sizes,
            tuple(primitive_clusters.label_map.shape),
        )

    def _filter_small_clusters(self, clusters: _ClusterScanResult) -> _ClusterScanResult:
        min_size = int(max(1, self.config.min_cluster_size))
        if min_size <= 1 or clusters.cluster_count <= 0:
            return clusters

        keep = clusters.sizes >= min_size
        if np.all(keep):
            return clusters
        if not np.any(keep):
            return self._empty_cluster_scan(tuple(clusters.label_map.shape))

        filtered_sizes = clusters.sizes[keep].astype(np.int32, copy=False)
        filtered_offsets = np.empty((filtered_sizes.shape[0] + 1,), dtype=np.int32)
        filtered_offsets[0] = 0
        filtered_offsets[1:] = np.cumsum(filtered_sizes, dtype=np.int32)

        pixel_keep = keep[clusters.sorted_cluster_ids]
        filtered_rows = clusters.rows_sorted[pixel_keep]
        filtered_cols = clusters.cols_sorted[pixel_keep]
        filtered_ids = np.repeat(np.arange(filtered_sizes.shape[0], dtype=np.int32), filtered_sizes)

        return self._build_cluster_scan_result(
            filtered_rows,
            filtered_cols,
            filtered_ids,
            filtered_sizes,
            tuple(clusters.label_map.shape),
        )

    @staticmethod
    def _integral_image(mask: np.ndarray) -> np.ndarray:
        prefix = np.pad(mask.astype(np.int32), ((1, 0), (1, 0)), mode="constant", constant_values=0)
        return prefix.cumsum(axis=0).cumsum(axis=1)

    @staticmethod
    def _box_mean_from_integral(
        prefix: np.ndarray,
        r1: np.ndarray,
        r2: np.ndarray,
        c1: np.ndarray,
        c2: np.ndarray,
    ) -> np.ndarray:
        sums = prefix[r2, c2] - prefix[r1, c2] - prefix[r2, c1] + prefix[r1, c1]
        area = np.maximum(1, (r2 - r1) * (c2 - c1))
        return sums.astype(np.float32) / area.astype(np.float32)

    @staticmethod
    def _empty_feature_cache(
        frontier: np.ndarray,
        frontier_source_uid: int,
        frontier_revision: int,
        primitive_cluster_count_raw: int,
        primitive_cluster_count: int,
    ) -> _FrontierFeatureCache:
        empty_f32 = np.zeros((0,), dtype=np.float32)
        empty_i32 = np.zeros((0,), dtype=np.int32)
        return _FrontierFeatureCache(
            frontier_source_uid=int(frontier_source_uid),
            frontier_revision=int(frontier_revision),
            primitive_cluster_count_raw=int(primitive_cluster_count_raw),
            primitive_cluster_count=int(primitive_cluster_count),
            cluster_count=0,
            label_map=np.full(frontier.shape, -1, dtype=np.int32),
            rows_sorted=empty_i32,
            cols_sorted=empty_i32,
            offsets=np.zeros((1,), dtype=np.int32),
            sizes=empty_f32,
            log_sizes=empty_f32,
            centroid_rows_world=empty_f32,
            centroid_cols_world=empty_f32,
            obstacle_density=empty_f32,
        )

    def _build_feature_cache(
        self,
        cum_map,
        frontier: np.ndarray,
        frontier_source_uid: int,
        frontier_revision: int,
    ) -> tuple[_FrontierFeatureCache, float]:
        cached = self._feature_cache
        # Cluster/feature reuse is keyed by the explicit frontier cache identity
        # (map instance uid + frontier revision), not by ndarray object identity.
        if (
            cached is not None
            and int(cached.frontier_source_uid) == int(frontier_source_uid)
            and int(cached.frontier_revision) == int(frontier_revision)
            and tuple(cached.label_map.shape) == tuple(frontier.shape)
        ):
            return cached, 0.0

        cluster_t0 = time.perf_counter() if self._timing_enabled else 0.0
        raw_primitive_clusters = self._extract_frontier_clusters(frontier)
        primitive_clusters = self._filter_small_clusters(raw_primitive_clusters)
        unknown_bool = (cum_map.map == INVISIBLE)
        semantic_clusters = self._merge_primitive_clusters_to_semantic(primitive_clusters, unknown_bool)
        cluster_elapsed = (time.perf_counter() - cluster_t0) if self._timing_enabled else 0.0

        if semantic_clusters.cluster_count <= 0:
            cache = self._empty_feature_cache(
                frontier,
                frontier_source_uid=frontier_source_uid,
                frontier_revision=frontier_revision,
                primitive_cluster_count_raw=raw_primitive_clusters.cluster_count,
                primitive_cluster_count=primitive_clusters.cluster_count,
            )
            self._feature_cache = cache
            return cache, cluster_elapsed

        sizes = semantic_clusters.sizes.astype(np.float32, copy=False)
        size_denom = np.maximum(sizes, np.float32(1.0))
        orr, orc = cum_map.origin_world_rc
        centroid_rows_world = (semantic_clusters.sum_rows / size_denom).astype(np.float32, copy=False) + np.float32(orr)
        centroid_cols_world = (semantic_clusters.sum_cols / size_denom).astype(np.float32, copy=False) + np.float32(orc)

        obstacle_prefix = cum_map.get_obstacle_integral(refresh=False)
        pad_obs = int(max(0, self.config.neighborhood_radius))
        obs_r1 = np.maximum(0, semantic_clusters.min_rows - pad_obs).astype(np.int32, copy=False)
        obs_r2 = np.minimum(cum_map.map.shape[0], semantic_clusters.max_rows + pad_obs + 1).astype(np.int32, copy=False)
        obs_c1 = np.maximum(0, semantic_clusters.min_cols - pad_obs).astype(np.int32, copy=False)
        obs_c2 = np.minimum(cum_map.map.shape[1], semantic_clusters.max_cols + pad_obs + 1).astype(np.int32, copy=False)
        obstacle_density = np.clip(
            self._box_mean_from_integral(obstacle_prefix, obs_r1, obs_r2, obs_c1, obs_c2),
            0.0,
            1.0,
        ).astype(np.float32, copy=False)

        cache = _FrontierFeatureCache(
            frontier_source_uid=int(frontier_source_uid),
            frontier_revision=int(frontier_revision),
            primitive_cluster_count_raw=raw_primitive_clusters.cluster_count,
            primitive_cluster_count=primitive_clusters.cluster_count,
            cluster_count=semantic_clusters.cluster_count,
            label_map=semantic_clusters.label_map,
            rows_sorted=semantic_clusters.rows_sorted,
            cols_sorted=semantic_clusters.cols_sorted,
            offsets=semantic_clusters.offsets,
            sizes=sizes,
            log_sizes=np.log1p(sizes).astype(np.float32, copy=False),
            centroid_rows_world=centroid_rows_world,
            centroid_cols_world=centroid_cols_world,
            obstacle_density=obstacle_density,
        )
        self._feature_cache = cache
        return cache, cluster_elapsed

    def build(
        self,
        cum_map,
        agent_state: Tuple[int, int],
        frontier_u8: Optional[np.ndarray] = None,
        frontier_stats=None,
        shared_artifacts=None,
        world_window_shape: Tuple[int, int] = (128, 128),
        return_meta: bool = False,
    ):
        build_t0 = time.perf_counter() if self._timing_enabled else 0.0
        k = int(self.config.top_k)
        if k <= 0:
            raise ValueError("top_k must be positive")

        tokens = np.zeros((k, FRONTIER_REGION_TOKEN_FIELD_COUNT), dtype=np.float32)
        token_mask = np.zeros((k,), dtype=bool)

        frontier_stats_use, frontier = self._resolve_frontier_stats(
            cum_map,
            frontier_u8=frontier_u8,
            frontier_stats=frontier_stats,
            shared_artifacts=shared_artifacts,
        )

        feature_cache, cluster_elapsed = self._build_feature_cache(
            cum_map,
            frontier,
            frontier_source_uid=int(frontier_stats_use.frontier_source_uid),
            frontier_revision=int(frontier_stats_use.frontier_revision),
        )
        if self._timing_enabled:
            self.cluster_extract_time += cluster_elapsed

        if feature_cache.cluster_count <= 0:
            if self._timing_enabled:
                build_elapsed = time.perf_counter() - build_t0
                self.build_total_time += build_elapsed
            if not return_meta:
                return tokens, token_mask

            meta = {
                "primitive_cluster_count_raw": feature_cache.primitive_cluster_count_raw,
                "primitive_cluster_count": feature_cache.primitive_cluster_count,
                "semantic_cluster_count": 0,
                "cluster_count_raw": feature_cache.primitive_cluster_count_raw,
                "cluster_count": 0,
                "selected_count": 0,
                "priority_scores": [],
            }
            if self._timing_enabled:
                meta["timing"] = {
                    "cluster_extract_time": float(cluster_elapsed),
                    "build_total_time": float(build_elapsed),
                }
            return tokens, token_mask, meta

        wh, ww = int(world_window_shape[0]), int(world_window_shape[1])
        half_h = max(1.0, wh / 2.0)
        half_w = max(1.0, ww / 2.0)

        ar = float(agent_state[0])
        ac = float(agent_state[1])
        dx = np.clip((feature_cache.centroid_cols_world - ac) / half_w, -1.0, 1.0).astype(np.float32, copy=False)
        dy = np.clip((feature_cache.centroid_rows_world - ar) / half_h, -1.0, 1.0).astype(np.float32, copy=False)

        frontier_contact_span = np.clip(
            feature_cache.log_sizes / np.log1p(float(max(2, wh + ww))),
            0.0,
            1.0,
        ).astype(np.float32, copy=False)
        obstacle_density = feature_cache.obstacle_density
        geom_dist_norm = np.clip(np.sqrt((dx * dx) + (dy * dy)) / np.sqrt(2.0), 0.0, 1.0).astype(np.float32)

        unknown_bool = (cum_map.map == INVISIBLE)
        min_local_gain_area = int(max(1, self.config.min_local_gain_area))

        candidates: list[_FrontierTokenCandidate] = []
        eligible_semantic_frontier_cluster_count = 0

        for cluster_id in range(feature_cache.cluster_count):
            start = int(feature_cache.offsets[cluster_id])
            end = int(feature_cache.offsets[cluster_id + 1])
            cluster_rows = feature_cache.rows_sorted[start:end]
            cluster_cols = feature_cache.cols_sorted[start:end]

            local_gain_area_count = self._cluster_local_gain_area(
                unknown_bool,
                cluster_rows,
                cluster_cols,
            )
            if local_gain_area_count < min_local_gain_area:
                continue

            eligible_semantic_frontier_cluster_count += 1
            local_potential_gain = float(
                np.clip(
                    np.log1p(float(local_gain_area_count)) / np.log1p(float(max(2, wh + ww))),
                    0.0,
                    1.0,
                )
            )
            priority = float(
                (0.50 * local_potential_gain) +
                (0.08 * float(frontier_contact_span[cluster_id])) -
                (0.12 * float(obstacle_density[cluster_id])) -
                (0.30 * float(geom_dist_norm[cluster_id]))
            )
            candidates.append(
                _FrontierTokenCandidate(
                    cluster_id=int(cluster_id),
                    dx=float(dx[cluster_id]),
                    dy=float(dy[cluster_id]),
                    local_potential_gain=local_potential_gain,
                    frontier_contact_span=float(frontier_contact_span[cluster_id]),
                    obstacle_density=float(obstacle_density[cluster_id]),
                    geom_dist_norm=float(geom_dist_norm[cluster_id]),
                    priority=priority,
                )
            )

        candidates.sort(
            key=lambda item: (
                -item.priority,
                item.geom_dist_norm,
                item.obstacle_density,
                -item.frontier_contact_span,
                item.cluster_id,
            )
        )

        n = min(k, len(candidates))
        for token_idx in range(n):
            candidate = candidates[token_idx]
            tokens[token_idx, 0] = np.float32(candidate.dx)
            tokens[token_idx, 1] = np.float32(candidate.dy)
            tokens[token_idx, 2] = np.float32(candidate.local_potential_gain)
            tokens[token_idx, 3] = np.float32(candidate.frontier_contact_span)
            tokens[token_idx, 4] = np.float32(candidate.obstacle_density)
            token_mask[token_idx] = True

        if self._timing_enabled:
            build_elapsed = time.perf_counter() - build_t0
            self.build_total_time += build_elapsed

        if not return_meta:
            return tokens, token_mask

        meta = {
            "primitive_cluster_count_raw": feature_cache.primitive_cluster_count_raw,
            "primitive_cluster_count": feature_cache.primitive_cluster_count,
            "semantic_cluster_count": feature_cache.cluster_count,
            "cluster_count_raw": feature_cache.primitive_cluster_count_raw,
            "cluster_count": feature_cache.cluster_count,
            "eligible_frontier_cluster_count": eligible_semantic_frontier_cluster_count,
            "candidate_count": len(candidates),
            "candidate_overflow": int(len(candidates) > int(k)),
            "selected_count": n,
            "candidate_local_potential_gain_mean": (
                float(np.mean([candidate.local_potential_gain for candidate in candidates]))
                if len(candidates) > 0 else float("nan")
            ),
            "candidate_frontier_contact_span_mean": (
                float(np.mean([candidate.frontier_contact_span for candidate in candidates]))
                if len(candidates) > 0 else float("nan")
            ),
            "candidate_geom_dist_norm_mean": (
                float(np.mean([candidate.geom_dist_norm for candidate in candidates]))
                if len(candidates) > 0 else float("nan")
            ),
            "candidate_obstacle_density_mean": (
                float(np.mean([candidate.obstacle_density for candidate in candidates]))
                if len(candidates) > 0 else float("nan")
            ),
            "selected_local_potential_gain_mean": (
                float(np.mean([candidate.local_potential_gain for candidate in candidates[:n]]))
                if n > 0 else float("nan")
            ),
            "selected_frontier_contact_span_mean": (
                float(np.mean([candidate.frontier_contact_span for candidate in candidates[:n]]))
                if n > 0 else float("nan")
            ),
            "selected_geom_dist_norm_mean": (
                float(np.mean([candidate.geom_dist_norm for candidate in candidates[:n]]))
                if n > 0 else float("nan")
            ),
            "selected_obstacle_density_mean": (
                float(np.mean([candidate.obstacle_density for candidate in candidates[:n]]))
                if n > 0 else float("nan")
            ),
            "priority_scores": [float(candidates[idx].priority) for idx in range(n)],
        }
        if self._timing_enabled:
            meta["timing"] = {
                "cluster_extract_time": float(cluster_elapsed),
                "build_total_time": float(build_elapsed),
            }
        return tokens, token_mask, meta

    def build_visualization_meta(
        self,
        cum_map,
        agent_state: Tuple[int, int],
        frontier_u8: Optional[np.ndarray] = None,
        frontier_stats=None,
        shared_artifacts=None,
        world_window_shape: Tuple[int, int] = (128, 128),
    ) -> dict[str, object]:
        frontier_stats_use = frontier_stats
        frontier_bool = None
        if frontier_u8 is not None:
            frontier_bool = np.asarray(frontier_u8, dtype=np.uint8) > 0
            frontier_source_uid = int(self._shared_artifact_value(shared_artifacts, "frontier_source_uid") or -1)
            frontier_revision = int(self._shared_artifact_value(shared_artifacts, "frontier_revision") or -1)
        else:
            frontier_stats_use, frontier_bool = self._resolve_frontier_stats(
                cum_map,
                frontier_u8=frontier_u8,
                frontier_stats=frontier_stats,
                shared_artifacts=shared_artifacts,
            )
            frontier_source_uid = int(frontier_stats_use.frontier_source_uid)
            frontier_revision = int(frontier_stats_use.frontier_revision)

        feature_cache, _ = self._build_feature_cache(
            cum_map,
            np.asarray(frontier_bool, dtype=bool),
            frontier_source_uid=frontier_source_uid,
            frontier_revision=frontier_revision,
        )

        meta: dict[str, object] = {
            "primitive_cluster_count_raw": feature_cache.primitive_cluster_count_raw,
            "primitive_cluster_count": feature_cache.primitive_cluster_count,
            "semantic_cluster_count": feature_cache.cluster_count,
            "cluster_count_raw": feature_cache.primitive_cluster_count_raw,
            "cluster_count": feature_cache.cluster_count,
            "candidate_count": 0,
            "selected_count": 0,
            "selected_cluster_ids": [],
            "clusters": [],
        }
        if feature_cache.cluster_count <= 0:
            return meta

        wh, ww = int(world_window_shape[0]), int(world_window_shape[1])
        half_h = max(1.0, wh / 2.0)
        half_w = max(1.0, ww / 2.0)
        ar = float(agent_state[0])
        ac = float(agent_state[1])

        dx = np.clip((feature_cache.centroid_cols_world - ac) / half_w, -1.0, 1.0).astype(np.float32, copy=False)
        dy = np.clip((feature_cache.centroid_rows_world - ar) / half_h, -1.0, 1.0).astype(np.float32, copy=False)
        frontier_contact_span = np.clip(
            feature_cache.log_sizes / np.log1p(float(max(2, wh + ww))),
            0.0,
            1.0,
        ).astype(np.float32, copy=False)
        obstacle_density = feature_cache.obstacle_density
        geom_dist_norm = np.clip(np.sqrt((dx * dx) + (dy * dy)) / np.sqrt(2.0), 0.0, 1.0).astype(np.float32)

        unknown_bool = (cum_map.map == INVISIBLE)
        min_local_gain_area = int(max(1, self.config.min_local_gain_area))

        clusters: list[dict[str, object]] = []
        candidates: list[_FrontierTokenCandidate] = []
        for cluster_id in range(feature_cache.cluster_count):
            start = int(feature_cache.offsets[cluster_id])
            end = int(feature_cache.offsets[cluster_id + 1])
            cluster_rows = feature_cache.rows_sorted[start:end]
            cluster_cols = feature_cache.cols_sorted[start:end]
            local_gain_rows, local_gain_cols = self._cluster_local_gain_coords(
                unknown_bool,
                cluster_rows,
                cluster_cols,
            )
            local_gain_area_count = int(local_gain_rows.shape[0])
            if local_gain_area_count < min_local_gain_area:
                continue

            local_potential_gain = float(
                np.clip(
                    np.log1p(float(local_gain_area_count)) / np.log1p(float(max(2, wh + ww))),
                    0.0,
                    1.0,
                )
            )
            priority = float(
                (0.50 * local_potential_gain) +
                (0.08 * float(frontier_contact_span[cluster_id])) -
                (0.12 * float(obstacle_density[cluster_id])) -
                (0.30 * float(geom_dist_norm[cluster_id]))
            )
            candidate = _FrontierTokenCandidate(
                cluster_id=int(cluster_id),
                dx=float(dx[cluster_id]),
                dy=float(dy[cluster_id]),
                local_potential_gain=local_potential_gain,
                frontier_contact_span=float(frontier_contact_span[cluster_id]),
                obstacle_density=float(obstacle_density[cluster_id]),
                geom_dist_norm=float(geom_dist_norm[cluster_id]),
                priority=priority,
            )
            candidates.append(candidate)
            clusters.append(
                {
                    "cluster_id": int(cluster_id),
                    "frontier_rows": np.asarray(cluster_rows, dtype=np.int32).copy(),
                    "frontier_cols": np.asarray(cluster_cols, dtype=np.int32).copy(),
                    "local_gain_rows": np.asarray(local_gain_rows, dtype=np.int32).copy(),
                    "local_gain_cols": np.asarray(local_gain_cols, dtype=np.int32).copy(),
                    "dx": float(candidate.dx),
                    "dy": float(candidate.dy),
                    "local_potential_gain": float(candidate.local_potential_gain),
                    "frontier_contact_span": float(candidate.frontier_contact_span),
                    "obstacle_density": float(candidate.obstacle_density),
                    "geom_dist_norm": float(candidate.geom_dist_norm),
                    "priority": float(candidate.priority),
                }
            )

        order = sorted(
            range(len(candidates)),
            key=lambda idx: (
                -candidates[idx].priority,
                candidates[idx].geom_dist_norm,
                candidates[idx].obstacle_density,
                -candidates[idx].frontier_contact_span,
                candidates[idx].cluster_id,
            ),
        )
        selected_count = min(int(self.config.top_k), len(order))
        selected_cluster_ids: list[int] = []
        ordered_clusters: list[dict[str, object]] = []
        for rank, idx in enumerate(order, start=1):
            cluster_info = dict(clusters[idx])
            cluster_info["selection_rank"] = int(rank)
            cluster_info["selected"] = bool(rank <= selected_count)
            if rank <= selected_count:
                selected_cluster_ids.append(int(cluster_info["cluster_id"]))
            ordered_clusters.append(cluster_info)

        meta["candidate_count"] = len(ordered_clusters)
        meta["selected_count"] = int(selected_count)
        meta["selected_cluster_ids"] = selected_cluster_ids
        meta["clusters"] = ordered_clusters
        return meta

    def get_timing_stats(self) -> dict[str, float]:
        return {
            "cluster_extract_time": float(self.cluster_extract_time),
            "build_total_time": float(self.build_total_time),
        }

    def get_cluster_path_stats(self) -> dict[str, int]:
        return {
            "fast_path_hits": int(self.fast_path_hits),
            "fallback_hits": int(self.fallback_hits),
        }


def _smoke_test() -> None:
    from env.agent_version import LocalObservationModel
    from env.block_random_g import RandomMapGenerator
    from env.core_cummap import CumulativeBeliefMap

    g, s = RandomMapGenerator(30, 40, 5, 0.2).generate_map()
    obs = LocalObservationModel(g, s)
    snap, _ = obs.observe(s)
    cm = CumulativeBeliefMap(g, s, snap)
    frontier_u8 = cm.get_frontier_u8(refresh=True)

    builder = FrontierRegionTokenBuilder(FrontierRegionTokenConfig(top_k=12))
    tokens, mask, meta = builder.build(cm, s, frontier_u8, world_window_shape=(96, 96), return_meta=True)

    assert tokens.shape == (12, FRONTIER_REGION_TOKEN_FIELD_COUNT)
    assert mask.shape == (12,)
    if meta["selected_count"] < 12:
        assert np.allclose(tokens[meta["selected_count"] :], 0.0)
        assert np.all(mask[meta["selected_count"] :] == 0)

    print("FrontierRegionTokenBuilder smoke test passed", tokens.shape, mask.sum())


if __name__ == "__main__":
    _smoke_test()
