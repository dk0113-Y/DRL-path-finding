from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

try:
    from scipy import ndimage as _scipy_ndimage
except Exception:
    _scipy_ndimage = None

from env.grid_topology import OBSTACLE

_EIGHT_CONNECTED_STRUCTURE = np.ones((3, 3), dtype=np.uint8)

FRONTIER_REGION_TOKEN_FIELDS = (
    "dx",
    "dy",
    "cluster_area",
    "obstacle_density",
)
FRONTIER_REGION_TOKEN_DIM = len(FRONTIER_REGION_TOKEN_FIELDS)
FRONTIER_REGION_TOKEN_FIELD_COUNT = FRONTIER_REGION_TOKEN_DIM


@dataclass(frozen=True)
class FrontierRegionTokenConfig:
    top_k: int = 8
    neighborhood_radius: int = 2
    min_cluster_size: int = 3
    enable_timing: bool = False


@dataclass(frozen=True)
class _ClusterScanResult:
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
    frontier_ref: np.ndarray
    cluster_count_raw: int
    cluster_count: int
    sizes: np.ndarray
    log_sizes: np.ndarray
    centroid_rows_world: np.ndarray
    centroid_cols_world: np.ndarray
    obstacle_density: np.ndarray


class FrontierRegionTokenBuilder:
    """
    Lightweight frontier candidate representation.

    Tokens now describe only relative candidate location, candidate size, and
    local obstacle complexity. The builder no longer explicitly encodes path
    cost or unknown-potential estimates; those higher-level value judgments are
    left to the near/mid context and the policy network.
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
    def _empty_cluster_scan() -> _ClusterScanResult:
        empty_i32 = np.zeros((0,), dtype=np.int32)
        empty_f32 = np.zeros((0,), dtype=np.float32)
        return _ClusterScanResult(
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

        return _ClusterScanResult(
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

    def _extract_frontier_clusters_fast(self, frontier_bool: np.ndarray) -> Optional[_ClusterScanResult]:
        if _scipy_ndimage is None:
            return None

        rows, cols = np.nonzero(frontier_bool)
        pixel_count = int(rows.shape[0])
        if pixel_count <= 0:
            return self._empty_cluster_scan()

        labels, component_count = _scipy_ndimage.label(
            frontier_bool,
            structure=_EIGHT_CONNECTED_STRUCTURE,
        )
        if int(component_count) <= 0:
            return self._empty_cluster_scan()

        cluster_ids = np.asarray(labels[rows, cols], dtype=np.int32)
        cluster_ids -= 1
        counts = np.bincount(cluster_ids, minlength=int(component_count))
        return self._build_cluster_scan_result(rows, cols, cluster_ids, counts)

    def _extract_frontier_clusters_fallback(self, frontier_bool: np.ndarray) -> _ClusterScanResult:
        rows, cols = np.nonzero(frontier_bool)
        pixel_count = int(rows.shape[0])
        if pixel_count <= 0:
            return self._empty_cluster_scan()

        h, w = frontier_bool.shape
        label_map = np.full((h, w), -1, dtype=np.int32)
        parents: list[int] = []

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

            if r > 0:
                if c > 0:
                    lab = int(label_map[r - 1, c - 1])
                    if lab >= 0:
                        neighbor_labels.append(lab)
                lab = int(label_map[r - 1, c])
                if lab >= 0:
                    neighbor_labels.append(lab)
                if c + 1 < w:
                    lab = int(label_map[r - 1, c + 1])
                    if lab >= 0:
                        neighbor_labels.append(lab)
            if c > 0:
                lab = int(label_map[r, c - 1])
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
        return self._build_cluster_scan_result(rows, cols, inverse, counts)

    def _extract_frontier_clusters(self, frontier_bool: np.ndarray) -> _ClusterScanResult:
        fast = self._extract_frontier_clusters_fast(frontier_bool)
        if fast is not None:
            self.fast_path_hits += 1
            return fast
        self.fallback_hits += 1
        return self._extract_frontier_clusters_fallback(frontier_bool)

    def _filter_small_clusters(self, clusters: _ClusterScanResult) -> _ClusterScanResult:
        min_size = int(max(1, self.config.min_cluster_size))
        if min_size <= 1 or clusters.cluster_count <= 0:
            return clusters

        keep = clusters.sizes >= min_size
        if np.all(keep):
            return clusters
        if not np.any(keep):
            return self._empty_cluster_scan()

        filtered_sizes = clusters.sizes[keep].astype(np.int32, copy=False)
        filtered_offsets = np.empty((filtered_sizes.shape[0] + 1,), dtype=np.int32)
        filtered_offsets[0] = 0
        filtered_offsets[1:] = np.cumsum(filtered_sizes, dtype=np.int32)

        pixel_keep = keep[clusters.sorted_cluster_ids]
        filtered_rows = clusters.rows_sorted[pixel_keep]
        filtered_cols = clusters.cols_sorted[pixel_keep]
        filtered_ids = np.repeat(np.arange(filtered_sizes.shape[0], dtype=np.int32), filtered_sizes)

        return _ClusterScanResult(
            rows_sorted=filtered_rows,
            cols_sorted=filtered_cols,
            sorted_cluster_ids=filtered_ids,
            offsets=filtered_offsets,
            sizes=filtered_sizes,
            sum_rows=clusters.sum_rows[keep].astype(np.float32, copy=False),
            sum_cols=clusters.sum_cols[keep].astype(np.float32, copy=False),
            min_rows=clusters.min_rows[keep].astype(np.int32, copy=False),
            max_rows=clusters.max_rows[keep].astype(np.int32, copy=False),
            min_cols=clusters.min_cols[keep].astype(np.int32, copy=False),
            max_cols=clusters.max_cols[keep].astype(np.int32, copy=False),
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
    def _priority_order(
        priority: np.ndarray,
        cluster_area: np.ndarray,
        obstacle_density: np.ndarray,
        geom_dist_norm: np.ndarray,
    ) -> np.ndarray:
        return np.lexsort((geom_dist_norm, obstacle_density, -cluster_area, -priority))

    @staticmethod
    def _empty_feature_cache(frontier_ref: np.ndarray, cluster_count_raw: int) -> _FrontierFeatureCache:
        empty_f32 = np.zeros((0,), dtype=np.float32)
        return _FrontierFeatureCache(
            frontier_ref=frontier_ref,
            cluster_count_raw=int(cluster_count_raw),
            cluster_count=0,
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
    ) -> tuple[_FrontierFeatureCache, float]:
        cached = self._feature_cache
        if cached is not None and frontier is cached.frontier_ref:
            return cached, 0.0

        cluster_t0 = time.perf_counter() if self._timing_enabled else 0.0
        raw_clusters = self._extract_frontier_clusters(frontier)
        clusters = self._filter_small_clusters(raw_clusters)
        cluster_elapsed = (time.perf_counter() - cluster_t0) if self._timing_enabled else 0.0

        if clusters.cluster_count <= 0:
            cache = self._empty_feature_cache(frontier, cluster_count_raw=raw_clusters.cluster_count)
            self._feature_cache = cache
            return cache, cluster_elapsed

        sizes = clusters.sizes.astype(np.float32, copy=False)
        size_denom = np.maximum(sizes, np.float32(1.0))
        orr, orc = cum_map.origin_world_rc
        centroid_rows_world = (clusters.sum_rows / size_denom).astype(np.float32, copy=False) + np.float32(orr)
        centroid_cols_world = (clusters.sum_cols / size_denom).astype(np.float32, copy=False) + np.float32(orc)

        obstacle_prefix = cum_map.get_obstacle_integral(refresh=False)
        pad_obs = int(max(0, self.config.neighborhood_radius))
        obs_r1 = np.maximum(0, clusters.min_rows - pad_obs).astype(np.int32, copy=False)
        obs_r2 = np.minimum(cum_map.map.shape[0], clusters.max_rows + pad_obs + 1).astype(np.int32, copy=False)
        obs_c1 = np.maximum(0, clusters.min_cols - pad_obs).astype(np.int32, copy=False)
        obs_c2 = np.minimum(cum_map.map.shape[1], clusters.max_cols + pad_obs + 1).astype(np.int32, copy=False)
        obstacle_density = np.clip(
            self._box_mean_from_integral(obstacle_prefix, obs_r1, obs_r2, obs_c1, obs_c2),
            0.0,
            1.0,
        ).astype(np.float32, copy=False)

        cache = _FrontierFeatureCache(
            frontier_ref=frontier,
            cluster_count_raw=raw_clusters.cluster_count,
            cluster_count=clusters.cluster_count,
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

        _, frontier = self._resolve_frontier_stats(
            cum_map,
            frontier_u8=frontier_u8,
            frontier_stats=frontier_stats,
            shared_artifacts=shared_artifacts,
        )

        feature_cache, cluster_elapsed = self._build_feature_cache(cum_map, frontier)
        if self._timing_enabled:
            self.cluster_extract_time += cluster_elapsed

        if feature_cache.cluster_count <= 0:
            if self._timing_enabled:
                build_elapsed = time.perf_counter() - build_t0
                self.build_total_time += build_elapsed
            if not return_meta:
                return tokens, token_mask

            meta = {
                "cluster_count_raw": feature_cache.cluster_count_raw,
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

        sizes = feature_cache.sizes
        wh, ww = int(world_window_shape[0]), int(world_window_shape[1])
        half_h = max(1.0, wh / 2.0)
        half_w = max(1.0, ww / 2.0)

        ar = float(agent_state[0])
        ac = float(agent_state[1])
        dx = np.clip((feature_cache.centroid_cols_world - ac) / half_w, -1.0, 1.0).astype(np.float32, copy=False)
        dy = np.clip((feature_cache.centroid_rows_world - ar) / half_h, -1.0, 1.0).astype(np.float32, copy=False)

        cluster_area = np.clip(
            feature_cache.log_sizes / np.log1p(float(max(2, wh * ww))),
            0.0,
            1.0,
        ).astype(np.float32, copy=False)
        obstacle_density = feature_cache.obstacle_density

        # Lightweight candidate-ordering heuristic only. It helps rank clusters
        # for top-k selection without reintroducing explicit path planning or
        # unknown-potential estimation into the token features themselves.
        geom_dist_norm = np.clip(np.sqrt((dx * dx) + (dy * dy)) / np.sqrt(2.0), 0.0, 1.0).astype(np.float32)
        priority = ((0.65 * cluster_area) - (0.20 * obstacle_density) - (0.15 * geom_dist_norm)).astype(np.float32)

        order = self._priority_order(priority, cluster_area, obstacle_density, geom_dist_norm)
        n = min(k, feature_cache.cluster_count)
        if n > 0:
            selected = order[:n]
            tokens[:n, 0] = dx[selected]
            tokens[:n, 1] = dy[selected]
            tokens[:n, 2] = cluster_area[selected]
            tokens[:n, 3] = obstacle_density[selected]
            token_mask[:n] = True

        if self._timing_enabled:
            build_elapsed = time.perf_counter() - build_t0
            self.build_total_time += build_elapsed

        if not return_meta:
            return tokens, token_mask

        meta = {
            "cluster_count_raw": feature_cache.cluster_count_raw,
            "cluster_count": feature_cache.cluster_count,
            "selected_count": n,
            "priority_scores": priority[order[:n]].astype(np.float32).tolist(),
        }
        if self._timing_enabled:
            meta["timing"] = {
                "cluster_extract_time": float(cluster_elapsed),
                "build_total_time": float(build_elapsed),
            }
        return tokens, token_mask, meta

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

    builder = FrontierRegionTokenBuilder(FrontierRegionTokenConfig(top_k=8))
    tokens, mask, meta = builder.build(cm, s, frontier_u8, world_window_shape=(96, 96), return_meta=True)

    assert tokens.shape == (8, FRONTIER_REGION_TOKEN_FIELD_COUNT)
    assert mask.shape == (8,)
    if meta["selected_count"] < 8:
        assert np.allclose(tokens[meta["selected_count"] :], 0.0)
        assert np.all(mask[meta["selected_count"] :] == 0)

    print("FrontierRegionTokenBuilder smoke test passed", tokens.shape, mask.sum())


if __name__ == "__main__":
    _smoke_test()
