from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

from env.shared_semantic_layer import SharedSemanticSnapshot


VALUE_BLOCK_FEATURES = (
    "block_area_ratio",
    "frontier_cluster_count",
    "representative_delta_r_ratio",
    "representative_delta_c_ratio",
    "representative_entry_width_ratio",
    "representative_support_obstacle_density",
)
VALUE_ENTRY_FEATURES = (
    "delta_r_ratio",
    "delta_c_ratio",
    "entry_width_ratio",
    "support_obstacle_density",
)
VALUE_BLOCK_FEATURE_COUNT = len(VALUE_BLOCK_FEATURES)
VALUE_ENTRY_FEATURE_COUNT = len(VALUE_ENTRY_FEATURES)


@dataclass(frozen=True)
class ValueStateConfig:
    max_accessible_blocks: int = 16
    max_entries_per_block: int = 6
    enable_timing: bool = False


class ValueStateBuilder:
    """
    Build the block-tree tensor state consumed by the value branch.

    Unknown blocks are the primary units. Frontier clusters remain attached as
    children under each block and are never flattened into a single shared token
    list. Block sorting is only used for stable tensor packing, not expert
    prioritization. Block summaries carry direct area/count statistics plus the
    nearest representative frontier-anchor summary. Frontier clusters carry
    local entry geometry, while SupportGeometry is reduced to a local
    obstacle-density descriptor.
    """

    def __init__(self, config: Optional[ValueStateConfig] = None):
        self.config = config if config is not None else ValueStateConfig()
        self._timing_enabled = bool(self.config.enable_timing)
        self.build_time = 0.0

    def build(
        self,
        semantic_snapshot: SharedSemanticSnapshot,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        t0 = time.perf_counter() if self._timing_enabled else 0.0
        max_blocks = int(self.config.max_accessible_blocks)
        max_entries = int(self.config.max_entries_per_block)
        block_features = np.zeros((max_blocks, VALUE_BLOCK_FEATURE_COUNT), dtype=np.float32)
        entry_features = np.zeros((max_blocks, max_entries, VALUE_ENTRY_FEATURE_COUNT), dtype=np.float32)
        block_mask = np.zeros((max_blocks,), dtype=bool)
        entry_mask = np.zeros((max_blocks, max_entries), dtype=bool)

        blocks = list(semantic_snapshot.accessible_blocks[:max_blocks])
        total_unknown_area = float(max(1, semantic_snapshot.total_accessible_unknown_area))
        box_h, box_w = semantic_snapshot.analysis_box.shape
        delta_r_scale = float(max(1, box_h))
        delta_c_scale = float(max(1, box_w))
        entry_width_scale = float(max(1, box_h + box_w))

        for block_slot, block in enumerate(blocks):
            block_mask[block_slot] = True
            block_features[block_slot, 0] = np.float32(float(block.block_area) / total_unknown_area)
            block_features[block_slot, 1] = np.float32(float(block.frontier_cluster_count))
            block_features[block_slot, 2] = np.float32(float(block.representative_delta_r) / delta_r_scale)
            block_features[block_slot, 3] = np.float32(float(block.representative_delta_c) / delta_c_scale)
            block_features[block_slot, 4] = np.float32(
                float(block.representative_entry_width) / entry_width_scale
            )
            block_features[block_slot, 5] = np.float32(float(block.representative_support_obstacle_density))

            for entry_slot, frontier_cluster in enumerate(block.frontier_clusters[:max_entries]):
                entry_mask[block_slot, entry_slot] = True
                entry_features[block_slot, entry_slot, 0] = np.float32(float(frontier_cluster.delta_r) / delta_r_scale)
                entry_features[block_slot, entry_slot, 1] = np.float32(float(frontier_cluster.delta_c) / delta_c_scale)
                entry_features[block_slot, entry_slot, 2] = np.float32(float(frontier_cluster.entry_width) / entry_width_scale)
                entry_features[block_slot, entry_slot, 3] = np.float32(float(frontier_cluster.support_obstacle_density))

        if self._timing_enabled:
            self.build_time += time.perf_counter() - t0
        return block_features, entry_features, block_mask, entry_mask

    def get_timing_stats(self) -> dict[str, float]:
        return {"build_time": float(self.build_time)}
