from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from env.advantage_state_builder import (  # noqa: E402
    FRONTIER_CHANNEL_MODE_LOCAL_BINARY,
    FRONTIER_CHANNEL_MODE_LOCAL_GLOBAL_AREA,
    AdvantageStateBuilder,
    AdvantageStateConfig,
)
from env.core_cummap import AnalysisBox  # noqa: E402
from env.grid_topology import EMPTY  # noqa: E402
from env.shared_semantic_layer import (  # noqa: E402
    FrontierCluster,
    SharedSemanticSnapshot,
    SparseMaskGeometry,
    SupportGeometry,
    UnknownBlock,
)


class DummyCumMap:
    def __init__(self) -> None:
        self.local_shape = (3, 3)
        self.map = np.full((5, 5), EMPTY, dtype=np.int8)
        self.visit_count = np.ones((5, 5), dtype=np.int32)
        self.frontier_u8 = np.zeros((5, 5), dtype=np.uint8)
        self.frontier_u8[1, 2] = 255
        self.frontier_u8[2, 3] = 255
        self.frontier_u8[3, 3] = 255
        self.frontier_u8[0, 0] = 255

    def world_to_array(self, world_rc: tuple[int, int]) -> tuple[int, int]:
        return int(world_rc[0]), int(world_rc[1])

    def get_frontier_u8(self, refresh: bool = False) -> np.ndarray:
        if refresh:
            raise AssertionError("local frontier variants must use refresh=False")
        return self.frontier_u8


def _geometry_from_cells(cells: list[tuple[int, int]]) -> SparseMaskGeometry:
    if not cells:
        return SparseMaskGeometry.empty()
    rows = np.asarray([cell[0] for cell in cells], dtype=np.int32)
    cols = np.asarray([cell[1] for cell in cells], dtype=np.int32)
    r0 = int(rows.min())
    c0 = int(cols.min())
    r1 = int(rows.max()) + 1
    c1 = int(cols.max()) + 1
    mask = np.zeros((r1 - r0, c1 - c0), dtype=bool)
    mask[rows - r0, cols - c0] = True
    return SparseMaskGeometry(r0=r0, c0=c0, mask=mask, count=int(mask.sum()))


def _cluster(cells: list[tuple[int, int]], *, block_index: int) -> FrontierCluster:
    support = SupportGeometry(
        local_box_bounds=(0, 0, 0, 0),
        support_free_geometry=SparseMaskGeometry.empty(),
        support_obstacle_density=0.0,
    )
    anchor = cells[0] if cells else (0, 0)
    return FrontierCluster(
        frontier_index=block_index,
        block_index=block_index,
        frontier_geometry=_geometry_from_cells(cells),
        support_geometry=support,
        frontier_anchor_rc=(int(anchor[0]), int(anchor[1])),
        delta_r=0.0,
        delta_c=0.0,
        entry_width=float(len(cells)),
    )


def _snapshot() -> SharedSemanticSnapshot:
    analysis_box = AnalysisBox(
        r0=0,
        r1=5,
        c0=0,
        c1=5,
        margin=0,
        known_r0=0,
        known_r1=5,
        known_c0=0,
        known_c1=5,
    )
    matched_block = UnknownBlock(
        block_index=0,
        unknown_geometry=SparseMaskGeometry.empty(),
        frontier_clusters=(_cluster([(1, 2), (2, 3)], block_index=0),),
        block_area=4,
        frontier_cluster_count=1,
    )
    outside_block = UnknownBlock(
        block_index=1,
        unknown_geometry=SparseMaskGeometry.empty(),
        frontier_clusters=(_cluster([(0, 0)], block_index=1),),
        block_area=6,
        frontier_cluster_count=1,
    )
    return SharedSemanticSnapshot(
        analysis_box=analysis_box,
        accessible_blocks=(matched_block, outside_block),
        total_accessible_unknown_area=10,
    )


def _assert_close(actual: np.ndarray, expected: np.ndarray) -> None:
    if not np.allclose(actual, expected):
        raise AssertionError(f"array mismatch\nactual:\n{actual}\nexpected:\n{expected}")


def main() -> int:
    cum_map = DummyCumMap()
    agent = (2, 2)
    snapshot = _snapshot()

    binary_builder = AdvantageStateBuilder(
        AdvantageStateConfig(frontier_channel_mode=FRONTIER_CHANNEL_MODE_LOCAL_BINARY)
    )
    binary_canvas, binary_meta = binary_builder.build(cum_map, agent, snapshot)
    expected_binary = np.asarray(
        [
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    _assert_close(binary_canvas[2], expected_binary)
    assert binary_meta["frontier_channel_mode"] == FRONTIER_CHANNEL_MODE_LOCAL_BINARY
    assert binary_meta["local_frontier_positive_count"] == 3.0
    assert binary_meta["local_frontier_block_area_mean"] == 0.0

    area_builder = AdvantageStateBuilder(
        AdvantageStateConfig(frontier_channel_mode=FRONTIER_CHANNEL_MODE_LOCAL_GLOBAL_AREA)
    )
    area_canvas, area_meta = area_builder.build(cum_map, agent, snapshot)
    expected_area = np.asarray(
        [
            [0.0, 0.4, 0.0],
            [0.0, 0.0, 0.4],
            [0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    _assert_close(area_canvas[2], expected_area)
    assert area_meta["frontier_channel_mode"] == FRONTIER_CHANNEL_MODE_LOCAL_GLOBAL_AREA
    assert area_meta["local_frontier_positive_count"] == 3.0
    assert area_meta["local_frontier_global_area_positive_count"] == 2.0
    assert area_meta["local_frontier_unmatched_count"] == 1.0
    assert abs(float(area_meta["local_frontier_block_area_mean"]) - 0.4) < 1.0e-6

    print("frontier channel variant checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
