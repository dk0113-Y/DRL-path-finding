from __future__ import annotations

from collections import deque
from typing import Iterator, Optional, Tuple

import numpy as np

INVISIBLE = -1
EMPTY = 0
OBSTACLE = 1

ACTIONS_8: Tuple[Tuple[int, int], ...] = (
    (-1, 0),
    (-1, 1),
    (0, 1),
    (1, 1),
    (1, 0),
    (1, -1),
    (0, -1),
    (-1, -1),
)


class GridTopology:
    """
    Canonical 1x1 geometry/topology semantics for environment modules.

    Notes on retained graph utilities:
    - `bfs_reachable` is used by simulator-side effective coverage denominator.
    - `largest_component_mask` is used by random map generation/start filtering.
    """

    @staticmethod
    def free_mask(grid: np.ndarray) -> np.ndarray:
        return np.asarray(grid == EMPTY, dtype=bool)

    @staticmethod
    def in_bounds(shape: Tuple[int, int], r: int, c: int) -> bool:
        return 0 <= r < int(shape[0]) and 0 <= c < int(shape[1])

    @staticmethod
    def can_occupy(free: np.ndarray, r: int, c: int) -> bool:
        return GridTopology.in_bounds(free.shape, r, c) and bool(free[r, c])

    @staticmethod
    def can_step(free: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> bool:
        if not GridTopology.can_occupy(free, r0, c0):
            return False
        if not GridTopology.can_occupy(free, r1, c1):
            return False

        dr = int(r1 - r0)
        dc = int(c1 - c0)
        if dr == 0 and dc == 0:
            return False
        if abs(dr) > 1 or abs(dc) > 1:
            return False

        # Diagonal corner-cut prevention.
        if dr != 0 and dc != 0:
            if not GridTopology.can_occupy(free, r0 + dr, c0):
                return False
            if not GridTopology.can_occupy(free, r0, c0 + dc):
                return False

        return True

    @staticmethod
    def valid_action_indices_fast(free: np.ndarray, state: Tuple[int, int]) -> Tuple[int, ...]:
        """
        Fast 8-action legality helper with the same corner-cut prevention semantics
        as can_step()/valid_action_indices().
        """
        r, c = int(state[0]), int(state[1])
        H, W = int(free.shape[0]), int(free.shape[1])
        if not (0 <= r < H and 0 <= c < W) or not bool(free[r, c]):
            return ()

        valid: list[int] = []

        north = (r > 0) and bool(free[r - 1, c])
        south = (r + 1 < H) and bool(free[r + 1, c])
        west = (c > 0) and bool(free[r, c - 1])
        east = (c + 1 < W) and bool(free[r, c + 1])

        if north:
            valid.append(0)
        if north and east and bool(free[r - 1, c + 1]):
            valid.append(1)
        if east:
            valid.append(2)
        if south and east and bool(free[r + 1, c + 1]):
            valid.append(3)
        if south:
            valid.append(4)
        if south and west and bool(free[r + 1, c - 1]):
            valid.append(5)
        if west:
            valid.append(6)
        if north and west and bool(free[r - 1, c - 1]):
            valid.append(7)

        return tuple(valid)

    @staticmethod
    def circular_turn_steps(prev_action_idx: int | None, curr_action_idx: int) -> int:
        """
        Circular action-index distance on the canonical ACTIONS_8 ring.

        Returns:
          0 for no turn / missing previous action
          1 for 45 degrees
          2 for 90 degrees
          3 for 135 degrees
          4 for 180 degrees
        """
        if prev_action_idx is None:
            return 0

        action_dim = len(ACTIONS_8)
        prev_idx = int(prev_action_idx)
        curr_idx = int(curr_action_idx)
        if not (0 <= prev_idx < action_dim):
            raise ValueError(f"prev_action_idx out of range: {prev_action_idx}")
        if not (0 <= curr_idx < action_dim):
            raise ValueError(f"curr_action_idx out of range: {curr_action_idx}")

        delta = (curr_idx - prev_idx) % action_dim
        return int(min(delta, action_dim - delta))

    @staticmethod
    def valid_action_indices(free: np.ndarray, state: Tuple[int, int]) -> set[int]:
        return set(GridTopology.valid_action_indices_fast(free, state))

    @staticmethod
    def neighbors(free: np.ndarray, state: Tuple[int, int]) -> Iterator[Tuple[int, int]]:
        r, c = int(state[0]), int(state[1])
        for dr, dc in ACTIONS_8:
            nr, nc = r + dr, c + dc
            if GridTopology.can_step(free, r, c, nr, nc):
                yield nr, nc

    @staticmethod
    def bfs_reachable(
        free: np.ndarray,
        start: Tuple[int, int],
        allowed: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Kinematics-aware reachability on free grid (8-neighborhood with corner constraints)."""
        H, W = free.shape
        vis = np.zeros((H, W), dtype=bool)

        sr, sc = int(start[0]), int(start[1])
        if not GridTopology.can_occupy(free, sr, sc):
            return vis
        if allowed is not None and not bool(allowed[sr, sc]):
            return vis

        dq = deque([(sr, sc)])
        vis[sr, sc] = True

        while dq:
            cur = dq.popleft()
            for nr, nc in GridTopology.neighbors(free, cur):
                if vis[nr, nc]:
                    continue
                if allowed is not None and not bool(allowed[nr, nc]):
                    continue
                vis[nr, nc] = True
                dq.append((nr, nc))

        return vis

    @staticmethod
    def bfs_reachable_4(
        free: np.ndarray,
        start: Tuple[int, int],
        allowed: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Orthogonal 4-neighborhood reachability on a free grid."""
        H, W = free.shape
        vis = np.zeros((H, W), dtype=bool)

        sr, sc = int(start[0]), int(start[1])
        if not GridTopology.can_occupy(free, sr, sc):
            return vis
        if allowed is not None and not bool(allowed[sr, sc]):
            return vis

        dq = deque([(sr, sc)])
        vis[sr, sc] = True

        while dq:
            r, c = dq.popleft()
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if not GridTopology.can_occupy(free, nr, nc):
                    continue
                if vis[nr, nc]:
                    continue
                if allowed is not None and not bool(allowed[nr, nc]):
                    continue
                vis[nr, nc] = True
                dq.append((nr, nc))

        return vis

    @staticmethod
    def bfs_distance_map(
        free: np.ndarray,
        start: Tuple[int, int],
        allowed: Optional[np.ndarray] = None,
        unreachable_value: int = -1,
    ) -> np.ndarray:
        """
        Kinematics-aware shortest-step distance map on free grid.

        Distances are computed with the same step legality as `can_step`/`neighbors`.
        Unreachable cells are filled with `unreachable_value`.
        """
        H, W = free.shape
        dist = np.full((H, W), int(unreachable_value), dtype=np.int32)

        sr, sc = int(start[0]), int(start[1])
        if not GridTopology.can_occupy(free, sr, sc):
            return dist
        if allowed is not None and not bool(allowed[sr, sc]):
            return dist

        dq = deque([(sr, sc)])
        dist[sr, sc] = 0

        while dq:
            cur = dq.popleft()
            base_d = int(dist[cur[0], cur[1]])
            nd = base_d + 1
            for nr, nc in GridTopology.neighbors(free, cur):
                if dist[nr, nc] >= 0:
                    continue
                if allowed is not None and not bool(allowed[nr, nc]):
                    continue
                dist[nr, nc] = nd
                dq.append((nr, nc))

        return dist

    @staticmethod
    def largest_component_mask(
        free: np.ndarray,
        allowed: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, int]:
        """Largest connected component under the same movement kinematics as agent transitions."""
        H, W = free.shape
        seen = np.zeros((H, W), dtype=bool)
        best = np.zeros((H, W), dtype=bool)
        best_size = 0

        for sr in range(H):
            for sc in range(W):
                if seen[sr, sc] or not bool(free[sr, sc]):
                    continue
                if allowed is not None and not bool(allowed[sr, sc]):
                    continue

                comp = np.zeros((H, W), dtype=bool)
                dq = deque([(sr, sc)])
                seen[sr, sc] = True
                comp[sr, sc] = True
                size = 1

                while dq:
                    cur = dq.popleft()
                    for nr, nc in GridTopology.neighbors(free, cur):
                        if seen[nr, nc]:
                            continue
                        if allowed is not None and not bool(allowed[nr, nc]):
                            continue
                        seen[nr, nc] = True
                        comp[nr, nc] = True
                        size += 1
                        dq.append((nr, nc))

                if size > best_size:
                    best_size = size
                    best = comp

        return best, best_size

    @staticmethod
    def frontier_mask(
        known_map: np.ndarray,
        min_unknown_neighbors: int = 1,
        connectivity: int = 8,
    ) -> np.ndarray:
        """
        Canonical frontier geometry on belief map:
          known_free cells adjacent to unknown cells.

        This is a boundary definition only; exploration value scoring is handled
        downstream (e.g., token/cluster ranking), not in this mask.
        """
        unknown = (known_map == INVISIBLE).astype(np.uint8)
        known_free = (known_map == EMPTY)

        p = np.pad(unknown, 1, mode="constant", constant_values=0)
        if int(connectivity) == 4:
            neigh_unknown = (
                p[:-2, 1:-1] +
                p[1:-1, :-2] +
                p[1:-1, 2:] +
                p[2:, 1:-1]
            )
        elif int(connectivity) == 8:
            neigh_unknown = (
                p[:-2, :-2] + p[:-2, 1:-1] + p[:-2, 2:] + p[1:-1, :-2] +
                p[1:-1, 2:] + p[2:, :-2] + p[2:, 1:-1] + p[2:, 2:]
            )
        else:
            raise ValueError(f"frontier connectivity must be 4 or 8, got {connectivity}")
        return known_free & (neigh_unknown >= int(min_unknown_neighbors))

    @staticmethod
    def local_to_global_grid(
        agent_rc: Tuple[int, int],
        local_shape: Tuple[int, int],
        center_rc: Tuple[int, int],
    ) -> Tuple[np.ndarray, np.ndarray]:
        ar, ac = int(agent_rc[0]), int(agent_rc[1])
        lr = np.arange(int(local_shape[0]), dtype=np.int32)
        lc = np.arange(int(local_shape[1]), dtype=np.int32)
        lrg, lcg = np.meshgrid(lr, lc, indexing="ij")
        gr = ar + (lrg - int(center_rc[0]))
        gc = ac + (lcg - int(center_rc[1]))
        return gr, gc
