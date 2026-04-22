from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

from env.core_radar import RadarSensor
from env.grid_topology import EMPTY, INVISIBLE, OBSTACLE


_NO_CORNER_CHECK = -1_000_000
_FastInteriorCell = tuple[int, int, int, int, int, int]
_FastInteriorRay = tuple[_FastInteriorCell, ...]
_FastBoundaryCell = tuple[int, int, int, int, int, int, int, int]
_FastBoundaryRay = tuple[int, int, tuple[_FastBoundaryCell, ...]]


class LocalObservationModel:
    """
    Compute local radar observation for a 1x1 agent.

    Local observation size is controlled by `sensor.scan_radius`.
    This affects local_snap only; policy local-state window is configured separately.
    """

    def __init__(self, grid: np.ndarray, agent_state: Tuple[int, int], sensor: Optional[RadarSensor] = None):
        self.grid = np.asarray(grid, dtype=np.int8)
        if self.grid.ndim != 2:
            raise ValueError("grid must be a 2D array")

        self.sensor = sensor if sensor is not None else RadarSensor()
        self.center_state = tuple(self.sensor.center_state)
        self.local_shape = tuple(self.sensor.local_shape)
        self._rows = int(self.grid.shape[0])
        self._cols = int(self.grid.shape[1])
        self._scan_radius = int(self.sensor.scan_r)
        self._block_corner_peeking = bool(getattr(self.sensor, "block_corner_peeking", True))
        self._interior_r_min = self._scan_radius
        self._interior_r_max_exclusive = self._rows - self._scan_radius
        self._interior_c_min = self._scan_radius
        self._interior_c_max_exclusive = self._cols - self._scan_radius
        self._fast_interior_rays, self._fast_boundary_rays = self._build_fast_observation_cache()

        self.local_snap = np.full(self.local_shape, INVISIBLE, dtype=np.int8)
        self.observe_fast(agent_state)

    def _build_fast_observation_cache(
        self,
    ) -> tuple[tuple[_FastInteriorRay, ...], tuple[_FastBoundaryRay, ...]]:
        center_r, center_c = int(self.center_state[0]), int(self.center_state[1])
        interior_rays: list[_FastInteriorRay] = []
        boundary_rays: list[_FastBoundaryRay] = []

        for ray in self.sensor.local_ray_templates:
            target_dr, target_dc, _, _ = ray[-1]
            interior_cells: list[_FastInteriorCell] = []
            boundary_cells: list[_FastBoundaryCell] = []
            prev_rel_r: int | None = None
            prev_rel_c: int | None = None

            for rel_r, rel_c, local_r, local_c in ray:
                rel_r_i = int(rel_r)
                rel_c_i = int(rel_c)
                local_r_i = int(local_r)
                local_c_i = int(local_c)

                side_a_rel_r = _NO_CORNER_CHECK
                side_a_rel_c = _NO_CORNER_CHECK
                side_b_rel_r = _NO_CORNER_CHECK
                side_b_rel_c = _NO_CORNER_CHECK
                side_a_local_r = _NO_CORNER_CHECK
                side_a_local_c = _NO_CORNER_CHECK
                side_b_local_r = _NO_CORNER_CHECK
                side_b_local_c = _NO_CORNER_CHECK

                if (
                    self._block_corner_peeking
                    and prev_rel_r is not None
                    and abs(rel_r_i - prev_rel_r) == 1
                    and abs(rel_c_i - prev_rel_c) == 1
                ):
                    side_a_rel_r = rel_r_i
                    side_a_rel_c = prev_rel_c
                    side_b_rel_r = prev_rel_r
                    side_b_rel_c = rel_c_i
                    side_a_local_r = center_r + side_a_rel_r
                    side_a_local_c = center_c + side_a_rel_c
                    side_b_local_r = center_r + side_b_rel_r
                    side_b_local_c = center_c + side_b_rel_c

                interior_cells.append(
                    (
                        local_r_i,
                        local_c_i,
                        side_a_local_r,
                        side_a_local_c,
                        side_b_local_r,
                        side_b_local_c,
                    )
                )
                boundary_cells.append(
                    (
                        rel_r_i,
                        rel_c_i,
                        local_r_i,
                        local_c_i,
                        side_a_rel_r,
                        side_a_rel_c,
                        side_b_rel_r,
                        side_b_rel_c,
                    )
                )
                prev_rel_r = rel_r_i
                prev_rel_c = rel_c_i

            interior_rays.append(tuple(interior_cells))
            boundary_rays.append((int(target_dr), int(target_dc), tuple(boundary_cells)))

        return tuple(interior_rays), tuple(boundary_rays)

    def _global_to_local(self, agent_state: Tuple[int, int], gr: int, gc: int) -> Optional[Tuple[int, int]]:
        ar, ac = int(agent_state[0]), int(agent_state[1])
        cr, cc = int(self.center_state[0]), int(self.center_state[1])
        lr = cr + (int(gr) - ar)
        lc = cc + (int(gc) - ac)
        if 0 <= lr < self.local_shape[0] and 0 <= lc < self.local_shape[1]:
            return int(lr), int(lc)
        return None

    def _corner_occluded_global(self, prev_rc: Tuple[int, int], cur_rc: Tuple[int, int]) -> bool:
        if not bool(getattr(self.sensor, "block_corner_peeking", True)):
            return False

        pr, pc = int(prev_rc[0]), int(prev_rc[1])
        cr, cc = int(cur_rc[0]), int(cur_rc[1])
        dr = cr - pr
        dc = cc - pc
        if abs(dr) != 1 or abs(dc) != 1:
            return False

        side_a = (cr, pc)
        side_b = (pr, cc)
        for rr, cc_ in (side_a, side_b):
            if not (0 <= rr < self._rows and 0 <= cc_ < self._cols):
                return False
        return bool(self.grid[side_a[0], side_a[1]] == OBSTACLE) and bool(
            self.grid[side_b[0], side_b[1]] == OBSTACLE
        )

    def _render_local_snap(self, agent_state: Tuple[int, int], los_lines: Sequence[Sequence[Tuple[int, int]]]) -> np.ndarray:
        """
        Full-disk visibility semantics:
        - each LOS line corresponds to one candidate target in Euclidean disk
        - if an obstacle is hit before target, target remains invisible
        - first obstacle on LOS is visible
        """
        snap = np.full(self.local_shape, INVISIBLE, dtype=np.int8)

        for line in los_lines:
            prev_global: Optional[Tuple[int, int]] = None
            for r, c in line:
                if prev_global is not None and self._corner_occluded_global(prev_global, (r, c)):
                    break

                local = self._global_to_local(agent_state, r, c)
                if local is None:
                    break

                lr, lc = local
                value = int(self.grid[r, c])
                if value == OBSTACLE:
                    snap[lr, lc] = OBSTACLE
                    break
                snap[lr, lc] = EMPTY
                prev_global = (int(r), int(c))

        return snap

    def observe_fast(self, agent_state: Tuple[int, int]) -> np.ndarray:
        """
        Hot-path local observation rendering.

        Reuses the persistent local_snap buffer and iterates precomputed local-space
        LOS templates, avoiding per-step LOS list construction and global->local remapping.
        """
        ar, ac = int(agent_state[0]), int(agent_state[1])
        snap = self.local_snap
        snap.fill(INVISIBLE)
        grid = self.grid
        obstacle = OBSTACLE

        if (
            self._interior_r_min <= ar < self._interior_r_max_exclusive
            and self._interior_c_min <= ac < self._interior_c_max_exclusive
        ):
            local_grid = grid[
                ar - self._scan_radius:ar + self._scan_radius + 1,
                ac - self._scan_radius:ac + self._scan_radius + 1,
            ]
            if self._block_corner_peeking:
                for ray in self._fast_interior_rays:
                    for local_r, local_c, side_a_r, side_a_c, side_b_r, side_b_c in ray:
                        if (
                            side_a_r >= 0
                            and local_grid[side_a_r, side_a_c] == obstacle
                            and local_grid[side_b_r, side_b_c] == obstacle
                        ):
                            break

                        value = int(local_grid[local_r, local_c])
                        if value == obstacle:
                            snap[local_r, local_c] = OBSTACLE
                            break
                        snap[local_r, local_c] = EMPTY
            else:
                for ray in self._fast_interior_rays:
                    for local_r, local_c, _, _, _, _ in ray:
                        value = int(local_grid[local_r, local_c])
                        if value == obstacle:
                            snap[local_r, local_c] = OBSTACLE
                            break
                        snap[local_r, local_c] = EMPTY
            return snap

        rows = self._rows
        cols = self._cols
        if self._block_corner_peeking:
            for target_dr, target_dc, ray in self._fast_boundary_rays:
                tr, tc = ar + target_dr, ac + target_dc
                if not (0 <= tr < rows and 0 <= tc < cols):
                    continue

                for rel_r, rel_c, local_r, local_c, side_a_r, side_a_c, side_b_r, side_b_c in ray:
                    if (
                        side_a_r != _NO_CORNER_CHECK
                        and grid[ar + side_a_r, ac + side_a_c] == obstacle
                        and grid[ar + side_b_r, ac + side_b_c] == obstacle
                    ):
                        break

                    value = int(grid[ar + rel_r, ac + rel_c])
                    if value == obstacle:
                        snap[local_r, local_c] = OBSTACLE
                        break
                    snap[local_r, local_c] = EMPTY
        else:
            for target_dr, target_dc, ray in self._fast_boundary_rays:
                tr, tc = ar + target_dr, ac + target_dc
                if not (0 <= tr < rows and 0 <= tc < cols):
                    continue

                for rel_r, rel_c, local_r, local_c, _, _, _, _ in ray:
                    value = int(grid[ar + rel_r, ac + rel_c])
                    if value == obstacle:
                        snap[local_r, local_c] = OBSTACLE
                        break
                    snap[local_r, local_c] = EMPTY

        return snap

    def observe(self, agent_state: Tuple[int, int]) -> Tuple[np.ndarray, List[List[Tuple[int, int]]]]:
        agent = (int(agent_state[0]), int(agent_state[1]))
        self.observe_fast(agent)
        los_lines = self.sensor.scan_area_cal(agent, self.grid.shape)
        return self.local_snap, los_lines
