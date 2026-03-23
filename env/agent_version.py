from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

from env.core_radar import RadarSensor
from env.grid_topology import EMPTY, INVISIBLE, OBSTACLE


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

        self.local_snap = np.full(self.local_shape, INVISIBLE, dtype=np.int8)
        self.observe_fast(agent_state)

    def _global_to_local(self, agent_state: Tuple[int, int], gr: int, gc: int) -> Optional[Tuple[int, int]]:
        ar, ac = int(agent_state[0]), int(agent_state[1])
        cr, cc = int(self.center_state[0]), int(self.center_state[1])
        lr = cr + (int(gr) - ar)
        lc = cc + (int(gc) - ac)
        if 0 <= lr < self.local_shape[0] and 0 <= lc < self.local_shape[1]:
            return int(lr), int(lc)
        return None

    def _render_local_snap(self, agent_state: Tuple[int, int], los_lines: Sequence[Sequence[Tuple[int, int]]]) -> np.ndarray:
        """
        Full-disk visibility semantics:
        - each LOS line corresponds to one candidate target in Euclidean disk
        - if an obstacle is hit before target, target remains invisible
        - first obstacle on LOS is visible
        """
        snap = np.full(self.local_shape, INVISIBLE, dtype=np.int8)

        for line in los_lines:
            for r, c in line:
                local = self._global_to_local(agent_state, r, c)
                if local is None:
                    break

                lr, lc = local
                value = int(self.grid[r, c])
                if value == OBSTACLE:
                    snap[lr, lc] = OBSTACLE
                    break
                snap[lr, lc] = EMPTY

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

        for ray in self.sensor.local_ray_templates:
            target_dr, target_dc, _, _ = ray[-1]
            tr, tc = ar + target_dr, ac + target_dc
            if not (0 <= tr < self._rows and 0 <= tc < self._cols):
                continue

            for rel_r, rel_c, local_r, local_c in ray:
                value = int(self.grid[ar + rel_r, ac + rel_c])
                if value == OBSTACLE:
                    snap[local_r, local_c] = OBSTACLE
                    break
                snap[local_r, local_c] = EMPTY

        return snap

    def observe(self, agent_state: Tuple[int, int]) -> Tuple[np.ndarray, List[List[Tuple[int, int]]]]:
        agent = (int(agent_state[0]), int(agent_state[1]))
        self.observe_fast(agent)
        los_lines = self.sensor.scan_area_cal(agent, self.grid.shape)
        return self.local_snap, los_lines
