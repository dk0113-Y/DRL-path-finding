from __future__ import annotations

import math
from typing import Dict, List, Tuple


def _bresenham_line(r0: int, c0: int, r1: int, c1: int) -> List[Tuple[int, int]]:
    dr = abs(r1 - r0)
    dc = abs(c1 - c0)
    sr = 1 if r0 < r1 else -1
    sc = 1 if c0 < c1 else -1

    r, c = r0, c0
    pts = [(r, c)]

    if dc > dr:
        err = dc // 2
        while c != c1:
            c += sc
            err -= dr
            if err < 0:
                r += sr
                err += dc
            pts.append((r, c))
    else:
        err = dr // 2
        while r != r1:
            r += sr
            err -= dc
            if err < 0:
                c += sc
                err += dr
            pts.append((r, c))

    return pts


class RadarSensor:
    """
    1x1 agent radar with full-disk candidate visibility semantics.

    local window shape is square: (2R+1, 2R+1)
    effective geometric FOV is Euclidean disk: dr^2 + dc^2 <= R^2

    Note:
    this sensor-local window only defines observation footprint/local_snap size.
    Policy local-state window size is configured independently in LocalStateConfig.
    """

    def __init__(self, scan_radius: int = 10):
        if scan_radius < 1:
            raise ValueError("scan_radius must be >= 1")

        self.scan_r = int(scan_radius)
        self.local_shape = (2 * self.scan_r + 1, 2 * self.scan_r + 1)
        self.center_state = (self.scan_r, self.scan_r)

        self._disk_offsets = self._build_disk_offsets()
        self._los_templates = self._build_los_templates()
        self._local_ray_templates = self._build_local_ray_templates()

    @property
    def theoretical_visible_cell_count(self) -> int:
        """Number of radar footprint cells implied by the current geometry."""
        return int(len(self._disk_offsets))

    def _build_disk_offsets(self) -> Tuple[Tuple[int, int], ...]:
        r = self.scan_r
        pts = set()
        rr = r * r

        for dr in range(-r, r + 1):
            for dc in range(-r, r + 1):
                if dr * dr + dc * dc <= rr:
                    pts.add((dr, dc))

        # Smooth single-cell cardinal tips with a symmetric 3-cell shoulder.
        # This only expands the outermost boundary near N/S/E/W and keeps the
        # interior fully covered.
        if r >= 2:
            shoulder = (
                (-r, -1), (-r, 1),
                (r, -1), (r, 1),
                (-1, -r), (1, -r),
                (-1, r), (1, r),
            )
            for p in shoulder:
                pts.add(p)

        # Stable order: distance first (near -> far), then angle.
        out = list(pts)
        out.sort(key=lambda p: (p[0] * p[0] + p[1] * p[1], math.atan2(p[0], p[1])))
        return tuple(out)

    def _build_los_templates(self) -> Dict[Tuple[int, int], Tuple[Tuple[int, int], ...]]:
        out: Dict[Tuple[int, int], Tuple[Tuple[int, int], ...]] = {}
        for dr, dc in self._disk_offsets:
            out[(dr, dc)] = tuple(_bresenham_line(0, 0, dr, dc))
        return out

    def _build_local_ray_templates(
        self,
    ) -> Tuple[Tuple[Tuple[int, int, int, int], ...], ...]:
        center_r, center_c = int(self.center_state[0]), int(self.center_state[1])
        rays: list[Tuple[Tuple[int, int, int, int], ...]] = []
        for dr, dc in self._disk_offsets:
            line = self._los_templates[(dr, dc)]
            rays.append(
                tuple(
                    (
                        int(rr),
                        int(cc),
                        int(center_r + rr),
                        int(center_c + cc),
                    )
                    for rr, cc in line
                )
            )
        return tuple(rays)

    @property
    def local_ray_templates(self) -> Tuple[Tuple[Tuple[int, int, int, int], ...], ...]:
        """
        Precomputed local-space LOS templates.

        Each point entry is (rel_r, rel_c, local_r, local_c), where:
          rel_*   : offset from agent in world coordinates
          local_* : write location inside the fixed local observation window
        """
        return self._local_ray_templates

    def scan_area_cal(self, agent_state: Tuple[int, int], map_shape: Tuple[int, int]) -> List[List[Tuple[int, int]]]:
        """
        Returns LOS lines to all in-bounds disk targets.
        Each returned line is full geometric LOS (not obstacle-truncated).
        """
        ar, ac = int(agent_state[0]), int(agent_state[1])
        H, W = int(map_shape[0]), int(map_shape[1])

        rays: List[List[Tuple[int, int]]] = []
        for ray in self._local_ray_templates:
            dr, dc, _, _ = ray[-1]
            tr, tc = ar + dr, ac + dc
            if not (0 <= tr < H and 0 <= tc < W):
                continue

            rays.append([(ar + rr, ac + cc) for rr, cc, _, _ in ray])

        return rays
