from __future__ import annotations

import hashlib
import random
from typing import Optional, Tuple

import numpy as np

from env.grid_topology import EMPTY, OBSTACLE, GridTopology


def compute_map_fingerprint(grid: np.ndarray, start: tuple[int, int]) -> str:
    arr = np.ascontiguousarray(np.asarray(grid, dtype=np.int8))
    start_rc = (int(start[0]), int(start[1]))
    payload = hashlib.sha1()
    payload.update(str(tuple(arr.shape)).encode("utf-8"))
    payload.update(b"|")
    payload.update(arr.tobytes())
    payload.update(b"|")
    payload.update(f"{start_rc[0]},{start_rc[1]}".encode("utf-8"))
    return payload.hexdigest()[:16]


class RandomMapGenerator:
    """Generate bounded random maps and valid 1x1 start states."""

    def __init__(self, rows: int, cols: int, obs_size: int, obstacle_ratio: float = 0.2):
        self.rows = int(rows)
        self.cols = int(cols)
        self.obs_size = int(obs_size)
        self.obstacle_ratio = float(obstacle_ratio)

        if self.rows < 6 or self.cols < 6:
            raise ValueError("rows and cols must be >= 6")
        if self.obs_size < 1:
            raise ValueError("obs_size must be >= 1")
        if not (0.0 <= self.obstacle_ratio < 0.85):
            raise ValueError("obstacle_ratio must be in [0.0, 0.85)")

        inner_rows = self.rows - 2
        inner_cols = self.cols - 2
        inner_area = inner_rows * inner_cols

        self.target_obstacles = int(round(inner_area * self.obstacle_ratio))
        self.max_rect_trials = max(800, 12 * inner_area)
        self.max_generate_tries = 32

        self.map: Optional[np.ndarray] = None

    @staticmethod
    def _normalize_np_seed(seed: int) -> int:
        return int(seed) % (2**32)

    def _generate_map_unseeded(self) -> Tuple[np.ndarray, Tuple[int, int]]:
        for _ in range(self.max_generate_tries):
            grid = self._generate_candidate()
            start = self._pick_start(grid)
            if start is not None:
                self.map = grid
                return grid, start

        grid = self._generate_candidate()
        self._force_center_clear(grid)
        start = self._pick_start(grid)
        if start is None:
            raise RuntimeError("failed to produce a valid map/start for 1x1 agent")

        self.map = grid
        return grid, start

    def generate_map(self, seed: int | None = None) -> Tuple[np.ndarray, Tuple[int, int]]:
        if seed is None:
            return self._generate_map_unseeded()

        python_state = random.getstate()
        numpy_state = np.random.get_state()
        try:
            random.seed(int(seed))
            np.random.seed(self._normalize_np_seed(int(seed)))
            return self._generate_map_unseeded()
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)

    def _generate_candidate(self) -> np.ndarray:
        grid = np.full((self.rows, self.cols), EMPTY, dtype=np.int8)
        self._add_border(grid)
        self._carve_obstacles(grid)
        self._keep_largest_free_component(grid)
        return grid

    def _add_border(self, grid: np.ndarray) -> None:
        grid[0, :] = OBSTACLE
        grid[-1, :] = OBSTACLE
        grid[:, 0] = OBSTACLE
        grid[:, -1] = OBSTACLE

    def _sample_rect_hw(self) -> Tuple[int, int]:
        mean = (self.obs_size + 1.0) / 2.0
        std = max(0.8, self.obs_size / 3.5)
        h = int(np.clip(np.round(np.random.normal(mean, std)), 1, self.obs_size))
        w = int(np.clip(np.round(np.random.normal(mean, std)), 1, self.obs_size))
        return h, w

    def _carve_obstacles(self, grid: np.ndarray) -> None:
        H, W = self.rows, self.cols
        placed = 0
        trials = 0

        while placed < self.target_obstacles and trials < self.max_rect_trials:
            trials += 1

            if placed > 0 and random.random() < 0.6:
                obs = np.argwhere(grid[1:-1, 1:-1] == OBSTACLE)
                if len(obs) > 0:
                    br, bc = obs[np.random.randint(0, len(obs))]
                    r = int(np.clip(br + 1 + np.random.randint(-2, 3), 1, H - 2))
                    c = int(np.clip(bc + 1 + np.random.randint(-2, 3), 1, W - 2))
                else:
                    r = int(np.random.randint(1, H - 1))
                    c = int(np.random.randint(1, W - 1))
            else:
                r = int(np.random.randint(1, H - 1))
                c = int(np.random.randint(1, W - 1))

            h, w = self._sample_rect_hw()
            if random.random() < 0.5:
                h, w = w, h

            r2 = min(r + h, H - 1)
            c2 = min(c + w, W - 1)
            patch = grid[r:r2, c:c2]

            newly = int((patch == EMPTY).sum())
            if newly == 0:
                continue
            patch[:, :] = OBSTACLE
            placed += newly

        self._match_target_density(grid)

    def _match_target_density(self, grid: np.ndarray) -> None:
        inner = grid[1:-1, 1:-1]
        cur = int((inner == OBSTACLE).sum())
        diff = self.target_obstacles - cur
        if diff == 0:
            return

        if diff > 0:
            empties = np.argwhere(inner == EMPTY)
            if len(empties) == 0:
                return
            k = min(diff, len(empties))
            pts = empties[np.random.choice(len(empties), size=k, replace=False)]
            inner[pts[:, 0], pts[:, 1]] = OBSTACLE
        else:
            obstacles = np.argwhere(inner == OBSTACLE)
            if len(obstacles) == 0:
                return
            k = min(-diff, len(obstacles))
            pts = obstacles[np.random.choice(len(obstacles), size=k, replace=False)]
            inner[pts[:, 0], pts[:, 1]] = EMPTY

    def _keep_largest_free_component(self, grid: np.ndarray) -> None:
        free = (grid == EMPTY)
        allowed = np.zeros_like(free, dtype=bool)
        allowed[1:-1, 1:-1] = True

        # Keep generator behavior stable: retain only the largest inner free component.
        # Coverage denominator is computed separately in simulator-side effective coverage
        # and does not rely on this heuristic to infer full-map reachability.
        largest, size = GridTopology.largest_component_mask(free, allowed=allowed)
        if size > 0:
            grid[allowed & (~largest)] = OBSTACLE

    def _pick_start(self, grid: np.ndarray) -> Optional[Tuple[int, int]]:
        free = (grid == EMPTY)
        allowed = np.zeros_like(free, dtype=bool)
        allowed[1:-1, 1:-1] = True

        largest, size = GridTopology.largest_component_mask(free, allowed=allowed)
        if size <= 0:
            return None

        candidates = np.argwhere(largest)
        if len(candidates) == 0:
            return None

        best_score = None
        best_states = []

        for r, c in candidates:
            r = int(r)
            c = int(c)
            margin = min(r, self.rows - 1 - r, c, self.cols - 1 - c)

            r1, r2 = max(1, r - 2), min(self.rows - 1, r + 3)
            c1, c2 = max(1, c - 2), min(self.cols - 1, c + 3)
            local_free = int((grid[r1:r2, c1:c2] == EMPTY).sum())

            score = 2.0 * margin + local_free
            if best_score is None or score > best_score:
                best_score = score
                best_states = [(r, c)]
            elif score == best_score:
                best_states.append((r, c))

        return random.choice(best_states) if best_states else None

    def _force_center_clear(self, grid: np.ndarray) -> None:
        r = max(1, min(self.rows // 2, self.rows - 2))
        c = max(1, min(self.cols // 2, self.cols - 2))
        grid[r, c] = EMPTY
