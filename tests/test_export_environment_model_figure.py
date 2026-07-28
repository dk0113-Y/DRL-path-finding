from __future__ import annotations

import unittest

import matplotlib.pyplot as plt
import numpy as np

from env.core_radar import RadarSensor
from env.grid_topology import ACTIONS_8, EMPTY, INVISIBLE, OBSTACLE
from tools.export_environment_model_figure import (
    EnvironmentFigureStyle,
    _clip_ray_to_observation,
    _draw_eight_neighbor_action_arrows,
    _normalized_ray_angle,
    _select_representative_rays,
)


class EnvironmentModelFigureHelpersTest(unittest.TestCase):
    def test_representative_rays_are_bounded_uniform_and_non_mutating(self) -> None:
        sensor = RadarSensor(scan_radius=10)
        templates = sensor.local_ray_templates
        endpoint_snapshot = tuple((ray[-1][0], ray[-1][1]) for ray in templates)

        selected = _select_representative_rays(templates, visual_ray_count=32)

        self.assertLessEqual(len(selected), 32)
        self.assertEqual(len(selected), 32)
        self.assertEqual(endpoint_snapshot, tuple((ray[-1][0], ray[-1][1]) for ray in templates))

        angles = np.sort(np.asarray([_normalized_ray_angle(ray) for ray in selected], dtype=np.float64))
        circular_gaps = np.diff(np.concatenate([angles, angles[:1] + (2.0 * np.pi)]))
        self.assertLess(float(np.max(circular_gaps)), 0.35)
        quadrant_counts = [
            int(np.count_nonzero((angles >= start) & (angles < start + (np.pi / 2.0))))
            for start in (0.0, np.pi / 2.0, np.pi, 3.0 * np.pi / 2.0)
        ]
        self.assertLessEqual(max(quadrant_counts) - min(quadrant_counts), 1)

    def test_eight_neighbor_arrows_use_canonical_action_order_and_directions(self) -> None:
        fig, ax = plt.subplots()
        try:
            directions = _draw_eight_neighbor_action_arrows(
                ax,
                center_rc=(4.0, 4.0),
                style=EnvironmentFigureStyle(),
            )
        finally:
            plt.close(fig)

        self.assertEqual(directions, ACTIONS_8)
        self.assertEqual(len(directions), 8)
        self.assertEqual(len(set(directions)), 8)

    def test_ray_clipping_stops_at_first_obstacle(self) -> None:
        observation = np.full((9, 9), EMPTY, dtype=np.int8)
        observation[4, 6] = OBSTACLE
        observation[4, 7] = INVISIBLE
        observation[4, 8] = INVISIBLE
        ray = (
            (0, 0, 4, 4),
            (0, 1, 4, 5),
            (0, 2, 4, 6),
            (0, 3, 4, 7),
            (0, 4, 4, 8),
        )

        clipped = _clip_ray_to_observation(ray, observation)

        self.assertEqual(clipped, ray[:3])
        self.assertEqual(int(observation[clipped[-1][2], clipped[-1][3]]), OBSTACLE)


if __name__ == "__main__":
    unittest.main()
