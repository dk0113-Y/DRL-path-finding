from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from env.grid_topology import ACTIONS_8, INVISIBLE, OBSTACLE
from tools.export_architecture_pictures import ExportConfig
from tools.export_environment_model_figure import EnvironmentFigureStyle
from tools.export_figure_demo_blueprint import (
    blueprint_manifest,
    build_figure_demo_blueprint,
    clip_ray_to_local_snap,
)
from tools.export_online_workflow_assets import (
    OnlineWorkflowStyle,
    _action_arrow_specs,
    _assets_from_blueprint,
    _belief_background,
    export_online_workflow_assets,
)
from tools.paper_figure_style import (
    draw_topdown_robot,
    load_paper_figure_style,
    robot_envelope_diameter_cells,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


class FigureDemoBlueprintTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.blueprint = build_figure_demo_blueprint(
            seed=1,
            preferred_step=8,
            scan_radius=10,
            visual_ray_count=32,
        )

    def test_seed1_snapshot_contract(self) -> None:
        blueprint = self.blueprint
        self.assertEqual(blueprint.seed, 1)
        self.assertEqual(blueprint.step, 8)
        self.assertGreaterEqual(len(blueprint.valid_action_indices), 1)
        self.assertGreaterEqual(len(blueprint.invalid_action_indices), 1)
        self.assertIn(blueprint.selected_action, blueprint.valid_action_indices)
        self.assertGreater(
            int(np.count_nonzero(blueprint.local_observation.local_snap == OBSTACLE)),
            0,
        )
        manifest = blueprint_manifest(blueprint)
        self.assertEqual(manifest["local_shape"], [21, 21])
        self.assertEqual(manifest["center_state"], [10, 10])
        self.assertEqual(manifest["scan_radius"], 10)

    def test_action_colors_widths_and_normalized_lengths(self) -> None:
        blueprint = self.blueprint
        style = OnlineWorkflowStyle()
        specs = _action_arrow_specs(
            center_row=10.0,
            center_col=10.0,
            valid_action_indices=blueprint.valid_action_indices,
            chosen_action_index=blueprint.selected_action,
            style=style,
        )
        self.assertEqual(len(specs), 8)
        expected_colors = style.paper.fig1_action_palette
        for spec in specs:
            self.assertEqual(spec.color, expected_colors[spec.state])
        selected = [spec for spec in specs if spec.state == "selected"]
        legal = [spec for spec in specs if spec.state == "legal"]
        illegal = [spec for spec in specs if spec.state == "illegal"]
        self.assertEqual(len(selected), 1)
        self.assertGreaterEqual(len(legal), 1)
        self.assertGreaterEqual(len(illegal), 1)
        normal_width = max(spec.linewidth_pt for spec in specs if spec.state != "selected")
        self.assertGreaterEqual(selected[0].linewidth_pt, 1.6 * normal_width)
        lengths = np.asarray([spec.length_cells for spec in specs], dtype=np.float64)
        self.assertLessEqual(float(lengths.max() - lengths.min()), 0.01 * float(lengths.mean()))
        self.assertTrue(np.allclose(lengths, 1.5))

    def test_belief_update_and_environment_share_exact_b_t(self) -> None:
        blueprint = self.blueprint
        assets = _assets_from_blueprint(blueprint)
        belief_background = _belief_background(assets, blueprint.belief_canvas)
        environment_background = _belief_background(assets, blueprint.belief_canvas)
        self.assertTrue(np.array_equal(belief_background, blueprint.belief_display))
        self.assertTrue(np.array_equal(environment_background, blueprint.belief_display))
        self.assertEqual(assets.belief_after_update.agent_world, tuple(blueprint_manifest(blueprint)["p_t"]))
        with tempfile.TemporaryDirectory() as temporary_dir:
            result = export_online_workflow_assets(
                temporary_dir,
                config=ExportConfig(
                    seed=1,
                    scan_radius=10,
                    step_late=8,
                    dpi=40,
                    output_dir=Path(temporary_dir),
                ),
                step=8,
            )
            manifest = json.loads(Path(result["manifest_path"]).read_text(encoding="utf-8"))
        self.assertEqual(manifest["belief_update_panel_count"], 1)
        self.assertEqual(manifest["belief_update_semantics"], "B_t with one robot at p_t")
        self.assertEqual(manifest["belief_update_robot_count"], 1)
        self.assertFalse(manifest["belief_update_has_trajectory"])
        self.assertFalse(manifest["belief_update_has_new_cell_highlight"])
        self.assertEqual(manifest["environment_robot_count"], 2)
        self.assertEqual(manifest["environment_motion_arrow_count"], 0)
        self.assertEqual(
            manifest["belief_update_background_sha256"],
            manifest["environment_background_sha256"],
        )

    def test_online_cycle_temporal_contract(self) -> None:
        blueprint = self.blueprint
        p_t = blueprint.belief_after_update.agent_world
        p_next = blueprint.environment_after_action.agent_world
        self.assertEqual(blueprint.local_observation.agent_world, p_t)
        self.assertEqual(blueprint.belief_before_update.agent_world, p_t)
        self.assertEqual(blueprint.belief_after_update.agent_world, p_t)
        actual_delta = (p_next[0] - p_t[0], p_next[1] - p_t[1])
        self.assertEqual(actual_delta, ACTIONS_8[blueprint.selected_action])
        self.assertTrue(
            np.array_equal(
                blueprint.belief_after_update.belief_map,
                blueprint.environment_after_action.belief_map,
            )
        )

    def test_shared_robot_contract_is_used_by_python_and_visio(self) -> None:
        style = load_paper_figure_style()
        self.assertEqual(
            dict(OnlineWorkflowStyle().paper.robot_palette),
            dict(EnvironmentFigureStyle().paper.robot_palette),
        )
        self.assertEqual(
            dict(OnlineWorkflowStyle().paper.robot_geometry_cell_relative),
            dict(EnvironmentFigureStyle().paper.robot_geometry_cell_relative),
        )
        self.assertLessEqual(
            robot_envelope_diameter_cells(style),
            style.robot_geometry_cell_relative["envelope_target_cells"],
        )
        fig, ax = plt.subplots()
        try:
            parts = draw_topdown_robot(
                ax,
                row=0.0,
                col=0.0,
                heading_action=self.blueprint.selected_action,
                style=style,
            )
        finally:
            plt.close(fig)
        self.assertEqual(len(parts), 7)  # body + four wheels + radar + heading

        common_source = (REPO_ROOT / "tools/visio/visio_common.ps1").read_text(encoding="utf-8-sig")
        fig1_source = (REPO_ROOT / "tools/visio/build_fig1_online_interaction.ps1").read_text(
            encoding="utf-8-sig"
        )
        fig2_source = (REPO_ROOT / "tools/visio/build_fig2_environment_model.ps1").read_text(
            encoding="utf-8-sig"
        )
        self.assertIn("paper_figure_style.json", common_source)
        self.assertIn("function Add-PaperFigureRobotParts", common_source)
        self.assertIn("Import-PaperFigureStyle", fig1_source)
        self.assertIn("Import-PaperFigureStyle", fig2_source)
        self.assertIn("Add-PaperFigureRobotParts", fig1_source)
        self.assertIn("Add-PaperFigureRobotParts", fig2_source)

    def test_real_los_rays_are_template_derived_and_truncated(self) -> None:
        blueprint = self.blueprint
        self.assertEqual(blueprint.sensor.scan_r, 10)
        self.assertEqual(blueprint.sensor.local_shape, (21, 21))
        self.assertEqual(blueprint.sensor.center_state, (10, 10))
        self.assertEqual(len(blueprint.representative_rays), 32)
        lengths: list[float] = []
        for representative in blueprint.representative_rays:
            template = blueprint.sensor.local_ray_templates[representative.template_index]
            self.assertEqual(
                representative.source_endpoint_relative,
                (int(template[-1][0]), int(template[-1][1])),
            )
            self.assertEqual(
                representative.points,
                clip_ray_to_local_snap(template, blueprint.local_observation.local_snap),
            )
            end_row, end_col = representative.end_local_rc
            self.assertNotEqual(
                int(blueprint.local_observation.local_snap[end_row, end_col]),
                INVISIBLE,
            )
            obstacle_positions = [
                point_index
                for point_index, point in enumerate(representative.points)
                if int(blueprint.local_observation.local_snap[point[2], point[3]]) == OBSTACLE
            ]
            if obstacle_positions:
                self.assertEqual(obstacle_positions[0], len(representative.points) - 1)
            end = representative.points[-1]
            lengths.append(math.hypot(end[0], end[1]))
        self.assertGreater(max(lengths) - min(lengths), 0.5)
        self.assertIn(
            "variable ray lengths are expected",
            blueprint_manifest(blueprint)["ray_semantics"],
        )

    def test_legacy_figure2_hardcoding_is_absent(self) -> None:
        fig2_source = (REPO_ROOT / "tools/visio/build_fig2_environment_model.ps1").read_text(
            encoding="utf-8-sig"
        )
        self.assertNotIn("fixedObstacleKeys", fig2_source)
        self.assertNotIn("RadarRadiusCells", fig2_source)
        self.assertNotIn("RadarFillRadiusCells", fig2_source)
        self.assertNotIn("GridRows = 15", fig2_source)
        self.assertNotIn("GridCols = 15", fig2_source)


if __name__ == "__main__":
    unittest.main()
