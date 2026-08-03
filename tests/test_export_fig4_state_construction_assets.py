from __future__ import annotations

import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from PIL import Image

from tools.export_fig4_state_construction_assets import (
    ASSET_NAMES,
    MANIFEST_FILENAME,
    export_fig4_state_construction_assets,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
PAPER_REPO = REPO_ROOT.parent / "paper_work"


def _export(output_dir: Path):
    return export_fig4_state_construction_assets(
        output_dir,
        seed=1,
        step=8,
        rows=40,
        cols=60,
        obstacle_ratio=0.20,
        obs_size=6,
        scan_radius=10,
        dpi=72,
        include_svg=True,
        paper_repo=PAPER_REPO,
    )


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256_float32(array: np.ndarray) -> str:
    return hashlib.sha256(
        np.ascontiguousarray(np.asarray(array, dtype=np.float32)).tobytes()
    ).hexdigest()


def _svg_ids(path: Path) -> set[str]:
    return {
        element_id
        for element in ET.parse(path).iter()
        if (element_id := element.attrib.get("id")) is not None
    }


def test_scene_reuses_fig3_seed_step_and_real_storage_expansion(tmp_path: Path) -> None:
    manifest = _export(tmp_path / "assets")["manifest"]
    assert manifest["scene_source"].endswith("build_fig3_overview_scene")
    assert manifest["seed"] == 1
    assert manifest["requested_step"] == 8
    assert manifest["resolved_step"] == 8
    assert manifest["previous_step"] == 7
    assert manifest["previous_storage_shape"] == [32, 31]
    assert manifest["expanded_storage_shape"] == [32, 43]
    assert manifest["storage_expansion_cells"]["right"] == 12
    assert sum(manifest["storage_expansion_cells"].values()) > 0
    assert manifest["storage_boundary_rendered"] is False
    assert manifest["previous_boundary_rendered"] is False
    assert manifest["local_window_boundary_rendered_in_belief_asset"] is False
    assert manifest["storage_expansion_recorded_in_manifest"] is True
    assert manifest["truth_map_rendered"] is False


def test_local_state_has_four_real_channels_and_linear_history(tmp_path: Path) -> None:
    result = _export(tmp_path / "assets")
    scene = result["scene"]
    manifest = result["manifest"]
    assert tuple(scene.shared.advantage_canvas.shape) == (4, 21, 21)
    assert manifest["local_channel_names"] == [
        "free",
        "obstacle",
        "visit_count_log_norm",
        "recent_trajectory_decay",
    ]
    assert manifest["local_window_shape"] == [21, 21]
    assert float(np.sum(scene.shared.advantage_canvas[0])) > 0.0
    assert float(np.sum(scene.shared.advantage_canvas[1])) > 0.0
    assert float(np.sum(scene.shared.advantage_canvas[3])) > 0.0
    channel_assets = (
        "local_channel_free_space",
        "local_channel_obstacle",
        "local_channel_visit_count",
        "local_channel_recent_trajectory",
    )
    for channel_index, asset_name in enumerate(channel_assets):
        contract = manifest["assets"][asset_name]["layer_contract"]
        assert contract["source"] == "scene.shared.advantage_canvas"
        assert contract["channel_index"] == channel_index
        assert contract["source_array_shape"] == [21, 21]
        assert contract["source_array_sha256_float32"] == _sha256_float32(
            scene.shared.advantage_canvas[channel_index]
        )
    weights = manifest["recent_trajectory_semantics"]["weights_old_to_new"]
    assert weights == sorted(weights)
    assert np.allclose(
        weights,
        [(index + 1) / len(weights) for index in range(len(weights))],
    )
    assert manifest["recent_trajectory_semantics"]["current_robot_cell_repainted"] is False


def test_asset_layer_contracts_and_svg_gids_are_clean(tmp_path: Path) -> None:
    result = _export(tmp_path / "assets")
    scene = result["scene"]
    manifest = result["manifest"]
    assets = manifest["assets"]

    dynamic = assets["dynamic_cumulative_belief_map"]["layer_contract"]
    assert dynamic["cumulative_belief_occupancy"] is True
    assert dynamic["robot"] is False
    assert dynamic["trajectory"] is False
    assert dynamic["storage_boundary"] is False
    assert dynamic["local_window_boundary"] is False

    belief = assets["belief_map_with_robot_and_history"]["layer_contract"]
    assert belief["current_robot"] is True
    assert belief["current_robot_count"] == 1
    assert belief["previous_robot"] is False
    assert belief["next_robot"] is False
    assert belief["executed_trajectory"] is True
    assert belief["storage_boundary"] is False
    assert belief["local_window_boundary"] is False
    assert belief["trajectory_style"] == "fig3_shared_blue"
    assert belief["belief_matrix_sha256"] == dynamic["belief_matrix_sha256"]
    assert belief["executed_trajectory_point_count"] == len(
        scene.shared.trajectory_world
    )
    assert belief["executed_trajectory_endpoint_world"] == list(
        scene.shared.agent_world
    )
    assert tuple(scene.shared.trajectory_world[-1]) == tuple(scene.shared.agent_world)

    local_window = assets["robot_centered_local_window"]["layer_contract"]
    assert local_window["local_sampled_occupancy"] is True
    assert local_window["recent_history"] is True
    assert local_window["current_robot_count"] == 1
    assert local_window["external_window_outline"] is False
    assert local_window["trajectory_style"] == "fig3_shared_blue"

    frontier = assets["frontier_unknown_region_extraction"]["layer_contract"]
    assert frontier["unknown_region_overlay"] is True
    assert frontier["frontier_cluster_overlay"] is True
    assert frontier["unknown_region_cell_count"] > 0
    assert frontier["frontier_cluster_cell_count"] > 0
    assert frontier["robot_marker"] is False
    assert frontier["trajectory"] is False
    assert frontier["semantic_analysis_box"] is False

    style = manifest["trajectory_style_contract"]
    assert style == {
        "name": "fig3_shared_blue",
        "line_color": "#315F91",
        "marker_color": "#5185C0",
        "marker_edge_color": "#FFFFFF",
        "line_width_pt": 2.0,
        "marker_size_pt": 3.2,
        "marker_edge_width_pt": 0.45,
        "alpha": 0.94,
        "solid_capstyle": "round",
        "solid_joinstyle": "round",
        "marker_size_or_alpha_gradient": False,
    }

    dynamic_ids = _svg_ids(Path(assets["dynamic_cumulative_belief_map"]["svg_path"]))
    belief_ids = _svg_ids(Path(assets["belief_map_with_robot_and_history"]["svg_path"]))
    local_ids = _svg_ids(Path(assets["robot_centered_local_window"]["svg_path"]))
    frontier_ids = _svg_ids(
        Path(assets["frontier_unknown_region_extraction"]["svg_path"])
    )
    assert "fig4_dynamic_cumulative_belief" in dynamic_ids
    assert "fig4_executed_trajectory" in belief_ids
    assert any(value.startswith("fig4_current_robot_part_") for value in belief_ids)
    assert "fig4_local_window_recent_trajectory" in local_ids
    assert any(
        value.startswith("fig4_local_window_current_robot_part_")
        for value in local_ids
    )
    assert "fig4_unknown_region_overlay" in frontier_ids
    assert "fig4_frontier_cluster_overlay" in frontier_ids

    boundary_ids = {
        "fig4_previous_storage_boundary",
        "fig4_expanded_storage_boundary",
        "fig4_robot_centered_local_window",
    }
    assert dynamic_ids.isdisjoint(boundary_ids)
    assert belief_ids.isdisjoint(boundary_ids)
    assert "fig4_semantic_analysis_box" not in frontier_ids
    assert "fig4_frontier_robot_marker" not in frontier_ids
    assert "fig4_executed_trajectory" not in frontier_ids


def test_global_hierarchy_and_five_fixed_inputs_match_active_contract(tmp_path: Path) -> None:
    result = _export(tmp_path / "assets")
    scene = result["scene"]
    manifest = result["manifest"]
    assert scene.block_features.shape == (16, 2)
    assert scene.entry_features.shape == (16, 8, 4)
    assert scene.block_mask.shape == (16,)
    assert scene.entry_mask.shape == (16, 8)
    assert manifest["five_policy_inputs"] == {
        "advantage_canvas": [4, 21, 21],
        "value_block_features": [16, 2],
        "value_entry_features": [16, 8, 4],
        "value_block_mask": [16],
        "value_entry_mask": [16, 8],
    }
    global_state = manifest["global_semantic_state"]
    assert global_state["frontier_associated_unknown_region_block_count"] >= 1
    assert global_state["frontier_entrance_count"] >= 1
    assert global_state["frontier_cluster_connectivity"] == 8
    assert global_state["unknown_region_connectivity"] == 4
    assert manifest["feature_semantics"]["block_features"] == [
        "relative area",
        "number of entrances",
    ]
    assert manifest["feature_semantics"]["entrance_features"][2] == (
        "frontier-cluster scale"
    )


def test_assets_are_deterministic_openable_png_and_parseable_svg(tmp_path: Path) -> None:
    first = _export(tmp_path / "first")
    second = _export(tmp_path / "second")
    for name in ASSET_NAMES:
        first_record = first["manifest"]["assets"][name]
        second_record = second["manifest"]["assets"][name]
        assert first_record["png_sha256"] == second_record["png_sha256"]
        assert first_record["svg_sha256"] == second_record["svg_sha256"]
        png_path = Path(first_record["png_path"])
        svg_path = Path(first_record["svg_path"])
        assert png_path.stat().st_size > 0
        assert svg_path.stat().st_size > 0
        assert _sha256_file(png_path) == first_record["png_sha256"]
        assert _sha256_file(svg_path) == first_record["svg_sha256"]
        with Image.open(png_path) as image:
            image.verify()
        ET.parse(svg_path)
        assert all(
            line == line.rstrip()
            for line in svg_path.read_text(encoding="utf-8").splitlines()
        )

    manifest_path = tmp_path / "first" / MANIFEST_FILENAME
    parsed = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert set(parsed["assets"]) == set(ASSET_NAMES)
    assert parsed["claim_boundaries"] == {
        "reachability_asserted": False,
        "network_structure_rendered": False,
        "figure_stops_at_fixed_dimensional_state_interface": True,
    }
