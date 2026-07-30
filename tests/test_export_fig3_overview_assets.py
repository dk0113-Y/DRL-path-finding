from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from PIL import Image

from env.advantage_state_builder import FINAL_4CH_ADVANTAGE_CANVAS_CHANNELS
from tools.export_fig3_overview_assets import (
    MANIFEST_FILENAME,
    export_fig3_overview_assets,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
PAPER_REPO = REPO_ROOT.parent / "paper_work"
SOURCE_MANIFEST = (
    REPO_ROOT.parent
    / "figure_assets"
    / "fig1_seed1"
    / "online_workflow_assets_manifest.json"
)
SHARED_ASSET_NAMES = (
    "dynamic_cumulative_belief_map",
    "belief_map_with_robot_and_history",
    "interaction_history",
)


def _export(output_dir: Path):
    return export_fig3_overview_assets(
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
        source_manifest=SOURCE_MANIFEST,
    )


def test_deterministic_hashes_and_shared_scene_manifest(
    tmp_path: Path,
) -> None:
    first = _export(tmp_path / "first")
    second = _export(tmp_path / "second")
    first_manifest = first["manifest"]
    second_manifest = second["manifest"]
    for key in (
        "seed",
        "requested_step",
        "resolved_step",
        "agent_world_position",
        "selected_action_index",
        "selected_action_key",
        "belief_origin_world",
        "belief_canvas_shape",
        "belief_matrix_sha256",
        "trajectory_length",
        "trajectory_positions_world",
        "shared_source_scene",
        "advantage_channel_names",
        "advantage_tensor_shape",
        "total_unknown_block_count",
        "total_frontier_entrance_count",
    ):
        assert first_manifest[key] == second_manifest[key]
    for asset_name in first_manifest["assets"]:
        assert (
            first_manifest["assets"][asset_name]["png_sha256"]
            == second_manifest["assets"][asset_name]["png_sha256"]
        )
        assert (
            first_manifest["assets"][asset_name]["svg_sha256"]
            == second_manifest["assets"][asset_name]["svg_sha256"]
        )


def test_shared_assets_have_one_canvas_projection_and_exact_layers(
    tmp_path: Path,
) -> None:
    result = _export(tmp_path / "assets")
    manifest = result["manifest"]
    assets = manifest["assets"]
    dimensions = {
        (assets[name]["width_px"], assets[name]["height_px"])
        for name in SHARED_ASSET_NAMES
    }
    origins = {
        tuple(assets[name]["belief_origin_world"])
        for name in SHARED_ASSET_NAMES
    }
    canvas_shapes = {
        tuple(assets[name]["belief_canvas_shape"])
        for name in SHARED_ASSET_NAMES
    }
    projections = {
        assets[name]["coordinate_projection"]
        for name in SHARED_ASSET_NAMES
    }
    assert len(dimensions) == 1
    assert origins == {tuple(manifest["belief_origin_world"])}
    assert canvas_shapes == {tuple(manifest["belief_canvas_shape"])}
    assert len(projections) == 1
    assert assets["dynamic_cumulative_belief_map"]["render_layers"] == [
        "cumulative_belief_occupancy"
    ]
    assert assets["belief_map_with_robot_and_history"]["render_layers"] == [
        "cumulative_belief_occupancy",
        "executed_trajectory",
        "topdown_robot",
    ]
    assert assets["interaction_history"]["render_layers"] == [
        "executed_trajectory"
    ]
    assert manifest["same_belief_background"] is True
    assert manifest["same_world_canvas"] is True
    assert manifest["same_coordinate_projection"] is True
    assert manifest["same_trajectory_geometry"] is True


def test_history_is_transparent_path_only_and_composite_uses_shared_robot(
    tmp_path: Path,
) -> None:
    result = _export(tmp_path / "assets")
    assets = result["manifest"]["assets"]
    history = assets["interaction_history"]
    with Image.open(history["png_path"]) as image:
        assert image.mode == "RGBA"
        alpha = np.asarray(image.getchannel("A"), dtype=np.uint8)
    transparent_count = int(np.count_nonzero(alpha == 0))
    nontransparent_count = int(np.count_nonzero(alpha > 0))
    assert transparent_count > 0
    assert nontransparent_count > 0
    assert history["transparent_background"] is True
    assert history["transparent_pixel_count"] == transparent_count
    assert history["nontransparent_pixel_count"] == nontransparent_count

    dynamic_svg = Path(
        assets["dynamic_cumulative_belief_map"]["svg_path"]
    ).read_text(encoding="utf-8")
    composite_svg = Path(
        assets["belief_map_with_robot_and_history"]["svg_path"]
    ).read_text(encoding="utf-8")
    history_svg = Path(history["svg_path"]).read_text(encoding="utf-8")
    assert "fig3_executed_trajectory" not in dynamic_svg
    assert "fig3_topdown_robot" not in dynamic_svg
    assert "fig3_executed_trajectory" in composite_svg
    assert "fig3_topdown_robot_part_0" in composite_svg
    assert "fig3_executed_trajectory" in history_svg
    assert "fig3_topdown_robot" not in history_svg
    assert "cumulative_belief_raster" not in history_svg


def test_robot_position_asset_is_removed_and_known_stale_files_are_cleaned(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "assets"
    output_dir.mkdir()
    (output_dir / "robot_position.png").write_bytes(b"stale-png")
    (output_dir / "robot_position.svg").write_text(
        "<svg>stale</svg>",
        encoding="utf-8",
    )
    result = _export(output_dir)
    manifest = result["manifest"]
    assert "robot_position" not in manifest["assets"]
    assert "robot_position" not in result["files"]
    assert not (output_dir / "robot_position.png").exists()
    assert not (output_dir / "robot_position.svg").exists()
    assert manifest["robot_position_asset_generated"] is False
    assert (
        manifest["robot_position_source"]
        == "manual crop from belief_map_with_robot_and_history"
    )
    assert "crop_bounds" not in manifest


def test_complete_trajectory_and_belief_match_the_shared_blueprint(
    tmp_path: Path,
) -> None:
    result = _export(tmp_path / "assets")
    scene = result["scene"]
    manifest = result["manifest"]
    trajectory = np.asarray(scene.trajectory_world, dtype=np.int32)
    assert manifest["seed"] == 1
    assert manifest["resolved_step"] == 8
    assert manifest["agent_world_position"] == [13, 40]
    assert len(trajectory) == 9
    assert tuple(trajectory[-1]) == tuple(scene.agent_world)
    assert len(trajectory) == manifest["trajectory_length"]
    assert trajectory.tolist() == manifest["trajectory_positions_world"]
    assert np.array_equal(
        trajectory,
        scene.blueprint.belief_after_update.trajectory_world,
    )
    assert np.array_equal(
        scene.cum_map.map,
        scene.blueprint.belief_after_update.belief_map,
    )
    assert manifest["shared_source_scene"]["trajectory_positions_world"] == (
        manifest["trajectory_positions_world"]
    )
    composite_positions = manifest["assets"][
        "belief_map_with_robot_and_history"
    ]["trajectory_positions_world"]
    history_positions = manifest["assets"]["interaction_history"][
        "trajectory_positions_world"
    ]
    assert composite_positions == history_positions
    assert composite_positions == manifest["trajectory_positions_world"]


def test_local_and_global_states_keep_the_active_project_contract(
    tmp_path: Path,
) -> None:
    result = _export(tmp_path / "assets")
    scene = result["scene"]
    manifest = result["manifest"]
    assert tuple(scene.advantage_canvas.shape) == (4, 21, 21)
    assert tuple(FINAL_4CH_ADVANTAGE_CANVAS_CHANNELS) == (
        "free",
        "obstacle",
        "visit_count_log_norm",
        "recent_trajectory_decay",
    )
    assert scene.advantage_meta["advantage_canvas_channel_count"] == 4.0
    assert scene.advantage_meta["frontier_raster_used"] is False
    assert float(np.sum(scene.advantage_canvas[0])) > 0.0
    assert float(np.sum(scene.advantage_canvas[1])) > 0.0
    assert float(np.sum(scene.advantage_canvas[3])) > 0.0

    blocks = tuple(scene.semantic_snapshot.accessible_blocks)
    entries = sum(int(block.frontier_cluster_count) for block in blocks)
    assert manifest["total_unknown_block_count"] == len(blocks)
    assert manifest["total_frontier_entrance_count"] == entries
    assert manifest["displayed_unknown_block_count"] <= len(blocks)
    assert manifest["displayed_frontier_entrance_count"] <= entries
    assert int(scene.value_meta["value_total_block_count_before_cap"]) == len(
        blocks
    )
    assert int(scene.value_meta["value_total_entry_count_before_cap"]) == entries


def test_all_files_open_and_figure2_source_is_verified(tmp_path: Path) -> None:
    result = _export(tmp_path / "assets")
    manifest = json.loads(
        (tmp_path / "assets" / MANIFEST_FILENAME).read_text(encoding="utf-8")
    )
    assert manifest["truth_map_rendered"] is False
    assert manifest["same_snapshot_as_source_belief_asset"] is True
    assert (
        manifest["figure2_or_belief_map_source"]["manifest_verified"] is True
    )
    assert (
        manifest["figure2_or_belief_map_source"][
            "same_belief_matrix_verified"
        ]
        is True
    )
    for record in manifest["assets"].values():
        png_path = Path(record["png_path"])
        svg_path = Path(record["svg_path"])
        assert png_path.stat().st_size > 0
        assert svg_path.stat().st_size > 0
        with Image.open(png_path) as image:
            image.verify()
        ET.parse(svg_path)
