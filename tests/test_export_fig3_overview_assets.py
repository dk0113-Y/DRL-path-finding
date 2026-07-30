from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from PIL import Image

from env.advantage_state_builder import FINAL_4CH_ADVANTAGE_CANVAS_CHANNELS
from tools.export_fig3_overview_assets import (
    MANIFEST_FILENAME,
    build_fig3_overview_scene,
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


def test_deterministic_hashes_and_key_manifest_fields(tmp_path: Path) -> None:
    first = _export(tmp_path / "first")
    second = _export(tmp_path / "second")
    first_manifest = first["manifest"]
    second_manifest = second["manifest"]
    for key in (
        "seed",
        "requested_step",
        "resolved_step",
        "agent_world_position",
        "belief_origin_world",
        "belief_shape",
        "belief_matrix_sha256",
        "crop_bounds",
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


def test_dynamic_map_has_no_robot_or_trajectory_layer(tmp_path: Path) -> None:
    result = _export(tmp_path / "assets")
    manifest = result["manifest"]
    record = manifest["assets"]["dynamic_cumulative_belief_map"]
    assert record["render_layers"] == ["cumulative_belief_occupancy"]
    svg_text = Path(record["svg_path"]).read_text(encoding="utf-8")
    assert "robot_marker" not in svg_text
    assert "trajectory_segment" not in svg_text


def test_position_and_history_share_crop_and_history_is_real(tmp_path: Path) -> None:
    result = _export(tmp_path / "assets")
    manifest = result["manifest"]
    crop = manifest["crop_bounds"]
    assert crop["shared_by"] == ["robot_position", "interaction_history"]
    assert crop["shape"] == [11, 11]
    assert manifest["recent_trajectory_length"] >= 2
    points = manifest["recent_trajectory_positions_world"]
    assert len(points) == manifest["recent_trajectory_length"]
    assert points[-1] == manifest["agent_world_position"]
    assert len({tuple(point) for point in points}) >= 2


def test_local_state_is_actual_four_channel_schema_without_frontier() -> None:
    scene = build_fig3_overview_scene(seed=1, step=8)
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


def test_global_parent_child_counts_match_semantic_snapshot(tmp_path: Path) -> None:
    result = _export(tmp_path / "assets")
    scene = result["scene"]
    manifest = result["manifest"]
    blocks = tuple(scene.semantic_snapshot.accessible_blocks)
    entries = sum(int(block.frontier_cluster_count) for block in blocks)
    assert manifest["total_unknown_block_count"] == len(blocks)
    assert manifest["total_frontier_entrance_count"] == entries
    assert manifest["displayed_unknown_block_count"] <= len(blocks)
    assert manifest["displayed_frontier_entrance_count"] <= entries
    assert int(scene.value_meta["value_total_block_count_before_cap"]) == len(blocks)
    assert int(scene.value_meta["value_total_entry_count_before_cap"]) == entries


def test_all_png_and_svg_files_open_and_truth_is_not_rendered(
    tmp_path: Path,
) -> None:
    result = _export(tmp_path / "assets")
    manifest = json.loads(
        (tmp_path / "assets" / MANIFEST_FILENAME).read_text(encoding="utf-8")
    )
    assert manifest["truth_map_rendered"] is False
    assert manifest["same_snapshot_as_source_belief_asset"] is True
    assert (
        manifest["figure2_or_belief_map_source"]["manifest_verified"] is True
    )
    for record in manifest["assets"].values():
        png_path = Path(record["png_path"])
        svg_path = Path(record["svg_path"])
        assert png_path.stat().st_size > 0
        assert svg_path.stat().st_size > 0
        with Image.open(png_path) as image:
            image.verify()
        ET.parse(svg_path)
