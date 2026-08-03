from __future__ import annotations

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
    weights = manifest["recent_trajectory_semantics"]["weights_old_to_new"]
    assert weights == sorted(weights)
    assert np.allclose(
        weights,
        [(index + 1) / len(weights) for index in range(len(weights))],
    )
    assert manifest["recent_trajectory_semantics"]["current_robot_cell_repainted"] is False


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
