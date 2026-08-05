from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from tools.export_all_visio_ready_png_assets import (
    BATCH_MANIFEST_FILENAME,
    export_all_visio_ready_png_assets,
)
from tools.export_fig3_overview_assets import _git_head as fig3_git_head
from tools.export_fig3_overview_assets import export_fig3_overview_assets
from tools.export_fig4_state_construction_assets import _git_head as fig4_git_head


def test_missing_optional_provenance_inputs_are_skipped(tmp_path: Path) -> None:
    missing_repo = tmp_path / "missing-paper-repo"
    missing_manifest = tmp_path / "missing-source-manifest.json"

    assert fig3_git_head(missing_repo) is None
    assert fig4_git_head(missing_repo) is None
    result = export_fig3_overview_assets(
        tmp_path / "fig3-without-external-inputs",
        dpi=40,
        include_svg=False,
        paper_repo=missing_repo,
        source_manifest=missing_manifest,
    )
    validation = result["manifest"]["figure2_or_belief_map_source"]
    assert len(result["files"]) == 5
    assert all(Path(path).is_file() for path in result["files"].values())
    assert result["manifest"]["paper_repo_commit_before_change"] is None
    assert validation["manifest_verified"] is False
    assert validation["validation_status"] == "skipped"
    assert validation["validation_skipped"] is True
    assert validation["skip_reason"] == "source_manifest_not_found"


def test_batch_export_reuses_exporters_and_writes_openable_pngs(tmp_path: Path) -> None:
    output_root = tmp_path / "visio-assets"
    missing_repo = tmp_path / "missing-paper-repo"
    result = export_all_visio_ready_png_assets(
        output_root,
        dpi=40,
        include_svg=False,
        paper_repo=missing_repo,
    )

    manifest = json.loads(
        (output_root / BATCH_MANIFEST_FILENAME).read_text(encoding="utf-8")
    )
    assert manifest["dpi"] == 40
    assert manifest["include_svg"] is False
    assert manifest["paper_repo"]["available"] is False
    assert manifest["paper_repo"]["fig3_commit"] is None
    assert manifest["paper_repo"]["fig4_commit"] is None
    assert manifest["fig3_source_manifest_validation"]["validation_status"] == "verified"
    assert set(manifest["figures"]) == {"fig1", "fig3", "fig4"}

    for figure_name, expected_count in (("fig1", 4), ("fig3", 5), ("fig4", 10)):
        figure_dir = output_root / figure_name
        assert figure_dir.is_dir()
        png_paths = sorted(figure_dir.glob("*.png"))
        assert len(png_paths) == expected_count
        assert manifest["figures"][figure_name]["png_count"] == expected_count
        for png_path in png_paths:
            with Image.open(png_path) as image:
                image.verify()

    assert Path(result["manifest_path"]) == output_root / BATCH_MANIFEST_FILENAME
