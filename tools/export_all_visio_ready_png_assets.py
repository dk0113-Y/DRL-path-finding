from __future__ import annotations

"""Batch-export the Figure 1, 3, and 4 PNG assets used for Visio assembly."""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.export_architecture_pictures import ExportConfig  # noqa: E402
from tools.export_fig3_overview_assets import (  # noqa: E402
    DEFAULT_PAPER_REPO,
    export_fig3_overview_assets,
)
from tools.export_fig4_state_construction_assets import (  # noqa: E402
    export_fig4_state_construction_assets,
)
from tools.export_online_workflow_assets import export_online_workflow_assets  # noqa: E402


DEFAULT_OUTPUT_ROOT = Path(r"C:\Users\Dk\Desktop\SCI\论文0\New\绘图\fig_fix_1")
BATCH_MANIFEST_FILENAME = "visio_ready_export_manifest.json"
FIGURE_NAMES = ("fig1", "fig3", "fig4")


def _verify_png(path: Path, *, expected_dpi: int) -> dict[str, object]:
    if not path.is_file() or path.stat().st_size <= 0:
        raise RuntimeError(f"PNG was not created or is empty: {path}")
    with Image.open(path) as image:
        if image.format != "PNG":
            raise RuntimeError(f"Expected PNG data in {path}, got {image.format}")
        width_px, height_px = (int(value) for value in image.size)
        mode = str(image.mode)
        reported_dpi_raw = image.info.get("dpi")
        reported_dpi = (
            [float(value) for value in reported_dpi_raw]
            if isinstance(reported_dpi_raw, tuple)
            else None
        )
        image.verify()
    if width_px <= 0 or height_px <= 0:
        raise RuntimeError(f"PNG has invalid dimensions: {path}")
    if reported_dpi is not None and any(
        abs(value - float(expected_dpi)) > 1.0 for value in reported_dpi
    ):
        raise RuntimeError(
            f"PNG DPI metadata mismatch for {path}: expected {expected_dpi}, "
            f"got {reported_dpi}"
        )
    return {
        "path": str(path.resolve()),
        "width_px": width_px,
        "height_px": height_px,
        "mode": mode,
        "reported_dpi": reported_dpi,
        "size_bytes": int(path.stat().st_size),
    }


def export_all_visio_ready_png_assets(
    output_root: Path | str = DEFAULT_OUTPUT_ROOT,
    *,
    dpi: int = 600,
    include_svg: bool = False,
    paper_repo: Path | str = DEFAULT_PAPER_REPO,
) -> dict[str, object]:
    """Run the existing Figure 1/3/4 exporters and verify every PNG output."""

    if int(dpi) <= 0:
        raise ValueError("dpi must be a positive integer")

    root = Path(output_root)
    figure_dirs = {name: root / name for name in FIGURE_NAMES}
    for directory in figure_dirs.values():
        directory.mkdir(parents=True, exist_ok=True)

    fig1_config = ExportConfig(dpi=int(dpi), output_dir=figure_dirs["fig1"])
    fig1 = export_online_workflow_assets(
        figure_dirs["fig1"],
        config=fig1_config,
        step=int(fig1_config.step_late),
        include_svg=bool(include_svg),
    )
    fig3 = export_fig3_overview_assets(
        figure_dirs["fig3"],
        dpi=int(dpi),
        include_svg=bool(include_svg),
        paper_repo=paper_repo,
        source_manifest=fig1["manifest_path"],
    )
    fig4 = export_fig4_state_construction_assets(
        figure_dirs["fig4"],
        dpi=int(dpi),
        include_svg=bool(include_svg),
        paper_repo=paper_repo,
    )

    exporter_results = {"fig1": fig1, "fig3": fig3, "fig4": fig4}
    verified_pngs: dict[str, dict[str, dict[str, object]]] = {}
    for figure_name, result in exporter_results.items():
        verified_pngs[figure_name] = {
            asset_name: _verify_png(Path(path), expected_dpi=int(dpi))
            for asset_name, path in result["files"].items()
        }

    batch_manifest = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "output_root": str(root.resolve()),
        "dpi": int(dpi),
        "include_svg": bool(include_svg),
        "paper_repo": {
            "path": str(Path(paper_repo).resolve()),
            "available": Path(paper_repo).is_dir(),
            "fig3_commit": fig3["manifest"]["paper_repo_commit_before_change"],
            "fig4_commit": fig4["manifest"]["paper_repo_commit_before_change"],
        },
        "fig3_source_manifest_validation": fig3["manifest"][
            "figure2_or_belief_map_source"
        ],
        "figures": {
            figure_name: {
                "output_dir": str(figure_dirs[figure_name].resolve()),
                "manifest_path": str(result["manifest_path"].resolve()),
                "png_count": len(verified_pngs[figure_name]),
                "pngs": verified_pngs[figure_name],
            }
            for figure_name, result in exporter_results.items()
        },
    }
    manifest_path = root / BATCH_MANIFEST_FILENAME
    manifest_path.write_text(
        json.dumps(batch_manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return {
        "output_root": root,
        "figures": exporter_results,
        "verified_pngs": verified_pngs,
        "manifest": batch_manifest,
        "manifest_path": manifest_path,
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export Visio-ready high-resolution PNG assets for Figures 1, 3, and 4."
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument(
        "--include-svg",
        action="store_true",
        help="Also retain SVG copies. Disabled by default because PNG is the primary output.",
    )
    parser.add_argument(
        "--paper-repo",
        type=Path,
        default=DEFAULT_PAPER_REPO,
        help="Optional paper repository used only for provenance metadata.",
    )
    return parser


def cli_main() -> None:
    args = _build_arg_parser().parse_args()
    result = export_all_visio_ready_png_assets(
        args.output_root,
        dpi=int(args.dpi),
        include_svg=bool(args.include_svg),
        paper_repo=args.paper_repo,
    )
    print("mode=all-visio-ready-png-assets")
    print(f"output_root={Path(result['output_root']).resolve()}")
    print(f"dpi={result['manifest']['dpi']}")
    print(f"include_svg={result['manifest']['include_svg']}")
    for figure_name in FIGURE_NAMES:
        figure = result["manifest"]["figures"][figure_name]
        print(f"{figure_name}_png_count={figure['png_count']}")
        print(f"{figure_name}_manifest={figure['manifest_path']}")
    validation = result["manifest"]["fig3_source_manifest_validation"]
    print(f"fig3_source_manifest_status={validation['validation_status']}")
    print(f"manifest={Path(result['manifest_path']).resolve()}")


if __name__ == "__main__":
    cli_main()
