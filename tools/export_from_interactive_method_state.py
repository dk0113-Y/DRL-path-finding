from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("DRL_PAPER_FIGURE_INTERACTIVE", "0")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.export_architecture_pictures import ExportConfig, _format_output_path
from tools.interactive_method_figure_export import (
    DEFAULT_STATE_DIR,
    InteractiveMethodFigureExporter,
)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export paper method assets directly from a saved interactive method .npz state."
    )
    parser.add_argument("--load-state", type=Path, required=True, help="Saved .npz state from interactive_method_figure_export.py.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for exported PNG assets.")
    parser.add_argument("--dpi", type=int, default=240, help="Export DPI. Default: 240.")
    parser.add_argument(
        "--recent-trajectory-length",
        type=int,
        default=None,
        help="Deprecated alias for --override-recent-trajectory-length.",
    )
    parser.add_argument(
        "--override-recent-trajectory-length",
        type=int,
        default=None,
        help="Override the recent trajectory length stored in older .npz states.",
    )
    parser.add_argument(
        "--trajectory-decay-length",
        type=int,
        default=10,
        help="Recent trajectory window length for trajectory_decay_10step_local.png. Default: 10.",
    )
    parser.add_argument(
        "--local-semantic-crop-radius",
        type=int,
        default=None,
        help="Agent-centered cumulative-belief crop radius for local_semantic_crop.png. Defaults to the loaded state's scan radius.",
    )
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    recent_override = (
        args.override_recent_trajectory_length
        if args.override_recent_trajectory_length is not None
        else args.recent_trajectory_length
    )
    config = ExportConfig(
        dpi=int(args.dpi),
        output_dir=Path(args.output_dir),
    )
    exporter = InteractiveMethodFigureExporter(
        output_dir=Path(args.output_dir),
        recent_trajectory_length=10,
        state_dir=DEFAULT_STATE_DIR,
        load_state=Path(args.load_state),
        config=config,
        trajectory_decay_length=int(args.trajectory_decay_length),
        local_semantic_crop_radius=args.local_semantic_crop_radius,
        override_recent_trajectory_length=recent_override,
    )
    if exporter.last_transition is None:
        raise RuntimeError(
            "current state has no exportable transition; execute at least one action before saving the state"
        )

    outputs = exporter.export_current()
    transition = exporter.last_transition
    print("mode=export-from-interactive-method-state")
    print(f"state={_format_output_path(Path(args.load_state))}")
    print(f"step={transition.step if transition else 0}")
    print(f"last_action={transition.action_key if transition else ''}")
    print(f"recent_trajectory_length={exporter.recent_trajectory_length}")
    print(f"trajectory_decay_length={exporter.trajectory_decay_length}")
    for name, path in outputs.items():
        print(f"{name}={_format_output_path(path)}")


if __name__ == "__main__":
    main()
