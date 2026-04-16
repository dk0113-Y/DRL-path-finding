from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np

if "--export-and-exit" not in sys.argv:
    os.environ.setdefault("DRL_PAPER_FIGURE_INTERACTIVE", "1")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from env.agent_version import LocalObservationModel
from env.block_random_g import RandomMapGenerator
from env.core_cummap import CumulativeBeliefMap
from env.core_radar import RadarSensor
from env.grid_topology import ACTIONS_8, GridTopology
from env.shared_semantic_layer import SharedSemanticLayer
from tools.export_architecture_pictures import (
    ACTION_TO_KEY,
    KEY_TO_ACTION,
    ExportConfig,
    MethodFigureStyle,
    Snapshot,
    _build_method_world_canvas,
    _capture_snapshot,
    _export_executed_action_arrow,
    _export_method_belief_map,
    _export_method_local_observation,
    _export_method_overlay,
    _format_output_path,
    _render_method_belief_axis,
    _render_method_local_axis,
    _set_global_seed,
)
from tools.export_shared_semantic_layer_assets import (
    RAW_FRONTIER_COLOR,
    SharedSemanticAssetStyle,
    _crop_belief,
    _crop_from_box,
    _export_frontier_parsing_overlay,
    _export_semantic_input_belief_map,
    _frontier_crop,
    _overlay_mask,
    _render_base_map,
)

import matplotlib.pyplot as plt
from matplotlib.widgets import Button


DEFAULT_OUTPUT_DIR = REPO_ROOT / "run_picture" / "interactive_method_assets"
ACTION_KEYS = tuple(KEY_TO_ACTION.keys())


@dataclass(frozen=True)
class CachedTransition:
    step: int
    action_key: str
    before_snapshot: Snapshot
    after_snapshot: Snapshot


def get_recent_trajectory_window(history: np.ndarray | list[tuple[int, int]], length: int) -> np.ndarray:
    trajectory = np.asarray(history, dtype=np.int32)
    if trajectory.ndim != 2 or trajectory.shape[1] != 2:
        trajectory = trajectory.reshape((-1, 2))
    recent_steps = max(0, int(length))
    max_points = int(recent_steps) + 1
    if trajectory.shape[0] <= max_points:
        return trajectory.copy()
    return trajectory[-max_points:].copy()


def _parse_action_sequence(value: str | None) -> list[str]:
    if value is None or str(value).strip() == "":
        return []
    normalized = str(value).replace(",", " ").strip().lower()
    tokens = normalized.split()
    if len(tokens) == 1 and len(tokens[0]) > 1:
        tokens = list(tokens[0])
    invalid = [token for token in tokens if token not in KEY_TO_ACTION]
    if invalid:
        allowed = " ".join(ACTION_KEYS)
        raise argparse.ArgumentTypeError(f"invalid action key(s): {invalid}; allowed: {allowed}")
    return tokens


def _format_clean_axis(ax, shape: tuple[int, int]) -> None:
    ax.set_aspect("equal")
    ax.set_xlim(-0.5, float(shape[1]) - 0.5)
    ax.set_ylim(float(shape[0]) - 0.5, -0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


class InteractiveMethodFigureExporter:
    def __init__(
        self,
        *,
        output_dir: Path,
        recent_trajectory_length: int,
        config: ExportConfig,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.recent_trajectory_length = max(0, int(recent_trajectory_length))
        self.config = config
        self.method_style = MethodFigureStyle()
        self.semantic_style = SharedSemanticAssetStyle(dpi=int(config.dpi))

        _set_global_seed(int(config.seed))
        generator = RandomMapGenerator(
            rows=int(config.rows),
            cols=int(config.cols),
            obs_size=int(config.obs_size),
            obstacle_ratio=float(config.obstacle_ratio),
        )
        self.true_grid, self.start_state = generator.generate_map()
        self.free_mask = GridTopology.free_mask(self.true_grid)
        self.sensor = RadarSensor(scan_radius=int(config.scan_radius))
        self.obs_model = LocalObservationModel(self.true_grid, self.start_state, sensor=self.sensor)
        self.agent_state = (int(self.start_state[0]), int(self.start_state[1]))
        self.local_snap = np.asarray(self.obs_model.local_snap, dtype=np.int8).copy()
        self.cum_map = CumulativeBeliefMap(self.true_grid, self.agent_state, self.local_snap)
        self.semantic_layer = SharedSemanticLayer()
        self.trajectory_world: list[tuple[int, int]] = [self.agent_state]
        self.step = 0
        self.last_transition: CachedTransition | None = None
        self.status_message = "Press an action key, then press p or Export."

        self.fig = None
        self.axes = None
        self.status_text = None
        self.export_button = None

    def current_snapshot(self) -> Snapshot:
        return _capture_snapshot(
            step=int(self.step),
            agent_state=self.agent_state,
            trajectory_world=self.trajectory_world,
            local_snap=self.local_snap,
            cum_map=self.cum_map,
        )

    def current_recent_trajectory(self) -> np.ndarray:
        return get_recent_trajectory_window(self.trajectory_world, self.recent_trajectory_length)

    def _semantic_scene(self, snapshot: Snapshot):
        semantic_snapshot = self.semantic_layer.analyze(self.cum_map, self.agent_state)
        frontier_mask = np.asarray(self.cum_map.get_frontier_u8(refresh=False), dtype=np.uint8) > 0
        return SimpleNamespace(
            snapshot=snapshot,
            frontier_mask=frontier_mask,
            semantic_snapshot=semantic_snapshot,
        )

    def execute_action(self, action_key: str, *, strict: bool = False) -> bool:
        key = str(action_key).strip().lower()
        if key not in KEY_TO_ACTION:
            self.status_message = f"Unsupported action key: {action_key}"
            if strict:
                raise ValueError(self.status_message)
            return False

        valid_actions = GridTopology.valid_action_indices_fast(self.free_mask, self.agent_state)
        chosen_action = int(KEY_TO_ACTION[key])
        if chosen_action not in valid_actions:
            valid_keys = " ".join(ACTION_TO_KEY[int(action_idx)] for action_idx in valid_actions)
            self.status_message = f"Illegal action '{key}' at step {self.step + 1}; valid: {valid_keys}"
            if strict:
                raise RuntimeError(self.status_message)
            return False

        next_step = int(self.step) + 1
        before_snapshot = _capture_snapshot(
            step=next_step,
            agent_state=self.agent_state,
            trajectory_world=self.trajectory_world,
            local_snap=self.local_snap,
            cum_map=self.cum_map,
        )

        dr, dc = ACTIONS_8[chosen_action]
        self.agent_state = (int(self.agent_state[0] + dr), int(self.agent_state[1] + dc))
        self.trajectory_world.append(self.agent_state)
        self.local_snap = np.asarray(self.obs_model.observe_fast(self.agent_state), dtype=np.int8).copy()
        self.cum_map.update(self.agent_state, self.local_snap)
        self.step = next_step

        after_snapshot = _capture_snapshot(
            step=next_step,
            agent_state=self.agent_state,
            trajectory_world=self.trajectory_world,
            local_snap=self.local_snap,
            cum_map=self.cum_map,
        )
        self.last_transition = CachedTransition(
            step=next_step,
            action_key=key,
            before_snapshot=before_snapshot,
            after_snapshot=after_snapshot,
        )
        before_steps = max(0, before_snapshot.trajectory_world.shape[0] - 1)
        after_steps = max(0, after_snapshot.trajectory_world.shape[0] - 1)
        self.status_message = (
            f"step={self.step}, action={key}, before_steps={before_steps}, "
            f"after_steps={after_steps}, recent={self.recent_trajectory_length}"
        )
        return True

    def _draw_shared_entry_axis(self, ax, snapshot: Snapshot) -> None:
        scene = self._semantic_scene(snapshot)
        crop = _crop_from_box(scene.semantic_snapshot.analysis_box)
        belief_crop = _crop_belief(snapshot, crop)
        ax.clear()
        _render_base_map(ax, belief_crop)
        _overlay_mask(ax, _frontier_crop(scene, crop), color=RAW_FRONTIER_COLOR, alpha=float(self.semantic_style.frontier_alpha))
        _format_clean_axis(ax, crop.shape)
        ax.set_title("Shared Semantic Input: analysis domain + raw frontier", fontsize=9)

    def refresh(self) -> None:
        if self.axes is None:
            return
        snapshot = self.current_snapshot()
        recent_trajectory = self.current_recent_trajectory()

        self.axes[0].clear()
        canvas = _build_method_world_canvas(snapshot, snapshot, self.sensor)
        _render_method_belief_axis(
            self.axes[0],
            snapshot=snapshot,
            canvas=canvas,
            sensor=self.sensor,
            style=self.method_style,
            show_analysis_box=True,
            trajectory_world=recent_trajectory,
        )
        self.axes[0].set_title("Dynamic Cumulative Belief Map", fontsize=9)

        self.axes[1].clear()
        _render_method_local_axis(
            self.axes[1],
            snapshot=snapshot,
            sensor=self.sensor,
            style=self.method_style,
            trajectory_world=recent_trajectory,
        )
        self.axes[1].set_title("Local LiDAR Observation", fontsize=9)

        self._draw_shared_entry_axis(self.axes[2], snapshot)
        if self.status_text is not None:
            self.status_text.set_text(self.status_message)
        if self.fig is not None:
            self.fig.canvas.draw_idle()

    def export_current(self) -> dict[str, Path]:
        if self.last_transition is None:
            raise RuntimeError("execute at least one action before exporting transition assets")

        transition = self.last_transition
        self.output_dir.mkdir(parents=True, exist_ok=True)
        before_recent = get_recent_trajectory_window(
            transition.before_snapshot.trajectory_world,
            self.recent_trajectory_length,
        )
        after_recent = get_recent_trajectory_window(
            transition.after_snapshot.trajectory_world,
            self.recent_trajectory_length,
        )
        canvas = _build_method_world_canvas(
            transition.before_snapshot,
            transition.after_snapshot,
            self.sensor,
        )

        outputs = {
            "local_lidar_observation": self.output_dir / "local_lidar_observation.png",
            "belief_before_update": self.output_dir / "belief_before_update.png",
            "belief_after_update": self.output_dir / "belief_after_update.png",
            "observation_overlay": self.output_dir / "observation_overlay.png",
            "executed_action_arrow": self.output_dir / "executed_action_arrow.png",
            "semantic_input_belief_map": self.output_dir / "semantic_input_belief_map.png",
            "frontier_parsing_overlay": self.output_dir / "frontier_parsing_overlay.png",
        }

        _export_method_local_observation(
            outputs["local_lidar_observation"],
            snapshot=transition.after_snapshot,
            sensor=self.sensor,
            style=self.method_style,
            dpi=int(self.config.dpi),
            trajectory_world=after_recent,
        )
        _export_method_belief_map(
            outputs["belief_before_update"],
            snapshot=transition.before_snapshot,
            canvas=canvas,
            sensor=self.sensor,
            style=self.method_style,
            dpi=int(self.config.dpi),
            trajectory_world=before_recent,
        )
        _export_method_belief_map(
            outputs["belief_after_update"],
            snapshot=transition.after_snapshot,
            canvas=canvas,
            sensor=self.sensor,
            style=self.method_style,
            dpi=int(self.config.dpi),
            show_analysis_box=True,
            trajectory_world=after_recent,
        )
        _export_method_overlay(
            outputs["observation_overlay"],
            before_snapshot=transition.before_snapshot,
            after_snapshot=transition.after_snapshot,
            canvas=canvas,
            sensor=self.sensor,
            style=self.method_style,
            dpi=int(self.config.dpi),
            trajectory_world=after_recent,
        )
        _export_executed_action_arrow(
            outputs["executed_action_arrow"],
            before_snapshot=transition.before_snapshot,
            after_snapshot=transition.after_snapshot,
            style=self.method_style,
            dpi=int(self.config.dpi),
        )

        scene = self._semantic_scene(transition.after_snapshot)
        _export_semantic_input_belief_map(outputs["semantic_input_belief_map"], scene, style=self.semantic_style)
        _export_frontier_parsing_overlay(outputs["frontier_parsing_overlay"], scene, style=self.semantic_style)

        manifest = {
            "step": int(transition.step),
            "action_key": str(transition.action_key),
            "recent_trajectory_length": int(self.recent_trajectory_length),
            "before_display_steps": int(before_recent.shape[0] - 1),
            "after_display_steps": int(after_recent.shape[0] - 1),
            "before_agent_world": [int(v) for v in transition.before_snapshot.agent_world],
            "after_agent_world": [int(v) for v in transition.after_snapshot.agent_world],
            "files": {name: _format_output_path(path) for name, path in outputs.items()},
        }
        manifest_path = self.output_dir / "interactive_method_assets_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        outputs["manifest"] = manifest_path

        self.status_message = f"Exported step {transition.step} action {transition.action_key} to {_format_output_path(self.output_dir)}"
        return outputs

    def _on_key(self, event) -> None:
        key = str(event.key or "").lower()
        if key in KEY_TO_ACTION:
            self.execute_action(key)
            self.refresh()
            return
        if key == "p":
            try:
                self.export_current()
            except Exception as exc:
                self.status_message = f"Export failed: {exc}"
            self.refresh()
            return
        if key in {"escape", "ctrl+q"}:
            plt.close(self.fig)

    def _on_export_clicked(self, _event) -> None:
        try:
            self.export_current()
        except Exception as exc:
            self.status_message = f"Export failed: {exc}"
        self.refresh()

    def run(self) -> None:
        self.fig, self.axes = plt.subplots(1, 3, figsize=(13.8, 4.8), num="Interactive Method Figure Export")
        self.fig.subplots_adjust(left=0.02, right=0.98, top=0.86, bottom=0.18, wspace=0.12)
        button_ax = self.fig.add_axes([0.86, 0.05, 0.10, 0.06])
        self.export_button = Button(button_ax, "Export")
        self.export_button.on_clicked(self._on_export_clicked)
        self.status_text = self.fig.text(
            0.02,
            0.06,
            "",
            ha="left",
            va="center",
            fontsize=9,
        )
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        self.refresh()
        plt.show()


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Interactively control agent actions and export paper method assets.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--recent-trajectory-length", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rows", type=int, default=40)
    parser.add_argument("--cols", type=int, default=60)
    parser.add_argument("--obstacle-ratio", type=float, default=0.20)
    parser.add_argument("--obs-size", type=int, default=6)
    parser.add_argument("--scan-radius", type=int, default=10)
    parser.add_argument("--dpi", type=int, default=240)
    parser.add_argument(
        "--scripted-actions",
        type=_parse_action_sequence,
        default=None,
        help="Optional action sequence for validation, e.g. qwwewdded or q,w,w,e,w,d,d,e,d.",
    )
    parser.add_argument(
        "--export-and-exit",
        action="store_true",
        help="Run scripted actions, export current transition assets, and exit without opening the GUI.",
    )
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    config = ExportConfig(
        rows=int(args.rows),
        cols=int(args.cols),
        obstacle_ratio=float(args.obstacle_ratio),
        obs_size=int(args.obs_size),
        scan_radius=int(args.scan_radius),
        seed=int(args.seed),
        dpi=int(args.dpi),
        output_dir=Path(args.output_dir),
    )
    exporter = InteractiveMethodFigureExporter(
        output_dir=Path(args.output_dir),
        recent_trajectory_length=int(args.recent_trajectory_length),
        config=config,
    )

    for action_key in args.scripted_actions or []:
        exporter.execute_action(action_key, strict=True)

    if args.export_and_exit:
        outputs = exporter.export_current()
        transition = exporter.last_transition
        print("mode=interactive-method-assets")
        print(f"step={transition.step if transition else 0}")
        print(f"last_action={transition.action_key if transition else ''}")
        print(f"recent_trajectory_length={exporter.recent_trajectory_length}")
        for name, path in outputs.items():
            print(f"{name}={_format_output_path(path)}")
        return

    exporter.run()


if __name__ == "__main__":
    main()
