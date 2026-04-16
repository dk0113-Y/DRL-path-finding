from __future__ import annotations

import argparse
import json
import os
import sys
from collections import deque
from dataclasses import dataclass
from datetime import datetime
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
    _draw_cropped_trajectory_and_agent,
    _export_cluster_analysis_boxes,
    _export_frontier_parsing_overlay,
    _export_frontier_cluster_overlay,
    _export_semantic_input_belief_map,
    _frontier_crop,
    _overlay_mask,
    _render_base_map,
)

import matplotlib.pyplot as plt
from matplotlib.widgets import Button


DEFAULT_OUTPUT_DIR = REPO_ROOT / "run_picture" / "interactive_method_assets"
DEFAULT_STATE_DIR = REPO_ROOT / "outputs" / "interactive_method_states"
ACTION_KEYS = tuple(KEY_TO_ACTION.keys())


@dataclass(frozen=True)
class CachedTransition:
    step: int
    action_key: str
    before_snapshot: Snapshot
    after_snapshot: Snapshot


@dataclass(frozen=True)
class MethodFigureRuntimeState:
    true_grid: np.ndarray
    free_mask: np.ndarray
    local_snap: np.ndarray
    trajectory_world: np.ndarray
    cum_map_map: np.ndarray
    cum_map_visit_count: np.ndarray
    cum_map_frontier_bool: np.ndarray
    cum_map_frontier_u8: np.ndarray
    start_state: tuple[int, int]
    agent_state: tuple[int, int]
    cum_map_origin_world_rc: tuple[int, int]
    cum_map_step_count: int
    cum_map_coverage_rate: float
    cum_map_kpm_count: int
    cum_map_tpm_count: int
    cum_map_frontier_revision: int
    step: int
    recent_trajectory_length: int
    scan_radius: int
    status_message: str
    last_transition: CachedTransition | None


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
        state_dir: Path,
        load_state: Path | None,
        config: ExportConfig,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.state_dir = Path(state_dir)
        self.recent_trajectory_length = max(0, int(recent_trajectory_length))
        self.config = config
        self.method_style = MethodFigureStyle()
        self.semantic_style = SharedSemanticAssetStyle(dpi=int(config.dpi))
        self.semantic_layer = SharedSemanticLayer()
        self.status_message = "Action keys move; p/Export exports images; k saves state; Ctrl+Z undoes."
        self.undo_history: deque[MethodFigureRuntimeState] = deque(maxlen=10)

        self.fig = None
        self.axes = None
        self.status_text = None
        self.export_button = None

        if load_state is not None:
            self.load_runtime_state(load_state)
            return

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
        self.trajectory_world: list[tuple[int, int]] = [self.agent_state]
        self.step = 0
        self.last_transition: CachedTransition | None = None

    @staticmethod
    def _copy_snapshot(snapshot: Snapshot) -> Snapshot:
        return Snapshot(
            step=int(snapshot.step),
            agent_world=(int(snapshot.agent_world[0]), int(snapshot.agent_world[1])),
            agent_array=(int(snapshot.agent_array[0]), int(snapshot.agent_array[1])),
            belief_origin_world=(
                int(snapshot.belief_origin_world[0]),
                int(snapshot.belief_origin_world[1]),
            ),
            analysis_box=tuple(int(v) for v in snapshot.analysis_box),
            trajectory_world=np.asarray(snapshot.trajectory_world, dtype=np.int32).copy(),
            trajectory_array=np.asarray(snapshot.trajectory_array, dtype=np.int32).copy(),
            local_snap=np.asarray(snapshot.local_snap, dtype=np.int8).copy(),
            belief_map=np.asarray(snapshot.belief_map, dtype=np.int8).copy(),
        )

    @staticmethod
    def _copy_transition(transition: CachedTransition | None) -> CachedTransition | None:
        if transition is None:
            return None
        return CachedTransition(
            step=int(transition.step),
            action_key=str(transition.action_key),
            before_snapshot=InteractiveMethodFigureExporter._copy_snapshot(transition.before_snapshot),
            after_snapshot=InteractiveMethodFigureExporter._copy_snapshot(transition.after_snapshot),
        )

    @staticmethod
    def _snapshot_metadata(snapshot: Snapshot) -> dict[str, object]:
        return {
            "step": int(snapshot.step),
            "agent_world": [int(v) for v in snapshot.agent_world],
            "agent_array": [int(v) for v in snapshot.agent_array],
            "belief_origin_world": [int(v) for v in snapshot.belief_origin_world],
            "analysis_box": [int(v) for v in snapshot.analysis_box],
        }

    @staticmethod
    def _snapshot_arrays(prefix: str, snapshot: Snapshot) -> dict[str, np.ndarray]:
        return {
            f"{prefix}_trajectory_world": np.asarray(snapshot.trajectory_world, dtype=np.int32),
            f"{prefix}_trajectory_array": np.asarray(snapshot.trajectory_array, dtype=np.int32),
            f"{prefix}_local_snap": np.asarray(snapshot.local_snap, dtype=np.int8),
            f"{prefix}_belief_map": np.asarray(snapshot.belief_map, dtype=np.int8),
        }

    @staticmethod
    def _tuple2(metadata: dict[str, object], key: str) -> tuple[int, int]:
        values = metadata[key]
        if not isinstance(values, (list, tuple)) or len(values) != 2:
            raise ValueError(f"state metadata field {key!r} must contain two integers")
        return (int(values[0]), int(values[1]))

    @staticmethod
    def _tuple4(metadata: dict[str, object], key: str) -> tuple[int, int, int, int]:
        values = metadata[key]
        if not isinstance(values, (list, tuple)) or len(values) != 4:
            raise ValueError(f"state metadata field {key!r} must contain four integers")
        return tuple(int(v) for v in values)

    @classmethod
    def _snapshot_from_npz(cls, metadata: dict[str, object], data, prefix: str) -> Snapshot:
        return Snapshot(
            step=int(metadata["step"]),
            agent_world=cls._tuple2(metadata, "agent_world"),
            agent_array=cls._tuple2(metadata, "agent_array"),
            belief_origin_world=cls._tuple2(metadata, "belief_origin_world"),
            analysis_box=cls._tuple4(metadata, "analysis_box"),
            trajectory_world=np.asarray(data[f"{prefix}_trajectory_world"], dtype=np.int32).copy(),
            trajectory_array=np.asarray(data[f"{prefix}_trajectory_array"], dtype=np.int32).copy(),
            local_snap=np.asarray(data[f"{prefix}_local_snap"], dtype=np.int8).copy(),
            belief_map=np.asarray(data[f"{prefix}_belief_map"], dtype=np.int8).copy(),
        )

    def capture_runtime_state(self) -> MethodFigureRuntimeState:
        return MethodFigureRuntimeState(
            true_grid=np.asarray(self.true_grid, dtype=np.int8).copy(),
            free_mask=np.asarray(self.free_mask, dtype=bool).copy(),
            local_snap=np.asarray(self.local_snap, dtype=np.int8).copy(),
            trajectory_world=np.asarray(self.trajectory_world, dtype=np.int32).reshape((-1, 2)).copy(),
            cum_map_map=np.asarray(self.cum_map.map, dtype=np.int8).copy(),
            cum_map_visit_count=np.asarray(self.cum_map.visit_count, dtype=np.int32).copy(),
            cum_map_frontier_bool=np.asarray(self.cum_map.frontier_bool, dtype=bool).copy(),
            cum_map_frontier_u8=np.asarray(self.cum_map.frontier_u8, dtype=np.uint8).copy(),
            start_state=(int(self.start_state[0]), int(self.start_state[1])),
            agent_state=(int(self.agent_state[0]), int(self.agent_state[1])),
            cum_map_origin_world_rc=(
                int(self.cum_map.origin_world_rc[0]),
                int(self.cum_map.origin_world_rc[1]),
            ),
            cum_map_step_count=int(self.cum_map.step_count),
            cum_map_coverage_rate=float(self.cum_map.coverage_rate),
            cum_map_kpm_count=int(self.cum_map.kpm_count),
            cum_map_tpm_count=int(self.cum_map.tpm_count),
            cum_map_frontier_revision=int(self.cum_map.frontier_revision),
            step=int(self.step),
            recent_trajectory_length=int(self.recent_trajectory_length),
            scan_radius=int(self.sensor.scan_r),
            status_message=str(self.status_message),
            last_transition=self._copy_transition(self.last_transition),
        )

    def _runtime_state_metadata(self, state: MethodFigureRuntimeState) -> dict[str, object]:
        transition = state.last_transition
        return {
            "format": "interactive_method_figure_export_state",
            "version": 1,
            "start_state": [int(v) for v in state.start_state],
            "agent_state": [int(v) for v in state.agent_state],
            "cum_map_origin_world_rc": [int(v) for v in state.cum_map_origin_world_rc],
            "cum_map_step_count": int(state.cum_map_step_count),
            "cum_map_coverage_rate": float(state.cum_map_coverage_rate),
            "cum_map_kpm_count": int(state.cum_map_kpm_count),
            "cum_map_tpm_count": int(state.cum_map_tpm_count),
            "cum_map_frontier_revision": int(state.cum_map_frontier_revision),
            "step": int(state.step),
            "recent_trajectory_length": int(state.recent_trajectory_length),
            "scan_radius": int(state.scan_radius),
            "status_message": str(state.status_message),
            "last_transition": None
            if transition is None
            else {
                "step": int(transition.step),
                "action_key": str(transition.action_key),
                "before_snapshot": self._snapshot_metadata(transition.before_snapshot),
                "after_snapshot": self._snapshot_metadata(transition.after_snapshot),
            },
        }

    @classmethod
    def _state_from_npz(cls, path: Path) -> MethodFigureRuntimeState:
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata"].item()))
            if metadata.get("format") != "interactive_method_figure_export_state":
                raise ValueError(f"unsupported interactive method state format in {path}")
            if int(metadata.get("version", 0)) != 1:
                raise ValueError(f"unsupported interactive method state version in {path}")
            transition_meta = metadata.get("last_transition")
            transition = None
            if transition_meta is not None:
                transition = CachedTransition(
                    step=int(transition_meta["step"]),
                    action_key=str(transition_meta["action_key"]),
                    before_snapshot=cls._snapshot_from_npz(
                        transition_meta["before_snapshot"],
                        data,
                        "last_before",
                    ),
                    after_snapshot=cls._snapshot_from_npz(
                        transition_meta["after_snapshot"],
                        data,
                        "last_after",
                    ),
                )
            return MethodFigureRuntimeState(
                true_grid=np.asarray(data["true_grid"], dtype=np.int8).copy(),
                free_mask=np.asarray(data["free_mask"], dtype=bool).copy(),
                local_snap=np.asarray(data["local_snap"], dtype=np.int8).copy(),
                trajectory_world=np.asarray(data["trajectory_world"], dtype=np.int32).reshape((-1, 2)).copy(),
                cum_map_map=np.asarray(data["cum_map_map"], dtype=np.int8).copy(),
                cum_map_visit_count=np.asarray(data["cum_map_visit_count"], dtype=np.int32).copy(),
                cum_map_frontier_bool=np.asarray(data["cum_map_frontier_bool"], dtype=bool).copy(),
                cum_map_frontier_u8=np.asarray(data["cum_map_frontier_u8"], dtype=np.uint8).copy(),
                start_state=cls._tuple2(metadata, "start_state"),
                agent_state=cls._tuple2(metadata, "agent_state"),
                cum_map_origin_world_rc=cls._tuple2(metadata, "cum_map_origin_world_rc"),
                cum_map_step_count=int(metadata["cum_map_step_count"]),
                cum_map_coverage_rate=float(metadata["cum_map_coverage_rate"]),
                cum_map_kpm_count=int(metadata["cum_map_kpm_count"]),
                cum_map_tpm_count=int(metadata["cum_map_tpm_count"]),
                cum_map_frontier_revision=int(metadata["cum_map_frontier_revision"]),
                step=int(metadata["step"]),
                recent_trajectory_length=int(metadata["recent_trajectory_length"]),
                scan_radius=int(metadata["scan_radius"]),
                status_message=str(metadata.get("status_message", "")),
                last_transition=transition,
            )

    def restore_runtime_state(self, state: MethodFigureRuntimeState, *, preserve_undo: bool = False) -> None:
        self.true_grid = np.asarray(state.true_grid, dtype=np.int8).copy()
        self.free_mask = np.asarray(state.free_mask, dtype=bool).copy()
        self.start_state = (int(state.start_state[0]), int(state.start_state[1]))
        self.agent_state = (int(state.agent_state[0]), int(state.agent_state[1]))
        self.sensor = RadarSensor(scan_radius=int(state.scan_radius))
        self.obs_model = LocalObservationModel(self.true_grid, self.agent_state, sensor=self.sensor)
        self.local_snap = np.asarray(state.local_snap, dtype=np.int8).copy()
        self.obs_model.local_snap[:, :] = self.local_snap
        self.cum_map = CumulativeBeliefMap(self.true_grid, self.start_state, self.local_snap)
        self.cum_map.map = np.asarray(state.cum_map_map, dtype=np.int8).copy()
        self.cum_map.visit_count = np.asarray(state.cum_map_visit_count, dtype=np.int32).copy()
        self.cum_map.frontier_bool = np.asarray(state.cum_map_frontier_bool, dtype=bool).copy()
        self.cum_map.frontier_u8 = np.asarray(state.cum_map_frontier_u8, dtype=np.uint8).copy()
        self.cum_map.origin_world_rc = (
            int(state.cum_map_origin_world_rc[0]),
            int(state.cum_map_origin_world_rc[1]),
        )
        self.cum_map.step_count = int(state.cum_map_step_count)
        self.cum_map.coverage_rate = float(state.cum_map_coverage_rate)
        self.cum_map.kpm_count = int(state.cum_map_kpm_count)
        self.cum_map.tpm_count = int(state.cum_map_tpm_count)
        self.cum_map.frontier_revision = int(state.cum_map_frontier_revision)
        self.cum_map._invalidate_visit_cache()
        self.cum_map._invalidate_map_state_caches()
        self.cum_map._update_analysis_box()

        trajectory = np.asarray(state.trajectory_world, dtype=np.int32).reshape((-1, 2))
        self.trajectory_world = [(int(row[0]), int(row[1])) for row in trajectory]
        if not self.trajectory_world:
            self.trajectory_world = [self.agent_state]
        self.step = int(state.step)
        self.recent_trajectory_length = max(0, int(state.recent_trajectory_length))
        self.last_transition = self._copy_transition(state.last_transition)
        self.status_message = str(state.status_message)
        if not preserve_undo:
            self.undo_history.clear()

    def _default_state_path(self, directory: Path) -> Path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return Path(directory) / f"interactive_method_state_step{int(self.step):04d}_{timestamp}.npz"

    def save_runtime_state(self, path: Path | None = None) -> Path:
        state = self.capture_runtime_state()
        save_path = Path(path) if path is not None else self._default_state_path(self.state_dir)
        if save_path.suffix == "":
            save_path = self._default_state_path(save_path)
        elif save_path.suffix.lower() != ".npz":
            save_path = save_path.with_suffix(".npz")
        save_path.parent.mkdir(parents=True, exist_ok=True)
        metadata = self._runtime_state_metadata(state)
        arrays = {
            "metadata": np.asarray(json.dumps(metadata, ensure_ascii=False)),
            "true_grid": state.true_grid,
            "free_mask": state.free_mask,
            "local_snap": state.local_snap,
            "trajectory_world": state.trajectory_world,
            "cum_map_map": state.cum_map_map,
            "cum_map_visit_count": state.cum_map_visit_count,
            "cum_map_frontier_bool": state.cum_map_frontier_bool,
            "cum_map_frontier_u8": state.cum_map_frontier_u8,
        }
        if state.last_transition is not None:
            arrays.update(self._snapshot_arrays("last_before", state.last_transition.before_snapshot))
            arrays.update(self._snapshot_arrays("last_after", state.last_transition.after_snapshot))
        np.savez_compressed(save_path, **arrays)
        self.status_message = f"Saved runtime state to {_format_output_path(save_path)}"
        return save_path

    def load_runtime_state(self, path: Path) -> Path:
        load_path = Path(path)
        if not load_path.exists() and load_path.suffix == "":
            candidate = load_path.with_suffix(".npz")
            if candidate.exists():
                load_path = candidate
        if not load_path.exists():
            raise FileNotFoundError(f"interactive method state not found: {load_path}")
        state = self._state_from_npz(load_path)
        self.restore_runtime_state(state)
        self.status_message = f"Loaded runtime state from {_format_output_path(load_path)}"
        return load_path

    def undo_last_state(self) -> bool:
        if not self.undo_history:
            self.status_message = "Undo history is empty."
            return False
        state = self.undo_history.pop()
        self.restore_runtime_state(state, preserve_undo=True)
        self.status_message = f"Undid one step; current step={self.step}."
        return True

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
        frontier_mask = self.cum_map.compute_analysis_box_frontier_bool()
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

        self.undo_history.append(self.capture_runtime_state())
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
        _draw_cropped_trajectory_and_agent(ax, snapshot, crop, trajectory_world=self.current_recent_trajectory())
        _format_clean_axis(ax, crop.shape)
        ax.set_title("Shared Semantic Input: analysis domain + analysis frontier", fontsize=9)

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
            "frontier_cluster_overlay": self.output_dir / "frontier_cluster_overlay.png",
            "cluster_analysis_boxes": self.output_dir / "cluster_analysis_boxes.png",
            "frontier_parsing_overlay": self.output_dir / "frontier_parsing_overlay.png",
        }

        _export_method_local_observation(
            outputs["local_lidar_observation"],
            snapshot=transition.after_snapshot,
            sensor=self.sensor,
            style=self.method_style,
            dpi=int(self.config.dpi),
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
        _export_semantic_input_belief_map(
            outputs["semantic_input_belief_map"],
            scene,
            style=self.semantic_style,
            trajectory_world=after_recent,
        )
        _export_frontier_cluster_overlay(outputs["frontier_cluster_overlay"], scene, style=self.semantic_style)
        _export_cluster_analysis_boxes(outputs["cluster_analysis_boxes"], scene, style=self.semantic_style)
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
        key = str(event.key or "").lower().replace("control+", "ctrl+")
        if key in {"k", "ctrl+shift+s", "shift+ctrl+s", "cmd+shift+s", "shift+cmd+s"}:
            try:
                self.save_runtime_state()
            except Exception as exc:
                self.status_message = f"State save failed: {exc}"
            self.refresh()
            return
        if key in {"ctrl+z", "cmd+z"}:
            self.undo_last_state()
            self.refresh()
            return
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
    parser.add_argument("--recent-trajectory-length", type=int, default=10)
    parser.add_argument("--state-dir", type=Path, default=DEFAULT_STATE_DIR, help="Directory for k state snapshots.")
    parser.add_argument("--load-state", type=Path, default=None, help="Load a saved .npz interactive method state.")
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
        state_dir=Path(args.state_dir),
        load_state=None if args.load_state is None else Path(args.load_state),
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
