from __future__ import annotations

"""Build the shared deterministic seed-1 blueprint used by Figures 1 and 2."""

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from env.agent_version import LocalObservationModel
from env.block_random_g import RandomMapGenerator
from env.core_cummap import CumulativeBeliefMap
from env.core_radar import RadarSensor
from env.grid_topology import ACTIONS_8, INVISIBLE, OBSTACLE, GridTopology
from tools.export_architecture_pictures import (
    ACTION_TO_KEY,
    ExportConfig,
    FIXED_ACTION_PREFERENCES,
    KEY_TO_ACTION,
    Snapshot,
    WorldCanvas,
    _build_method_world_canvas,
    _capture_snapshot,
    _project_belief_to_canvas,
    _select_fallback_action,
)
from tools.paper_figure_style import load_paper_figure_style


@dataclass(frozen=True, slots=True)
class RepresentativeRay:
    template_index: int
    source_endpoint_relative: tuple[int, int]
    points: tuple[tuple[int, int, int, int], ...]
    terminal_state: int

    @property
    def end_local_rc(self) -> tuple[int, int]:
        return int(self.points[-1][2]), int(self.points[-1][3])


@dataclass(frozen=True, slots=True)
class FigureDemoBlueprint:
    seed: int
    step: int
    scan_radius: int
    sensor: RadarSensor
    selected_action: int
    valid_action_indices: tuple[int, ...]
    invalid_action_indices: tuple[int, ...]
    local_observation: Snapshot
    belief_before_update: Snapshot
    belief_after_update: Snapshot
    environment_after_action: Snapshot
    belief_canvas: WorldCanvas
    belief_display: np.ndarray
    representative_rays: tuple[RepresentativeRay, ...]
    style_contract_version: str


def _normalized_ray_angle(ray: Sequence[Sequence[int]]) -> float:
    if not ray:
        raise ValueError("ray template must not be empty")
    endpoint = ray[-1]
    return float(np.mod(np.arctan2(float(endpoint[0]), float(endpoint[1])), 2.0 * np.pi))


def select_representative_ray_indices(
    ray_templates: Sequence[Sequence[Sequence[int]]],
    visual_ray_count: int = 32,
) -> tuple[int, ...]:
    """Select angle-uniform outer templates without inventing sensor channels."""

    requested = max(0, int(visual_ray_count))
    if requested == 0:
        return tuple()
    outermost_by_angle: dict[float, tuple[float, int]] = {}
    for template_index, ray in enumerate(tuple(ray_templates)):
        if not ray:
            continue
        rel_row, rel_col = int(ray[-1][0]), int(ray[-1][1])
        if rel_row == 0 and rel_col == 0:
            continue
        angle = _normalized_ray_angle(ray)
        distance_sq = float(rel_row * rel_row + rel_col * rel_col)
        key = round(angle, 12)
        previous = outermost_by_angle.get(key)
        if previous is None or distance_sq > previous[0]:
            outermost_by_angle[key] = (distance_sq, int(template_index))

    candidates = sorted((angle, item[1]) for angle, item in outermost_by_angle.items())
    target_count = min(requested, len(candidates))
    unused = set(range(len(candidates)))
    selected: list[tuple[float, int]] = []
    for target_index in range(target_count):
        target_angle = (2.0 * np.pi * float(target_index)) / float(target_count)

        def distance(candidate_index: int) -> tuple[float, float]:
            candidate_angle = candidates[candidate_index][0]
            delta = abs(candidate_angle - target_angle)
            return min(delta, 2.0 * np.pi - delta), candidate_angle

        chosen = min(unused, key=distance)
        selected.append(candidates[chosen])
        unused.remove(chosen)
    selected.sort(key=lambda item: item[0])
    return tuple(int(item[1]) for item in selected)


def clip_ray_to_local_snap(
    ray: Sequence[Sequence[int]],
    local_snap: np.ndarray,
) -> tuple[tuple[int, int, int, int], ...]:
    """Apply the plotted obstacle/invisible truncation to one real LOS template."""

    observation = np.asarray(local_snap, dtype=np.int8)
    clipped: list[tuple[int, int, int, int]] = []
    for point_index, point in enumerate(ray):
        rel_row, rel_col, local_row, local_col = (int(v) for v in point[:4])
        if not (0 <= local_row < observation.shape[0] and 0 <= local_col < observation.shape[1]):
            break
        state = int(observation[local_row, local_col])
        if point_index > 0 and state == INVISIBLE:
            break
        clipped.append((rel_row, rel_col, local_row, local_col))
        if state == OBSTACLE:
            break
    return tuple(clipped)


def _choose_action(
    *,
    planned_key: str,
    forced_key: str | None,
    valid_actions: tuple[int, ...],
    agent_state: tuple[int, int],
    visit_counts: dict[tuple[int, int], int],
) -> int:
    desired_action = int(KEY_TO_ACTION[forced_key or planned_key])
    if forced_key is not None and desired_action not in valid_actions:
        raise RuntimeError(f"forced action '{forced_key}' is illegal at {agent_state}")
    if desired_action in valid_actions:
        return desired_action
    return _select_fallback_action(
        valid_actions,
        agent_state=agent_state,
        visit_counts=visit_counts,
    )


def _capture_candidate(
    config: ExportConfig,
    *,
    target_step: int,
    forced_method_action: str | None,
    visual_ray_count: int,
) -> FigureDemoBlueprint:
    if int(target_step) < 1:
        raise ValueError("target_step must be >= 1")
    forced_key = None if forced_method_action is None else str(forced_method_action).strip().lower()
    if forced_key is not None and forced_key not in KEY_TO_ACTION:
        raise ValueError(f"forced_method_action must be one of: {', '.join(sorted(KEY_TO_ACTION))}")

    generator = RandomMapGenerator(
        rows=int(config.rows),
        cols=int(config.cols),
        obs_size=int(config.obs_size),
        obstacle_ratio=float(config.obstacle_ratio),
    )
    true_grid, start_state = generator.generate_map(seed=int(config.seed))
    free_mask = GridTopology.free_mask(true_grid)
    sensor = RadarSensor(scan_radius=int(config.scan_radius))
    observation_model = LocalObservationModel(true_grid, start_state, sensor=sensor)
    agent_state = (int(start_state[0]), int(start_state[1]))
    local_snap = np.asarray(observation_model.local_snap, dtype=np.int8).copy()
    belief = CumulativeBeliefMap(true_grid, agent_state, local_snap)
    trajectory_world = [agent_state]
    visit_counts: dict[tuple[int, int], int] = {agent_state: 1}

    for step_index in range(1, int(target_step) + 1):
        incoming_valid = GridTopology.valid_action_indices_fast(free_mask, agent_state)
        if not incoming_valid:
            raise RuntimeError(f"agent has no legal incoming move at step {step_index}")
        incoming_action = _choose_action(
            planned_key=FIXED_ACTION_PREFERENCES[(step_index - 1) % len(FIXED_ACTION_PREFERENCES)],
            forced_key=None,
            valid_actions=incoming_valid,
            agent_state=agent_state,
            visit_counts=visit_counts,
        )
        delta_row, delta_col = ACTIONS_8[incoming_action]
        agent_state = (agent_state[0] + int(delta_row), agent_state[1] + int(delta_col))
        trajectory_world.append(agent_state)
        visit_counts[agent_state] = int(visit_counts.get(agent_state, 0) + 1)
        local_snap = np.asarray(observation_model.observe_fast(agent_state), dtype=np.int8).copy()
        if step_index < int(target_step):
            belief.update(agent_state, local_snap)
            continue

        belief_before = _capture_snapshot(
            step=step_index,
            agent_state=agent_state,
            trajectory_world=trajectory_world,
            local_snap=local_snap,
            cum_map=belief,
        )
        belief.update(agent_state, local_snap)
        belief_after = _capture_snapshot(
            step=step_index,
            agent_state=agent_state,
            trajectory_world=trajectory_world,
            local_snap=local_snap,
            cum_map=belief,
        )
        valid_actions = GridTopology.valid_action_indices_fast(free_mask, agent_state)
        selected_action = _choose_action(
            planned_key=FIXED_ACTION_PREFERENCES[step_index % len(FIXED_ACTION_PREFERENCES)],
            forced_key=forced_key,
            valid_actions=valid_actions,
            agent_state=agent_state,
            visit_counts=visit_counts,
        )
        next_delta = ACTIONS_8[selected_action]
        next_state = (
            agent_state[0] + int(next_delta[0]),
            agent_state[1] + int(next_delta[1]),
        )
        next_snap = np.asarray(observation_model.observe_fast(next_state), dtype=np.int8).copy()
        environment_after = _capture_snapshot(
            step=step_index + 1,
            agent_state=next_state,
            trajectory_world=[*trajectory_world, next_state],
            local_snap=next_snap,
            cum_map=belief,
        )
        canvas = _build_method_world_canvas(belief_before, belief_after, sensor)
        belief_display = _project_belief_to_canvas(belief_after, canvas)
        representative_rays: list[RepresentativeRay] = []
        for template_index in select_representative_ray_indices(
            sensor.local_ray_templates,
            visual_ray_count=int(visual_ray_count),
        ):
            template = sensor.local_ray_templates[template_index]
            clipped = clip_ray_to_local_snap(template, local_snap)
            if not clipped:
                continue
            end_row, end_col = int(clipped[-1][2]), int(clipped[-1][3])
            representative_rays.append(
                RepresentativeRay(
                    template_index=int(template_index),
                    source_endpoint_relative=(int(template[-1][0]), int(template[-1][1])),
                    points=clipped,
                    terminal_state=int(local_snap[end_row, end_col]),
                )
            )
        style = load_paper_figure_style()
        return FigureDemoBlueprint(
            seed=int(config.seed),
            step=int(step_index),
            scan_radius=int(sensor.scan_r),
            sensor=sensor,
            selected_action=int(selected_action),
            valid_action_indices=tuple(int(index) for index in valid_actions),
            invalid_action_indices=tuple(index for index in range(len(ACTIONS_8)) if index not in valid_actions),
            local_observation=belief_before,
            belief_before_update=belief_before,
            belief_after_update=belief_after,
            environment_after_action=environment_after,
            belief_canvas=canvas,
            belief_display=belief_display,
            representative_rays=tuple(representative_rays),
            style_contract_version=style.version,
        )
    raise RuntimeError("failed to capture requested demo step")


def _candidate_is_publishable(blueprint: FigureDemoBlueprint) -> bool:
    legal_count = len(blueprint.valid_action_indices)
    return bool(
        1 <= legal_count <= 7
        and len(blueprint.invalid_action_indices) >= 1
        and blueprint.selected_action in blueprint.valid_action_indices
        and np.any(blueprint.local_observation.local_snap == OBSTACLE)
    )


def build_figure_demo_blueprint(
    *,
    seed: int = 1,
    preferred_step: int = 8,
    scan_radius: int = 10,
    rows: int = 40,
    cols: int = 60,
    obstacle_ratio: float = 0.20,
    obs_size: int = 6,
    visual_ray_count: int = 32,
    forced_method_action: str | None = None,
) -> FigureDemoBlueprint:
    """Check the requested step first, then search steps 1..30 if necessary."""

    config = ExportConfig(
        rows=int(rows),
        cols=int(cols),
        obstacle_ratio=float(obstacle_ratio),
        obs_size=int(obs_size),
        scan_radius=int(scan_radius),
        seed=int(seed),
        step_mid=4,
        step_late=int(preferred_step),
        dpi=300,
        output_dir=Path("."),
    )
    candidate_steps = [int(preferred_step), *(step for step in range(1, 31) if step != int(preferred_step))]
    for candidate_step in candidate_steps:
        candidate = _capture_candidate(
            config,
            target_step=candidate_step,
            forced_method_action=forced_method_action,
            visual_ray_count=int(visual_ray_count),
        )
        if _candidate_is_publishable(candidate):
            return candidate
    raise RuntimeError("seed did not yield a publishable figure snapshot in steps 1..30")


def blueprint_manifest(blueprint: FigureDemoBlueprint) -> dict[str, object]:
    local_snap = np.asarray(blueprint.local_observation.local_snap, dtype=np.int8)
    ray_lengths = [
        float(math.hypot(ray.points[-1][0], ray.points[-1][1]))
        for ray in blueprint.representative_rays
    ]
    return {
        "seed": int(blueprint.seed),
        "step": int(blueprint.step),
        "scan_radius": int(blueprint.scan_radius),
        "local_shape": [int(v) for v in local_snap.shape],
        "center_state": [int(v) for v in blueprint.sensor.center_state],
        "p_t": [int(v) for v in blueprint.belief_after_update.agent_world],
        "p_t_plus_1": [int(v) for v in blueprint.environment_after_action.agent_world],
        "selected_action": int(blueprint.selected_action),
        "selected_action_delta_row_col": [int(v) for v in ACTIONS_8[blueprint.selected_action]],
        "valid_action_indices": [int(v) for v in blueprint.valid_action_indices],
        "invalid_action_indices": [int(v) for v in blueprint.invalid_action_indices],
        "local_obstacle_count": int(np.count_nonzero(local_snap == OBSTACLE)),
        "local_snap_t": local_snap.tolist(),
        "belief_t": {
            "matrix": np.asarray(blueprint.belief_display, dtype=np.int8).tolist(),
            "origin_world_rc": [int(v) for v in blueprint.belief_canvas.origin_world],
            "shape": [int(v) for v in blueprint.belief_canvas.shape],
            "display_range_world_half_open": [
                int(blueprint.belief_canvas.origin_world[0]),
                int(blueprint.belief_canvas.origin_world[0] + blueprint.belief_canvas.shape[0]),
                int(blueprint.belief_canvas.origin_world[1]),
                int(blueprint.belief_canvas.origin_world[1] + blueprint.belief_canvas.shape[1]),
            ],
        },
        "representative_rays": [
            {
                "template_index": int(ray.template_index),
                "source_endpoint_relative": [int(v) for v in ray.source_endpoint_relative],
                "points_relative_and_local": [[int(v) for v in point] for point in ray.points],
                "end_local_rc": [int(v) for v in ray.end_local_rc],
                "terminal_state": int(ray.terminal_state),
                "length_cells": float(math.hypot(ray.points[-1][0], ray.points[-1][1])),
            }
            for ray in blueprint.representative_rays
        ],
        "representative_ray_count": int(len(blueprint.representative_rays)),
        "representative_ray_length_min_cells": float(min(ray_lengths) if ray_lengths else 0.0),
        "representative_ray_length_max_cells": float(max(ray_lengths) if ray_lengths else 0.0),
        "ray_semantics": (
            "variable ray lengths are expected under obstacle-truncated LOS semantics; "
            "rays are visualization samples from RadarSensor.local_ray_templates, not sensor channels"
        ),
        "style_contract_version": str(blueprint.style_contract_version),
        "style_contract_path": str(load_paper_figure_style().contract_path),
    }


def write_blueprint_json(blueprint: FigureDemoBlueprint, output_path: Path | str) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(blueprint_manifest(blueprint), indent=2), encoding="utf-8")
    return path


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export the shared deterministic blueprint for paper Figures 1 and 2.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--step", type=int, default=8)
    parser.add_argument("--scan-radius", type=int, default=10)
    parser.add_argument("--rows", type=int, default=40)
    parser.add_argument("--cols", type=int, default=60)
    parser.add_argument("--obstacle-ratio", type=float, default=0.20)
    parser.add_argument("--obs-size", type=int, default=6)
    parser.add_argument("--visual-ray-count", type=int, default=32)
    return parser


def cli_main() -> None:
    args = _build_arg_parser().parse_args()
    blueprint = build_figure_demo_blueprint(
        seed=int(args.seed),
        preferred_step=int(args.step),
        scan_radius=int(args.scan_radius),
        rows=int(args.rows),
        cols=int(args.cols),
        obstacle_ratio=float(args.obstacle_ratio),
        obs_size=int(args.obs_size),
        visual_ray_count=int(args.visual_ray_count),
    )
    path = write_blueprint_json(blueprint, args.output)
    print(f"blueprint={path.resolve()}")
    print(f"seed={blueprint.seed}")
    print(f"step={blueprint.step}")
    print(f"p_t={blueprint.belief_after_update.agent_world}")
    print(f"p_t_plus_1={blueprint.environment_after_action.agent_world}")


if __name__ == "__main__":
    cli_main()
