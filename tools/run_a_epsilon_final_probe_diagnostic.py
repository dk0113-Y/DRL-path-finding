from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
import time
import traceback
from collections import deque
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import tools.run_final_probe_matrix as final_probe_matrix
from agents.q_value_agent import ExplorationQConfig, ExplorationQNetwork, StateTensorAdapter, select_greedy_action
from env.agent_version import LocalObservationModel
from env.core_cummap import CumulativeBeliefMap
from env.grid_topology import ACTIONS_8, GridTopology
from env.shared_semantic_layer import build_semantic_visualization_payload
from train_q_agent import configure_torch_runtime, set_seed
from training.collector import SEMANTIC_EPISODE_FIELDS, summarize_semantic_records
from training.evaluator import EvaluatorConfig, GreedyEvaluator
from training.rewarding import (
    REWARD_BREAKDOWN_FIELDS,
    REWARD_EVENT_SUMMARY_FIELDS,
    add_reward_breakdown,
    finalize_reward_event_summary,
    info_gain_components,
    reward_from_breakdown,
    timeout_penalty_breakdown,
    turn_penalty_weight_from_steps,
    valid_step_reward_breakdown,
    zero_reward_breakdown,
    zero_reward_event_summary,
)


DEFAULT_EPISODES = 100
DEFAULT_EPSILON = 0.04
DEFAULT_OFFICIAL_SEED_BASE = 20261323
DEFAULT_TRAIN_SEED_BASE = 20259323
DEFAULT_OUTPUT_ROOT = Path("experiment_records/final_probe_epsilon_diagnostic")
RANDOM_ACTION_SEED_OFFSET = 1_000_003

CHECKPOINT_PATH = REPO_ROOT / "checkpoint_store" / "full_method_main" / "A_full_method.pt"
CONFIG_SNAPSHOT_PATH = REPO_ROOT / "experiment_records" / "full_method_main" / "logs" / "config_snapshot.json"
METRIC_SNAPSHOT_PATH = REPO_ROOT / "experiment_records" / "full_method_main" / "logs" / "metric_snapshot.json"
OFFICIAL_FINAL_PROBE_ROOT = REPO_ROOT / "experiment_records" / "final_probe"
OFFICIAL_PROTOCOL_PATH = OFFICIAL_FINAL_PROBE_ROOT / "final_probe_protocol.json"
OFFICIAL_SUMMARY_CSV = OFFICIAL_FINAL_PROBE_ROOT / "final_probe_summary.csv"
TRAIN_SEED_DIAGNOSTIC_ROOT = REPO_ROOT / "experiment_records" / "final_probe_train_seed_diagnostic"
TRAIN_SEED_COMPARISON_CSV = TRAIN_SEED_DIAGNOSTIC_ROOT / "comparison_with_existing.csv"

EXPECTED_ENV_PARAMS = {
    "rows": 40,
    "cols": 60,
    "obs_size": 6,
    "scan_radius": 10,
    "obstacle_ratio": 0.20,
    "max_episode_steps": 600,
    "coverage_stop_threshold": 0.95,
    "trajectory_history_steps": 10,
}

A_METHOD = {
    "method_id": "A",
    "group": "A",
    "display_name": "full_method_main",
    "checkpoint_required": True,
    "checkpoint_path": str(CHECKPOINT_PATH.resolve()),
    "config_snapshot_path": str(CONFIG_SNAPSHOT_PATH.resolve()),
    "model_factory": "ExplorationQNetwork",
    "state_adapter_factory": "StateTensorAdapter",
    "evaluation_order": 1,
}


class DiagnosticReadinessError(RuntimeError):
    pass


class EpsilonGreedyEvaluator(GreedyEvaluator):
    """A-only epsilon-greedy diagnostic evaluator with GreedyEvaluator-compatible metrics."""

    def __init__(self, *args, epsilon: float, **kwargs):
        super().__init__(*args, **kwargs)
        if not (0.0 <= float(epsilon) <= 1.0):
            raise ValueError("epsilon must be in [0, 1]")
        self.epsilon = float(epsilon)

    def _select_epsilon_action(self, model, state_tensors, valid_before: tuple[int, ...], rng: np.random.Generator) -> tuple[int, str]:
        if len(valid_before) <= 0:
            raise RuntimeError("Cannot select an action from an empty valid-action set.")
        if float(rng.random()) < self.epsilon:
            return int(rng.choice(np.asarray(valid_before, dtype=np.int64))), "random"

        policy_state = self.state_adapter.move_state_batch(
            state_tensors,
            target_device=self._policy_device,
            non_blocking=True,
        )
        with torch.inference_mode():
            with self._inference_autocast_context():
                q_values = model(
                    policy_state["advantage_canvas"],
                    policy_state["value_block_features"],
                    policy_state["value_entry_features"],
                    policy_state["value_block_mask"],
                    policy_state["value_entry_mask"],
                    return_aux=False,
                )
            action_mask = self._get_current_action_mask(policy_device=True)
            action = select_greedy_action(q_values, action_mask=action_mask)
        return int(action.item()), "greedy"

    def _run_episode(self, model, episode_seed: int | None = None) -> Dict[str, object]:
        action_rng_seed = RANDOM_ACTION_SEED_OFFSET if episode_seed is None else int(episode_seed) + RANDOM_ACTION_SEED_OFFSET
        action_rng = np.random.default_rng(action_rng_seed)

        with self._seeded_map_generation(episode_seed):
            grid, agent = self.generator.generate_map()
        free_mask = GridTopology.free_mask(grid)
        obs_model = LocalObservationModel(grid, agent, sensor=self.sensor)
        local_snap = obs_model.local_snap
        self._refresh_valid_action_cache(GridTopology.valid_action_indices_fast(free_mask, agent))

        cum_map = CumulativeBeliefMap(grid, agent, local_snap)
        frontier_u8 = cum_map.get_frontier_u8(refresh=False)
        self._check_incremental_frontier_consistency(
            cum_map,
            context="epsilon_evaluator_reset",
            episode_seed=episode_seed,
            episode_len=0,
        )
        shared_artifacts = self.state_adapter.build_shared_step_artifacts(
            cum_map,
            agent,
            frontier_u8=frontier_u8,
        )
        model_device = self._resolve_module_device(model)
        self._set_policy_device(model_device)

        episode_reward = 0.0
        episode_len = 0
        random_action_count = 0
        greedy_action_count = 0
        episode_breakdown = zero_reward_breakdown()
        episode_event_summary = zero_reward_event_summary()
        episode_semantic_records: list[dict[str, float]] = []
        recent_positions: deque[tuple[int, int]] = deque(
            [(int(agent[0]), int(agent[1]))],
            maxlen=self._recent_revisit_horizon,
        )
        recent_trajectory_positions: deque[tuple[int, int]] = deque(
            [(int(agent[0]), int(agent[1]))],
            maxlen=max(1, int(self.cfg.trajectory_history_steps)) + 1,
        )
        stall_streak = 0
        prev_action_idx: int | None = None
        trajectory_positions: list[tuple[int, int]] = [(int(agent[0]), int(agent[1]))]

        while True:
            valid_before = self.valid_action_indices
            step_breakdown = zero_reward_breakdown()

            if len(valid_before) <= 0:
                raise RuntimeError(
                    "Encountered an empty valid-action set before evaluation step. "
                    "This is treated as a defensive invariant violation, not a normal episode outcome."
                )

            state_tensors, state_meta = self._build_state_tensors(
                cum_map,
                agent,
                shared_artifacts=shared_artifacts,
                recent_trajectory_positions=tuple(recent_trajectory_positions),
            )
            if isinstance(state_meta, dict):
                episode_semantic_records.append(
                    {field: float(state_meta.get(field, float("nan"))) for field in SEMANTIC_EPISODE_FIELDS}
                )

            action_idx, action_source = self._select_epsilon_action(model, state_tensors, valid_before, action_rng)
            if action_source == "random":
                random_action_count += 1
            else:
                greedy_action_count += 1

            if action_idx not in valid_before:
                raise RuntimeError(
                    f"Selected epsilon diagnostic action {action_idx} outside the valid-action set {sorted(valid_before)}. "
                    "This is treated as a defensive invariant violation."
                )

            turn_steps = GridTopology.circular_turn_steps(prev_action_idx, action_idx)
            turn_penalty_weight = float(
                turn_penalty_weight_from_steps(
                    turn_steps,
                    weight_45=float(self.cfg.reward_turn_weight_45),
                    weight_90=float(self.cfg.reward_turn_weight_90),
                    weight_135=float(self.cfg.reward_turn_weight_135),
                    weight_180=float(self.cfg.reward_turn_weight_180),
                )
            )
            dr, dc = ACTIONS_8[action_idx]
            agent = (int(agent[0] + dr), int(agent[1] + dc))
            recent_trajectory_positions.append((int(agent[0]), int(agent[1])))
            trajectory_positions.append((int(agent[0]), int(agent[1])))

            local_snap = obs_model.observe_fast(agent)
            self._refresh_valid_action_cache(GridTopology.valid_action_indices_fast(free_mask, agent))
            updated, delta_empty, delta_obstacle = cum_map.update(agent, local_snap)
            if int(updated) != int(delta_empty + delta_obstacle):
                raise RuntimeError("belief-map update returned inconsistent information-gain counts")
            frontier_u8 = cum_map.get_frontier_u8(refresh=False)
            self._check_incremental_frontier_consistency(
                cum_map,
                context="epsilon_evaluator_step_post_update",
                episode_seed=episode_seed,
                episode_len=episode_len + 1,
            )
            shared_artifacts = self.state_adapter.build_shared_step_artifacts(
                cum_map,
                agent,
                frontier_u8=frontier_u8,
            )

            recent_revisit = bool((int(agent[0]), int(agent[1])) in recent_positions)
            recent_positions.append((int(agent[0]), int(agent[1])))
            if int(delta_empty) == 0 and int(delta_obstacle) == 0:
                stall_streak += 1
            else:
                stall_streak = 0
            stall_triggered = bool(stall_streak >= self._stall_diagnostic_window)
            info_metrics = info_gain_components(
                delta_empty=delta_empty,
                delta_obstacle=delta_obstacle,
                obstacle_weight=float(self.cfg.reward_obstacle_weight),
                info_norm=self.reward_info_norm,
                reward_info_scale=float(self.cfg.reward_info_scale),
            )
            episode_event_summary["delta_empty_sum"] += float(delta_empty)
            episode_event_summary["delta_obstacle_sum"] += float(delta_obstacle)
            episode_event_summary["empty_info_gain_sum"] += float(info_metrics["empty_info_gain_sum"])
            episode_event_summary["obstacle_info_gain_sum"] += float(info_metrics["obstacle_info_gain_sum"])
            episode_event_summary["weighted_obstacle_info_gain_sum"] += float(info_metrics["weighted_obstacle_info_gain_sum"])
            episode_event_summary["weighted_info_gain_sum"] += float(info_metrics["weighted_info_gain_sum"])
            episode_event_summary["empty_info_reward_sum"] += float(info_metrics["empty_info_reward_sum"])
            episode_event_summary["obstacle_info_reward_sum"] += float(info_metrics["obstacle_info_reward_sum"])
            episode_event_summary["recent_revisit_trigger_count"] += float(bool(recent_revisit))
            episode_event_summary["stall_trigger_count"] += float(bool(stall_triggered))
            if int(delta_empty) == 0 and int(delta_obstacle) == 0:
                episode_event_summary["zero_info_step_count"] += 1.0
            if int(turn_steps) >= 2:
                episode_event_summary["turn_ge_90_count"] += 1.0
            if int(turn_steps) == 3:
                episode_event_summary["turn_135_count"] += 1.0
            if int(turn_steps) == 4:
                episode_event_summary["turn_180_count"] += 1.0
            episode_event_summary["turn_penalty_weight_sum"] += float(turn_penalty_weight)
            success = bool(cum_map.coverage_rate >= float(self.cfg.coverage_stop_threshold))
            no_valid_after_step = bool((not success) and (len(self.valid_action_indices) <= 0))

            step_breakdown = valid_step_reward_breakdown(
                self.cfg,
                delta_empty=delta_empty,
                delta_obstacle=delta_obstacle,
                info_norm=self.reward_info_norm,
                recent_revisit=recent_revisit,
                turn_penalty_weight=turn_penalty_weight,
                success=success,
            )
            reward = reward_from_breakdown(step_breakdown)
            prev_action_idx = int(action_idx)
            done = False
            done_reason = ""

            if success:
                done = True
                done_reason = "coverage_reached"
            elif no_valid_after_step:
                raise RuntimeError(
                    "Encountered an empty valid-action set after a valid evaluation move "
                    "without reaching coverage target. This is treated as a defensive "
                    "environment invariant violation."
                )

            episode_len += 1
            if (not done) and (episode_len >= int(self.cfg.max_episode_steps)):
                done = True
                done_reason = "max_episode_steps"
                timeout_breakdown = timeout_penalty_breakdown(self.cfg)
                add_reward_breakdown(step_breakdown, timeout_breakdown)
                reward += reward_from_breakdown(timeout_breakdown)

            episode_reward += reward
            add_reward_breakdown(episode_breakdown, step_breakdown)

            if done:
                episode_event_summary = finalize_reward_event_summary(dict(episode_event_summary))
                episode_event_summary["timeout_flag"] = float(done_reason == "max_episode_steps")
                semantic_snapshot = getattr(shared_artifacts, "semantic_snapshot", None)
                semantic_viz = (
                    build_semantic_visualization_payload(semantic_snapshot)
                    if semantic_snapshot is not None else None
                )
                total_actions = max(1, int(random_action_count + greedy_action_count))
                return {
                    "episode_reward": float(episode_reward),
                    "episode_length": int(episode_len),
                    "final_coverage": float(cum_map.coverage_rate),
                    "success": int(done_reason == "coverage_reached"),
                    "repeat_visit_ratio": float(self._repeat_visit_ratio(cum_map)),
                    "done_reason": str(done_reason),
                    "epsilon": float(self.epsilon),
                    "random_action_count": int(random_action_count),
                    "greedy_action_count": int(greedy_action_count),
                    "random_action_ratio": float(random_action_count) / float(total_actions),
                    **summarize_semantic_records(episode_semantic_records),
                    "true_grid": np.asarray(grid, dtype=np.int8).copy(),
                    "trajectory_positions": list(trajectory_positions),
                    "belief_map": np.asarray(cum_map.map, dtype=np.int8).copy(),
                    "belief_origin_world_rc": (
                        int(cum_map.origin_world_rc[0]), int(cum_map.origin_world_rc[1])
                    ),
                    "semantic_viz": semantic_viz,
                    **{field: float(episode_breakdown[field]) for field in REWARD_BREAKDOWN_FIELDS},
                    **{field: float(episode_event_summary[field]) for field in REWARD_EVENT_SUMMARY_FIELDS},
                }

    def evaluate(self, model, num_episodes: int = 5, seed_base: int | None = None) -> Dict[str, object]:
        if num_episodes <= 0:
            raise ValueError("num_episodes must be > 0")

        was_training = bool(model.training)
        model.eval()
        episodes = [
            self._run_episode(
                model,
                episode_seed=(None if seed_base is None else int(seed_base) + idx),
            )
            for idx in range(int(num_episodes))
        ]
        if was_training:
            model.train()

        result: Dict[str, object] = {
            "eval_episodes": int(num_episodes),
            "eval_mean_reward": _mean_episode_field(episodes, "episode_reward"),
            "eval_mean_coverage": _mean_episode_field(episodes, "final_coverage"),
            "eval_success_rate": _mean_episode_field(episodes, "success"),
            "eval_mean_episode_length": _mean_episode_field(episodes, "episode_length"),
            "eval_mean_repeat_visit_ratio": _mean_episode_field(episodes, "repeat_visit_ratio"),
            "eval_epsilon": float(self.epsilon),
            "eval_mean_random_action_count": _mean_episode_field(episodes, "random_action_count"),
            "eval_mean_greedy_action_count": _mean_episode_field(episodes, "greedy_action_count"),
            "eval_mean_random_action_ratio": _mean_episode_field(episodes, "random_action_ratio"),
            "episodes": episodes,
        }
        for field in SEMANTIC_EPISODE_FIELDS:
            result[f"eval_mean_{field}"] = _mean_episode_field(episodes, field)
        for field in REWARD_BREAKDOWN_FIELDS:
            result[f"eval_mean_{field}"] = _mean_episode_field(episodes, field)
        for field in REWARD_EVENT_SUMMARY_FIELDS:
            result[f"eval_mean_{field}"] = _mean_episode_field(episodes, field)
        return result


def _mean_episode_field(episodes: Sequence[Mapping[str, Any]], field: str) -> float:
    values = np.asarray([float(ep[field]) for ep in episodes], dtype=np.float32)
    return float(np.nanmean(values))


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _row_by_source(path: Path, source: str) -> dict[str, str]:
    for row in _read_csv_rows(path):
        if row.get("source") == source:
            return row
    raise DiagnosticReadinessError(f"No source={source!r} row found in {path}")


def _first_a_row(path: Path) -> dict[str, str]:
    for row in _read_csv_rows(path):
        if row.get("method_id") == "A" or row.get("group") == "A":
            return row
    raise DiagnosticReadinessError(f"No A row found in {path}")


def _float_or_none(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _summary_metric(summary: Mapping[str, Any], metric: str) -> Any:
    metrics = summary.get("summary", {}).get("metrics", {})
    if isinstance(metrics, Mapping):
        return metrics.get(metric)
    return None


def _validate_expected_env_params(cfg: Any) -> dict[str, Any]:
    checked: dict[str, Any] = {}
    mismatches: list[str] = []
    for key, expected in EXPECTED_ENV_PARAMS.items():
        actual = getattr(cfg, key)
        checked[key] = actual
        if isinstance(expected, float):
            if abs(float(actual) - expected) > 1e-9:
                mismatches.append(f"{key}: expected {expected}, got {actual}")
        elif int(actual) != int(expected):
            mismatches.append(f"{key}: expected {expected}, got {actual}")
    if mismatches:
        raise DiagnosticReadinessError("Checkpoint train_config env parameter mismatch: " + "; ".join(mismatches))
    return checked


def readiness_check(device: torch.device) -> dict[str, Any]:
    missing: list[dict[str, str]] = []
    for kind, path in (
        ("checkpoint", CHECKPOINT_PATH),
        ("config_snapshot", CONFIG_SNAPSHOT_PATH),
        ("metric_snapshot", METRIC_SNAPSHOT_PATH),
        ("official_protocol", OFFICIAL_PROTOCOL_PATH),
        ("official_summary_csv", OFFICIAL_SUMMARY_CSV),
        ("train_seed_comparison_csv", TRAIN_SEED_COMPARISON_CSV),
    ):
        if not path.exists():
            missing.append({"kind": kind, "path": str(path)})
    if missing:
        raise DiagnosticReadinessError("Missing required artifacts: " + json.dumps(missing, ensure_ascii=False))

    payload = final_probe_matrix._load_checkpoint_payload(CHECKPOINT_PATH)
    cfg = final_probe_matrix.train_config_from_payload(payload, device)
    env_params = _validate_expected_env_params(cfg)

    model = ExplorationQNetwork(ExplorationQConfig())
    load_result = model.load_state_dict(payload["online_state_dict"], strict=True)
    if load_result.missing_keys or load_result.unexpected_keys:
        raise DiagnosticReadinessError(
            "Strict load_state_dict was not clean: "
            + json.dumps(
                {
                    "missing_keys": list(load_result.missing_keys),
                    "unexpected_keys": list(load_result.unexpected_keys),
                },
                ensure_ascii=False,
            )
        )

    adapter_cfg = final_probe_matrix.state_adapter_config_from_train_config(cfg)
    _ = StateTensorAdapter(cfg=adapter_cfg, device="cpu")

    return {
        "checkpoint_path": str(CHECKPOINT_PATH.resolve()),
        "checkpoint_status": "ok",
        "config_snapshot_path": str(CONFIG_SNAPSHOT_PATH.resolve()),
        "config_snapshot_status": "ok",
        "metric_snapshot_path": str(METRIC_SNAPSHOT_PATH.resolve()),
        "metric_snapshot_status": "ok",
        "official_protocol_path": str(OFFICIAL_PROTOCOL_PATH.resolve()),
        "official_summary_csv": str(OFFICIAL_SUMMARY_CSV.resolve()),
        "train_seed_comparison_csv": str(TRAIN_SEED_COMPARISON_CSV.resolve()),
        "model_factory": "ExplorationQNetwork",
        "model_factory_status": "ok",
        "state_adapter_factory": "StateTensorAdapter",
        "state_adapter_factory_status": "ok",
        "load_state_dict": {
            "missing_keys": list(load_result.missing_keys),
            "unexpected_keys": list(load_result.unexpected_keys),
        },
        "checkpoint_metadata": {
            "env_steps": payload.get("env_steps"),
            "learn_steps": payload.get("learn_steps"),
            "train_episode_idx": payload.get("train_episode_idx"),
        },
        "env_params_from_checkpoint_train_config": env_params,
    }


def write_protocol(
    output_root: Path,
    *,
    episodes: int,
    epsilon: float,
    official_seed_base: int,
    train_seed_base: int,
    device_text: str,
) -> None:
    payload = {
        "protocol_name": "a_epsilon_final_probe_diagnostic_v1",
        "created_at": final_probe_matrix._now_iso(),
        "diagnostic_scope": "A_only_epsilon_probe",
        "purpose": "Diagnose whether greedy determinization contributes to A_full_method_main final-probe drop.",
        "not_replacement_for": str(OFFICIAL_FINAL_PROBE_ROOT.resolve()),
        "method_id": "A",
        "episodes": int(episodes),
        "epsilon": float(epsilon),
        "policy": "epsilon_greedy",
        "epsilon_rule": "per step, sample a random valid action with probability epsilon; otherwise choose the greedy valid action",
        "random_action_seed_rule": f"numpy.default_rng(episode_seed + {RANDOM_ACTION_SEED_OFFSET})",
        "seed_blocks": {
            "official_final_seed_block": int(official_seed_base),
            "train_seed_block": int(train_seed_base),
        },
        "episode_seed_rule": "seed_base + zero_based_episode_index",
        "device": str(device_text),
        "checkpoint_path": str(CHECKPOINT_PATH.resolve()),
        "config_snapshot_path": str(CONFIG_SNAPSHOT_PATH.resolve()),
        "method_loading": {
            "model_factory": "ExplorationQNetwork",
            "state_adapter_factory": "StateTensorAdapter",
        },
        "env_params": dict(EXPECTED_ENV_PARAMS),
        "checkpoint_copy_policy": "checkpoint is referenced in place and is not copied into experiment_records",
        "official_final_probe_reference": str(OFFICIAL_FINAL_PROBE_ROOT.resolve()),
        "train_seed_greedy_reference": str(TRAIN_SEED_DIAGNOSTIC_ROOT.resolve()),
    }
    final_probe_matrix._write_json(output_root / "protocol.json", payload)


def _build_model_and_evaluator(device: torch.device, epsilon: float) -> tuple[torch.nn.Module, EpsilonGreedyEvaluator, dict[str, Any]]:
    payload = final_probe_matrix._load_checkpoint_payload(CHECKPOINT_PATH)
    cfg = final_probe_matrix.train_config_from_payload(payload, device)
    _validate_expected_env_params(cfg)
    set_seed(int(cfg.seed))
    configure_torch_runtime(cfg)

    model = ExplorationQNetwork(ExplorationQConfig())
    model.load_state_dict(payload["online_state_dict"], strict=True)
    model.to(device)
    model.eval()

    state_adapter = StateTensorAdapter(
        cfg=final_probe_matrix.state_adapter_config_from_train_config(cfg),
        device="cpu",
    )
    collector_cfg = final_probe_matrix.collector_config_from_train_config(cfg)
    evaluator_cfg = EvaluatorConfig(
        rows=collector_cfg.rows,
        cols=collector_cfg.cols,
        obs_size=collector_cfg.obs_size,
        scan_radius=collector_cfg.scan_radius,
        obstacle_ratio=collector_cfg.obstacle_ratio,
        max_episode_steps=collector_cfg.max_episode_steps,
        coverage_stop_threshold=collector_cfg.coverage_stop_threshold,
        trajectory_history_steps=collector_cfg.trajectory_history_steps,
        reward_info_scale=collector_cfg.reward_info_scale,
        reward_obstacle_weight=collector_cfg.reward_obstacle_weight,
        reward_step_penalty=collector_cfg.reward_step_penalty,
        reward_terminal_bonus=collector_cfg.reward_terminal_bonus,
        reward_revisit_penalty=collector_cfg.reward_revisit_penalty,
        reward_turn_penalty_scale=collector_cfg.reward_turn_penalty_scale,
        reward_turn_weight_45=collector_cfg.reward_turn_weight_45,
        reward_turn_weight_90=collector_cfg.reward_turn_weight_90,
        reward_turn_weight_135=collector_cfg.reward_turn_weight_135,
        reward_turn_weight_180=collector_cfg.reward_turn_weight_180,
        reward_timeout_penalty=collector_cfg.reward_timeout_penalty,
        enable_inference_amp=collector_cfg.enable_inference_amp,
        inference_amp_dtype=collector_cfg.inference_amp_dtype,
        debug_check_incremental_frontier=collector_cfg.debug_check_incremental_frontier,
    )
    evaluator = EpsilonGreedyEvaluator(evaluator_cfg, state_adapter=state_adapter, device=str(device), epsilon=float(epsilon))
    return model, evaluator, payload


def evaluate_block(
    *,
    block_name: str,
    episodes: int,
    seed_base: int,
    epsilon: float,
    device: torch.device,
    output_root: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    model, evaluator, payload = _build_model_and_evaluator(device, epsilon)
    probe = evaluator.evaluate(model, num_episodes=int(episodes), seed_base=int(seed_base))

    episode_rows: list[dict[str, Any]] = []
    for idx, row in enumerate(probe.get("episodes", []), start=1):
        out = final_probe_matrix._episode_csv_row(row)
        out["block"] = str(block_name)
        out["method_id"] = "A"
        out["group"] = "A"
        out["display_name"] = "full_method_main"
        out["checkpoint_path"] = str(CHECKPOINT_PATH.resolve())
        out.setdefault("episode_idx", idx)
        out.setdefault("episode_seed", int(seed_base) + idx - 1)
        episode_rows.append(out)

    summary = final_probe_matrix._summary_from_probe(probe)
    metrics = summary.setdefault("metrics", {})
    metrics["epsilon"] = float(epsilon)
    metrics["random_action_count"] = probe.get("eval_mean_random_action_count")
    metrics["greedy_action_count"] = probe.get("eval_mean_greedy_action_count")
    metrics["random_action_ratio"] = probe.get("eval_mean_random_action_ratio")

    summary_payload = {
        "method_id": "A",
        "group": "A",
        "display_name": "full_method_main",
        "block": str(block_name),
        "status": "ok",
        "checkpoint_path": str(CHECKPOINT_PATH.resolve()),
        "config_snapshot_path": str(CONFIG_SNAPSHOT_PATH.resolve()),
        "episodes": int(episodes),
        "seed_base": int(seed_base),
        "episode_seed_rule": "seed_base + zero_based_episode_index",
        "policy": "epsilon_greedy",
        "epsilon": float(epsilon),
        "random_action_seed_rule": f"numpy.default_rng(episode_seed + {RANDOM_ACTION_SEED_OFFSET})",
        "checkpoint_metadata": {
            "env_steps": payload.get("env_steps"),
            "learn_steps": payload.get("learn_steps"),
            "train_episode_idx": payload.get("train_episode_idx"),
        },
        "method_loading": {
            "model_factory": "ExplorationQNetwork",
            "state_adapter_factory": "StateTensorAdapter",
        },
        "summary": summary,
    }

    block_dir = output_root / block_name
    final_probe_matrix._write_csv(
        block_dir / "per_episode.csv",
        episode_rows,
        preferred=(
            "block",
            "method_id",
            "group",
            "display_name",
            "episode_idx",
            "episode_seed",
            "epsilon",
            "success",
            "final_coverage",
            "episode_reward",
            "episode_length",
            "repeat_visit_ratio",
            "timeout_flag",
            "done_reason",
            "random_action_count",
            "greedy_action_count",
            "random_action_ratio",
        ),
    )
    final_probe_matrix._write_json(block_dir / "summary.json", summary_payload)
    return summary_payload, episode_rows


def _train_recent_row_from_metric_snapshot() -> dict[str, Any]:
    snapshot = _read_json(METRIC_SNAPSHOT_PATH)
    recent = snapshot.get("recent_train", {})
    metrics = recent.get("metrics", {})
    semantic = recent.get("semantic_monitoring", {})
    return {
        "source": "A_train_recent_from_metric_snapshot",
        "seed_base": "",
        "policy": "train_epsilon_greedy",
        "epsilon": recent.get("optimizer_monitoring", {}).get("epsilon"),
        "success_rate": metrics.get("success_rate"),
        "coverage": metrics.get("coverage"),
        "reward": metrics.get("reward"),
        "episode_length": metrics.get("episode_length"),
        "repeat_visit_ratio": metrics.get("repeat_visit_ratio"),
        "timeout_rate": semantic.get("timeout_rate"),
        "timeout_flag": "",
        "random_action_ratio": "",
    }


def build_comparison_rows(
    *,
    official_epsilon_summary: Mapping[str, Any],
    train_epsilon_summary: Mapping[str, Any],
    official_seed_base: int,
    train_seed_base: int,
) -> list[dict[str, Any]]:
    official_greedy = _first_a_row(OFFICIAL_SUMMARY_CSV)
    train_greedy = _row_by_source(TRAIN_SEED_COMPARISON_CSV, "A_train_seed_greedy_probe_20259323")
    return [
        _train_recent_row_from_metric_snapshot(),
        {
            "source": "A_official_greedy_final_probe_seed_20261323",
            "seed_base": 20261323,
            "policy": "greedy",
            "epsilon": 0.0,
            "success_rate": _float_or_none(official_greedy.get("success_rate")),
            "coverage": _float_or_none(official_greedy.get("coverage")),
            "reward": _float_or_none(official_greedy.get("reward")),
            "episode_length": _float_or_none(official_greedy.get("episode_length")),
            "repeat_visit_ratio": _float_or_none(official_greedy.get("repeat_visit_ratio")),
            "timeout_rate": "",
            "timeout_flag": _float_or_none(official_greedy.get("timeout_flag")),
            "random_action_ratio": 0.0,
        },
        {
            "source": "A_train_seed_greedy_probe_20259323",
            "seed_base": 20259323,
            "policy": "greedy",
            "epsilon": 0.0,
            "success_rate": _float_or_none(train_greedy.get("success_rate")),
            "coverage": _float_or_none(train_greedy.get("coverage")),
            "reward": _float_or_none(train_greedy.get("reward")),
            "episode_length": _float_or_none(train_greedy.get("episode_length")),
            "repeat_visit_ratio": _float_or_none(train_greedy.get("repeat_visit_ratio")),
            "timeout_rate": "",
            "timeout_flag": _float_or_none(train_greedy.get("timeout_flag")),
            "random_action_ratio": 0.0,
        },
        _comparison_row_from_summary(
            "A_epsilon004_final_seed_probe_20261323",
            official_epsilon_summary,
            seed_base=official_seed_base,
        ),
        _comparison_row_from_summary(
            "A_epsilon004_train_seed_probe_20259323",
            train_epsilon_summary,
            seed_base=train_seed_base,
        ),
    ]


def _comparison_row_from_summary(source: str, summary: Mapping[str, Any], *, seed_base: int) -> dict[str, Any]:
    return {
        "source": source,
        "seed_base": int(seed_base),
        "policy": "epsilon_greedy",
        "epsilon": _summary_metric(summary, "epsilon"),
        "success_rate": _summary_metric(summary, "success_rate"),
        "coverage": _summary_metric(summary, "coverage"),
        "reward": _summary_metric(summary, "reward"),
        "episode_length": _summary_metric(summary, "episode_length"),
        "repeat_visit_ratio": _summary_metric(summary, "repeat_visit_ratio"),
        "timeout_rate": "",
        "timeout_flag": _summary_metric(summary, "timeout_flag"),
        "random_action_ratio": _summary_metric(summary, "random_action_ratio"),
    }


def _delta(new_value: Any, old_value: Any) -> float | None:
    new_float = _float_or_none(new_value)
    old_float = _float_or_none(old_value)
    if new_float is None or old_float is None:
        return None
    return new_float - old_float


def write_interpretation(output_root: Path, comparison_rows: Sequence[Mapping[str, Any]]) -> None:
    by_source = {str(row["source"]): row for row in comparison_rows}
    train = by_source["A_train_recent_from_metric_snapshot"]
    official_greedy = by_source["A_official_greedy_final_probe_seed_20261323"]
    train_greedy = by_source["A_train_seed_greedy_probe_20259323"]
    official_eps = by_source["A_epsilon004_final_seed_probe_20261323"]
    train_eps = by_source["A_epsilon004_train_seed_probe_20259323"]
    metrics = ("success_rate", "coverage", "reward", "episode_length", "repeat_visit_ratio")
    lines = [
        "# A Epsilon Final Probe Diagnostic",
        "",
        "- This diagnostic uses epsilon-greedy evaluation and is not a replacement for the official greedy final probe.",
        "- The official final probe remains experiment_records/final_probe/.",
        "- The train-seed greedy diagnostic remains experiment_records/final_probe_train_seed_diagnostic/.",
        "- If epsilon=0.04 improves A on the official final seed block, then greedy determinization likely contributes to A's final-probe drop.",
        "- If epsilon=0.04 does not improve A, the remaining likely causes are endpoint checkpoint quality, last-vs-peak mismatch, or train-window/checkpoint mismatch.",
        "- Epsilon diagnostic should not be used as the main SCI result unless all learning methods are evaluated with the same epsilon protocol.",
        "",
        "## Official Seed Block Deltas",
        "",
        "| metric | epsilon_minus_official_greedy | epsilon_minus_train_recent |",
        "| --- | ---: | ---: |",
    ]
    for metric in metrics:
        lines.append(
            f"| {metric} | {_delta(official_eps.get(metric), official_greedy.get(metric))} | "
            f"{_delta(official_eps.get(metric), train.get(metric))} |"
        )
    lines.append(
        f"| timeout_flag_or_rate | {_delta(official_eps.get('timeout_flag'), official_greedy.get('timeout_flag'))} | "
        f"{_delta(official_eps.get('timeout_flag'), train.get('timeout_rate'))} |"
    )
    lines.extend(
        [
            "",
            "## Train Seed Block Deltas",
            "",
            "| metric | epsilon_minus_train_seed_greedy | epsilon_minus_train_recent |",
            "| --- | ---: | ---: |",
        ]
    )
    for metric in metrics:
        lines.append(
            f"| {metric} | {_delta(train_eps.get(metric), train_greedy.get(metric))} | "
            f"{_delta(train_eps.get(metric), train.get(metric))} |"
        )
    lines.append(
        f"| timeout_flag_or_rate | {_delta(train_eps.get('timeout_flag'), train_greedy.get('timeout_flag'))} | "
        f"{_delta(train_eps.get('timeout_flag'), train.get('timeout_rate'))} |"
    )
    lines.append("")
    lines.append("Positive deltas mean the epsilon diagnostic value is higher than the reference row.")
    (output_root / "interpretation.md").write_text("\n".join(str(line) for line in lines) + "\n", encoding="utf-8")


def write_run_manifest(
    output_root: Path,
    *,
    status: str,
    args: argparse.Namespace,
    device_info: Mapping[str, Any],
    readiness: Mapping[str, Any] | None,
    runtime_sec: float,
    error: str | None = None,
    error_trace: str | None = None,
) -> None:
    seed_blocks = {
        "official_final_seed_block": int(args.official_seed_base),
        "train_seed_block": int(args.train_seed_base),
    }
    manifest = {
        "schema_version": "a_epsilon_final_probe_run_manifest/v1",
        "created_at": final_probe_matrix._now_iso(),
        "status": status,
        "repo_root": str(REPO_ROOT.resolve()),
        "git_sha": final_probe_matrix._git_output(["rev-parse", "HEAD"]),
        "git_branch": final_probe_matrix._git_output(["rev-parse", "--abbrev-ref", "HEAD"]),
        "python": sys.version,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "episodes": int(args.episodes),
        "epsilon": float(args.epsilon),
        "seed_blocks": seed_blocks,
        "episode_seed_rule": "seed_base + zero_based_episode_index",
        "checkpoint_path": str(CHECKPOINT_PATH.resolve()),
        "config_snapshot_path": str(CONFIG_SNAPSHOT_PATH.resolve()),
        "policy": "epsilon_greedy",
        "diagnostic_scope": "A_only_epsilon_probe",
        "official_final_probe_reference": str(OFFICIAL_FINAL_PROBE_ROOT.resolve()),
        "train_seed_greedy_reference": str(TRAIN_SEED_DIAGNOSTIC_ROOT.resolve()),
        "arguments": vars(args),
        "device": dict(device_info),
        "readiness": readiness,
        "runtime_sec": runtime_sec,
        "output_files": {
            "protocol_json": str((output_root / "protocol.json").resolve()),
            "run_manifest_json": str((output_root / "run_manifest.json").resolve()),
            "official_per_episode_csv": str((output_root / "official_final_seed_block" / "per_episode.csv").resolve()),
            "official_summary_json": str((output_root / "official_final_seed_block" / "summary.json").resolve()),
            "train_seed_per_episode_csv": str((output_root / "train_seed_block" / "per_episode.csv").resolve()),
            "train_seed_summary_json": str((output_root / "train_seed_block" / "summary.json").resolve()),
            "comparison_with_existing_csv": str((output_root / "comparison_with_existing.csv").resolve()),
            "interpretation_md": str((output_root / "interpretation.md").resolve()),
        },
    }
    if error:
        manifest["error"] = error
    if error_trace:
        manifest["traceback"] = error_trace
    final_probe_matrix._write_json(output_root / "run_manifest.json", manifest)


def run(args: argparse.Namespace) -> int:
    output_root = Path(args.output_root).expanduser().resolve()
    device, device_info = final_probe_matrix.resolve_device(str(args.device), require_available=not bool(args.dry_run))
    readiness = readiness_check(device)
    seed_blocks = {
        "official_final_seed_block": int(args.official_seed_base),
        "train_seed_block": int(args.train_seed_base),
    }

    if args.dry_run:
        plan = {
            "dry_run": True,
            "episodes": int(args.episodes),
            "epsilon": float(args.epsilon),
            "policy": "epsilon_greedy",
            "seed_blocks": seed_blocks,
            "episode_seed_rule": "seed_base + zero_based_episode_index",
            "device": device_info,
            "output_root": str(output_root),
            "checkpoint_path": str(CHECKPOINT_PATH.resolve()),
            "config_snapshot_path": str(CONFIG_SNAPSHOT_PATH.resolve()),
            "official_final_probe_reference": str(OFFICIAL_FINAL_PROBE_ROOT.resolve()),
            "train_seed_greedy_reference": str(TRAIN_SEED_DIAGNOSTIC_ROOT.resolve()),
            "diagnostic_scope": "A_only_epsilon_probe",
            "readiness": readiness,
            "planned_outputs": {
                "protocol_json": str((output_root / "protocol.json").resolve()),
                "run_manifest_json": str((output_root / "run_manifest.json").resolve()),
                "official_per_episode_csv": str((output_root / "official_final_seed_block" / "per_episode.csv").resolve()),
                "official_summary_json": str((output_root / "official_final_seed_block" / "summary.json").resolve()),
                "train_seed_per_episode_csv": str((output_root / "train_seed_block" / "per_episode.csv").resolve()),
                "train_seed_summary_json": str((output_root / "train_seed_block" / "summary.json").resolve()),
                "comparison_with_existing_csv": str((output_root / "comparison_with_existing.csv").resolve()),
                "interpretation_md": str((output_root / "interpretation.md").resolve()),
            },
        }
        print("[a_epsilon_final_probe] dry_run=true")
        print(json.dumps(final_probe_matrix._json_safe(plan), ensure_ascii=False, indent=2))
        return 0

    output_root.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    try:
        write_protocol(
            output_root,
            episodes=int(args.episodes),
            epsilon=float(args.epsilon),
            official_seed_base=int(args.official_seed_base),
            train_seed_base=int(args.train_seed_base),
            device_text=str(args.device),
        )
        official_summary, _ = evaluate_block(
            block_name="official_final_seed_block",
            episodes=int(args.episodes),
            seed_base=int(args.official_seed_base),
            epsilon=float(args.epsilon),
            device=device,
            output_root=output_root,
        )
        train_summary, _ = evaluate_block(
            block_name="train_seed_block",
            episodes=int(args.episodes),
            seed_base=int(args.train_seed_base),
            epsilon=float(args.epsilon),
            device=device,
            output_root=output_root,
        )
        comparison_rows = build_comparison_rows(
            official_epsilon_summary=official_summary,
            train_epsilon_summary=train_summary,
            official_seed_base=int(args.official_seed_base),
            train_seed_base=int(args.train_seed_base),
        )
        final_probe_matrix._write_csv(
            output_root / "comparison_with_existing.csv",
            comparison_rows,
            preferred=(
                "source",
                "seed_base",
                "policy",
                "epsilon",
                "success_rate",
                "coverage",
                "reward",
                "episode_length",
                "repeat_visit_ratio",
                "timeout_rate",
                "timeout_flag",
                "random_action_ratio",
            ),
        )
        write_interpretation(output_root, comparison_rows)
        write_run_manifest(
            output_root,
            status="ok",
            args=args,
            device_info=device_info,
            readiness=readiness,
            runtime_sec=time.perf_counter() - start,
        )
        print("[a_epsilon_final_probe] ok")
        for block_name, summary in (
            ("official_final_seed_block", official_summary),
            ("train_seed_block", train_summary),
        ):
            metrics = summary["summary"]["metrics"]
            print(
                "[a_epsilon_final_probe] "
                f"{block_name} "
                f"success_rate={metrics.get('success_rate')} "
                f"coverage={metrics.get('coverage')} "
                f"reward={metrics.get('reward')} "
                f"episode_length={metrics.get('episode_length')} "
                f"repeat_visit_ratio={metrics.get('repeat_visit_ratio')} "
                f"timeout_flag={metrics.get('timeout_flag')} "
                f"random_action_ratio={metrics.get('random_action_ratio')}"
            )
        return 0
    except Exception as exc:
        error_text = f"{type(exc).__name__}: {exc}"
        error_trace = traceback.format_exc()
        write_run_manifest(
            output_root,
            status="error",
            args=args,
            device_info=device_info,
            readiness=readiness,
            runtime_sec=time.perf_counter() - start,
            error=error_text,
            error_trace=error_trace,
        )
        print(f"[a_epsilon_final_probe] error: {error_text}", file=sys.stderr)
        return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the A-only epsilon-greedy final probe diagnostic.")
    parser.add_argument("--dry-run", action="store_true", help="Check artifacts and print the plan without running episodes.")
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES)
    parser.add_argument("--epsilon", type=float, default=DEFAULT_EPSILON)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--official-seed-base", type=int, default=DEFAULT_OFFICIAL_SEED_BASE)
    parser.add_argument("--train-seed-base", type=int, default=DEFAULT_TRAIN_SEED_BASE)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if int(args.episodes) <= 0:
        raise SystemExit("--episodes must be > 0")
    if not (0.0 <= float(args.epsilon) <= 1.0):
        raise SystemExit("--epsilon must be in [0, 1]")
    try:
        return run(args)
    except DiagnosticReadinessError as exc:
        print(f"[a_epsilon_final_probe] readiness error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
