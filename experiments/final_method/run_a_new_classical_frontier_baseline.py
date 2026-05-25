from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
import time
from collections import Counter, deque
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import train_q_agent  # noqa: E402
from env.agent_version import LocalObservationModel  # noqa: E402
from env.block_random_g import RandomMapGenerator, compute_map_fingerprint  # noqa: E402
from env.core_cummap import CumulativeBeliefMap  # noqa: E402
from env.core_radar import RadarSensor  # noqa: E402
from env.grid_topology import ACTIONS_8, GridTopology  # noqa: E402
from env.shared_semantic_layer import SharedSemanticLayer  # noqa: E402
from env.value_state_builder import ValueStateBuilder  # noqa: E402
from experiments.final_method.a_new_classical_frontier_greedy_policy import (  # noqa: E402
    ClassicalFrontierGreedyPolicy,
)
from training.collector import (  # noqa: E402
    DERIVED_TRAIN_DIAGNOSTIC_FIELDS,
    SEMANTIC_EPISODE_FIELDS,
    derive_train_episode_diagnostics,
    summarize_semantic_records,
)
from training.rewarding import (  # noqa: E402
    REWARD_BREAKDOWN_FIELDS,
    REWARD_EVENT_SUMMARY_FIELDS,
    STALL_DIAGNOSTIC_WINDOW,
    add_reward_breakdown,
    finalize_reward_event_summary,
    fixed_half_perimeter_info_norm,
    info_gain_components,
    reward_from_breakdown,
    timeout_penalty_breakdown,
    turn_penalty_weight_from_steps,
    valid_step_reward_breakdown,
    zero_reward_breakdown,
    zero_reward_event_summary,
)


EXPERIMENT_ID = "Anew_B"
METHOD_ID = "Anew_B_classical_frontier_greedy"
METHOD_NAME = "classical_frontier_greedy"
BASELINE_GROUP = "classical"
BASELINE_TYPE = "traditional_non_learning"
RUNNER_ENTRYPOINT = "experiments/final_method/run_a_new_classical_frontier_baseline.py"
DEFAULT_RUN_NAME = METHOD_ID
STAGE_EPISODES = {"smoke": 2, "pilot": 10, "formal": 100}
PLANNED_ARTIFACTS = (
    "logs/config_snapshot.json",
    "logs/baseline_manifest.json",
    "logs/baseline_policy_summary.json",
    "logs/benchmark_summary.json",
    "logs/metric_snapshot.json",
    "logs/reproducibility_contract.json",
    "logs/artifact_index.json",
    "logs/final_probe.csv",
    "logs/baseline_summary.txt",
)


@dataclass(frozen=True)
class ClassicalBaselineConfig:
    run_stage: str
    device: str
    output_root: str
    run_name: str
    episodes: int
    seed_base: int
    rows: int
    cols: int
    obs_size: int
    scan_radius: int
    obstacle_ratio: float
    max_episode_steps: int
    coverage_stop_threshold: float
    trajectory_history_steps: int
    reward_info_scale: float
    reward_obstacle_weight: float
    reward_step_penalty: float
    reward_terminal_bonus: float
    reward_revisit_penalty: float
    reward_turn_penalty_scale: float
    reward_turn_weight_45: float
    reward_turn_weight_90: float
    reward_turn_weight_135: float
    reward_turn_weight_180: float
    reward_timeout_penalty: float
    debug_check_incremental_frontier: bool = False


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        number = float(value)
        return None if math.isnan(number) or math.isinf(number) else number
    if isinstance(value, float):
        return None if math.isnan(value) or math.isinf(value) else value
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(dict(payload)), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _git_output(repo_dir: Path, args: list[str]) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=str(repo_dir),
            text=True,
            capture_output=True,
            check=True,
        )
    except Exception:
        return None
    text = result.stdout.strip()
    return text or None


def _command_text(command: list[str]) -> str:
    try:
        return subprocess.list2cmdline(command)
    except Exception:
        return " ".join(str(item) for item in command)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _environment_config(cfg: ClassicalBaselineConfig) -> dict[str, Any]:
    return {
        "rows": int(cfg.rows),
        "cols": int(cfg.cols),
        "obs_size": int(cfg.obs_size),
        "scan_radius": int(cfg.scan_radius),
        "obstacle_ratio": float(cfg.obstacle_ratio),
        "max_episode_steps": int(cfg.max_episode_steps),
        "coverage_stop_threshold": float(cfg.coverage_stop_threshold),
        "trajectory_history_steps": int(cfg.trajectory_history_steps),
    }


def _reward_config(cfg: ClassicalBaselineConfig) -> dict[str, Any]:
    return {
        "reward_info_scale": float(cfg.reward_info_scale),
        "reward_obstacle_weight": float(cfg.reward_obstacle_weight),
        "reward_step_penalty": float(cfg.reward_step_penalty),
        "reward_terminal_bonus": float(cfg.reward_terminal_bonus),
        "reward_revisit_penalty": float(cfg.reward_revisit_penalty),
        "reward_turn_penalty_scale": float(cfg.reward_turn_penalty_scale),
        "reward_turn_weight_45": float(cfg.reward_turn_weight_45),
        "reward_turn_weight_90": float(cfg.reward_turn_weight_90),
        "reward_turn_weight_135": float(cfg.reward_turn_weight_135),
        "reward_turn_weight_180": float(cfg.reward_turn_weight_180),
        "reward_timeout_penalty": float(cfg.reward_timeout_penalty),
    }


def build_config(
    *,
    run_stage: str,
    device: str,
    output_root: str,
    run_name: str,
    episodes: int | None,
    seed_base: int | None,
) -> ClassicalBaselineConfig:
    reference = train_q_agent.TrainConfig()
    resolved_stage = str(run_stage)
    if resolved_stage not in STAGE_EPISODES:
        raise ValueError(f"Unsupported run_stage: {run_stage!r}")
    resolved_episodes = int(episodes) if episodes is not None else int(STAGE_EPISODES[resolved_stage])
    if resolved_episodes <= 0:
        raise ValueError("episodes must be > 0")
    resolved_seed_base = (
        int(seed_base)
        if seed_base is not None
        else int(reference.fixed_final_probe_seed_base)
    )
    return ClassicalBaselineConfig(
        run_stage=resolved_stage,
        device=str(device),
        output_root=str(output_root),
        run_name=str(run_name or DEFAULT_RUN_NAME),
        episodes=resolved_episodes,
        seed_base=resolved_seed_base,
        rows=int(reference.rows),
        cols=int(reference.cols),
        obs_size=int(reference.obs_size),
        scan_radius=int(reference.scan_radius),
        obstacle_ratio=float(reference.obstacle_ratio),
        max_episode_steps=int(reference.max_episode_steps),
        coverage_stop_threshold=float(reference.coverage_stop_threshold),
        trajectory_history_steps=int(reference.trajectory_history_steps),
        reward_info_scale=float(reference.reward_info_scale),
        reward_obstacle_weight=float(reference.reward_obstacle_weight),
        reward_step_penalty=float(reference.reward_step_penalty),
        reward_terminal_bonus=float(reference.reward_terminal_bonus),
        reward_revisit_penalty=float(reference.reward_revisit_penalty),
        reward_turn_penalty_scale=float(reference.reward_turn_penalty_scale),
        reward_turn_weight_45=float(reference.reward_turn_weight_45),
        reward_turn_weight_90=float(reference.reward_turn_weight_90),
        reward_turn_weight_135=float(reference.reward_turn_weight_135),
        reward_turn_weight_180=float(reference.reward_turn_weight_180),
        reward_timeout_penalty=float(reference.reward_timeout_penalty),
        debug_check_incremental_frontier=bool(reference.debug_check_incremental_frontier),
    )


def _base_manifest(cfg: ClassicalBaselineConfig, policy_summary: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "a_new_classical_frontier_baseline_manifest/v1",
        "experiment_id": EXPERIMENT_ID,
        "method_id": METHOD_ID,
        "method_name": METHOD_NAME,
        "baseline_id": METHOD_ID,
        "baseline_group": BASELINE_GROUP,
        "baseline_name": METHOD_NAME,
        "baseline_type": BASELINE_TYPE,
        "is_learning_baseline": False,
        "is_ablation": False,
        "no_ground_truth_map_for_decision": True,
        "advantage_canvas_schema": "not_applicable",
        "frontier_raster_used": False,
        "value_tree_enabled": "not_applicable",
        "value_tree_used_for_decision": False,
        "reward_override": {},
        "checkpoint_used": False,
        "trainable_parameters": 0,
        "model_class": "not_applicable",
        "uses_exploration_q_network": False,
        "run_stage": cfg.run_stage,
        "episodes": int(cfg.episodes),
        "seed_base": int(cfg.seed_base),
        "environment_config": _environment_config(cfg),
        "reward_config": _reward_config(cfg),
        "policy_summary": dict(policy_summary),
        "runner_entrypoint": RUNNER_ENTRYPOINT,
        "final_probe_table_note": (
            "logs/final_probe.csv is a non-learning baseline benchmark episode table, "
            "not a neural model final probe."
        ),
        "legacy_inheritance": "none",
        "restores_legacy_baselines": False,
    }


def build_dry_run_payload(cfg: ClassicalBaselineConfig, command: list[str]) -> dict[str, Any]:
    policy_summary = ClassicalFrontierGreedyPolicy().policy_summary()
    manifest = _base_manifest(cfg, policy_summary)
    payload = {
        **manifest,
        "dry_run": True,
        "command": command,
        "command_text": _command_text(command),
        "planned_artifacts": list(PLANNED_ARTIFACTS),
        "baseline_config": asdict(cfg),
        "reward_info_scale": float(cfg.reward_info_scale),
        "reward_obstacle_weight": float(cfg.reward_obstacle_weight),
        "max_episode_steps": int(cfg.max_episode_steps),
        "coverage_stop_threshold": float(cfg.coverage_stop_threshold),
    }
    return payload


class ClassicalFrontierBenchmark:
    def __init__(self, cfg: ClassicalBaselineConfig):
        self.cfg = cfg
        self.policy = ClassicalFrontierGreedyPolicy()
        self.sensor = RadarSensor(scan_radius=int(cfg.scan_radius))
        self.generator = RandomMapGenerator(
            rows=int(cfg.rows),
            cols=int(cfg.cols),
            obs_size=int(cfg.obs_size),
            obstacle_ratio=float(cfg.obstacle_ratio),
        )
        self.shared_semantics = SharedSemanticLayer()
        self.value_state_builder = ValueStateBuilder()
        self.reward_info_norm = fixed_half_perimeter_info_norm(int(cfg.scan_radius))
        self._recent_revisit_horizon = max(1, int(cfg.trajectory_history_steps))
        self._stall_diagnostic_window = int(STALL_DIAGNOSTIC_WINDOW)

    @staticmethod
    def _repeat_visit_ratio(cum_map) -> float:
        total_visits = int(np.sum(cum_map.visit_count))
        unique_visited = int(np.sum(cum_map.visit_count > 0))
        if total_visits <= 0:
            return 0.0
        return float(max(0, total_visits - unique_visited)) / float(total_visits)

    def _semantic_record(self, shared_snapshot) -> dict[str, float]:
        record = dict(shared_snapshot.metrics())
        try:
            _, _, _, _, value_meta = self.value_state_builder.build(shared_snapshot)
            record.update(value_meta)
        except Exception:
            pass
        return {field: float(record.get(field, float("nan"))) for field in SEMANTIC_EPISODE_FIELDS}

    def run_episode(self, *, episode_index: int, seed: int) -> dict[str, Any]:
        grid, agent = self.generator.generate_map(seed=int(seed))
        map_fingerprint = compute_map_fingerprint(grid, agent)
        traversable = GridTopology.free_mask(grid)
        obs_model = LocalObservationModel(grid, agent, sensor=self.sensor)
        local_snap = obs_model.local_snap
        cum_map = CumulativeBeliefMap(grid, agent, local_snap)
        shared_snapshot = self.shared_semantics.analyze(cum_map, agent)

        episode_reward = 0.0
        episode_len = 0
        episode_breakdown = zero_reward_breakdown()
        episode_event_summary = zero_reward_event_summary()
        episode_semantic_records: list[dict[str, float]] = []
        recent_positions: deque[tuple[int, int]] = deque(
            [(int(agent[0]), int(agent[1]))],
            maxlen=self._recent_revisit_horizon,
        )
        stall_streak = 0
        prev_action_idx: int | None = None
        decision_counts: Counter[str] = Counter()
        invalid_action_count = 0
        collision_count = 0
        done_reason = ""

        while True:
            valid_before = GridTopology.valid_action_indices_fast(traversable, agent)
            episode_semantic_records.append(self._semantic_record(shared_snapshot))
            if len(valid_before) <= 0:
                decision_counts["no_valid_action"] += 1
                done_reason = "no_valid_actions"
                break

            agent_array_rc = cum_map.world_to_array(agent)
            decision = self.policy.decide(
                belief_map=cum_map.map,
                agent_array_rc=agent_array_rc,
                valid_action_indices=valid_before,
                semantic_snapshot=shared_snapshot,
                scan_radius=int(self.cfg.scan_radius),
            )
            decision_counts[str(decision.decision_mode)] += 1
            action_idx = decision.action_idx
            if action_idx is None:
                action_idx = int(valid_before[0])
                decision_counts["runner_safe_action_patch"] += 1
            if int(action_idx) not in valid_before:
                invalid_action_count += 1
                action_idx = int(valid_before[0])
                decision_counts["runner_invalid_action_patch"] += 1

            turn_steps = GridTopology.circular_turn_steps(prev_action_idx, int(action_idx))
            turn_penalty_weight = float(
                turn_penalty_weight_from_steps(
                    turn_steps,
                    weight_45=float(self.cfg.reward_turn_weight_45),
                    weight_90=float(self.cfg.reward_turn_weight_90),
                    weight_135=float(self.cfg.reward_turn_weight_135),
                    weight_180=float(self.cfg.reward_turn_weight_180),
                )
            )
            dr, dc = ACTIONS_8[int(action_idx)]
            agent = (int(agent[0] + dr), int(agent[1] + dc))

            local_snap = obs_model.observe_fast(agent)
            updated, delta_empty, delta_obstacle = cum_map.update(agent, local_snap)
            if int(updated) != int(delta_empty + delta_obstacle):
                raise RuntimeError("belief-map update returned inconsistent information-gain counts")
            shared_snapshot = self.shared_semantics.analyze(cum_map, agent)

            recent_revisit = bool((int(agent[0]), int(agent[1])) in recent_positions)
            recent_positions.append((int(agent[0]), int(agent[1])))
            if int(delta_empty) == 0 and int(delta_obstacle) == 0:
                stall_streak += 1
            else:
                stall_streak = 0
            stall_triggered = bool(stall_streak >= self._stall_diagnostic_window)
            info_metrics = info_gain_components(
                delta_empty=int(delta_empty),
                delta_obstacle=int(delta_obstacle),
                obstacle_weight=float(self.cfg.reward_obstacle_weight),
                info_norm=float(self.reward_info_norm),
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
            step_breakdown = valid_step_reward_breakdown(
                self.cfg,
                delta_empty=int(delta_empty),
                delta_obstacle=int(delta_obstacle),
                info_norm=float(self.reward_info_norm),
                recent_revisit=recent_revisit,
                turn_penalty_weight=turn_penalty_weight,
                success=success,
            )
            reward = reward_from_breakdown(step_breakdown)
            prev_action_idx = int(action_idx)
            episode_len += 1

            if success:
                done_reason = "coverage_reached"
            elif episode_len >= int(self.cfg.max_episode_steps):
                done_reason = "max_episode_steps"
                timeout_breakdown = timeout_penalty_breakdown(self.cfg)
                add_reward_breakdown(step_breakdown, timeout_breakdown)
                reward += reward_from_breakdown(timeout_breakdown)
            elif len(GridTopology.valid_action_indices_fast(traversable, agent)) <= 0:
                done_reason = "no_valid_actions"

            episode_reward += float(reward)
            add_reward_breakdown(episode_breakdown, step_breakdown)

            if done_reason:
                break

        episode_event_summary = finalize_reward_event_summary(dict(episode_event_summary))
        episode_event_summary["timeout_flag"] = float(done_reason == "max_episode_steps")
        result = {
            "episode_index": int(episode_index),
            "seed": int(seed),
            "episode_seed": int(seed),
            "map_fingerprint": str(map_fingerprint),
            "episode_reward": float(episode_reward),
            "episode_length": int(episode_len),
            "final_coverage": float(cum_map.coverage_rate),
            "success": int(done_reason == "coverage_reached"),
            "repeat_visit_ratio": float(self._repeat_visit_ratio(cum_map)),
            "done_reason": str(done_reason),
            "invalid_action_count": int(invalid_action_count),
            "collision_count": int(collision_count),
            "policy_frontier_decision_count": int(decision_counts.get("frontier_greedy", 0)),
            "policy_fallback_info_gain_count": int(decision_counts.get("immediate_info_gain", 0)),
            "policy_safe_fallback_count": int(decision_counts.get("safe_fallback", 0)),
            "policy_no_valid_action_count": int(decision_counts.get("no_valid_action", 0)),
            **summarize_semantic_records(episode_semantic_records),
            **{field: float(episode_breakdown[field]) for field in REWARD_BREAKDOWN_FIELDS},
            **{field: float(episode_event_summary[field]) for field in REWARD_EVENT_SUMMARY_FIELDS},
        }
        result.update(derive_train_episode_diagnostics(result))
        return result

    def run(self) -> list[dict[str, Any]]:
        return [
            self.run_episode(
                episode_index=idx,
                seed=int(self.cfg.seed_base) + idx,
            )
            for idx in range(int(self.cfg.episodes))
        ]


def _mean_field(rows: list[Mapping[str, Any]], field: str) -> float | None:
    values: list[float] = []
    for row in rows:
        try:
            value = float(row.get(field, float("nan")))
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            values.append(value)
    if not values:
        return None
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def build_metric_snapshot(cfg: ClassicalBaselineConfig, episodes: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = {
        "reward": _mean_field(episodes, "episode_reward"),
        "coverage": _mean_field(episodes, "final_coverage"),
        "success_rate": _mean_field(episodes, "success"),
        "episode_length": _mean_field(episodes, "episode_length"),
        "repeat_visit_ratio": _mean_field(episodes, "repeat_visit_ratio"),
    }
    reward_breakdown = {field: _mean_field(episodes, field) for field in REWARD_BREAKDOWN_FIELDS}
    reward_events = {field: _mean_field(episodes, field) for field in REWARD_EVENT_SUMMARY_FIELDS}
    semantic_monitoring = {
        field: _mean_field(episodes, field)
        for field in (*SEMANTIC_EPISODE_FIELDS, *DERIVED_TRAIN_DIAGNOSTIC_FIELDS)
    }
    return {
        "schema_version": "a_new_classical_frontier_metric_snapshot/v1",
        "experiment_id": EXPERIMENT_ID,
        "method_id": METHOD_ID,
        "run_stage": cfg.run_stage,
        "episodes": int(cfg.episodes),
        "seed_base": int(cfg.seed_base),
        "source": "logs/final_probe.csv",
        "source_note": "non-learning baseline benchmark episode table",
        "metrics": metrics,
        "reward_breakdown": reward_breakdown,
        "reward_events": reward_events,
        "semantic_monitoring": semantic_monitoring,
    }


def build_benchmark_summary(
    *,
    cfg: ClassicalBaselineConfig,
    run_dir: Path,
    episodes: list[dict[str, Any]],
    total_runtime_sec: float,
) -> dict[str, Any]:
    metric_snapshot = build_metric_snapshot(cfg, episodes)
    return {
        "schema_version": "a_new_classical_frontier_benchmark_summary/v1",
        "experiment_id": EXPERIMENT_ID,
        "method_id": METHOD_ID,
        "method_name": METHOD_NAME,
        "baseline_type": BASELINE_TYPE,
        "run_stage": cfg.run_stage,
        "run_dir": str(run_dir.resolve()),
        "episodes": int(cfg.episodes),
        "seed_base": int(cfg.seed_base),
        "total_runtime_sec": float(total_runtime_sec),
        "checkpoint_used": False,
        "trainable_parameters": 0,
        "metrics": metric_snapshot["metrics"],
        "reward_breakdown": metric_snapshot["reward_breakdown"],
        "reward_events": metric_snapshot["reward_events"],
    }


def _csv_fieldnames(episodes: list[dict[str, Any]]) -> list[str]:
    preferred = [
        "episode_index",
        "seed",
        "episode_seed",
        "map_fingerprint",
        "episode_reward",
        "episode_length",
        "final_coverage",
        "success",
        "repeat_visit_ratio",
        "done_reason",
        "timeout_flag",
        "weighted_info_gain_sum",
        "invalid_action_count",
        "collision_count",
        "policy_frontier_decision_count",
        "policy_fallback_info_gain_count",
        "policy_safe_fallback_count",
    ]
    remaining = sorted({key for row in episodes for key in row.keys()} - set(preferred))
    return [field for field in preferred if any(field in row for row in episodes)] + remaining


def _write_episode_csv(path: Path, episodes: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = _csv_fieldnames(episodes)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in episodes:
            writer.writerow({field: _json_safe(row.get(field)) for field in fieldnames})


def _artifact_index(run_dir: Path) -> dict[str, Any]:
    artifacts = []
    for rel_path in PLANNED_ARTIFACTS:
        path = run_dir / rel_path
        artifacts.append(
            {
                "path": rel_path,
                "required": True,
                "exists": bool(path.exists()) or rel_path == "logs/artifact_index.json",
                "category": "a_new_classical_frontier_baseline",
            }
        )
    return {
        "schema_version": "a_new_classical_frontier_artifact_index/v1",
        "experiment_id": EXPERIMENT_ID,
        "method_id": METHOD_ID,
        "artifacts": artifacts,
    }


def _baseline_summary_text(
    *,
    run_dir: Path,
    cfg: ClassicalBaselineConfig,
    benchmark_summary: Mapping[str, Any],
) -> str:
    metrics = benchmark_summary.get("metrics", {})
    return "\n".join(
        [
            "A_new Classical Frontier Baseline Summary",
            f"run_dir: {run_dir.resolve()}",
            f"method_id: {METHOD_ID}",
            f"run_stage: {cfg.run_stage}",
            f"episodes: {cfg.episodes}",
            f"seed_base: {cfg.seed_base}",
            f"mean_reward: {metrics.get('reward')}",
            f"mean_coverage: {metrics.get('coverage')}",
            f"success_rate: {metrics.get('success_rate')}",
            f"mean_episode_length: {metrics.get('episode_length')}",
            f"mean_repeat_visit_ratio: {metrics.get('repeat_visit_ratio')}",
            "checkpoint_used: false",
            "trainable_parameters: 0",
            "final_probe_note: non-learning baseline benchmark table, not neural final probe",
        ]
    ) + "\n"


def create_run_dir(cfg: ClassicalBaselineConfig) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(cfg.output_root) / f"{cfg.run_name}_{cfg.run_stage}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)
    return run_dir


def write_artifacts(
    *,
    cfg: ClassicalBaselineConfig,
    run_dir: Path,
    episodes: list[dict[str, Any]],
    total_runtime_sec: float,
    raw_argv: list[str],
) -> dict[str, Path]:
    repo_dir = _repo_root()
    logs_dir = run_dir / "logs"
    policy_summary = ClassicalFrontierGreedyPolicy().policy_summary()
    manifest = _base_manifest(cfg, policy_summary)
    manifest.update(
        {
            "generated_at": _now_iso(),
            "source_commit": _git_output(repo_dir, ["rev-parse", "HEAD"]),
            "source_branch": _git_output(repo_dir, ["branch", "--show-current"]),
            "source_remote": _git_output(repo_dir, ["remote", "get-url", "origin"]),
        }
    )
    config_snapshot = {
        "schema_version": "a_new_classical_frontier_config_snapshot/v1",
        "experiment_id": EXPERIMENT_ID,
        "method_id": METHOD_ID,
        "baseline_config": asdict(cfg),
        "environment_config": _environment_config(cfg),
        "reward_config": _reward_config(cfg),
        "policy_summary": policy_summary,
        "reward_override": {},
    }
    metric_snapshot = build_metric_snapshot(cfg, episodes)
    benchmark_summary = build_benchmark_summary(
        cfg=cfg,
        run_dir=run_dir,
        episodes=episodes,
        total_runtime_sec=total_runtime_sec,
    )
    reproducibility_contract = {
        "schema_version": "a_new_classical_frontier_reproducibility_contract/v1",
        "experiment_id": EXPERIMENT_ID,
        "method_id": METHOD_ID,
        "runner_entrypoint": RUNNER_ENTRYPOINT,
        "raw_argv": list(raw_argv),
        "launch_command": _command_text([sys.executable, *raw_argv]),
        "source_commit": manifest.get("source_commit"),
        "source_branch": manifest.get("source_branch"),
        "source_remote": manifest.get("source_remote"),
        "seed_base": int(cfg.seed_base),
        "episodes": int(cfg.episodes),
        "checkpoint_used": False,
        "trainable_parameters": 0,
    }
    paths = {
        "config_snapshot": logs_dir / "config_snapshot.json",
        "baseline_manifest": logs_dir / "baseline_manifest.json",
        "baseline_policy_summary": logs_dir / "baseline_policy_summary.json",
        "benchmark_summary": logs_dir / "benchmark_summary.json",
        "metric_snapshot": logs_dir / "metric_snapshot.json",
        "reproducibility_contract": logs_dir / "reproducibility_contract.json",
        "artifact_index": logs_dir / "artifact_index.json",
        "final_probe": logs_dir / "final_probe.csv",
        "baseline_summary": logs_dir / "baseline_summary.txt",
    }
    _write_episode_csv(paths["final_probe"], episodes)
    _write_json(paths["config_snapshot"], config_snapshot)
    _write_json(paths["baseline_manifest"], manifest)
    _write_json(paths["baseline_policy_summary"], policy_summary)
    _write_json(paths["benchmark_summary"], benchmark_summary)
    _write_json(paths["metric_snapshot"], metric_snapshot)
    _write_json(paths["reproducibility_contract"], reproducibility_contract)
    paths["baseline_summary"].write_text(
        _baseline_summary_text(run_dir=run_dir, cfg=cfg, benchmark_summary=benchmark_summary),
        encoding="utf-8",
    )
    _write_json(paths["artifact_index"], _artifact_index(run_dir))
    return paths


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the A_new-aligned classical frontier greedy baseline.")
    parser.add_argument("--run-stage", choices=("smoke", "pilot", "formal"), default="smoke")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--seed-base", type=int, default=None)
    parser.add_argument("--run-name", default=DEFAULT_RUN_NAME)
    return parser.parse_args(argv)


def build_command(args: argparse.Namespace) -> list[str]:
    command = [
        sys.executable,
        RUNNER_ENTRYPOINT,
        "--run-stage",
        str(args.run_stage),
        "--device",
        str(args.device),
        "--output-root",
        str(args.output_root),
        "--run-name",
        str(args.run_name),
    ]
    if args.episodes is not None:
        command.extend(["--episodes", str(args.episodes)])
    if args.seed_base is not None:
        command.extend(["--seed-base", str(args.seed_base)])
    if bool(args.dry_run):
        command.append("--dry-run")
    return command


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = build_config(
        run_stage=args.run_stage,
        device=args.device,
        output_root=args.output_root,
        run_name=args.run_name,
        episodes=args.episodes,
        seed_base=args.seed_base,
    )
    command = build_command(args)
    print(f"[A_new_B] command: {_command_text(command)}", flush=True)
    if args.dry_run:
        print(json.dumps(build_dry_run_payload(cfg, command), indent=2, ensure_ascii=False))
        return 0

    print("[A_new_B] runtime_contract:")
    print(json.dumps(_base_manifest(cfg, ClassicalFrontierGreedyPolicy().policy_summary()), indent=2, ensure_ascii=False))
    run_dir = create_run_dir(cfg)
    start = time.perf_counter()
    episodes = ClassicalFrontierBenchmark(cfg).run()
    runtime_sec = float(time.perf_counter() - start)
    paths = write_artifacts(
        cfg=cfg,
        run_dir=run_dir,
        episodes=episodes,
        total_runtime_sec=runtime_sec,
        raw_argv=[RUNNER_ENTRYPOINT, *(argv if argv is not None else sys.argv[1:])],
    )
    print(f"baseline_run_dir: {run_dir}")
    print(f"baseline_manifest_json: {paths['baseline_manifest']}")
    print(f"baseline_final_probe_csv: {paths['final_probe']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
