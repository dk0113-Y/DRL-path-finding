from __future__ import annotations

import random
from collections import Counter, deque
from contextlib import contextmanager
from typing import Dict, Optional

import numpy as np

from agents.q_value_agent import StateTensorAdapter
from baselines.frontier_greedy_policy import FrontierGreedyBeliefView, FrontierGreedyPolicy
from env.agent_version import LocalObservationModel
from env.block_random_g import RandomMapGenerator, compute_map_fingerprint
from env.core_cummap import CumulativeBeliefMap
from env.core_radar import RadarSensor
from env.grid_topology import ACTIONS_8, GridTopology
from training.collector import (
    DERIVED_TRAIN_DIAGNOSTIC_FIELDS,
    SEMANTIC_EPISODE_FIELDS,
    derive_train_episode_diagnostics,
    summarize_semantic_records,
)
from training.evaluator import EvaluatorConfig
from training.rewarding import (
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


class FrontierGreedyBaselineEvaluator:
    """Final-probe evaluator for the no-training frontier greedy baseline."""

    def __init__(
        self,
        cfg: EvaluatorConfig,
        policy: FrontierGreedyPolicy,
        *,
        state_adapter: Optional[StateTensorAdapter] = None,
    ):
        self.cfg = cfg
        self.policy = policy
        self.state_adapter = state_adapter if state_adapter is not None else StateTensorAdapter(device="cpu")
        self.sensor = RadarSensor(scan_radius=int(cfg.scan_radius))
        self.reward_info_norm = fixed_half_perimeter_info_norm(int(cfg.scan_radius))
        self._recent_revisit_horizon = max(1, int(cfg.trajectory_history_steps))
        self._stall_diagnostic_window = int(STALL_DIAGNOSTIC_WINDOW)
        self.generator = RandomMapGenerator(
            rows=cfg.rows,
            cols=cfg.cols,
            obs_size=cfg.obs_size,
            obstacle_ratio=cfg.obstacle_ratio,
        )
        self.valid_action_indices: tuple[int, ...] = ()

    @staticmethod
    @contextmanager
    def _seeded_map_generation(seed: int | None):
        if seed is None:
            yield
            return
        py_state = random.getstate()
        np_state = np.random.get_state()
        random.seed(int(seed))
        np.random.seed(int(seed))
        try:
            yield
        finally:
            random.setstate(py_state)
            np.random.set_state(np_state)

    @staticmethod
    def _repeat_visit_ratio(cum_map) -> float:
        total_visits = int(np.sum(cum_map.visit_count))
        unique_visited = int(np.sum(cum_map.visit_count > 0))
        if total_visits <= 0:
            return 0.0
        repeat = max(0, total_visits - unique_visited)
        return float(repeat) / float(total_visits)

    def _refresh_valid_action_cache(self, valid_indices) -> None:
        self.valid_action_indices = tuple(int(v) for v in valid_indices)

    def _check_incremental_frontier_consistency(
        self,
        cum_map,
        *,
        context: str,
        episode_seed: int | None,
        episode_len: int,
    ) -> None:
        if not bool(self.cfg.debug_check_incremental_frontier):
            return
        stats = cum_map.debug_frontier_consistency_stats()
        if stats.consistent:
            return
        raise RuntimeError(
            "Incremental frontier cache mismatch in baseline evaluator: "
            f"context={context} "
            f"episode_seed={episode_seed!r} "
            f"episode_step={int(episode_len)} "
            f"mismatch_count={int(stats.mismatch_count)} "
            f"frontier_revision={int(stats.frontier_revision)} "
            f"frontier_source_uid={int(stats.frontier_source_uid)} "
            f"map_shape={tuple(stats.map_shape)}"
        )

    def _build_state_meta(self, cum_map, agent_state, shared_artifacts, recent_trajectory_positions):
        _, state_meta = self.state_adapter.build_single_state_tensors(
            cum_map,
            agent_state,
            recent_trajectory_positions=recent_trajectory_positions,
            shared_artifacts=shared_artifacts,
            target_device=None,
            return_state_meta=True,
        )
        return state_meta

    def _run_episode(self, episode_idx: int, episode_seed: int | None = None) -> Dict[str, object]:
        self.policy.reset_episode()
        with self._seeded_map_generation(episode_seed):
            grid, agent = self.generator.generate_map()
        map_fingerprint = compute_map_fingerprint(grid, agent)
        free_mask = GridTopology.free_mask(grid)
        obs_model = LocalObservationModel(grid, agent, sensor=self.sensor)
        local_snap = obs_model.local_snap
        self._refresh_valid_action_cache(GridTopology.valid_action_indices_fast(free_mask, agent))

        cum_map = CumulativeBeliefMap(grid, agent, local_snap)
        frontier_u8 = cum_map.get_frontier_u8(refresh=False)
        self._check_incremental_frontier_consistency(
            cum_map,
            context="baseline_reset",
            episode_seed=episode_seed,
            episode_len=0,
        )
        shared_artifacts = self.state_adapter.build_shared_step_artifacts(
            cum_map,
            agent,
            frontier_u8=frontier_u8,
        )

        episode_reward = 0.0
        episode_len = 0
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
        policy_mode_counts: Counter[str] = Counter()

        while True:
            valid_before = tuple(self.valid_action_indices)
            if len(valid_before) <= 0:
                raise RuntimeError(
                    "Encountered an empty valid-action set before baseline evaluation step. "
                    "This is treated as a defensive invariant violation."
                )

            state_meta = self._build_state_meta(
                cum_map,
                agent,
                shared_artifacts=shared_artifacts,
                recent_trajectory_positions=tuple(recent_trajectory_positions),
            )
            if isinstance(state_meta, dict):
                episode_semantic_records.append(
                    {field: float(state_meta.get(field, float("nan"))) for field in SEMANTIC_EPISODE_FIELDS}
                )

            belief_view = FrontierGreedyBeliefView.from_cumulative_map(
                cum_map,
                frontier_u8=frontier_u8,
            )
            action_idx = self.policy.select_action(
                belief=belief_view,
                agent_state=agent,
                valid_actions=valid_before,
                shared_semantic_snapshot=shared_artifacts.semantic_snapshot,
                recent_trajectory_positions=tuple(recent_trajectory_positions),
            )
            if action_idx not in valid_before:
                raise RuntimeError(
                    f"Selected baseline action {action_idx} outside the valid-action set {sorted(valid_before)}."
                )
            if self.policy.last_decision is not None:
                policy_mode_counts[str(self.policy.last_decision.mode)] += 1

            step_breakdown = zero_reward_breakdown()
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

            local_snap = obs_model.observe_fast(agent)
            self._refresh_valid_action_cache(GridTopology.valid_action_indices_fast(free_mask, agent))
            updated, delta_empty, delta_obstacle = cum_map.update(agent, local_snap)
            if int(updated) != int(delta_empty + delta_obstacle):
                raise RuntimeError("belief-map update returned inconsistent information-gain counts")
            frontier_u8 = cum_map.get_frontier_u8(refresh=False)
            self._check_incremental_frontier_consistency(
                cum_map,
                context="baseline_step_post_update",
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
            episode_event_summary["weighted_obstacle_info_gain_sum"] += float(
                info_metrics["weighted_obstacle_info_gain_sum"]
            )
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
                    "Encountered an empty valid-action set after a valid baseline move "
                    "without reaching coverage target."
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
                event_summary = finalize_reward_event_summary(dict(episode_event_summary))
                event_summary["timeout_flag"] = float(done_reason == "max_episode_steps")
                row = {
                    "episode_idx": int(episode_idx),
                    "episode_seed": None if episode_seed is None else int(episode_seed),
                    "map_fingerprint": str(map_fingerprint),
                    "episode_reward": float(episode_reward),
                    "episode_length": int(episode_len),
                    "final_coverage": float(cum_map.coverage_rate),
                    "success": int(done_reason == "coverage_reached"),
                    "repeat_visit_ratio": float(self._repeat_visit_ratio(cum_map)),
                    "done_reason": str(done_reason),
                    "policy_frontier_target_step_count": float(policy_mode_counts.get("frontier_target", 0)),
                    "policy_fallback_step_count": float(policy_mode_counts.get("fallback_information_gain", 0)),
                    **summarize_semantic_records(episode_semantic_records),
                    **{field: float(episode_breakdown[field]) for field in REWARD_BREAKDOWN_FIELDS},
                    **{field: float(event_summary[field]) for field in REWARD_EVENT_SUMMARY_FIELDS},
                }
                row.update(derive_train_episode_diagnostics(row))
                return row

    def evaluate(self, num_episodes: int = 5, seed_base: int | None = None) -> Dict[str, object]:
        if int(num_episodes) <= 0:
            raise ValueError("num_episodes must be > 0")

        episodes = [
            self._run_episode(
                episode_idx=idx + 1,
                episode_seed=(None if seed_base is None else int(seed_base) + idx),
            )
            for idx in range(int(num_episodes))
        ]

        result: Dict[str, object] = {
            "eval_episodes": int(num_episodes),
            "episodes": episodes,
            "policy_decision_counts": {
                "frontier_target": int(sum(float(ep.get("policy_frontier_target_step_count", 0.0)) for ep in episodes)),
                "fallback_information_gain": int(sum(float(ep.get("policy_fallback_step_count", 0.0)) for ep in episodes)),
            },
        }
        mean_specs = {
            "eval_mean_reward": "episode_reward",
            "eval_mean_coverage": "final_coverage",
            "eval_success_rate": "success",
            "eval_mean_episode_length": "episode_length",
            "eval_mean_repeat_visit_ratio": "repeat_visit_ratio",
        }
        for out_name, episode_field in mean_specs.items():
            values = np.asarray([float(ep[episode_field]) for ep in episodes], dtype=np.float32)
            result[out_name] = float(np.mean(values))
        for field in SEMANTIC_EPISODE_FIELDS:
            result[f"eval_mean_{field}"] = _nanmean([float(ep.get(field, float("nan"))) for ep in episodes])
        for field in REWARD_BREAKDOWN_FIELDS:
            result[f"eval_mean_{field}"] = _nanmean([float(ep.get(field, float("nan"))) for ep in episodes])
        for field in REWARD_EVENT_SUMMARY_FIELDS:
            result[f"eval_mean_{field}"] = _nanmean([float(ep.get(field, float("nan"))) for ep in episodes])
        for field in DERIVED_TRAIN_DIAGNOSTIC_FIELDS:
            result[f"eval_mean_{field}"] = _nanmean([float(ep.get(field, float("nan"))) for ep in episodes])
        return result


def _nanmean(values: list[float]) -> float:
    arr = np.asarray(values, dtype=np.float32)
    if arr.size <= 0 or not np.any(np.isfinite(arr)):
        return float("nan")
    return float(np.nanmean(arr))
