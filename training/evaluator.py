from __future__ import annotations

import random
from collections import deque
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import torch

from agents.q_value_agent import select_greedy_action
from env.agent_version import LocalObservationModel
from env.block_random_g import RandomMapGenerator
from env.core_cummap import CumulativeBeliefMap
from env.core_radar import RadarSensor
from env.grid_topology import ACTIONS_8, GridTopology
from env.shared_semantic_layer import build_semantic_visualization_payload
from training.collector import CollectorConfig, SEMANTIC_EPISODE_FIELDS, summarize_semantic_records
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


@dataclass(frozen=True)
class EvaluatorConfig:
    rows: int = 40
    cols: int = 60
    obs_size: int = 6
    scan_radius: int = 10
    obstacle_ratio: float = 0.20

    max_episode_steps: int = 300
    coverage_stop_threshold: float = 0.98
    trajectory_history_steps: int = 10

    reward_info_scale: float = 10.0
    reward_obstacle_weight: float = 0.25
    reward_step_penalty: float = 0.01
    reward_terminal_bonus: float = 0.5
    reward_revisit_penalty: float = 0.05
    reward_turn_penalty_scale: float = 0.0
    reward_turn_weight_45: float = 0.0
    reward_turn_weight_90: float = 1.0 / 3.0
    reward_turn_weight_135: float = 2.0 / 3.0
    reward_turn_weight_180: float = 1.0
    reward_timeout_penalty: float = 0.5
    enable_inference_amp: bool = False
    inference_amp_dtype: str = "fp16"
    debug_check_incremental_frontier: bool = False


ACTION_DIM = len(ACTIONS_8)


class GreedyEvaluator:
    """Greedy held-out evaluator for the final probe on the last checkpoint / online last network."""

    def __init__(self, cfg: EvaluatorConfig, state_adapter, device: str = "cpu"):
        self.cfg = cfg
        self.state_adapter = state_adapter
        self.device = torch.device(device)
        self._cpu_device = torch.device("cpu")
        self._policy_device = torch.device(device)
        self._pin_cpu_action_mask = bool(torch.cuda.is_available()) and self._policy_device.type == "cuda"
        self._enable_inference_amp = (
            bool(cfg.enable_inference_amp)
            and bool(torch.cuda.is_available())
            and self._policy_device.type == "cuda"
        )
        self._inference_amp_dtype = self._resolve_amp_dtype(cfg.inference_amp_dtype)

        self.sensor = RadarSensor(scan_radius=int(cfg.scan_radius))
        self.reward_info_norm = fixed_half_perimeter_info_norm(int(cfg.scan_radius))
        # Keep revisit-penalty horizon aligned with the explicit recent-trajectory branch.
        self._recent_revisit_horizon = max(1, int(cfg.trajectory_history_steps))
        # Diagnostic-only threshold; stall events no longer contribute to formal reward.
        self._stall_diagnostic_window = int(STALL_DIAGNOSTIC_WINDOW)
        self.generator = RandomMapGenerator(
            rows=cfg.rows,
            cols=cfg.cols,
            obs_size=cfg.obs_size,
            obstacle_ratio=cfg.obstacle_ratio,
        )
        self.valid_action_indices: tuple[int, ...] = ()
        self._valid_action_list: list[int] = []
        self._current_action_mask_cpu: Optional[torch.Tensor] = None
        self._current_action_mask_policy: Optional[torch.Tensor] = None

    @staticmethod
    def from_collector_config(cfg: CollectorConfig, state_adapter, device: str = "cpu") -> "GreedyEvaluator":
        e_cfg = EvaluatorConfig(
            rows=cfg.rows,
            cols=cfg.cols,
            obs_size=cfg.obs_size,
            scan_radius=cfg.scan_radius,
            obstacle_ratio=cfg.obstacle_ratio,
            max_episode_steps=cfg.max_episode_steps,
            coverage_stop_threshold=cfg.coverage_stop_threshold,
            trajectory_history_steps=cfg.trajectory_history_steps,
            reward_info_scale=cfg.reward_info_scale,
            reward_obstacle_weight=cfg.reward_obstacle_weight,
            reward_step_penalty=cfg.reward_step_penalty,
            reward_terminal_bonus=cfg.reward_terminal_bonus,
            reward_revisit_penalty=cfg.reward_revisit_penalty,
            reward_turn_penalty_scale=cfg.reward_turn_penalty_scale,
            reward_turn_weight_45=cfg.reward_turn_weight_45,
            reward_turn_weight_90=cfg.reward_turn_weight_90,
            reward_turn_weight_135=cfg.reward_turn_weight_135,
            reward_turn_weight_180=cfg.reward_turn_weight_180,
            reward_timeout_penalty=cfg.reward_timeout_penalty,
            enable_inference_amp=cfg.enable_inference_amp,
            inference_amp_dtype=cfg.inference_amp_dtype,
            debug_check_incremental_frontier=cfg.debug_check_incremental_frontier,
        )
        return GreedyEvaluator(e_cfg, state_adapter=state_adapter, device=device)

    @staticmethod
    def _repeat_visit_ratio(cum_map) -> float:
        total_visits = int(np.sum(cum_map.visit_count))
        unique_visited = int(np.sum(cum_map.visit_count > 0))
        if total_visits <= 0:
            return 0.0
        repeat = max(0, total_visits - unique_visited)
        return float(repeat) / float(total_visits)

    @staticmethod
    def _build_action_mask(
        valid_list: list[int],
        device: torch.device,
        *,
        pin_memory: bool = False,
    ) -> torch.Tensor:
        mask = torch.zeros((1, ACTION_DIM), dtype=torch.bool, device=device)
        if len(valid_list) > 0:
            mask[0, valid_list] = True
        if pin_memory and device.type == "cpu" and not mask.is_pinned():
            mask = mask.pin_memory()
        return mask

    def _refresh_valid_action_cache(self, valid_indices) -> None:
        valid_list = [int(v) for v in valid_indices]
        self.valid_action_indices = tuple(valid_list)
        self._valid_action_list = valid_list
        self._current_action_mask_cpu = None
        self._current_action_mask_policy = None

    def _get_current_action_mask(self, *, policy_device: bool = False) -> torch.Tensor:
        if (not policy_device) or self._policy_device.type == "cpu":
            if self._current_action_mask_cpu is None:
                self._current_action_mask_cpu = self._build_action_mask(
                    self._valid_action_list,
                    self._cpu_device,
                    pin_memory=self._pin_cpu_action_mask,
                )
            return self._current_action_mask_cpu

        if self._current_action_mask_policy is None:
            cpu_mask = self._get_current_action_mask()
            self._current_action_mask_policy = cpu_mask.to(
                self._policy_device,
                non_blocking=cpu_mask.is_pinned(),
            )
        return self._current_action_mask_policy

    @staticmethod
    def _resolve_module_device(module) -> torch.device:
        param = next(module.parameters(), None)
        if param is not None:
            return torch.device(param.device)

        buffer = next(module.buffers(), None)
        if buffer is not None:
            return torch.device(buffer.device)
        return torch.device("cpu")

    @staticmethod
    def _resolve_amp_dtype(amp_dtype: str) -> torch.dtype:
        text = str(amp_dtype).strip().lower()
        if text == "fp16":
            return torch.float16
        if text == "bf16":
            return torch.bfloat16
        raise ValueError(f"Unsupported inference_amp_dtype: {amp_dtype!r}; expected 'fp16' or 'bf16'")

    def _set_policy_device(self, device: torch.device | str) -> None:
        resolved = torch.device(device)
        if resolved != self._policy_device:
            self._policy_device = resolved
            self._current_action_mask_policy = None
        self._pin_cpu_action_mask = bool(torch.cuda.is_available()) and self._policy_device.type == "cuda"
        self._enable_inference_amp = (
            bool(self.cfg.enable_inference_amp)
            and bool(torch.cuda.is_available())
            and self._policy_device.type == "cuda"
        )

    def _inference_autocast_context(self):
        if not self._enable_inference_amp:
            return nullcontext()
        return torch.autocast(device_type="cuda", dtype=self._inference_amp_dtype)

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

    def _build_state_tensors(self, cum_map, agent_state, shared_artifacts, recent_trajectory_positions):
        return self.state_adapter.build_single_state_tensors(
            cum_map,
            agent_state,
            recent_trajectory_positions=recent_trajectory_positions,
            shared_artifacts=shared_artifacts,
            target_device=None,
            return_state_meta=True,
        )

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
            "Incremental frontier cache mismatch in evaluator: "
            f"context={context} "
            f"episode_seed={episode_seed!r} "
            f"episode_step={int(episode_len)} "
            f"mismatch_count={int(stats.mismatch_count)} "
            f"frontier_revision={int(stats.frontier_revision)} "
            f"frontier_source_uid={int(stats.frontier_source_uid)} "
            f"map_shape={tuple(stats.map_shape)}"
        )

    def _run_episode(self, model, episode_seed: int | None = None) -> Dict[str, object]:
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
            context="evaluator_reset",
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
                action_idx = int(action.item())

            if action_idx not in valid_before:
                raise RuntimeError(
                    f"Selected evaluation action {action_idx} outside the valid-action set {sorted(valid_before)}. "
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
                context="evaluator_step_post_update",
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
                return {
                    "episode_reward": float(episode_reward),
                    "episode_length": int(episode_len),
                    "final_coverage": float(cum_map.coverage_rate),
                    "success": int(done_reason == "coverage_reached"),
                    "repeat_visit_ratio": float(self._repeat_visit_ratio(cum_map)),
                    "done_reason": str(done_reason),
                    **summarize_semantic_records(episode_semantic_records),
                    "true_grid": np.asarray(grid, dtype=np.int8).copy(),
                    "trajectory_positions": list(trajectory_positions),
                    "belief_map": np.asarray(cum_map.map, dtype=np.int8).copy(),
                    "belief_origin_world_rc": (
                        int(cum_map.origin_world_rc[0]),
                        int(cum_map.origin_world_rc[1]),
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

        rewards = np.asarray([float(ep["episode_reward"]) for ep in episodes], dtype=np.float32)
        coverages = np.asarray([float(ep["final_coverage"]) for ep in episodes], dtype=np.float32)
        successes = np.asarray([float(ep["success"]) for ep in episodes], dtype=np.float32)
        lengths = np.asarray([float(ep["episode_length"]) for ep in episodes], dtype=np.float32)
        repeats = np.asarray([float(ep["repeat_visit_ratio"]) for ep in episodes], dtype=np.float32)

        result: Dict[str, object] = {
            "eval_episodes": int(num_episodes),
            "eval_mean_reward": float(np.mean(rewards)),
            "eval_mean_coverage": float(np.mean(coverages)),
            "eval_success_rate": float(np.mean(successes)),
            "eval_mean_episode_length": float(np.mean(lengths)),
            "eval_mean_repeat_visit_ratio": float(np.mean(repeats)),
            "episodes": episodes,
        }
        for field in SEMANTIC_EPISODE_FIELDS:
            result[f"eval_mean_{field}"] = float(
                np.nanmean(np.asarray([float(ep[field]) for ep in episodes], dtype=np.float32))
            )
        for field in REWARD_BREAKDOWN_FIELDS:
            result[f"eval_mean_{field}"] = float(
                np.nanmean(np.asarray([float(ep[field]) for ep in episodes], dtype=np.float32))
            )
        for field in REWARD_EVENT_SUMMARY_FIELDS:
            result[f"eval_mean_{field}"] = float(
                np.nanmean(np.asarray([float(ep[field]) for ep in episodes], dtype=np.float32))
            )
        return result
