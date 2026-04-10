from __future__ import annotations

import random
import time
from collections import deque
from contextlib import nullcontext
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
from env.value_state_builder import VALUE_DIAGNOSTIC_FIELDS
from training.rewarding import (
    REWARD_EVENT_SUMMARY_FIELDS,
    add_reward_breakdown,
    resolve_reward_info_norm,
    reward_from_breakdown,
    timeout_penalty_breakdown,
    turn_penalty_weight_from_steps,
    valid_step_reward_breakdown,
    weighted_info_gain,
    zero_reward_breakdown,
    zero_reward_event_summary,
)
from training.replay_buffer import NStepTransitionBuilder, ReplayBuffer


ACTION_DIM = len(ACTIONS_8)
SEMANTIC_EPISODE_FIELDS = (
    "accessible_block_count",
    "total_accessible_unknown_area",
    "total_frontier_cluster_count",
    "mean_block_area",
    "local_frontier_coverage",
    "local_frontier_block_area_mean",
    *VALUE_DIAGNOSTIC_FIELDS,
)


def _nanmean_or_nan(values: list[float]) -> float:
    if len(values) <= 0:
        return float("nan")
    arr = np.asarray(values, dtype=np.float32)
    if not np.any(np.isfinite(arr)):
        return float("nan")
    return float(np.nanmean(arr))


def summarize_semantic_records(records: list[dict[str, float]]) -> dict[str, float]:
    if len(records) <= 0:
        return {field: float("nan") for field in SEMANTIC_EPISODE_FIELDS}

    # Episode-level semantic/value-cap diagnostics use per-step means so they
    # stay directly comparable with recent_* train summaries and eval_mean_* logs.
    return {
        field: _nanmean_or_nan([float(item.get(field, float("nan"))) for item in records])
        for field in SEMANTIC_EPISODE_FIELDS
    }


@dataclass(frozen=True)
class CollectorConfig:
    rows: int = 40
    cols: int = 60
    obs_size: int = 6
    scan_radius: int = 10  # radar sensor radius only
    obstacle_ratio: float = 0.20

    max_episode_steps: int = 300  # tune with map scale as needed
    coverage_stop_threshold: float = 0.98
    trajectory_history_steps: int = 10

    reward_info_scale: float = 10.0
    reward_obstacle_weight: float = 0.25
    reward_info_norm: float | None = None
    reward_recent_revisit_window: int = 15
    reward_stall_window: int = 4
    reward_step_penalty: float = 0.01
    reward_terminal_bonus: float = 0.5
    reward_revisit_penalty: float = 0.05
    reward_stall_penalty: float = 0.02
    reward_turn_penalty_scale: float = 0.0
    reward_timeout_penalty: float = 0.5

    n_step: int = 3
    gamma: float = 0.99
    enable_timing: bool = False
    enable_cummap_timing: bool = False
    enable_inference_amp: bool = False
    inference_amp_dtype: str = "fp16"
    prefer_batch_replay_add: bool = True
    record_episode_artifacts: bool = False
    debug_check_incremental_frontier: bool = False


class TransitionCollector:
    """
    Rollout collector:
      - interacts with environment simulator
      - selects action via epsilon-greedy under legal-action mask
      - builds transitions and pushes n-step transitions into replay
      - emits episode-level metrics for monitoring
    """

    def __init__(self, cfg: CollectorConfig, online_net, state_adapter, replay: ReplayBuffer):
        self.cfg = cfg
        self.online_net = online_net
        self.state_adapter = state_adapter
        self.replay = replay
        self.online_net.eval()
        self._timing_enabled = bool(cfg.enable_timing)
        self._prefer_batch_replay_add = bool(cfg.prefer_batch_replay_add) and hasattr(replay, "add_many")
        self._record_episode_artifacts = bool(cfg.record_episode_artifacts)
        self._cpu_device = torch.device("cpu")
        self._policy_device = self._resolve_module_device(self.online_net)
        self._pin_cpu_action_mask = bool(torch.cuda.is_available()) and self._policy_device.type == "cuda"
        self._enable_inference_amp = (
            bool(cfg.enable_inference_amp)
            and bool(torch.cuda.is_available())
            and self._policy_device.type == "cuda"
        )
        self._inference_amp_dtype = self._resolve_amp_dtype(cfg.inference_amp_dtype)

        self.sensor = RadarSensor(scan_radius=int(cfg.scan_radius))
        self.reward_info_norm = resolve_reward_info_norm(
            cfg.reward_info_norm,
            self.sensor.theoretical_visible_cell_count,
        )
        self._recent_revisit_window = max(1, int(cfg.reward_recent_revisit_window))
        self._stall_window = max(1, int(cfg.reward_stall_window))
        self._trajectory_history_steps = max(1, int(cfg.trajectory_history_steps))
        self.generator = RandomMapGenerator(
            rows=cfg.rows,
            cols=cfg.cols,
            obs_size=cfg.obs_size,
            obstacle_ratio=cfg.obstacle_ratio,
        )
        self.nstep = NStepTransitionBuilder(n_step=cfg.n_step, gamma=cfg.gamma)

        self.grid = None
        self.agent = None
        self.obs_model = None
        self.cum_map = None
        self.free_mask = None
        self.local_snap = None
        self.valid_action_indices: tuple[int, ...] = ()
        self._valid_action_list: list[int] = []
        self._current_action_mask_cpu: Optional[torch.Tensor] = None
        self._current_action_mask_policy: Optional[torch.Tensor] = None
        self.frontier_u8 = None
        self._current_shared_artifacts = None
        self._current_state_meta = None

        self.episode_steps = 0
        self.total_env_steps = 0
        self.total_episodes = 0
        self._episode_reward = 0.0
        self._episode_reward_breakdown = zero_reward_breakdown()
        self._episode_event_summary = zero_reward_event_summary()
        self._recent_positions: deque[tuple[int, int]] = deque(maxlen=self._recent_revisit_window)
        self._recent_trajectory_positions: deque[tuple[int, int]] = deque(
            maxlen=self._trajectory_history_steps + 1
        )
        self._stall_streak = 0
        self._prev_action_idx: int | None = None
        self._current_state_tensors = None
        self._trajectory_positions: list[tuple[int, int]] = []
        self.state_build_time = 0.0
        self.policy_forward_time = 0.0
        self.env_step_time = 0.0

        self.reset_episode()

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

    def _inference_autocast_context(self):
        if not self._enable_inference_amp:
            return nullcontext()
        return torch.autocast(device_type="cuda", dtype=self._inference_amp_dtype)

    def reset_episode(self) -> None:
        self.grid, self.agent = self.generator.generate_map()
        self.free_mask = GridTopology.free_mask(self.grid)
        self.obs_model = LocalObservationModel(self.grid, self.agent, sensor=self.sensor)
        self.local_snap = self.obs_model.local_snap
        self._refresh_valid_action_cache(GridTopology.valid_action_indices_fast(self.free_mask, self.agent))

        self.cum_map = CumulativeBeliefMap(
            self.grid,
            self.agent,
            self.local_snap,
            enable_timing=bool(self.cfg.enable_cummap_timing),
        )
        self.frontier_u8 = self.cum_map.get_frontier_u8(refresh=False)
        self._check_incremental_frontier_consistency(context="collector_reset")
        self._current_shared_artifacts = self.state_adapter.build_shared_step_artifacts(
            self.cum_map,
            self.agent,
            frontier_u8=self.frontier_u8,
        )

        self.episode_steps = 0
        self._episode_reward = 0.0
        self._episode_reward_breakdown = zero_reward_breakdown()
        self._episode_event_summary = zero_reward_event_summary()
        self._recent_positions = deque(
            [(int(self.agent[0]), int(self.agent[1]))],
            maxlen=self._recent_revisit_window,
        )
        self._recent_trajectory_positions = deque(
            [(int(self.agent[0]), int(self.agent[1]))],
            maxlen=self._trajectory_history_steps + 1,
        )
        self._stall_streak = 0
        self._prev_action_idx = None
        self._trajectory_positions = (
            [(int(self.agent[0]), int(self.agent[1]))] if self._record_episode_artifacts else []
        )
        self._episode_semantic_records: list[dict[str, float]] = []
        t0 = time.perf_counter() if self._timing_enabled else 0.0
        self._current_state_tensors, self._current_state_meta = self._build_state_tensors()
        if self._timing_enabled:
            self.state_build_time += time.perf_counter() - t0
        self.total_episodes += 1

    def _is_recent_revisit(self, position: tuple[int, int]) -> bool:
        pos = (int(position[0]), int(position[1]))
        revisit = bool(pos in self._recent_positions)
        self._recent_positions.append(pos)
        return revisit

    def _update_stall_streak(self, delta_empty: int, delta_obstacle: int) -> bool:
        if int(delta_empty) == 0 and int(delta_obstacle) == 0:
            self._stall_streak += 1
        else:
            self._stall_streak = 0
        return bool(self._stall_streak >= self._stall_window)

    def _build_state_tensors(self):
        if self._current_shared_artifacts is None:
            self._current_shared_artifacts = self.state_adapter.build_shared_step_artifacts(
                self.cum_map,
                self.agent,
                frontier_u8=self.frontier_u8,
            )
        return self.state_adapter.build_single_state_tensors(
            self.cum_map,
            self.agent,
            recent_trajectory_positions=tuple(self._recent_trajectory_positions),
            shared_artifacts=self._current_shared_artifacts,
            target_device=None,
            return_state_meta=True,
        )

    def _check_incremental_frontier_consistency(self, *, context: str) -> None:
        if not bool(self.cfg.debug_check_incremental_frontier):
            return
        stats = self.cum_map.debug_frontier_consistency_stats()
        if stats.consistent:
            return
        episode_step = int(self.episode_steps) + (0 if context == "collector_reset" else 1)
        raise RuntimeError(
            "Incremental frontier cache mismatch in collector: "
            f"context={context} "
            f"episode_idx={int(self.total_episodes)} "
            f"episode_step={episode_step} "
            f"total_env_steps={int(self.total_env_steps)} "
            f"mismatch_count={int(stats.mismatch_count)} "
            f"frontier_revision={int(stats.frontier_revision)} "
            f"frontier_source_uid={int(stats.frontier_source_uid)} "
            f"map_shape={tuple(stats.map_shape)}"
        )

    def _record_current_state_meta(self) -> None:
        meta = self._current_state_meta
        if not isinstance(meta, dict):
            return
        self._episode_semantic_records.append(
            {field: float(meta.get(field, float("nan"))) for field in SEMANTIC_EPISODE_FIELDS}
        )

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

    def _refresh_valid_action_cache(
        self,
        valid_indices,
    ) -> None:
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
    def _assert_cpu_state_batch(state_tensors: Dict[str, torch.Tensor], name: str) -> None:
        for key in (
            "advantage_canvas",
            "value_block_features",
            "value_entry_features",
            "value_block_mask",
            "value_entry_mask",
        ):
            if state_tensors[key].device.type != "cpu":
                raise RuntimeError(f"{name}.{key} must stay on CPU before replay insertion")

    def _select_action(
        self,
        state_tensors: Optional[Dict[str, torch.Tensor]],
        epsilon: float,
        random_only: bool = False,
    ) -> Optional[int]:
        valid = self._valid_action_list
        if len(valid) <= 0:
            return None

        if random_only:
            return int(random.choice(valid))

        if random.random() < float(epsilon):
            return int(random.choice(valid))

        if state_tensors is None:
            raise ValueError("state_tensors is required when random_only=False and greedy branch is used")

        policy_state = self.state_adapter.move_state_batch(
            state_tensors,
            target_device=self._policy_device,
            non_blocking=True,
        )
        action_mask = self._get_current_action_mask(policy_device=True)
        t0 = time.perf_counter() if self._timing_enabled else 0.0
        with torch.inference_mode():
            with self._inference_autocast_context():
                q_values = self.online_net(
                    policy_state["advantage_canvas"],
                    policy_state["value_block_features"],
                    policy_state["value_entry_features"],
                    policy_state["value_block_mask"],
                    policy_state["value_entry_mask"],
                    return_aux=False,
                )
            action = select_greedy_action(q_values, action_mask=action_mask)
        if self._timing_enabled:
            self.policy_forward_time += time.perf_counter() - t0
        return int(action.item())

    def _step_env(self, action_idx: Optional[int], valid_before: list[int]) -> tuple[float, bool, str]:
        done_reason = ""
        step_breakdown = zero_reward_breakdown()

        if len(valid_before) <= 0:
            raise RuntimeError(
                "Encountered an empty valid-action set before stepping. "
                "This is treated as a defensive invariant violation, not a normal episode outcome."
            )
        elif action_idx is None:
            raise RuntimeError(
                "Action selection returned None despite a non-empty valid-action set. "
                "This is treated as a defensive policy-path violation."
            )
        elif action_idx not in valid_before:
            raise RuntimeError(
                f"Selected action {action_idx} outside the valid-action set {valid_before}. "
                "This is treated as a defensive invariant violation."
            )
        else:
            turn_steps = GridTopology.circular_turn_steps(self._prev_action_idx, int(action_idx))
            turn_penalty_weight = float(turn_penalty_weight_from_steps(turn_steps))
            dr, dc = ACTIONS_8[int(action_idx)]
            self.agent = (int(self.agent[0] + dr), int(self.agent[1] + dc))
            self._recent_trajectory_positions.append((int(self.agent[0]), int(self.agent[1])))
            if self._record_episode_artifacts:
                self._trajectory_positions.append((int(self.agent[0]), int(self.agent[1])))

            self.local_snap = self.obs_model.observe_fast(self.agent)
            self._refresh_valid_action_cache(GridTopology.valid_action_indices_fast(self.free_mask, self.agent))
            updated, delta_empty, delta_obstacle = self.cum_map.update(self.agent, self.local_snap)
            if int(updated) != int(delta_empty + delta_obstacle):
                raise RuntimeError("belief-map update returned inconsistent information-gain counts")
            self.frontier_u8 = self.cum_map.get_frontier_u8(refresh=False)
            self._check_incremental_frontier_consistency(context="collector_step_post_update")
            self._current_shared_artifacts = self.state_adapter.build_shared_step_artifacts(
                self.cum_map,
                self.agent,
                frontier_u8=self.frontier_u8,
            )

            recent_revisit = self._is_recent_revisit(self.agent)
            stall_triggered = self._update_stall_streak(delta_empty, delta_obstacle)
            event_summary = self._episode_event_summary
            event_summary["delta_empty_sum"] += float(delta_empty)
            event_summary["delta_obstacle_sum"] += float(delta_obstacle)
            event_summary["weighted_info_gain_sum"] += float(
                weighted_info_gain(
                    delta_empty=delta_empty,
                    delta_obstacle=delta_obstacle,
                    obstacle_weight=float(self.cfg.reward_obstacle_weight),
                    info_norm=self.reward_info_norm,
                )
            )
            event_summary["recent_revisit_count"] += float(bool(recent_revisit))
            event_summary["stall_trigger_count"] += float(bool(stall_triggered))
            if int(delta_empty) == 0 and int(delta_obstacle) == 0:
                event_summary["zero_info_step_count"] += 1.0
            if int(turn_steps) >= 2:
                event_summary["turn_ge_90_count"] += 1.0
            if int(turn_steps) == 3:
                event_summary["turn_135_count"] += 1.0
            if int(turn_steps) == 4:
                event_summary["turn_180_count"] += 1.0
            event_summary["turn_penalty_weight_sum"] += float(turn_penalty_weight)
            success = bool(self.cum_map.coverage_rate >= float(self.cfg.coverage_stop_threshold))
            no_valid_after_step = bool((not success) and (len(self.valid_action_indices) <= 0))
            done = False
            step_breakdown = valid_step_reward_breakdown(
                self.cfg,
                delta_empty=delta_empty,
                delta_obstacle=delta_obstacle,
                reward_info_norm=self.reward_info_norm,
                recent_revisit=recent_revisit,
                stall_triggered=stall_triggered,
                turn_penalty_weight=turn_penalty_weight,
                success=success,
            )
            reward = reward_from_breakdown(step_breakdown)
            self._prev_action_idx = int(action_idx)

            if success:
                done = True
                done_reason = "coverage_reached"

            if no_valid_after_step:
                raise RuntimeError(
                    "Encountered an empty valid-action set after a valid move without reaching coverage target. "
                    "This is treated as a defensive environment invariant violation."
                )

        self.episode_steps += 1

        if (not done) and (self.episode_steps >= int(self.cfg.max_episode_steps)):
            done = True
            done_reason = "max_episode_steps"
            timeout_breakdown = timeout_penalty_breakdown(self.cfg)
            add_reward_breakdown(step_breakdown, timeout_breakdown)
            reward += reward_from_breakdown(timeout_breakdown)

        self.total_env_steps += 1
        self._episode_reward += reward
        add_reward_breakdown(self._episode_reward_breakdown, step_breakdown)

        if done and done_reason == "":
            done_reason = "terminal"
        return reward, done, done_reason

    @staticmethod
    def _repeat_visit_ratio(cum_map) -> float:
        total_visits = int(np.sum(cum_map.visit_count))
        unique_visited = int(np.sum(cum_map.visit_count > 0))
        if total_visits <= 0:
            return 0.0
        repeat = max(0, total_visits - unique_visited)
        return float(repeat) / float(total_visits)

    def _episode_visual_artifacts(self) -> dict[str, object]:
        semantic_payload = (
            build_semantic_visualization_payload(self._current_shared_artifacts.semantic_snapshot)
            if self._current_shared_artifacts is not None else None
        )
        return {
            "true_grid": np.asarray(self.grid, dtype=np.int8).copy(),
            "trajectory_positions": list(self._trajectory_positions),
            "belief_map": np.asarray(self.cum_map.map, dtype=np.int8).copy(),
            "belief_origin_world_rc": (
                int(self.cum_map.origin_world_rc[0]),
                int(self.cum_map.origin_world_rc[1]),
            ),
            "semantic_viz": semantic_payload,
        }

    def _push_ready_transitions(self, ready: list[dict]) -> int:
        if len(ready) <= 0:
            return 0
        if self._prefer_batch_replay_add:
            self.replay.add_many(ready)
            return len(ready)

        for tr in ready:
            self.replay.add(tr)
        return len(ready)

    def get_timing_stats(self) -> Dict[str, float]:
        return {
            "state_build_time": float(self.state_build_time),
            "policy_forward_time": float(self.policy_forward_time),
            "env_step_time": float(self.env_step_time),
        }

    def collect_steps(self, num_steps: int, epsilon: float, random_only: bool = False) -> Dict[str, object]:
        if num_steps <= 0:
            raise ValueError("num_steps must be > 0")

        timing_enabled = self._timing_enabled
        self.online_net.eval()

        pushed = 0
        episode_done = 0
        reward_sum = 0.0
        last_episode_reward = 0.0
        episodes: list[dict] = []

        for _ in range(int(num_steps)):
            current_state = self._current_state_tensors
            if current_state is None:
                raise RuntimeError("current state tensors cache is not initialized")
            self._assert_cpu_state_batch(current_state, "current_state")
            self._record_current_state_meta()
            valid_before = self._valid_action_list
            current_action_mask = self._get_current_action_mask()

            state_for_policy = None if random_only else current_state
            action = self._select_action(state_for_policy, epsilon=epsilon, random_only=random_only)
            t0 = time.perf_counter() if timing_enabled else 0.0
            reward, done, done_reason = self._step_env(action, valid_before=valid_before)
            if timing_enabled:
                self.env_step_time += time.perf_counter() - t0
            reward_sum += reward

            t0 = time.perf_counter() if timing_enabled else 0.0
            next_state, next_state_meta = self._build_state_tensors()
            if timing_enabled:
                self.state_build_time += time.perf_counter() - t0
            self._assert_cpu_state_batch(next_state, "next_state")
            next_action_mask = self._get_current_action_mask()

            if action is None:
                action = 0

            # current_state and next_state come from separate _build_state_tensors() calls,
            # and collector does not mutate these tensors in place after creation, so
            # directly referencing them in n-step/replay storage is safe without extra clones.
            # The cached action masks are also replaced by re-assignment on each refresh rather than
            # mutated in place, so keeping direct references here is safe for replay storage.
            one_step = {
                "advantage_canvas": current_state["advantage_canvas"],
                "value_block_features": current_state["value_block_features"],
                "value_entry_features": current_state["value_entry_features"],
                "value_block_mask": current_state["value_block_mask"],
                "value_entry_mask": current_state["value_entry_mask"],
                "action_mask": current_action_mask,
                "action": int(action),
                "reward": float(reward),
                "next_advantage_canvas": next_state["advantage_canvas"],
                "next_value_block_features": next_state["value_block_features"],
                "next_value_entry_features": next_state["value_entry_features"],
                "next_value_block_mask": next_state["value_block_mask"],
                "next_value_entry_mask": next_state["value_entry_mask"],
                "next_action_mask": next_action_mask,
                "done": bool(done),
            }

            ready = self.nstep.append(one_step)
            pushed += self._push_ready_transitions(ready)

            if done:
                episode_done += 1
                last_episode_reward = self._episode_reward

                final_coverage = float(self.cum_map.coverage_rate)
                success = bool(done_reason == "coverage_reached")
                episode_event_summary = dict(self._episode_event_summary)
                episode_event_summary["timeout_flag"] = float(done_reason == "max_episode_steps")
                episodes.append(
                    {
                        "episode_idx": int(self.total_episodes),
                        "env_steps": int(self.total_env_steps),
                        "episode_reward": float(self._episode_reward),
                        "episode_length": int(self.episode_steps),
                        "final_coverage": final_coverage,
                        "success": int(success),
                        "repeat_visit_ratio": float(self._repeat_visit_ratio(self.cum_map)),
                        "done_reason": str(done_reason),
                        **summarize_semantic_records(self._episode_semantic_records),
                        **{k: float(self._episode_reward_breakdown[k]) for k in self._episode_reward_breakdown},
                        **{k: float(episode_event_summary.get(k, 0.0)) for k in REWARD_EVENT_SUMMARY_FIELDS},
                        **(self._episode_visual_artifacts() if self._record_episode_artifacts else {}),
                    }
                )
                self.reset_episode()
            else:
                self._current_state_tensors = next_state
                self._current_state_meta = next_state_meta

        out = {
            "env_steps": float(num_steps),
            "pushed_transitions": float(pushed),
            "episodes_done": float(episode_done),
            "reward_sum": float(reward_sum),
            "last_episode_reward": float(last_episode_reward),
            "replay_size": float(len(self.replay)),
            "episodes": episodes,
        }
        if timing_enabled:
            out["state_build_time"] = float(self.state_build_time)
            out["policy_forward_time"] = float(self.policy_forward_time)
            out["env_step_time"] = float(self.env_step_time)
        return out
