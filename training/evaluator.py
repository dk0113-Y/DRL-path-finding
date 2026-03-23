from __future__ import annotations

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
from training.collector import CollectorConfig
from training.rewarding import (
    REWARD_BREAKDOWN_FIELDS,
    add_reward_breakdown,
    resolve_reward_info_norm,
    reward_from_breakdown,
    timeout_penalty_breakdown,
    valid_step_reward_breakdown,
    zero_reward_breakdown,
)


@dataclass(frozen=True)
class EvaluatorConfig:
    rows: int = 40
    cols: int = 60
    obs_size: int = 6
    scan_radius: int = 10  # radar sensor radius only
    obstacle_ratio: float = 0.20

    max_episode_steps: int = 300  # tune with map scale as needed
    coverage_stop_threshold: float = 0.98

    reward_info_scale: float = 10.0
    reward_obstacle_weight: float = 0.25
    reward_info_norm: float | None = None
    reward_recent_revisit_window: int = 15
    reward_stall_window: int = 4
    reward_step_penalty: float = 0.01
    reward_terminal_bonus: float = 0.5
    reward_revisit_penalty: float = 0.05
    reward_stall_penalty: float = 0.02
    reward_timeout_penalty: float = 0.5
    enable_inference_amp: bool = False
    inference_amp_dtype: str = "fp16"


ACTION_DIM = len(ACTIONS_8)


class GreedyEvaluator:
    """Periodic greedy evaluation with the same reward mainline as training."""

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
        self.reward_info_norm = resolve_reward_info_norm(
            cfg.reward_info_norm,
            self.sensor.theoretical_visible_cell_count,
        )
        self._recent_revisit_window = max(1, int(cfg.reward_recent_revisit_window))
        self._stall_window = max(1, int(cfg.reward_stall_window))
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
            reward_info_scale=cfg.reward_info_scale,
            reward_obstacle_weight=cfg.reward_obstacle_weight,
            reward_info_norm=cfg.reward_info_norm,
            reward_recent_revisit_window=cfg.reward_recent_revisit_window,
            reward_stall_window=cfg.reward_stall_window,
            reward_step_penalty=cfg.reward_step_penalty,
            reward_terminal_bonus=cfg.reward_terminal_bonus,
            reward_revisit_penalty=cfg.reward_revisit_penalty,
            reward_stall_penalty=cfg.reward_stall_penalty,
            reward_timeout_penalty=cfg.reward_timeout_penalty,
            enable_inference_amp=cfg.enable_inference_amp,
            inference_amp_dtype=cfg.inference_amp_dtype,
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

    def _build_state_tensors(self, cum_map, agent_state, shared_artifacts) -> Dict[str, torch.Tensor]:
        return self.state_adapter.build_single_state_tensors(
            cum_map,
            agent_state,
            frontier_tokens=None,
            frontier_token_mask=None,
            shared_artifacts=shared_artifacts,
            target_device=None,
        )

    def _run_episode(self, model) -> Dict[str, object]:
        grid, agent = self.generator.generate_map()
        free_mask = GridTopology.free_mask(grid)
        obs_model = LocalObservationModel(grid, agent, sensor=self.sensor)
        local_snap = obs_model.local_snap
        self._refresh_valid_action_cache(GridTopology.valid_action_indices_fast(free_mask, agent))

        cum_map = CumulativeBeliefMap(grid, agent, local_snap)
        frontier_u8 = cum_map.get_frontier_u8(refresh=True)
        shared_artifacts = self.state_adapter.build_shared_step_artifacts(
            cum_map,
            frontier_u8=frontier_u8,
        )
        model_device = self._resolve_module_device(model)
        self._set_policy_device(model_device)

        episode_reward = 0.0
        episode_len = 0
        episode_breakdown = zero_reward_breakdown()
        recent_positions: deque[tuple[int, int]] = deque(
            [(int(agent[0]), int(agent[1]))],
            maxlen=self._recent_revisit_window,
        )
        stall_streak = 0
        trajectory_positions: list[tuple[int, int]] = [(int(agent[0]), int(agent[1]))]

        while True:
            valid_before = self.valid_action_indices
            step_breakdown = zero_reward_breakdown()

            if len(valid_before) <= 0:
                raise RuntimeError(
                    "Encountered an empty valid-action set before evaluation step. "
                    "This is treated as a defensive invariant violation, not a normal episode outcome."
                )
            else:
                state_tensors = self._build_state_tensors(
                    cum_map,
                    agent,
                    shared_artifacts=shared_artifacts,
                )
                policy_state = self.state_adapter.move_state_batch(
                    state_tensors,
                    target_device=self._policy_device,
                    non_blocking=True,
                )
                with torch.inference_mode():
                    with self._inference_autocast_context():
                        q_values = model(
                            policy_state["near_map"],
                            policy_state["mid_map"],
                            policy_state["frontier_tokens"],
                            frontier_token_mask=policy_state["frontier_token_mask"],
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
                else:
                    dr, dc = ACTIONS_8[action_idx]
                    agent = (int(agent[0] + dr), int(agent[1] + dc))
                    trajectory_positions.append((int(agent[0]), int(agent[1])))

                    local_snap = obs_model.observe_fast(agent)
                    self._refresh_valid_action_cache(GridTopology.valid_action_indices_fast(free_mask, agent))
                    updated, delta_empty, delta_obstacle = cum_map.update(agent, local_snap)
                    if int(updated) != int(delta_empty + delta_obstacle):
                        raise RuntimeError("belief-map update returned inconsistent information-gain counts")
                    frontier_u8 = cum_map.get_frontier_u8(refresh=True)
                    shared_artifacts = self.state_adapter.build_shared_step_artifacts(
                        cum_map,
                        frontier_u8=frontier_u8,
                    )

                    recent_revisit = bool((int(agent[0]), int(agent[1])) in recent_positions)
                    recent_positions.append((int(agent[0]), int(agent[1])))
                    if int(delta_empty) == 0 and int(delta_obstacle) == 0:
                        stall_streak += 1
                    else:
                        stall_streak = 0
                    stall_triggered = bool(stall_streak >= self._stall_window)
                    success = bool(cum_map.coverage_rate >= float(self.cfg.coverage_stop_threshold))
                    no_valid_after_step = bool((not success) and (len(self.valid_action_indices) <= 0))

                    step_breakdown = valid_step_reward_breakdown(
                        self.cfg,
                        delta_empty=delta_empty,
                        delta_obstacle=delta_obstacle,
                        reward_info_norm=self.reward_info_norm,
                        recent_revisit=recent_revisit,
                        stall_triggered=stall_triggered,
                        success=success,
                    )
                    reward = reward_from_breakdown(step_breakdown)
                    done = True
                    done_reason = ""

                    if success:
                        done_reason = "coverage_reached"
                    elif no_valid_after_step:
                        raise RuntimeError(
                            "Encountered an empty valid-action set after a valid evaluation move "
                            "without reaching coverage target. This is treated as a defensive "
                            "environment invariant violation."
                        )
                    else:
                        done = False

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
                final_coverage = float(cum_map.coverage_rate)
                success = bool(done_reason == "coverage_reached")
                return {
                    "episode_reward": float(episode_reward),
                    "episode_length": int(episode_len),
                    "final_coverage": final_coverage,
                    "success": int(success),
                    "repeat_visit_ratio": float(self._repeat_visit_ratio(cum_map)),
                    "done_reason": str(done_reason),
                    "true_grid": np.asarray(grid, dtype=np.int8).copy(),
                    "trajectory_positions": list(trajectory_positions),
                    **{field: float(episode_breakdown[field]) for field in REWARD_BREAKDOWN_FIELDS},
                }

    def evaluate(self, model, num_episodes: int = 5) -> Dict[str, object]:
        if num_episodes <= 0:
            raise ValueError("num_episodes must be > 0")

        was_training = bool(model.training)
        model.eval()

        episodes = [self._run_episode(model) for _ in range(int(num_episodes))]

        if was_training:
            model.train()

        rewards = np.asarray([float(ep["episode_reward"]) for ep in episodes], dtype=np.float32)
        coverages = np.asarray([float(ep["final_coverage"]) for ep in episodes], dtype=np.float32)
        successes = np.asarray([float(ep["success"]) for ep in episodes], dtype=np.float32)
        lengths = np.asarray([float(ep["episode_length"]) for ep in episodes], dtype=np.float32)
        repeats = np.asarray([float(ep["repeat_visit_ratio"]) for ep in episodes], dtype=np.float32)

        return {
            "eval_episodes": int(num_episodes),
            "eval_mean_reward": float(np.mean(rewards)),
            "eval_mean_coverage": float(np.mean(coverages)),
            "eval_success_rate": float(np.mean(successes)),
            "eval_mean_episode_length": float(np.mean(lengths)),
            "eval_mean_repeat_visit_ratio": float(np.mean(repeats)),
            **{
                f"eval_mean_{field}": float(
                    np.mean(np.asarray([float(ep[field]) for ep in episodes], dtype=np.float32))
                )
                for field in REWARD_BREAKDOWN_FIELDS
            },
            "episodes": episodes,
        }
