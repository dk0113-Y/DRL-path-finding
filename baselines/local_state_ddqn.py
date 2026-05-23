from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn

from agents.q_value_agent import ACTION_DIM, StateAdapterConfig, StateTensorAdapter
from env.grid_topology import EMPTY, INVISIBLE, OBSTACLE
from env.value_state_builder import (
    VALUE_BLOCK_FEATURE_COUNT,
    VALUE_DIAGNOSTIC_FIELDS,
    VALUE_ENTRY_FEATURE_COUNT,
)


LOCAL_STATE_BASELINE_ID = "C_baseline_local_state_ddqn"
LOCAL_STATE_BASELINE_NAME = LOCAL_STATE_BASELINE_ID
LOCAL_STATE_CHANNELS = ("known_free", "known_obstacle", "unknown")
LOCAL_STATE_CARRIER_KEY = "advantage_canvas"
DUMMY_VALUE_MASK_RULE = "all_false_masks; model ignores all value_* tensors"


@dataclass(frozen=True)
class LocalStateDDQNConfig:
    in_channels: int = len(LOCAL_STATE_CHANNELS)
    action_dim: int = ACTION_DIM
    hidden_channels: int = 64
    mlp_hidden_dim: int = 128


@dataclass(frozen=True)
class LocalStateStepArtifacts:
    """Marker object for the C baseline; no shared semantic snapshot is built."""

    semantic_snapshot: None = None


class LocalStateQNetwork(nn.Module):
    """
    Small CNN DDQN policy for the C learning baseline.

    It intentionally keeps the trainer-compatible forward signature while using
    only `advantage_canvas` as a local belief patch carrier. The value-tree
    tensors are accepted for replay/learner interface compatibility and ignored.
    """

    def __init__(self, cfg: Optional[LocalStateDDQNConfig] = None):
        super().__init__()
        self.cfg = cfg if cfg is not None else LocalStateDDQNConfig()
        if int(self.cfg.in_channels) != len(LOCAL_STATE_CHANNELS):
            raise ValueError(
                f"LocalStateQNetwork expects {len(LOCAL_STATE_CHANNELS)} local-state channels, "
                f"got {self.cfg.in_channels}"
            )
        if int(self.cfg.action_dim) != ACTION_DIM:
            raise ValueError(f"action_dim must be {ACTION_DIM}, got {self.cfg.action_dim}")

        hidden = int(self.cfg.hidden_channels)
        mlp_hidden = int(self.cfg.mlp_hidden_dim)
        self.encoder = nn.Sequential(
            nn.Conv2d(int(self.cfg.in_channels), 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, hidden, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
        self.q_head = nn.Sequential(
            nn.Linear(hidden, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, ACTION_DIM),
        )

    def forward(
        self,
        advantage_canvas: torch.Tensor,
        value_block_features: torch.Tensor | None = None,
        value_entry_features: torch.Tensor | None = None,
        value_block_mask: torch.Tensor | None = None,
        value_entry_mask: torch.Tensor | None = None,
        *,
        return_aux: bool = True,
    ):
        _ = value_block_features, value_entry_features, value_block_mask, value_entry_mask
        if advantage_canvas.dim() != 4:
            raise ValueError(f"advantage_canvas/local_state_canvas must be [B,C,H,W], got {tuple(advantage_canvas.shape)}")
        if int(advantage_canvas.shape[1]) != int(self.cfg.in_channels):
            raise ValueError(
                f"Local-state channel mismatch: expected {self.cfg.in_channels}, got {advantage_canvas.shape[1]}"
            )

        features = self.encoder(advantage_canvas)
        q_values = self.q_head(features)
        if not return_aux:
            return q_values
        aux = {
            "local_state_feature_l2": features.pow(2).mean(dim=1),
            "value_tensors_used_by_model": torch.zeros((q_values.shape[0],), device=q_values.device),
        }
        return q_values, aux


class LocalStateTensorAdapter(StateTensorAdapter):
    """
    Build C-baseline local belief patches without shared semantic/value-tree inputs.

    The output state dict keeps the existing replay keys. `advantage_canvas`
    carries a 3-channel agent-centered local belief patch; value tensors are
    zero dummy placeholders with all-false masks and are ignored by the model.
    """

    def __init__(self, cfg: Optional[StateAdapterConfig] = None, device: str = "cpu"):
        self.cfg = cfg if cfg is not None else StateAdapterConfig()
        self.device = torch.device(device)
        self._cpu_device = torch.device("cpu")
        self._pin_cpu_state = bool(self.cfg.pin_memory) and torch.cuda.is_available()
        self._non_blocking_transfer = bool(self.cfg.non_blocking_transfer)
        self._channels_last_on_cuda = bool(self.cfg.channels_last_on_cuda)
        self._timing_enabled = bool(self.cfg.enable_timing)

        self.shared_artifact_time = 0.0
        self.advantage_build_time = 0.0
        self.value_build_time = 0.0
        self.tensor_transfer_time = 0.0
        self.local_state_build_time = 0.0
        self.dummy_value_build_time = 0.0

    @staticmethod
    def _local_index_arrays(cum_map, agent_state: tuple[int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        h, w = int(cum_map.local_shape[0]), int(cum_map.local_shape[1])
        center_r = h // 2
        center_c = w // 2
        local_rows = np.arange(h, dtype=np.int32) - center_r
        local_cols = np.arange(w, dtype=np.int32) - center_c
        row_offsets, col_offsets = np.meshgrid(local_rows, local_cols, indexing="ij")
        agent_arr_r, agent_arr_c = cum_map.world_to_array(agent_state)
        arr_rows = row_offsets + int(agent_arr_r)
        arr_cols = col_offsets + int(agent_arr_c)
        inside = (
            (arr_rows >= 0) & (arr_rows < int(cum_map.map.shape[0])) &
            (arr_cols >= 0) & (arr_cols < int(cum_map.map.shape[1]))
        )
        return arr_rows, arr_cols, inside

    def build_shared_step_artifacts(self, cum_map, agent_state, frontier_u8=None) -> LocalStateStepArtifacts:
        _ = cum_map, agent_state, frontier_u8
        t0 = time.perf_counter() if self._timing_enabled else 0.0
        artifacts = LocalStateStepArtifacts()
        if self._timing_enabled:
            self.shared_artifact_time += time.perf_counter() - t0
        return artifacts

    def _build_local_state_canvas(self, cum_map, agent_state: tuple[int, int]) -> np.ndarray:
        local_shape = (int(cum_map.local_shape[0]), int(cum_map.local_shape[1]))
        arr_rows, arr_cols, inside = self._local_index_arrays(cum_map, agent_state)
        sampled_map = np.full(local_shape, INVISIBLE, dtype=np.int8)
        if np.any(inside):
            sampled_map[inside] = cum_map.map[arr_rows[inside], arr_cols[inside]]

        canvas = np.zeros((len(LOCAL_STATE_CHANNELS), local_shape[0], local_shape[1]), dtype=np.float32)
        canvas[0] = (sampled_map == EMPTY)
        canvas[1] = (sampled_map == OBSTACLE)
        canvas[2] = (sampled_map == INVISIBLE)
        return canvas

    def _build_dummy_value_state(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        max_blocks = int(self.cfg.value_state.max_accessible_blocks)
        max_entries = int(self.cfg.value_state.max_entries_per_block)
        block_features = np.zeros((max_blocks, VALUE_BLOCK_FEATURE_COUNT), dtype=np.float32)
        entry_features = np.zeros((max_blocks, max_entries, VALUE_ENTRY_FEATURE_COUNT), dtype=np.float32)
        block_mask = np.zeros((max_blocks,), dtype=bool)
        entry_mask = np.zeros((max_blocks, max_entries), dtype=bool)
        return block_features, entry_features, block_mask, entry_mask

    def build_single_state_tensors(
        self,
        cum_map,
        agent_state,
        recent_trajectory_positions: Optional[Sequence[tuple[int, int]]] = None,
        shared_artifacts: Optional[LocalStateStepArtifacts] = None,
        target_device: Optional[torch.device | str] = None,
        return_state_meta: bool = False,
    ):
        _ = recent_trajectory_positions, shared_artifacts
        t0 = time.perf_counter() if self._timing_enabled else 0.0
        local_state_canvas_np = self._build_local_state_canvas(cum_map, agent_state)
        if self._timing_enabled:
            elapsed = time.perf_counter() - t0
            self.advantage_build_time += elapsed
            self.local_state_build_time += elapsed

        t0 = time.perf_counter() if self._timing_enabled else 0.0
        block_np, entry_np, block_mask_np, entry_mask_np = self._build_dummy_value_state()
        if self._timing_enabled:
            elapsed = time.perf_counter() - t0
            self.value_build_time += elapsed
            self.dummy_value_build_time += elapsed

        canvas_t = self._to_cpu_batch_map(local_state_canvas_np, dtype=torch.float32)
        block_t, entry_t, block_mask_t, entry_mask_t = self._to_cpu_batch_tree(
            block_np,
            entry_np,
            block_mask_np,
            entry_mask_np,
        )

        state_batch = {
            "advantage_canvas": canvas_t,
            "value_block_features": block_t,
            "value_entry_features": entry_t,
            "value_block_mask": block_mask_t,
            "value_entry_mask": entry_mask_t,
        }

        local_shape = tuple(int(v) for v in local_state_canvas_np.shape)
        dummy_value_meta = {
            "accessible_block_count": 0.0,
            "total_accessible_unknown_area": 0.0,
            "total_frontier_cluster_count": 0.0,
            "mean_block_area": 0.0,
            "local_frontier_coverage": 0.0,
            "local_frontier_block_area_mean": 0.0,
            **{field: 0.0 for field in VALUE_DIAGNOSTIC_FIELDS},
        }
        state_meta = {
            **dummy_value_meta,
            "experiment_id": "C",
            "baseline_id": LOCAL_STATE_BASELINE_ID,
            "baseline_group": "learning_baseline",
            "baseline_name": "local_state_ddqn",
            "baseline_type": "simpler_drl_baseline",
            "is_ablation": False,
            "no_shared_semantic_dual_state": True,
            "no_value_tree": True,
            "no_frontier_cluster_input": True,
            "no_accessible_unknown_block_input": True,
            "no_ground_truth_map_for_decision": True,
            "local_state_channels": LOCAL_STATE_CHANNELS,
            "local_state_canvas_shape": local_shape,
            "local_state_patch_size": int(local_state_canvas_np.shape[-1]),
            "local_state_carrier_key": LOCAL_STATE_CARRIER_KEY,
            "dummy_value_tensors_for_interface": True,
            "value_tensors_used_by_model": False,
            "dummy_value_block_shape": tuple(int(v) for v in block_np.shape),
            "dummy_value_entry_shape": tuple(int(v) for v in entry_np.shape),
            "dummy_value_mask_rule": DUMMY_VALUE_MASK_RULE,
            "dummy_tensors_contain_real_value_tree_information": False,
        }

        if target_device is not None and self._resolve_device(target_device).type != "cpu":
            state_batch = self.move_state_batch(state_batch, target_device=target_device)

        if return_state_meta:
            return state_batch, state_meta
        return state_batch

    def get_timing_stats(self) -> dict[str, float]:
        return {
            "shared_artifact_time": float(self.shared_artifact_time),
            "advantage_build_time": float(self.advantage_build_time),
            "value_build_time": float(self.value_build_time),
            "local_state_build_time": float(self.local_state_build_time),
            "dummy_value_build_time": float(self.dummy_value_build_time),
            "tensor_transfer_time": float(self.tensor_transfer_time),
        }


def count_model_parameters(model: nn.Module) -> int:
    return int(sum(param.numel() for param in model.parameters()))


def build_baseline_manifest(
    *,
    cfg: Any,
    model: Optional[nn.Module] = None,
    git_sha: str | None = None,
) -> dict[str, Any]:
    scan_radius = int(getattr(cfg, "scan_radius", 10))
    patch_size = int(2 * scan_radius + 1)
    max_blocks = int(getattr(cfg, "max_accessible_blocks", 16))
    max_entries = int(getattr(cfg, "max_entries_per_block", 8))
    parameter_count = int(getattr(cfg, "model_parameter_count", 0) or 0)
    if model is not None:
        parameter_count = count_model_parameters(model)

    return {
        "experiment_id": "C",
        "baseline_id": LOCAL_STATE_BASELINE_ID,
        "baseline_group": "learning_baseline",
        "baseline_name": "local_state_ddqn",
        "baseline_type": "simpler_drl_baseline",
        "is_ablation": False,
        "no_shared_semantic_dual_state": True,
        "no_value_tree": True,
        "no_frontier_cluster_input": True,
        "no_accessible_unknown_block_input": True,
        "no_ground_truth_map_for_decision": True,
        "local_state_channels": list(LOCAL_STATE_CHANNELS),
        "local_state_patch_size": patch_size,
        "local_state_canvas_shape": [len(LOCAL_STATE_CHANNELS), patch_size, patch_size],
        "local_state_carrier_key": LOCAL_STATE_CARRIER_KEY,
        "local_state_carrier_note": (
            "For C, advantage_canvas carries a local belief patch, not A's semantic advantage canvas."
        ),
        "model_class": "LocalStateQNetwork",
        "model_parameter_count": parameter_count,
        "reward_overrides": "none",
        "channel_ablation": "none",
        "value_replacement_strategy": "not_applicable",
        "dummy_value_tensors_for_interface": True,
        "value_tensors_used_by_model": False,
        "dummy_value_block_shape": [max_blocks, VALUE_BLOCK_FEATURE_COUNT],
        "dummy_value_entry_shape": [max_blocks, max_entries, VALUE_ENTRY_FEATURE_COUNT],
        "dummy_value_mask_rule": DUMMY_VALUE_MASK_RULE,
        "dummy_tensors_contain_real_value_tree_information": False,
        "run_stage": getattr(cfg, "run_stage", "formal"),
        "rows": int(getattr(cfg, "rows", 40)),
        "cols": int(getattr(cfg, "cols", 60)),
        "obs_size": int(getattr(cfg, "obs_size", 6)),
        "scan_radius": scan_radius,
        "obstacle_ratio": float(getattr(cfg, "obstacle_ratio", 0.20)),
        "max_episode_steps": int(getattr(cfg, "max_episode_steps", 600)),
        "coverage_stop_threshold": float(getattr(cfg, "coverage_stop_threshold", 0.95)),
        "trajectory_history_steps": int(getattr(cfg, "trajectory_history_steps", 10)),
        "fixed_final_probe_seed_base": int(getattr(cfg, "fixed_final_probe_seed_base", 20261323)),
        "final_greedy_episodes": int(getattr(cfg, "final_greedy_episodes", 100)),
        "reward_info_scale": float(getattr(cfg, "reward_info_scale", 3.1)),
        "reward_obstacle_weight": float(getattr(cfg, "reward_obstacle_weight", 0.2)),
        "reward_step_penalty": float(getattr(cfg, "reward_step_penalty", 0.02)),
        "reward_terminal_bonus": float(getattr(cfg, "reward_terminal_bonus", 20.0)),
        "reward_revisit_penalty": float(getattr(cfg, "reward_revisit_penalty", 0.1)),
        "reward_turn_penalty_scale": float(getattr(cfg, "reward_turn_penalty_scale", 0.05)),
        "reward_timeout_penalty": float(getattr(cfg, "reward_timeout_penalty", 8.0)),
        "git_sha": git_sha,
    }
