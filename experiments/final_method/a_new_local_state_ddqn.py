from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import numpy as np
import torch

from agents.local_state_q_network import (
    LOCAL_STATE_CHANNELS,
    LOCAL_STATE_PATCH_SIZE,
    LocalStateQConfig,
    LocalStateQNetwork,
    local_state_model_parameter_count,
)
from env.grid_topology import EMPTY, INVISIBLE, OBSTACLE
from env.value_state_builder import VALUE_BLOCK_FEATURE_COUNT, VALUE_ENTRY_FEATURE_COUNT


EXPERIMENT_ID = "Anew_C"
METHOD_ID = "Anew_C_local_state_ddqn"
METHOD_NAME = "local_state_ddqn"
BASELINE_GROUP = "learning"
BASELINE_NAME = METHOD_NAME
BASELINE_TYPE = "learning_ddqn"
LOCAL_STATE_SOURCE = "cumulative_belief_patch"
LOCAL_STATE_CANVAS_SCHEMA = "local_state_3ch_belief_patch"
LOCAL_STATE_CARRIER_KEY = "advantage_canvas"
LOCAL_STATE_CANVAS_ROLE = "baseline_local_state_input"
VALUE_REPLACEMENT_STRATEGY = "not_applicable_to_baseline"
DUMMY_VALUE_MASK_RULE = "all_false"


@dataclass(frozen=True)
class LocalStateStepArtifacts:
    local_state_only: bool = True


def local_state_patch_size_from_scan_radius(scan_radius: int) -> int:
    return int(2 * int(scan_radius) + 1)


def build_local_state_patch(
    cum_map,
    agent_state: tuple[int, int],
    *,
    patch_size: int = LOCAL_STATE_PATCH_SIZE,
) -> np.ndarray:
    patch = int(patch_size)
    if patch <= 0 or patch % 2 == 0:
        raise ValueError(f"patch_size must be a positive odd integer, got {patch_size!r}")

    center = patch // 2
    local_rows = np.arange(patch, dtype=np.int32) - center
    local_cols = np.arange(patch, dtype=np.int32) - center
    row_offsets, col_offsets = np.meshgrid(local_rows, local_cols, indexing="ij")
    agent_arr_r, agent_arr_c = cum_map.world_to_array(agent_state)
    arr_rows = row_offsets + int(agent_arr_r)
    arr_cols = col_offsets + int(agent_arr_c)
    inside = (
        (arr_rows >= 0)
        & (arr_rows < int(cum_map.map.shape[0]))
        & (arr_cols >= 0)
        & (arr_cols < int(cum_map.map.shape[1]))
    )

    sampled_map = np.full((patch, patch), INVISIBLE, dtype=np.int8)
    if np.any(inside):
        sampled_map[inside] = cum_map.map[arr_rows[inside], arr_cols[inside]]

    local_state = np.empty((len(LOCAL_STATE_CHANNELS), patch, patch), dtype=np.float32)
    local_state[0] = (sampled_map == EMPTY)
    local_state[1] = (sampled_map == OBSTACLE)
    local_state[2] = (sampled_map == INVISIBLE)
    return local_state


class LocalStateTensorAdapter:
    """
    Adapter for Anew_C: local belief patch plus interface-compatible zero value tensors.

    The policy tensor is sampled from `cum_map.map` around the current pose. It
    deliberately ignores shared semantic snapshots, frontier rasters, visit
    counts, and recent trajectory history.
    """

    _STATE_BATCH_DTYPES = (
        ("advantage_canvas", torch.float32),
        ("value_block_features", torch.float32),
        ("value_entry_features", torch.float32),
        ("value_block_mask", torch.bool),
        ("value_entry_mask", torch.bool),
    )

    def __init__(
        self,
        cfg=None,
        device: str = "cpu",
        *,
        patch_size: int = LOCAL_STATE_PATCH_SIZE,
    ):
        self.cfg = cfg
        self.device = torch.device(device)
        self.patch_size = int(patch_size)
        self.max_accessible_blocks = int(getattr(getattr(cfg, "value_state", None), "max_accessible_blocks", 16))
        self.max_entries_per_block = int(getattr(getattr(cfg, "value_state", None), "max_entries_per_block", 8))
        self._pin_cpu_state = bool(getattr(cfg, "pin_memory", True)) and torch.cuda.is_available()
        self._non_blocking_transfer = bool(getattr(cfg, "non_blocking_transfer", True))
        self._channels_last_on_cuda = bool(getattr(cfg, "channels_last_on_cuda", False))
        self._timing_enabled = bool(getattr(cfg, "enable_timing", False))
        self.local_state_build_time = 0.0
        self.dummy_value_build_time = 0.0
        self.tensor_transfer_time = 0.0

    @staticmethod
    def _resolve_device(target_device: Optional[torch.device | str]) -> torch.device:
        return torch.device("cpu") if target_device is None else torch.device(target_device)

    def _finalize_cpu_tensor(self, tensor: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        if tensor.device.type != "cpu" or tensor.dtype != dtype:
            tensor = tensor.to(device="cpu", dtype=dtype)
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()
        if self._pin_cpu_state and not tensor.is_pinned():
            tensor = tensor.pin_memory()
        return tensor

    def _move_tensor_to_resolved_target(
        self,
        tensor: torch.Tensor,
        target: torch.device,
        *,
        target_dtype: torch.dtype,
        move_non_blocking: bool,
    ) -> torch.Tensor:
        if tensor.device == target and tensor.dtype == target_dtype and tensor.is_contiguous():
            return tensor
        t0 = time.perf_counter() if self._timing_enabled else 0.0
        if tensor.dtype != target_dtype:
            tensor = tensor.to(dtype=target_dtype)
        if tensor.device != target:
            tensor = tensor.to(target, non_blocking=move_non_blocking)
        if target.type == "cuda" and self._channels_last_on_cuda and tensor.dim() == 4:
            tensor = tensor.contiguous(memory_format=torch.channels_last)
        elif not tensor.is_contiguous():
            tensor = tensor.contiguous()
        if self._timing_enabled:
            self.tensor_transfer_time += time.perf_counter() - t0
        return tensor

    def _resolve_batch_move_non_blocking(
        self,
        tensors: Sequence[torch.Tensor],
        target: torch.device,
        non_blocking: Optional[bool],
    ) -> bool:
        if target.type != "cuda":
            return False
        if non_blocking is not None:
            return bool(non_blocking)
        if not self._non_blocking_transfer:
            return False
        return all(tensor.device.type == "cpu" and tensor.is_pinned() for tensor in tensors)

    def move_state_batch(
        self,
        state_batch: Dict[str, torch.Tensor],
        target_device: torch.device | str,
        non_blocking: Optional[bool] = None,
    ) -> Dict[str, torch.Tensor]:
        target = self._resolve_device(target_device)
        source_tensors: list[torch.Tensor] = []
        for key, _dtype in self._STATE_BATCH_DTYPES:
            tensor = state_batch.get(key)
            if tensor is None:
                raise KeyError(f"missing state batch key: {key}")
            if tensor.device.type != "cpu":
                raise ValueError(f"move_state_batch expects CPU state input for key={key}")
            source_tensors.append(tensor)
        move_non_blocking = self._resolve_batch_move_non_blocking(source_tensors, target, non_blocking)
        return {
            key: self._move_tensor_to_resolved_target(
                state_batch[key],
                target,
                target_dtype=dtype,
                move_non_blocking=move_non_blocking,
            )
            for key, dtype in self._STATE_BATCH_DTYPES
        }

    def build_shared_step_artifacts(
        self,
        cum_map,
        agent_state,
        frontier_u8=None,
    ) -> LocalStateStepArtifacts:
        _ = cum_map, agent_state, frontier_u8
        return LocalStateStepArtifacts()

    def _dummy_value_tensors(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        t0 = time.perf_counter() if self._timing_enabled else 0.0
        block = torch.zeros(
            (1, self.max_accessible_blocks, VALUE_BLOCK_FEATURE_COUNT),
            dtype=torch.float32,
            device="cpu",
        )
        entry = torch.zeros(
            (1, self.max_accessible_blocks, self.max_entries_per_block, VALUE_ENTRY_FEATURE_COUNT),
            dtype=torch.float32,
            device="cpu",
        )
        block_mask = torch.zeros((1, self.max_accessible_blocks), dtype=torch.bool, device="cpu")
        entry_mask = torch.zeros(
            (1, self.max_accessible_blocks, self.max_entries_per_block),
            dtype=torch.bool,
            device="cpu",
        )
        if self._timing_enabled:
            self.dummy_value_build_time += time.perf_counter() - t0
        return (
            self._finalize_cpu_tensor(block, torch.float32),
            self._finalize_cpu_tensor(entry, torch.float32),
            self._finalize_cpu_tensor(block_mask, torch.bool),
            self._finalize_cpu_tensor(entry_mask, torch.bool),
        )

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
        local_state_np = build_local_state_patch(cum_map, agent_state, patch_size=self.patch_size)
        if self._timing_enabled:
            self.local_state_build_time += time.perf_counter() - t0

        local_state_t = self._finalize_cpu_tensor(
            torch.from_numpy(np.ascontiguousarray(local_state_np)).unsqueeze(0),
            torch.float32,
        )
        block_t, entry_t, block_mask_t, entry_mask_t = self._dummy_value_tensors()
        state_batch = {
            "advantage_canvas": local_state_t,
            "value_block_features": block_t,
            "value_entry_features": entry_t,
            "value_block_mask": block_mask_t,
            "value_entry_mask": entry_mask_t,
        }
        if target_device is not None and self._resolve_device(target_device).type != "cpu":
            state_batch = self.move_state_batch(state_batch, target_device=target_device)

        state_meta = {
            "local_state_channel_count": float(len(LOCAL_STATE_CHANNELS)),
            "local_state_patch_size": float(self.patch_size),
            "local_state_known_free_ratio": float(local_state_np[0].mean()),
            "local_state_known_obstacle_ratio": float(local_state_np[1].mean()),
            "local_state_unknown_ratio": float(local_state_np[2].mean()),
            "value_packed_block_count": 0.0,
            "value_packed_entry_count": 0.0,
        }
        if return_state_meta:
            return state_batch, state_meta
        return state_batch

    def get_timing_stats(self) -> dict[str, float]:
        return {
            "local_state_build_time": float(self.local_state_build_time),
            "dummy_value_build_time": float(self.dummy_value_build_time),
            "tensor_transfer_time": float(self.tensor_transfer_time),
        }


def build_local_state_model(*, patch_size: int) -> LocalStateQNetwork:
    return LocalStateQNetwork(LocalStateQConfig(local_state_patch_size=int(patch_size)))
