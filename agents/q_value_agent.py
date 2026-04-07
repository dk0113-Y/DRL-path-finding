from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
import sys
from typing import Dict, Iterable, Optional, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.nn as nn

from encoders.advantage_encoder import AdvantageCanvasEncoder, AdvantageEncoderConfig
from encoders.value_encoder import ValueEncoderConfig, ValueTreeEncoder
from env.advantage_state_builder import (
    ADVANTAGE_CANVAS_CHANNEL_COUNT,
    AdvantageStateBuilder,
    AdvantageStateConfig,
)
from env.grid_topology import ACTIONS_8
from env.shared_semantic_layer import (
    SharedSemanticConfig,
    SharedSemanticLayer,
    SharedSemanticSnapshot,
)
from env.value_state_builder import (
    VALUE_BLOCK_FEATURE_COUNT,
    VALUE_ENTRY_FEATURE_COUNT,
    ValueStateBuilder,
    ValueStateConfig,
)
from heads.semantic_dueling_head import SemanticDuelingHead, SemanticDuelingHeadConfig


ACTION_DIM = len(ACTIONS_8)


@dataclass(frozen=True)
class ExplorationQConfig:
    advantage_encoder: AdvantageEncoderConfig = field(default_factory=AdvantageEncoderConfig)
    value_encoder: ValueEncoderConfig = field(default_factory=ValueEncoderConfig)
    decision_head: SemanticDuelingHeadConfig = field(default_factory=SemanticDuelingHeadConfig)


@dataclass(frozen=True)
class StateAdapterConfig:
    shared_semantics: SharedSemanticConfig = field(default_factory=SharedSemanticConfig)
    advantage_state: AdvantageStateConfig = field(default_factory=AdvantageStateConfig)
    value_state: ValueStateConfig = field(default_factory=ValueStateConfig)
    pin_memory: bool = True
    non_blocking_transfer: bool = True
    channels_last_on_cuda: bool = False
    enable_timing: bool = False


@dataclass(frozen=True)
class SharedStepArtifacts:
    semantic_snapshot: SharedSemanticSnapshot


class ExplorationQNetwork(nn.Module):
    """
    Shared-semantic dueling exploration network.

    Data flow:
      advantage_canvas -> advantage encoder -> per-action advantage states
      value block-tree (6D block summary + 4D frontier entries) -> value encoder -> state value context
      {value_state, advantage_state} -> dueling head -> Q(s, a)
    """

    def __init__(self, cfg: Optional[ExplorationQConfig] = None):
        super().__init__()
        self.cfg = cfg if cfg is not None else ExplorationQConfig()
        if int(self.cfg.advantage_encoder.action_dim) != ACTION_DIM:
            raise ValueError(
                f"Advantage encoder action_dim must be {ACTION_DIM}, got {self.cfg.advantage_encoder.action_dim}"
            )
        if int(self.cfg.decision_head.action_dim) != ACTION_DIM:
            raise ValueError(
                f"Decision head action_dim must be {ACTION_DIM}, got {self.cfg.decision_head.action_dim}"
            )
        if int(self.cfg.advantage_encoder.canvas_in_channels) != int(ADVANTAGE_CANVAS_CHANNEL_COUNT):
            raise ValueError(
                "Advantage canvas channel mismatch: "
                f"expected {ADVANTAGE_CANVAS_CHANNEL_COUNT}, got {self.cfg.advantage_encoder.canvas_in_channels}"
            )
        if int(self.cfg.value_encoder.block_input_dim) != int(VALUE_BLOCK_FEATURE_COUNT):
            raise ValueError(
                "Value block feature dim mismatch: "
                f"expected {VALUE_BLOCK_FEATURE_COUNT}, got {self.cfg.value_encoder.block_input_dim}"
            )
        if int(self.cfg.value_encoder.entry_input_dim) != int(VALUE_ENTRY_FEATURE_COUNT):
            raise ValueError(
                "Value entry feature dim mismatch: "
                f"expected {VALUE_ENTRY_FEATURE_COUNT}, got {self.cfg.value_encoder.entry_input_dim}"
            )
        if int(self.cfg.decision_head.value_state_dim) != int(self.cfg.value_encoder.value_state_dim):
            raise ValueError("Value encoder output dim must match decision head value_state_dim")
        if int(self.cfg.decision_head.advantage_state_dim) != int(self.cfg.advantage_encoder.action_state_dim):
            raise ValueError("Advantage encoder output dim must match decision head advantage_state_dim")

        self.advantage_encoder = AdvantageCanvasEncoder(self.cfg.advantage_encoder)
        self.value_encoder = ValueTreeEncoder(self.cfg.value_encoder)
        self.decision_head = SemanticDuelingHead(self.cfg.decision_head)

    def forward(
        self,
        advantage_canvas: torch.Tensor,
        value_block_features: torch.Tensor,
        value_entry_features: torch.Tensor,
        value_block_mask: torch.Tensor,
        value_entry_mask: torch.Tensor,
        *,
        return_aux: bool = True,
    ):
        if not return_aux:
            advantage_state = self.advantage_encoder(advantage_canvas, return_aux=False)
            value_state = self.value_encoder(
                value_block_features,
                value_entry_features,
                value_block_mask,
                value_entry_mask,
                return_aux=False,
            )
            return self.decision_head(value_state, advantage_state)

        advantage_state, advantage_aux = self.advantage_encoder(advantage_canvas, return_aux=True)
        value_state, value_aux = self.value_encoder(
            value_block_features,
            value_entry_features,
            value_block_mask,
            value_entry_mask,
            return_aux=True,
        )
        q_values = self.decision_head(value_state, advantage_state)
        aux: Dict[str, torch.Tensor] = {}
        aux.update(advantage_aux)
        aux.update(value_aux)
        return q_values, aux


class StateTensorAdapter:
    """
    Build the policy inputs from environment state objects.

    Outputs:
      advantage_canvas, value_block_features, value_entry_features,
      value_block_mask, value_entry_mask
    """

    _STATE_BATCH_KEYS = (
        "advantage_canvas",
        "value_block_features",
        "value_entry_features",
        "value_block_mask",
        "value_entry_mask",
    )
    _STATE_BATCH_DTYPES = (
        ("advantage_canvas", torch.float32),
        ("value_block_features", torch.float32),
        ("value_entry_features", torch.float32),
        ("value_block_mask", torch.bool),
        ("value_entry_mask", torch.bool),
    )

    def __init__(self, cfg: Optional[StateAdapterConfig] = None, device: str = "cpu"):
        self.cfg = cfg if cfg is not None else StateAdapterConfig()
        self.shared_semantic_layer = SharedSemanticLayer(self.cfg.shared_semantics)
        self.advantage_builder = AdvantageStateBuilder(self.cfg.advantage_state)
        self.value_builder = ValueStateBuilder(self.cfg.value_state)
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

    @staticmethod
    def _ensure_contiguous_numpy(np_array: np.ndarray) -> np.ndarray:
        return np_array if np_array.flags.c_contiguous else np.ascontiguousarray(np_array)

    @staticmethod
    def _resolve_device(target_device: Optional[torch.device | str]) -> torch.device:
        return torch.device("cpu") if target_device is None else torch.device(target_device)

    def _finalize_cpu_tensor(
        self,
        tensor: torch.Tensor,
        dtype: torch.dtype,
    ) -> torch.Tensor:
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
        if tensor.device.type == "cpu":
            if tensor.dtype != target_dtype:
                tensor = tensor.to(dtype=target_dtype)
            if tensor.device != target:
                tensor = tensor.to(target, non_blocking=move_non_blocking)
        else:
            tensor = tensor.to(device=target, dtype=target_dtype, non_blocking=move_non_blocking)

        if target.type == "cuda" and self._channels_last_on_cuda and tensor.dim() == 4:
            tensor = tensor.contiguous(memory_format=torch.channels_last)
        elif not tensor.is_contiguous():
            tensor = tensor.contiguous()

        if self._timing_enabled:
            self.tensor_transfer_time += time.perf_counter() - t0
        return tensor

    def _to_cpu_batch_map(self, np_array, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        if not isinstance(np_array, np.ndarray):
            raise TypeError("map input must be a numpy array")
        np_array = self._ensure_contiguous_numpy(np_array)
        map_t = torch.from_numpy(np_array)
        if map_t.dtype != dtype:
            map_t = map_t.to(dtype=dtype)
        map_t = map_t.unsqueeze(0)
        return self._finalize_cpu_tensor(map_t, dtype=dtype)

    def _to_cpu_batch_tree(
        self,
        block_features,
        entry_features,
        block_mask,
        entry_mask,
    ):
        if not isinstance(block_features, np.ndarray) or not isinstance(entry_features, np.ndarray):
            raise TypeError("tree features must be numpy arrays")
        if not isinstance(block_mask, np.ndarray) or not isinstance(entry_mask, np.ndarray):
            raise TypeError("tree masks must be numpy arrays")
        block_features_t = self._finalize_cpu_tensor(
            torch.from_numpy(self._ensure_contiguous_numpy(block_features)).unsqueeze(0),
            dtype=torch.float32,
        )
        entry_features_t = self._finalize_cpu_tensor(
            torch.from_numpy(self._ensure_contiguous_numpy(entry_features)).unsqueeze(0),
            dtype=torch.float32,
        )
        block_mask_t = self._finalize_cpu_tensor(
            torch.from_numpy(self._ensure_contiguous_numpy(block_mask)).unsqueeze(0),
            dtype=torch.bool,
        )
        entry_mask_t = self._finalize_cpu_tensor(
            torch.from_numpy(self._ensure_contiguous_numpy(entry_mask)).unsqueeze(0),
            dtype=torch.bool,
        )
        return block_features_t, entry_features_t, block_mask_t, entry_mask_t

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
        for key, _ in self._STATE_BATCH_DTYPES:
            tensor = state_batch.get(key)
            if tensor is None:
                raise KeyError(f"missing state batch key: {key}")
            if tensor.device.type != "cpu":
                raise ValueError(f"move_state_batch expects CPU state input for key={key}")
            source_tensors.append(tensor)

        move_non_blocking = self._resolve_batch_move_non_blocking(source_tensors, target, non_blocking=non_blocking)
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
    ) -> SharedStepArtifacts:
        _ = frontier_u8
        t0 = time.perf_counter() if self._timing_enabled else 0.0
        shared = SharedStepArtifacts(
            semantic_snapshot=self.shared_semantic_layer.analyze(cum_map, agent_state),
        )
        if self._timing_enabled:
            self.shared_artifact_time += time.perf_counter() - t0
        return shared

    def build_single_state_tensors(
        self,
        cum_map,
        agent_state,
        shared_artifacts: Optional[SharedStepArtifacts] = None,
        target_device: Optional[torch.device | str] = None,
        return_state_meta: bool = False,
    ):
        if shared_artifacts is None:
            shared_artifacts = self.build_shared_step_artifacts(cum_map, agent_state)

        t0 = time.perf_counter() if self._timing_enabled else 0.0
        advantage_canvas_np, local_meta = self.advantage_builder.build(
            cum_map,
            agent_state,
            shared_artifacts.semantic_snapshot,
        )
        if self._timing_enabled:
            self.advantage_build_time += time.perf_counter() - t0

        t0 = time.perf_counter() if self._timing_enabled else 0.0
        block_np, entry_np, block_mask_np, entry_mask_np = self.value_builder.build(
            shared_artifacts.semantic_snapshot,
        )
        if self._timing_enabled:
            self.value_build_time += time.perf_counter() - t0

        advantage_canvas_t = self._to_cpu_batch_map(advantage_canvas_np, dtype=torch.float32)
        block_t, entry_t, block_mask_t, entry_mask_t = self._to_cpu_batch_tree(
            block_np,
            entry_np,
            block_mask_np,
            entry_mask_np,
        )

        state_batch = {
            "advantage_canvas": advantage_canvas_t,
            "value_block_features": block_t,
            "value_entry_features": entry_t,
            "value_block_mask": block_mask_t,
            "value_entry_mask": entry_mask_t,
        }

        semantic_meta = dict(shared_artifacts.semantic_snapshot.metrics())
        state_meta = {**semantic_meta, **local_meta}
        if target_device is not None and self._resolve_device(target_device).type != "cpu":
            state_batch = self.move_state_batch(state_batch, target_device=target_device)

        if return_state_meta:
            return state_batch, state_meta
        return state_batch

    def get_timing_stats(self) -> Dict[str, float]:
        return {
            "shared_artifact_time": float(self.shared_artifact_time),
            "advantage_build_time": float(self.advantage_build_time),
            "value_build_time": float(self.value_build_time),
            "tensor_transfer_time": float(self.tensor_transfer_time),
        }


def action_mask_from_valid_indices(
    valid_action_indices: Sequence[Iterable[int]] | Iterable[int],
    action_dim: int = ACTION_DIM,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    if action_dim != ACTION_DIM:
        raise ValueError(f"action_dim must be {ACTION_DIM}")

    def _is_int_like(x) -> bool:
        return isinstance(x, (int, np.integer)) and not isinstance(x, bool)

    def _normalize_row(row_like) -> list[int]:
        if isinstance(row_like, torch.Tensor):
            if row_like.dim() == 0:
                items = [int(row_like.item())]
            else:
                items = [int(v) for v in row_like.detach().cpu().reshape(-1).tolist()]
        elif isinstance(row_like, np.ndarray):
            if row_like.ndim == 0:
                items = [int(row_like.item())]
            else:
                items = [int(v) for v in row_like.reshape(-1).tolist()]
        else:
            items = [int(v) for v in row_like]
        if len(items) == 0:
            raise ValueError("Each sample must contain at least one valid action index")
        for ai in items:
            if not (0 <= ai < action_dim):
                raise ValueError(f"invalid action index: {ai}")
        return items

    rows: list[list[int]]
    if isinstance(valid_action_indices, torch.Tensor):
        if valid_action_indices.dim() == 1:
            rows = [_normalize_row(valid_action_indices)]
        elif valid_action_indices.dim() == 2:
            rows = [_normalize_row(valid_action_indices[i]) for i in range(valid_action_indices.shape[0])]
        else:
            raise ValueError("torch valid_action_indices must be 1D or 2D")
    elif isinstance(valid_action_indices, np.ndarray):
        if valid_action_indices.ndim == 1:
            rows = [_normalize_row(valid_action_indices)]
        elif valid_action_indices.ndim == 2:
            rows = [_normalize_row(valid_action_indices[i]) for i in range(valid_action_indices.shape[0])]
        else:
            raise ValueError("numpy valid_action_indices must be 1D or 2D")
    else:
        seq = list(valid_action_indices)
        if len(seq) == 0:
            raise ValueError("valid_action_indices cannot be empty")
        if all(_is_int_like(v) for v in seq):
            rows = [[int(v) for v in seq]]
        else:
            rows = [_normalize_row(row) for row in seq]

    mask = torch.zeros((len(rows), action_dim), dtype=torch.bool, device=device)
    for i, row in enumerate(rows):
        mask[i, row] = True
    return mask


def masked_q_values(
    q_values: torch.Tensor,
    action_mask: torch.Tensor,
    invalid_fill: float = -1e9,
) -> torch.Tensor:
    if q_values.dim() == 1:
        q_values = q_values.unsqueeze(0)
        squeeze_back = True
    elif q_values.dim() == 2:
        squeeze_back = False
    else:
        raise ValueError(f"q_values must be [A] or [B,A], got shape={tuple(q_values.shape)}")

    if action_mask.dim() == 1:
        action_mask = action_mask.unsqueeze(0)
    if action_mask.shape != q_values.shape:
        raise ValueError(
            f"action_mask shape mismatch: expected {tuple(q_values.shape)}, got {tuple(action_mask.shape)}"
        )

    fill_value = float(invalid_fill)
    if torch.is_floating_point(q_values):
        fill_value = max(fill_value, float(torch.finfo(q_values.dtype).min))
    masked = q_values.masked_fill(~action_mask.to(dtype=torch.bool), fill_value)
    return masked.squeeze(0) if squeeze_back else masked


def select_greedy_action(
    q_values: torch.Tensor,
    action_mask: Optional[torch.Tensor] = None,
    valid_action_indices: Optional[Sequence[Iterable[int]] | Iterable[int]] = None,
) -> torch.Tensor:
    if action_mask is not None and valid_action_indices is not None:
        raise ValueError("Provide either action_mask or valid_action_indices, not both")

    squeeze_back = False
    if q_values.dim() == 1:
        q_values = q_values.unsqueeze(0)
        squeeze_back = True
    elif q_values.dim() != 2:
        raise ValueError(f"q_values must be [A] or [B,A], got shape={tuple(q_values.shape)}")

    if action_mask is None and valid_action_indices is not None:
        action_mask = action_mask_from_valid_indices(
            valid_action_indices,
            action_dim=q_values.shape[1],
            device=q_values.device,
        )

    if action_mask is not None:
        if action_mask.dim() == 1:
            action_mask = action_mask.unsqueeze(0)
        if action_mask.shape != q_values.shape:
            raise ValueError(
                f"action_mask shape mismatch: expected {tuple(q_values.shape)}, got {tuple(action_mask.shape)}"
            )
        valid_per_row = action_mask.to(dtype=torch.bool).sum(dim=1)
        if torch.any(valid_per_row <= 0):
            raise ValueError("Each sample must have at least one valid action")
        q_use = masked_q_values(q_values, action_mask)
    else:
        q_use = q_values

    action = torch.argmax(q_use, dim=1)
    return action.squeeze(0) if squeeze_back else action


def _smoke_test() -> None:
    from env.agent_version import LocalObservationModel
    from env.block_random_g import RandomMapGenerator
    from env.core_cummap import CumulativeBeliefMap
    from env.grid_topology import GridTopology

    torch.manual_seed(0)
    grid, start = RandomMapGenerator(30, 40, 5, 0.2).generate_map()
    obs = LocalObservationModel(grid, start)
    cum_map = CumulativeBeliefMap(grid, start, obs.local_snap)

    adapter = StateTensorAdapter(device="cpu")
    state_batch, state_meta = adapter.build_single_state_tensors(
        cum_map,
        start,
        return_state_meta=True,
    )

    net = ExplorationQNetwork()
    q_values, aux = net(
        state_batch["advantage_canvas"],
        state_batch["value_block_features"],
        state_batch["value_entry_features"],
        state_batch["value_block_mask"],
        state_batch["value_entry_mask"],
        return_aux=True,
    )
    valid = GridTopology.valid_action_indices_fast(GridTopology.free_mask(grid), start)
    action = select_greedy_action(q_values, valid_action_indices=valid)

    assert q_values.shape == (1, ACTION_DIM)
    assert torch.isfinite(q_values).all()
    assert isinstance(state_meta, dict) and "accessible_block_count" in state_meta
    assert isinstance(aux, dict) and "value_accessible_block_count" in aux
    assert int(action.item()) in valid
    print("ExplorationQNetwork semantic smoke test passed", tuple(q_values.shape))


if __name__ == "__main__":
    _smoke_test()
