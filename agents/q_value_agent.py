from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from encoders.global_encoder import GlobalSideEncoder, GlobalSideEncoderConfig
from encoders.local_encoder import NearMapEncoderConfig, RawNearSummaryEncoder
from env.core_cummap import (
    MID_MAP_CHANNEL_COUNT,
    FrontierDerivedStats,
    MidMapConfig,
)
from env.frontier_token_builder import (
    FRONTIER_REGION_TOKEN_FIELD_COUNT,
    FrontierRegionTokenBuilder,
    FrontierRegionTokenConfig,
)
from env.grid_topology import ACTIONS_8
from env.local_state_builder import LOCAL_STATE_CHANNEL_COUNT, LocalStateBuilder, LocalStateConfig
from heads.q_head import DecisionHeadConfig, SplitDuelingDecisionHead


ACTION_DIM = len(ACTIONS_8)


@dataclass(frozen=True)
class ExplorationQConfig:
    global_side_encoder: GlobalSideEncoderConfig = field(
        default_factory=lambda: GlobalSideEncoderConfig(
            frontier_token_input_dim=FRONTIER_REGION_TOKEN_FIELD_COUNT
        )
    )
    near_encoder: NearMapEncoderConfig = field(default_factory=NearMapEncoderConfig)
    decision_head: DecisionHeadConfig = field(default_factory=DecisionHeadConfig)


@dataclass(frozen=True)
class StateAdapterConfig:
    near_map: LocalStateConfig = field(default_factory=LocalStateConfig)
    mid_map: MidMapConfig = field(default_factory=MidMapConfig)
    frontier_tokens: FrontierRegionTokenConfig = field(default_factory=FrontierRegionTokenConfig)
    pin_memory: bool = True
    non_blocking_transfer: bool = True
    channels_last_on_cuda: bool = False
    enable_timing: bool = False


@dataclass(frozen=True)
class SharedStepArtifacts:
    frontier_u8: np.ndarray
    frontier_stats: FrontierDerivedStats


class ExplorationQNetwork(nn.Module):
    """
    A+ exploration backbone.

    Data flow:
      near_map -> near encoder -> raw_near_summary
      mid_map -> mid encoder
      frontier_tokens -> frontier-region token encoder
      global fusion -> global_context
      raw_near_summary + global_context -> split-input dueling decision head -> Q values

    Forward inputs:
      near_map             [B, C_near, H_near, W_near]
      mid_map              [B, C_mid, H_mid, W_mid]
      frontier_tokens      [B, K, D_token]
      frontier_token_mask  [B, K] bool (optional, True means valid)
    """

    def __init__(self, cfg: Optional[ExplorationQConfig] = None):
        super().__init__()
        self.cfg = cfg if cfg is not None else ExplorationQConfig()

        if self.cfg.decision_head.action_dim != ACTION_DIM:
            raise ValueError(
                f"DecisionHead action_dim must be {ACTION_DIM}, got {self.cfg.decision_head.action_dim}"
            )
        if self.cfg.near_encoder.near_in_channels != LOCAL_STATE_CHANNEL_COUNT:
            raise ValueError(
                "near_in_channels mismatch with LocalStateBuilder channels: "
                f"expected {LOCAL_STATE_CHANNEL_COUNT}, got {self.cfg.near_encoder.near_in_channels}"
            )
        if self.cfg.decision_head.global_context_dim != self.cfg.global_side_encoder.global_context_dim:
            raise ValueError(
                "global context dim mismatch between decision head and global side: "
                f"{self.cfg.decision_head.global_context_dim} vs {self.cfg.global_side_encoder.global_context_dim}"
            )
        if self.cfg.decision_head.raw_near_summary_dim != self.cfg.near_encoder.raw_near_summary_dim:
            raise ValueError(
                "raw near summary dim mismatch between near encoder and decision head: "
                f"{self.cfg.decision_head.raw_near_summary_dim} vs {self.cfg.near_encoder.raw_near_summary_dim}"
            )

        self.global_side_encoder = GlobalSideEncoder(self.cfg.global_side_encoder)
        self.raw_near_summary_encoder = RawNearSummaryEncoder(self.cfg.near_encoder)
        self.decision_head = SplitDuelingDecisionHead(self.cfg.decision_head)

    def forward(
        self,
        near_map: torch.Tensor,
        mid_map: torch.Tensor,
        frontier_tokens: torch.Tensor,
        frontier_token_mask: Optional[torch.Tensor] = None,
        return_aux: bool = True,
    ):
        if not return_aux:
            global_context = self.global_side_encoder(
                mid_map,
                frontier_tokens,
                frontier_token_mask=frontier_token_mask,
                return_aux=False,
            )
            raw_near_summary = self.raw_near_summary_encoder(near_map)
            return self.decision_head(raw_near_summary, global_context)

        global_context, global_aux = self.global_side_encoder(
            mid_map,
            frontier_tokens,
            frontier_token_mask=frontier_token_mask,
            return_aux=True,
        )
        raw_near_summary = self.raw_near_summary_encoder(near_map)
        q_values = self.decision_head(raw_near_summary, global_context)

        aux: Dict[str, torch.Tensor] = {
            "global_context": global_context,
            "mid_map_vector": global_aux["mid_map_vector"],
            "frontier_token_context": global_aux["frontier_token_context"],
            "global_source_gates": global_aux["global_source_gates"],
            "raw_near_summary": raw_near_summary,
        }
        return q_values, aux


class StateTensorAdapter:
    """
    Build the policy inputs from environment state objects.

    Outputs:
      near_map, mid_map, frontier_tokens, frontier_token_mask
    """

    _STATE_BATCH_KEYS = (
        "near_map",
        "mid_map",
        "frontier_tokens",
        "frontier_token_mask",
    )
    _STATE_BATCH_DTYPES = (
        ("near_map", torch.float32),
        ("mid_map", torch.float32),
        ("frontier_tokens", torch.float32),
        ("frontier_token_mask", torch.bool),
    )

    def __init__(self, cfg: Optional[StateAdapterConfig] = None, device: str = "cpu"):
        self.cfg = cfg if cfg is not None else StateAdapterConfig()
        self.near_builder = LocalStateBuilder(self.cfg.near_map)
        self.frontier_token_builder = FrontierRegionTokenBuilder(self.cfg.frontier_tokens)
        self.device = torch.device(device)
        self._cpu_device = torch.device("cpu")
        self._pin_cpu_state = bool(self.cfg.pin_memory) and torch.cuda.is_available()
        self._non_blocking_transfer = bool(self.cfg.non_blocking_transfer)
        self._channels_last_on_cuda = bool(self.cfg.channels_last_on_cuda)
        self._timing_enabled = bool(self.cfg.enable_timing)

        self.shared_artifact_time = 0.0
        self.near_build_time = 0.0
        self.mid_build_time = 0.0
        self.frontier_token_build_time = 0.0
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

    def _move_tensor_to_target(
        self,
        tensor: torch.Tensor,
        target_device: Optional[torch.device | str],
        dtype: Optional[torch.dtype] = None,
        non_blocking: Optional[bool] = None,
    ) -> torch.Tensor:
        target = self._resolve_device(target_device)
        target_dtype = tensor.dtype if dtype is None else dtype
        move_non_blocking = bool(self._non_blocking_transfer if non_blocking is None else non_blocking)
        if target.type != "cuda":
            move_non_blocking = False
        elif non_blocking is None:
            move_non_blocking = move_non_blocking and tensor.device.type == "cpu" and tensor.is_pinned()
        return self._move_tensor_to_resolved_target(
            tensor,
            target,
            target_dtype=target_dtype,
            move_non_blocking=move_non_blocking,
        )

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

        t0 = time.perf_counter() if self._timing_enabled else 0.0
        map_t = torch.from_numpy(np_array)
        if map_t.dtype != dtype:
            map_t = map_t.to(dtype=dtype)
        map_t = map_t.unsqueeze(0)
        map_t = self._finalize_cpu_tensor(map_t, dtype=dtype)
        if self._timing_enabled:
            self.tensor_transfer_time += time.perf_counter() - t0
        return map_t

    def _to_cpu_batch_frontier_tokens(self, frontier_tokens, frontier_token_mask=None):
        if isinstance(frontier_tokens, np.ndarray):
            frontier_tokens_t = torch.from_numpy(self._ensure_contiguous_numpy(frontier_tokens))
        elif isinstance(frontier_tokens, torch.Tensor):
            frontier_tokens_t = frontier_tokens.detach()
        else:
            raise TypeError("frontier_tokens must be np.ndarray or torch.Tensor")

        if frontier_tokens_t.dim() == 2:
            frontier_tokens_t = frontier_tokens_t.unsqueeze(0)
        if frontier_tokens_t.dim() != 3:
            raise ValueError(
                "frontier_tokens must be [B,K,D] or [K,D], "
                f"got {tuple(frontier_tokens_t.shape)}"
            )

        if frontier_token_mask is None:
            mask_t = torch.ones(frontier_tokens_t.shape[:2], dtype=torch.bool)
        else:
            if isinstance(frontier_token_mask, np.ndarray):
                mask_t = torch.from_numpy(self._ensure_contiguous_numpy(frontier_token_mask))
            elif isinstance(frontier_token_mask, torch.Tensor):
                mask_t = frontier_token_mask.detach()
            else:
                raise TypeError("frontier_token_mask must be np.ndarray or torch.Tensor")

            if mask_t.dim() == 1:
                mask_t = mask_t.unsqueeze(0)
            if mask_t.shape != frontier_tokens_t.shape[:2]:
                raise ValueError(
                    "frontier_token_mask shape mismatch: "
                    f"expected {tuple(frontier_tokens_t.shape[:2])}, got {tuple(mask_t.shape)}"
                )

        frontier_tokens_t = self._finalize_cpu_tensor(frontier_tokens_t, dtype=torch.float32)
        mask_t = self._finalize_cpu_tensor(mask_t, dtype=torch.bool)
        return frontier_tokens_t, mask_t

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

        move_non_blocking = self._resolve_batch_move_non_blocking(
            source_tensors,
            target,
            non_blocking=non_blocking,
        )
        return {
            key: self._move_tensor_to_resolved_target(
                state_batch[key],
                target,
                target_dtype=dtype,
                move_non_blocking=move_non_blocking,
            )
            for key, dtype in self._STATE_BATCH_DTYPES
        }

    def _build_shared_step_artifacts(
        self,
        cum_map,
        frontier_u8=None,
    ) -> SharedStepArtifacts:
        t0 = time.perf_counter() if self._timing_enabled else 0.0
        frontier_use = cum_map.get_frontier_u8(refresh=False) if frontier_u8 is None else frontier_u8
        frontier_stats = cum_map.get_frontier_derived_stats(refresh=False, frontier_u8=frontier_use)
        shared = SharedStepArtifacts(
            frontier_u8=frontier_stats.frontier_u8,
            frontier_stats=frontier_stats,
        )
        if self._timing_enabled:
            self.shared_artifact_time += time.perf_counter() - t0
        return shared

    def build_shared_step_artifacts(
        self,
        cum_map,
        frontier_u8=None,
    ) -> SharedStepArtifacts:
        """Public helper so callers can reuse same-step frontier artifacts across builders."""
        return self._build_shared_step_artifacts(cum_map, frontier_u8=frontier_u8)

    def build_single_state_tensors(
        self,
        cum_map,
        agent_state: Tuple[int, int],
        frontier_tokens=None,
        frontier_token_mask=None,
        frontier_u8=None,
        shared_artifacts: Optional[SharedStepArtifacts] = None,
        target_device: Optional[torch.device | str] = None,
    ) -> Dict[str, torch.Tensor]:
        if shared_artifacts is None:
            shared_artifacts = self._build_shared_step_artifacts(
                cum_map,
                frontier_u8=frontier_u8,
            )

        t0 = time.perf_counter() if self._timing_enabled else 0.0
        near_np = self.near_builder.build(
            cum_map,
            agent_state,
            shared_artifacts=shared_artifacts,
        )
        if self._timing_enabled:
            self.near_build_time += time.perf_counter() - t0

        t0 = time.perf_counter() if self._timing_enabled else 0.0
        mid_map_np = cum_map.build_mid_map(
            agent_state,
            config=self.cfg.mid_map,
            shared_artifacts=shared_artifacts,
        )
        if self._timing_enabled:
            self.mid_build_time += time.perf_counter() - t0

        if frontier_tokens is None:
            t0 = time.perf_counter() if self._timing_enabled else 0.0
            token_np, token_mask_np = self.frontier_token_builder.build(
                cum_map,
                agent_state,
                shared_artifacts=shared_artifacts,
                world_window_shape=self.cfg.mid_map.world_window_shape,
            )
            frontier_tokens = token_np
            frontier_token_mask = token_mask_np
            if self._timing_enabled:
                self.frontier_token_build_time += time.perf_counter() - t0

        near_t = self._to_cpu_batch_map(near_np, dtype=torch.float32)
        mid_map_t = self._to_cpu_batch_map(mid_map_np, dtype=torch.float32)
        frontier_tokens_t, frontier_token_mask_t = self._to_cpu_batch_frontier_tokens(
            frontier_tokens,
            frontier_token_mask=frontier_token_mask,
        )

        if frontier_tokens_t.shape[0] != 1:
            raise ValueError(
                "build_single_state_tensors expects single frontier-token batch [K,D] or [1,K,D]"
            )

        state_batch = {
            "near_map": near_t,
            "mid_map": mid_map_t,
            "frontier_tokens": frontier_tokens_t,
            "frontier_token_mask": frontier_token_mask_t,
        }
        if target_device is None:
            return state_batch
        target = self._resolve_device(target_device)
        if target.type == "cpu":
            return state_batch
        return self.move_state_batch(state_batch, target_device=target)

    def get_timing_stats(self) -> Dict[str, float]:
        return {
            "shared_artifact_time": float(self.shared_artifact_time),
            "near_build_time": float(self.near_build_time),
            "mid_build_time": float(self.mid_build_time),
            "frontier_token_build_time": float(self.frontier_token_build_time),
            "tensor_transfer_time": float(self.tensor_transfer_time),
        }


def action_mask_from_valid_indices(
    valid_action_indices: Sequence[Iterable[int]] | Iterable[int],
    action_dim: int = ACTION_DIM,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Build bool action mask from valid action indices.

    Input:
      - single sample: Iterable[int]
      - batch samples: Sequence[Iterable[int]]

    Output:
      mask [B, action_dim] bool
    """
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
    """
    Apply legal-action mask to Q values.

    action_mask: bool tensor, True means valid action.
    """
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

    masked = q_values.masked_fill(~action_mask.to(dtype=torch.bool), float(invalid_fill))
    return masked.squeeze(0) if squeeze_back else masked


def select_greedy_action(
    q_values: torch.Tensor,
    action_mask: Optional[torch.Tensor] = None,
    valid_action_indices: Optional[Sequence[Iterable[int]] | Iterable[int]] = None,
) -> torch.Tensor:
    """
    Greedy action selection with legal-action support.

    Priority:
      1) explicit action_mask
      2) valid_action_indices -> converted mask
      3) no mask -> plain argmax

    Returns:
      action_idx [B] (or scalar tensor for single sample after squeeze)
    """
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

    cfg = ExplorationQConfig(
        global_side_encoder=GlobalSideEncoderConfig(
            frontier_token_input_dim=FRONTIER_REGION_TOKEN_FIELD_COUNT,
            global_context_dim=256,
        ),
        near_encoder=NearMapEncoderConfig(
            near_in_channels=LOCAL_STATE_CHANNEL_COUNT,
            raw_near_summary_dim=128,
        ),
        decision_head=DecisionHeadConfig(
            raw_near_summary_dim=128,
            global_context_dim=256,
            hidden_dim=192,
            action_dim=ACTION_DIM,
        ),
    )

    net = ExplorationQNetwork(cfg)

    adapter_cfg = StateAdapterConfig(
        near_map=LocalStateConfig(local_window_shape=(21, 21)),
        mid_map=MidMapConfig(mid_map_shape=(24, 24), world_window_shape=(128, 128)),
        frontier_tokens=FrontierRegionTokenConfig(top_k=8),
    )

    bsz = 3
    near_h, near_w = adapter_cfg.near_map.local_window_shape
    mid_h, mid_w = adapter_cfg.mid_map.mid_map_shape

    near_map = torch.rand(bsz, LOCAL_STATE_CHANNEL_COUNT, near_h, near_w)
    mid_map = torch.rand(bsz, MID_MAP_CHANNEL_COUNT, mid_h, mid_w)
    frontier_tokens = torch.rand(bsz, 8, FRONTIER_REGION_TOKEN_FIELD_COUNT)
    frontier_token_mask = torch.ones(bsz, 8, dtype=torch.bool)
    frontier_token_mask[0, :] = False
    frontier_token_mask[1, 6:] = False

    q_values, aux = net(
        near_map,
        mid_map,
        frontier_tokens,
        frontier_token_mask=frontier_token_mask,
        return_aux=True,
    )
    assert q_values.shape == (bsz, ACTION_DIM)
    assert aux["raw_near_summary"].shape == (bsz, cfg.near_encoder.raw_near_summary_dim)
    assert torch.isfinite(aux["global_context"]).all()
    assert torch.isfinite(q_values).all()

    mask = action_mask_from_valid_indices([[0, 2, 3], [1, 4, 7], [0, 6]])
    act = select_greedy_action(q_values, action_mask=mask)
    assert act.shape == (bsz,)

    grid, start = RandomMapGenerator(30, 40, 5, 0.2).generate_map()
    obs = LocalObservationModel(grid, start)
    snap, _ = obs.observe(start)
    valid_idxs = GridTopology.valid_action_indices(GridTopology.free_mask(grid), start)
    cum_map = CumulativeBeliefMap(grid, start, snap)
    frontier_u8 = cum_map.get_frontier_u8(refresh=True)

    adapter = StateTensorAdapter(cfg=adapter_cfg, device="cpu")
    tensors = adapter.build_single_state_tensors(
        cum_map,
        start,
        frontier_tokens=None,
        frontier_token_mask=None,
        frontier_u8=frontier_u8,
    )

    q_single = net(
        tensors["near_map"],
        tensors["mid_map"],
        tensors["frontier_tokens"],
        frontier_token_mask=tensors["frontier_token_mask"],
        return_aux=False,
    )
    legal_mask = action_mask_from_valid_indices(valid_idxs, device=q_single.device)
    a_single = select_greedy_action(q_single, action_mask=legal_mask)

    assert q_single.shape == (1, ACTION_DIM)
    assert tensors["near_map"].shape == (1, LOCAL_STATE_CHANNEL_COUNT, near_h, near_w)
    assert tensors["mid_map"].shape == (1, MID_MAP_CHANNEL_COUNT, mid_h, mid_w)
    assert tensors["frontier_tokens"].shape[2] == FRONTIER_REGION_TOKEN_FIELD_COUNT
    assert int(a_single.item()) in set(valid_idxs)

    print("ExplorationQNetwork smoke test passed")
    print(
        "near_map:",
        tuple(tensors["near_map"].shape),
        "mid_map:",
        tuple(tensors["mid_map"].shape),
        "frontier_tokens:",
        tuple(tensors["frontier_tokens"].shape),
        "q_values:",
        tuple(q_values.shape),
        "single_action:",
        int(a_single.item()),
    )


if __name__ == "__main__":
    _smoke_test()
