from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn

from env.grid_topology import ACTIONS_8


LOCAL_STATE_CHANNELS = ("known_free", "known_obstacle", "unknown")
LOCAL_STATE_PATCH_SIZE = 21
ACTION_DIM = len(ACTIONS_8)


@dataclass(frozen=True)
class LocalStateQConfig:
    local_state_channels: tuple[str, ...] = LOCAL_STATE_CHANNELS
    local_state_patch_size: int = LOCAL_STATE_PATCH_SIZE
    action_dim: int = ACTION_DIM
    hidden_channels: int = 48
    feature_dim: int = 128
    dropout: float = 0.05

    def __post_init__(self) -> None:
        channels = tuple(str(channel) for channel in self.local_state_channels)
        if channels != LOCAL_STATE_CHANNELS:
            raise ValueError(f"LocalStateQNetwork expects channels {LOCAL_STATE_CHANNELS}, got {channels}")
        if int(self.local_state_patch_size) <= 0 or int(self.local_state_patch_size) % 2 == 0:
            raise ValueError("local_state_patch_size must be a positive odd integer")
        if int(self.action_dim) != ACTION_DIM:
            raise ValueError(f"action_dim must be {ACTION_DIM}, got {self.action_dim}")
        object.__setattr__(self, "local_state_channels", channels)


class _ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, *, stride: int = 1):
        super().__init__()
        groups = max(1, min(8, int(out_channels)))
        while int(out_channels) % groups != 0:
            groups -= 1
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class LocalStateQNetwork(nn.Module):
    """
    Lightweight DDQN Q-network for the Anew_C local-state learning baseline.

    The network consumes only a 3-channel belief patch carried through the
    training interface as the first tensor argument. Value-tree tensors are
    accepted for interface compatibility and intentionally ignored.
    """

    def __init__(self, cfg: Optional[LocalStateQConfig] = None):
        super().__init__()
        self.cfg = cfg if cfg is not None else LocalStateQConfig()
        in_channels = len(self.cfg.local_state_channels)
        width = int(self.cfg.hidden_channels)
        self.encoder = nn.Sequential(
            _ConvBlock(in_channels, width),
            _ConvBlock(width, width),
            _ConvBlock(width, width * 2, stride=2),
            _ConvBlock(width * 2, width * 2),
            _ConvBlock(width * 2, width * 2, stride=2),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
        self.head = nn.Sequential(
            nn.LayerNorm(width * 2),
            nn.Linear(width * 2, int(self.cfg.feature_dim)),
            nn.GELU(),
            nn.Dropout(float(self.cfg.dropout)),
            nn.Linear(int(self.cfg.feature_dim), int(self.cfg.action_dim)),
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
            raise ValueError(f"local state tensor must be [B,C,H,W], got {tuple(advantage_canvas.shape)}")
        if int(advantage_canvas.shape[1]) != len(self.cfg.local_state_channels):
            raise ValueError(
                "local state channel mismatch: "
                f"expected {len(self.cfg.local_state_channels)}, got {int(advantage_canvas.shape[1])}"
            )
        patch_size = int(self.cfg.local_state_patch_size)
        if tuple(int(v) for v in advantage_canvas.shape[-2:]) != (patch_size, patch_size):
            raise ValueError(
                "local state patch size mismatch: "
                f"expected {(patch_size, patch_size)}, got {tuple(advantage_canvas.shape[-2:])}"
            )

        features = self.encoder(advantage_canvas)
        q_values = self.head(features)
        if not return_aux:
            return q_values

        channel_index = {name: idx for idx, name in enumerate(self.cfg.local_state_channels)}
        aux: Dict[str, torch.Tensor] = {
            "local_state_known_free_mean": advantage_canvas[:, channel_index["known_free"]].mean(dim=(1, 2)),
            "local_state_known_obstacle_mean": advantage_canvas[:, channel_index["known_obstacle"]].mean(dim=(1, 2)),
            "local_state_unknown_mean": advantage_canvas[:, channel_index["unknown"]].mean(dim=(1, 2)),
        }
        return q_values, aux


def local_state_model_parameter_count(patch_size: int = LOCAL_STATE_PATCH_SIZE) -> int:
    model = LocalStateQNetwork(LocalStateQConfig(local_state_patch_size=int(patch_size)))
    return int(sum(parameter.numel() for parameter in model.parameters()))
