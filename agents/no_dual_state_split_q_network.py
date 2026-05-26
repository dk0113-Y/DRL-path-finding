from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from env.advantage_state_builder import FINAL_4CH_ADVANTAGE_CANVAS_CHANNELS
from env.grid_topology import ACTIONS_8
from env.value_state_builder import VALUE_BLOCK_FEATURE_COUNT, VALUE_ENTRY_FEATURE_COUNT


ACTION_DIM = len(ACTIONS_8)


@dataclass(frozen=True)
class NoDualStateSplitQConfig:
    canvas_in_channels: int = len(FINAL_4CH_ADVANTAGE_CANVAS_CHANNELS)
    canvas_channels: tuple[str, ...] = FINAL_4CH_ADVANTAGE_CANVAS_CHANNELS
    action_dim: int = ACTION_DIM
    local_base_dim: int = 64
    local_action_dim: int = 192
    value_model_dim: int = 128
    value_summary_dim: int = 160
    q_hidden_dim: int = 192
    dropout: float = 0.1

    def __post_init__(self) -> None:
        channels = tuple(str(channel) for channel in self.canvas_channels)
        if channels != FINAL_4CH_ADVANTAGE_CANVAS_CHANNELS:
            raise ValueError(
                "NoDualStateSplitQNetwork expects final A_new canvas channels "
                f"{FINAL_4CH_ADVANTAGE_CANVAS_CHANNELS}; got {channels}"
            )
        if int(self.canvas_in_channels) != len(channels):
            raise ValueError(
                "canvas_in_channels must match canvas_channels length: "
                f"{self.canvas_in_channels} != {len(channels)}"
            )
        if int(self.action_dim) != ACTION_DIM:
            raise ValueError(f"action_dim must be {ACTION_DIM}, got {self.action_dim}")
        object.__setattr__(self, "canvas_channels", channels)


class _ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        x = F.gelu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return F.gelu(x + identity)


def _masked_mean(values: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    mask = mask.to(dtype=torch.bool)
    while mask.dim() < values.dim():
        mask = mask.unsqueeze(-1)
    mask_f = mask.to(dtype=values.dtype)
    numerator = (values * mask_f).sum(dim=dim)
    denominator = mask_f.sum(dim=dim).clamp_min(1.0)
    return numerator / denominator


class _FlattenedValueSummaryEncoder(nn.Module):
    """
    Mask-aware flattened summary of the structured value-tree tensors.

    This encoder consumes the same block/entry feature tensors used by A_new, but
    collapses them into one conditioning vector instead of producing a separate
    state-value representation.
    """

    def __init__(self, cfg: NoDualStateSplitQConfig):
        super().__init__()
        self.cfg = cfg
        d = int(cfg.value_model_dim)
        self.block_encoder = nn.Sequential(
            nn.LayerNorm(VALUE_BLOCK_FEATURE_COUNT),
            nn.Linear(VALUE_BLOCK_FEATURE_COUNT, d),
            nn.GELU(),
            nn.Dropout(float(cfg.dropout)),
            nn.Linear(d, d),
            nn.LayerNorm(d),
        )
        self.entry_encoder = nn.Sequential(
            nn.LayerNorm(VALUE_ENTRY_FEATURE_COUNT),
            nn.Linear(VALUE_ENTRY_FEATURE_COUNT, d),
            nn.GELU(),
            nn.Dropout(float(cfg.dropout)),
            nn.Linear(d, d),
            nn.LayerNorm(d),
        )
        self.summary = nn.Sequential(
            nn.LayerNorm((d * 2) + 2),
            nn.Linear((d * 2) + 2, int(cfg.value_summary_dim)),
            nn.GELU(),
            nn.Dropout(float(cfg.dropout)),
            nn.Linear(int(cfg.value_summary_dim), int(cfg.value_summary_dim)),
            nn.LayerNorm(int(cfg.value_summary_dim)),
        )

    def forward(
        self,
        block_features: torch.Tensor,
        entry_features: torch.Tensor,
        block_mask: torch.Tensor,
        entry_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if block_features.dim() != 3:
            raise ValueError(f"value_block_features must be [B,N,D], got {tuple(block_features.shape)}")
        if entry_features.dim() != 4:
            raise ValueError(f"value_entry_features must be [B,N,M,D], got {tuple(entry_features.shape)}")
        if block_features.shape[-1] != VALUE_BLOCK_FEATURE_COUNT:
            raise ValueError(
                f"value block feature dim must be {VALUE_BLOCK_FEATURE_COUNT}, got {block_features.shape[-1]}"
            )
        if entry_features.shape[-1] != VALUE_ENTRY_FEATURE_COUNT:
            raise ValueError(
                f"value entry feature dim must be {VALUE_ENTRY_FEATURE_COUNT}, got {entry_features.shape[-1]}"
            )
        if block_mask.shape != block_features.shape[:2]:
            raise ValueError("value_block_mask shape mismatch")
        if entry_mask.shape != entry_features.shape[:3]:
            raise ValueError("value_entry_mask shape mismatch")

        block_mask = block_mask.to(dtype=torch.bool)
        entry_mask = entry_mask.to(dtype=torch.bool)
        block_token = self.block_encoder(block_features) * block_mask.to(dtype=block_features.dtype).unsqueeze(-1)
        entry_token = self.entry_encoder(entry_features) * entry_mask.to(dtype=entry_features.dtype).unsqueeze(-1)

        block_summary = _masked_mean(block_token, block_mask, dim=1)
        batch, block_count, entry_count, entry_dim = entry_token.shape
        entry_summary = _masked_mean(
            entry_token.reshape(batch, block_count * entry_count, entry_dim),
            entry_mask.reshape(batch, block_count * entry_count),
            dim=1,
        )
        block_count_norm = block_mask.to(dtype=block_features.dtype).sum(dim=1, keepdim=True) / float(
            max(1, int(block_features.shape[1]))
        )
        entry_count_norm = entry_mask.to(dtype=entry_features.dtype).sum(dim=(1, 2), keepdim=False).unsqueeze(-1) / float(
            max(1, int(entry_features.shape[1] * entry_features.shape[2]))
        )
        summary_input = torch.cat([block_summary, entry_summary, block_count_norm, entry_count_norm], dim=-1)
        value_summary = self.summary(summary_input)
        aux = {
            "no_dual_value_valid_block_count": block_mask.to(dtype=torch.float32).sum(dim=1),
            "no_dual_value_valid_entry_count": entry_mask.to(dtype=torch.float32).sum(dim=(1, 2)),
            "no_dual_value_summary_norm": value_summary.norm(dim=-1),
        }
        return value_summary, aux


class NoDualStateSplitQNetwork(nn.Module):
    """
    A_new E structural ablation network.

    The model keeps the final four-channel local canvas and the structured
    value-tree tensors, flattens the value-tree information into one conditioning
    vector, injects that vector into action-conditioned local CNN features, and
    directly predicts Q(s, a) for the eight actions.
    """

    def __init__(self, cfg: Optional[NoDualStateSplitQConfig] = None):
        super().__init__()
        self.cfg = cfg if cfg is not None else NoDualStateSplitQConfig()
        d = int(self.cfg.local_base_dim)
        self.local_encoder = nn.Sequential(
            nn.Conv2d(self.cfg.canvas_in_channels, d, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(d),
            nn.GELU(),
            _ResidualBlock(d),
            _ResidualBlock(d),
            nn.Conv2d(d, d, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(d),
            nn.GELU(),
            _ResidualBlock(d),
        )
        self.action_embedding = nn.Embedding(int(self.cfg.action_dim), d)
        self.local_action_projector = nn.Sequential(
            nn.LayerNorm(d * 4),
            nn.Linear(d * 4, int(self.cfg.local_action_dim)),
            nn.GELU(),
            nn.Dropout(float(self.cfg.dropout)),
            nn.Linear(int(self.cfg.local_action_dim), int(self.cfg.local_action_dim)),
            nn.LayerNorm(int(self.cfg.local_action_dim)),
        )
        self.value_summary_encoder = _FlattenedValueSummaryEncoder(self.cfg)
        self.q_head = nn.Sequential(
            nn.LayerNorm(int(self.cfg.local_action_dim) + int(self.cfg.value_summary_dim)),
            nn.Linear(int(self.cfg.local_action_dim) + int(self.cfg.value_summary_dim), int(self.cfg.q_hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(self.cfg.dropout)),
            nn.Linear(int(self.cfg.q_hidden_dim), int(self.cfg.q_hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(self.cfg.dropout)),
            nn.Linear(int(self.cfg.q_hidden_dim), 1),
        )
        self._direction_mask_cache: dict[tuple[int, int, torch.device, torch.dtype], torch.Tensor] = {}

    @staticmethod
    def _is_inference_tensor(tensor: torch.Tensor) -> bool:
        if hasattr(torch, "is_inference"):
            try:
                return bool(torch.is_inference(tensor))
            except Exception:
                pass
        marker = getattr(tensor, "is_inference", None)
        if callable(marker):
            try:
                return bool(marker())
            except Exception:
                return False
        return False

    def _directional_masks(
        self,
        h: int,
        w: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        key = (int(h), int(w), device, dtype)
        cached = self._direction_mask_cache.get(key)
        if cached is not None:
            if self._is_inference_tensor(cached):
                cached = cached.clone()
                self._direction_mask_cache[key] = cached
            return cached

        center_r = h // 2
        center_c = w // 2
        rows = torch.arange(h, device=device, dtype=torch.float32) - float(center_r)
        cols = torch.arange(w, device=device, dtype=torch.float32) - float(center_c)
        rr, cc = torch.meshgrid(rows, cols, indexing="ij")
        masks = []
        for dr, dc in ACTIONS_8:
            dr_f = float(dr)
            dc_f = float(dc)
            forward = (rr * dr_f) + (cc * dc_f)
            lateral = torch.abs((rr * dc_f) - (cc * dr_f))
            mask = torch.exp(-0.5 * (lateral / 1.2).square()) * torch.exp(-0.5 * ((forward - 1.5) / 1.5).square())
            mask = mask * (forward >= 0.0).to(mask.dtype)
            landing_r = int(center_r + dr)
            landing_c = int(center_c + dc)
            if 0 <= landing_r < h and 0 <= landing_c < w:
                landing_boost = torch.zeros_like(mask)
                landing_boost[landing_r, landing_c] = 1.0
                mask = mask + (2.0 * landing_boost)
            masks.append((mask / torch.clamp(mask.sum(), min=1e-6)).to(dtype=dtype))
        stacked = torch.stack(masks, dim=0)
        if self._is_inference_tensor(stacked):
            stacked = stacked.clone()
        self._direction_mask_cache[key] = stacked
        return stacked

    def _local_action_features(self, canvas: torch.Tensor) -> torch.Tensor:
        feat = self.local_encoder(canvas)
        batch, _channels, h, w = feat.shape
        direction_masks = self._directional_masks(h, w, device=feat.device, dtype=feat.dtype)
        pooled = torch.einsum("bchw,ahw->bac", feat, direction_masks)

        center_r = h // 2
        center_c = w // 2
        center_feat = feat[:, :, center_r, center_c].unsqueeze(1).expand(-1, int(self.cfg.action_dim), -1)
        landing_feats = []
        for dr, dc in ACTIONS_8:
            landing_r = max(0, min(h - 1, center_r + int(dr)))
            landing_c = max(0, min(w - 1, center_c + int(dc)))
            landing_feats.append(feat[:, :, landing_r, landing_c])
        landing_feat = torch.stack(landing_feats, dim=1)
        action_ids = torch.arange(int(self.cfg.action_dim), device=feat.device)
        action_embed = self.action_embedding(action_ids).unsqueeze(0).expand(batch, -1, -1)
        return self.local_action_projector(torch.cat([pooled, landing_feat, center_feat, action_embed], dim=-1))

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
        if advantage_canvas.dim() != 4:
            raise ValueError(f"advantage_canvas must be [B,C,H,W], got {tuple(advantage_canvas.shape)}")
        if int(advantage_canvas.shape[1]) != int(self.cfg.canvas_in_channels):
            raise ValueError(
                "advantage canvas channel mismatch: "
                f"expected {self.cfg.canvas_in_channels}, got {int(advantage_canvas.shape[1])}"
            )

        local_action_features = self._local_action_features(advantage_canvas)
        value_summary, value_aux = self.value_summary_encoder(
            value_block_features,
            value_entry_features,
            value_block_mask,
            value_entry_mask,
        )
        value_per_action = value_summary.unsqueeze(1).expand(-1, int(self.cfg.action_dim), -1)
        fused = torch.cat([local_action_features, value_per_action], dim=-1)
        q_values = self.q_head(fused).squeeze(-1)

        if not return_aux:
            return q_values

        channel_index = {name: idx for idx, name in enumerate(tuple(self.cfg.canvas_channels))}

        def channel_mean(channel_name: str) -> torch.Tensor:
            idx = channel_index.get(channel_name)
            if idx is None:
                return advantage_canvas.new_zeros((advantage_canvas.shape[0],))
            return advantage_canvas[:, idx].mean(dim=(1, 2))

        aux: Dict[str, torch.Tensor] = {
            "no_dual_local_action_feature_norm": local_action_features.norm(dim=-1).mean(dim=1),
            "advantage_canvas_visit_pressure_mean": channel_mean("visit_count_log_norm"),
            "advantage_canvas_trajectory_mean": channel_mean("recent_trajectory_decay"),
        }
        aux.update(value_aux)
        return q_values, aux


def no_dual_state_split_model_parameter_count() -> int:
    model = NoDualStateSplitQNetwork(NoDualStateSplitQConfig())
    return int(sum(parameter.numel() for parameter in model.parameters()))
