from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from env.advantage_state_builder import ADVANTAGE_CANVAS_CHANNEL_COUNT
from env.grid_topology import ACTIONS_8


@dataclass(frozen=True)
class AdvantageEncoderConfig:
    canvas_in_channels: int = ADVANTAGE_CANVAS_CHANNEL_COUNT
    action_dim: int = len(ACTIONS_8)
    base_dim: int = 64
    action_state_dim: int = 160
    dropout: float = 0.1


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


class AdvantageCanvasEncoder(nn.Module):
    """
    Encode the local decision canvas into one action-specific state per move.

    The same semantic canvas is shared across all actions, but each action reads
    it through its own directional spatial template and landing-cell view.
    """

    def __init__(self, cfg: Optional[AdvantageEncoderConfig] = None):
        super().__init__()
        self.cfg = cfg if cfg is not None else AdvantageEncoderConfig()
        d = int(self.cfg.base_dim)
        self.backbone = nn.Sequential(
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
        self.action_embedding = nn.Embedding(self.cfg.action_dim, d)
        self.action_state = nn.Sequential(
            nn.LayerNorm(d * 4),
            nn.Linear(d * 4, self.cfg.action_state_dim),
            nn.GELU(),
            nn.Dropout(self.cfg.dropout),
            nn.Linear(self.cfg.action_state_dim, self.cfg.action_state_dim),
            nn.LayerNorm(self.cfg.action_state_dim),
        )
        self._direction_mask_cache: dict[tuple[int, int, torch.device, torch.dtype], torch.Tensor] = {}

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
            mask_sum = torch.clamp(mask.sum(), min=1e-6)
            masks.append((mask / mask_sum).to(dtype=dtype))
        stacked = torch.stack(masks, dim=0)
        self._direction_mask_cache[key] = stacked
        return stacked

    def forward(
        self,
        canvas: torch.Tensor,
        *,
        return_aux: bool = False,
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]] | torch.Tensor:
        if canvas.dim() != 4:
            raise ValueError(f"canvas must be [B,C,H,W], got {tuple(canvas.shape)}")
        if int(canvas.shape[1]) != int(self.cfg.canvas_in_channels):
            raise ValueError(
                "canvas channel mismatch: "
                f"expected {self.cfg.canvas_in_channels}, got {int(canvas.shape[1])}"
            )

        feat = self.backbone(canvas)
        bsz, channels, h, w = feat.shape
        direction_masks = self._directional_masks(h, w, device=feat.device, dtype=feat.dtype)
        pooled = torch.einsum("bchw,ahw->bac", feat, direction_masks)

        center_r = h // 2
        center_c = w // 2
        center_feat = feat[:, :, center_r, center_c].unsqueeze(1).expand(-1, self.cfg.action_dim, -1)
        landing_feats = []
        for dr, dc in ACTIONS_8:
            landing_r = max(0, min(h - 1, center_r + int(dr)))
            landing_c = max(0, min(w - 1, center_c + int(dc)))
            landing_feats.append(feat[:, :, landing_r, landing_c])
        landing_feat = torch.stack(landing_feats, dim=1)

        action_ids = torch.arange(self.cfg.action_dim, device=feat.device)
        action_embed = self.action_embedding(action_ids).unsqueeze(0).expand(bsz, -1, -1)
        action_state = self.action_state(torch.cat([pooled, landing_feat, center_feat, action_embed], dim=-1))

        if not return_aux:
            return action_state

        aux: Dict[str, torch.Tensor] = {
            "advantage_canvas_frontier_mean": canvas[:, 3].mean(dim=(1, 2)),
            "advantage_canvas_frontier_block_area_mean": canvas[:, 4].mean(dim=(1, 2)),
        }
        return action_state, aux
