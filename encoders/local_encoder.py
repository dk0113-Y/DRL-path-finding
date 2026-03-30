from __future__ import annotations

"""Legacy near-map encoder path kept only as historical reference."""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from env.local_state_builder import LOCAL_STATE_CHANNEL_COUNT


@dataclass(frozen=True)
class NearMapEncoderConfig:
    near_in_channels: int = LOCAL_STATE_CHANNEL_COUNT
    near_base_dim: int = 64
    raw_near_summary_dim: int = 128
    dropout: float = 0.1


class NearResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.gelu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = x + identity
        x = F.gelu(x)
        return x


class NearMapEncoder(nn.Module):
    """
    Local geometry encoder for the near map.

    Input:
      near_map [B, C_near, H_near, W_near]

    Output:
      near_map_features [B, C_map, H', W']
    """

    def __init__(self, cfg: NearMapEncoderConfig):
        super().__init__()
        d = cfg.near_base_dim
        out_ch = d * 2

        self.stem = nn.Sequential(
            nn.Conv2d(cfg.near_in_channels, d, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(d),
            nn.GELU(),
        )
        self.block1 = NearResidualBlock(d)

        self.down1 = nn.Sequential(
            nn.Conv2d(d, out_ch, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        )
        self.block2 = NearResidualBlock(out_ch)

        self.down2 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        )
        self.block3 = NearResidualBlock(out_ch)

    def forward(self, near_map: torch.Tensor) -> torch.Tensor:
        x = self.stem(near_map)
        x = self.block1(x)
        x = self.down1(x)
        x = self.block2(x)
        x = self.down2(x)
        near_map_features = self.block3(x)
        return near_map_features


class RawNearSummaryHead(nn.Module):
    """
    Summarize near-map features into an unguided local execution vector.

    The resulting raw_near_summary is computed before any global-context
    interaction and is the only local vector consumed by the decision head.
    """

    def __init__(self, cfg: NearMapEncoderConfig):
        super().__init__()
        map_dim = cfg.near_base_dim * 2

        self.proj = nn.Sequential(
            nn.Linear(map_dim * 2, cfg.raw_near_summary_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.raw_near_summary_dim, cfg.raw_near_summary_dim),
            nn.LayerNorm(cfg.raw_near_summary_dim),
        )

    def forward(self, near_map_features: torch.Tensor) -> torch.Tensor:
        map_avg = F.adaptive_avg_pool2d(near_map_features, 1).flatten(1)
        map_max = F.adaptive_max_pool2d(near_map_features, 1).flatten(1)
        raw_near_summary = self.proj(torch.cat([map_avg, map_max], dim=-1))
        return raw_near_summary


class RawNearSummaryEncoder(nn.Module):
    """
    Local near-side encoder.

    Data flow:
      near_map -> NearMapEncoder -> near_map_features -> raw_near_summary
    """

    def __init__(self, cfg: Optional[NearMapEncoderConfig] = None):
        super().__init__()
        self.cfg = cfg if cfg is not None else NearMapEncoderConfig()

        self.near_map_encoder = NearMapEncoder(self.cfg)
        self.raw_summary_head = RawNearSummaryHead(self.cfg)

    def forward(self, near_map: torch.Tensor) -> torch.Tensor:
        if near_map.dim() != 4:
            raise ValueError(f"near_map must be 4D [B,C,H,W], got shape={tuple(near_map.shape)}")
        if near_map.shape[1] != self.cfg.near_in_channels:
            raise ValueError(
                f"near_map channel mismatch: expected {self.cfg.near_in_channels}, got {near_map.shape[1]}"
            )

        near_map_features = self.near_map_encoder(near_map)
        raw_near_summary = self.raw_summary_head(near_map_features)
        return raw_near_summary


def _smoke_test() -> None:
    bsz = 4
    near_cfg = NearMapEncoderConfig(
        near_in_channels=LOCAL_STATE_CHANNEL_COUNT,
        raw_near_summary_dim=128,
    )
    near_model = RawNearSummaryEncoder(near_cfg)

    near_map = torch.rand(bsz, LOCAL_STATE_CHANNEL_COUNT, 21, 21)
    raw_near_summary = near_model(near_map)

    assert raw_near_summary.shape == (bsz, near_cfg.raw_near_summary_dim)
    assert torch.isfinite(raw_near_summary).all()

    print("RawNearSummaryEncoder smoke test passed")
    print("raw_near_summary:", tuple(raw_near_summary.shape))


if __name__ == "__main__":
    _smoke_test()
