from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from env.core_cummap import MID_MAP_CHANNEL_COUNT
from env.frontier_token_builder import FRONTIER_REGION_TOKEN_FIELD_COUNT


@dataclass(frozen=True)
class GlobalSideEncoderConfig:
    """Configuration for the global side encoder."""

    mid_map_channels: int = MID_MAP_CHANNEL_COUNT
    frontier_token_input_dim: int = FRONTIER_REGION_TOKEN_FIELD_COUNT
    map_base_dim: int = 64
    map_vec_dim: int = 128
    token_model_dim: int = 128
    token_vec_dim: int = 128
    global_context_dim: int = 256
    fusion_hidden_dim: int = 256
    dropout: float = 0.1


class ResidualBlock(nn.Module):
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


class RasterMapEncoder(nn.Module):
    """Shared raster encoder template for agent-centered dense map inputs."""

    def __init__(self, in_channels: int, cfg: GlobalSideEncoderConfig):
        super().__init__()
        d = cfg.map_base_dim
        out_ch = d * 2

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, d, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(d),
            nn.GELU(),
        )
        self.block1 = ResidualBlock(d)

        self.down1 = nn.Sequential(
            nn.Conv2d(d, out_ch, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        )
        self.block2 = ResidualBlock(out_ch)

        self.down2 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        )
        self.block3 = ResidualBlock(out_ch)

        self.vec_head = nn.Sequential(
            nn.Linear(out_ch * 2, cfg.map_vec_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.map_vec_dim, cfg.map_vec_dim),
        )

    def forward(self, map_tensor: torch.Tensor) -> torch.Tensor:
        x = self.stem(map_tensor)
        x = self.block1(x)
        x = self.down1(x)
        x = self.block2(x)
        x = self.down2(x)
        x = self.block3(x)

        avg_pool = F.adaptive_avg_pool2d(x, 1).flatten(1)
        max_pool = F.adaptive_max_pool2d(x, 1).flatten(1)
        return self.vec_head(torch.cat([avg_pool, max_pool], dim=-1))


class MidMapEncoder(nn.Module):
    """Independent encoder for the regional mid-map branch."""

    def __init__(self, cfg: GlobalSideEncoderConfig):
        super().__init__()
        self.encoder = RasterMapEncoder(cfg.mid_map_channels, cfg)

    def forward(self, mid_map: torch.Tensor) -> torch.Tensor:
        return self.encoder(mid_map)


class FrontierTokenEncoder(nn.Module):
    """
    Frontier-region token encoder.

    token = sparse frontier candidate representation
    """

    def __init__(self, cfg: GlobalSideEncoderConfig):
        super().__init__()
        d_model = cfg.token_model_dim

        self.token_mlp = nn.Sequential(
            nn.Linear(cfg.frontier_token_input_dim, d_model),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(d_model, d_model),
            nn.GELU(),
        )
        self.token_score = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )
        self.vec_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, cfg.token_vec_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.token_vec_dim, cfg.token_vec_dim),
        )

    def forward(
        self,
        frontier_tokens: torch.Tensor,
        frontier_token_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        bsz, k, _ = frontier_tokens.shape

        token_embeddings = self.token_mlp(frontier_tokens)

        if frontier_token_mask is None:
            valid = torch.ones((bsz, k), dtype=torch.bool, device=frontier_tokens.device)
        else:
            if frontier_token_mask.shape != (bsz, k):
                raise ValueError(
                    "frontier_token_mask shape mismatch: "
                    f"expected {(bsz, k)}, got {tuple(frontier_token_mask.shape)}"
                )
            valid = frontier_token_mask.to(dtype=torch.bool)

        attn_logits = self.token_score(token_embeddings).squeeze(-1)
        attn_logits = attn_logits.masked_fill(~valid, torch.finfo(attn_logits.dtype).min)
        attn_weights = torch.softmax(attn_logits, dim=1)
        attn_weights = attn_weights * valid.to(dtype=attn_weights.dtype)
        attn_weights = attn_weights / attn_weights.sum(dim=1, keepdim=True).clamp_min(1e-6)

        pooled_tokens = torch.sum(token_embeddings * attn_weights.unsqueeze(-1), dim=1)
        return self.vec_head(pooled_tokens)


class GlobalContextFusion(nn.Module):
    """
    Fuse regional dense context and sparse frontier-token context.

    near = local geometry + short-term recency + global normalized position context
    mid = regional unknown/obstacle structure + coarse visitation memory
    token = sparse frontier candidate representation
    """

    def __init__(self, cfg: GlobalSideEncoderConfig):
        super().__init__()
        d_global = cfg.global_context_dim

        self.mid_proj = nn.Sequential(
            nn.Linear(cfg.map_vec_dim, d_global),
            nn.GELU(),
            nn.LayerNorm(d_global),
        )
        self.token_proj = nn.Sequential(
            nn.Linear(cfg.token_vec_dim, d_global),
            nn.GELU(),
            nn.LayerNorm(d_global),
        )

        self.source_gate = nn.Sequential(
            nn.Linear(d_global * 2, cfg.fusion_hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.fusion_hidden_dim, 2),
        )
        self.fuse = nn.Sequential(
            nn.Linear(d_global * 2, cfg.fusion_hidden_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.fusion_hidden_dim, d_global),
            nn.LayerNorm(d_global),
        )

    def forward(
        self,
        mid_map_vector: torch.Tensor,
        frontier_token_context: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        mid_summary = self.mid_proj(mid_map_vector)
        token_summary = self.token_proj(frontier_token_context)

        concat_summary = torch.cat([mid_summary, token_summary], dim=-1)
        gate_values = torch.sigmoid(self.source_gate(concat_summary))

        mid_gated = mid_summary * gate_values[:, 0:1]
        token_gated = token_summary * gate_values[:, 1:2]
        global_context = self.fuse(torch.cat([mid_gated, token_gated], dim=-1))
        return global_context, gate_values


class GlobalSideEncoder(nn.Module):
    """
    Global-side encoder.

    Data flow:
      mid_map -> mid encoder
      frontier_tokens -> frontier token encoder
      mid/token -> global fusion -> global_context
    """

    def __init__(self, cfg: Optional[GlobalSideEncoderConfig] = None):
        super().__init__()
        self.cfg = cfg if cfg is not None else GlobalSideEncoderConfig()

        self.mid_encoder = MidMapEncoder(self.cfg)
        self.frontier_token_encoder = FrontierTokenEncoder(self.cfg)
        self.fusion = GlobalContextFusion(self.cfg)

    def forward(
        self,
        mid_map: torch.Tensor,
        frontier_tokens: torch.Tensor,
        frontier_token_mask: Optional[torch.Tensor] = None,
        return_aux: bool = True,
    ):
        if mid_map.dim() != 4:
            raise ValueError(f"mid_map must be 4D [B,C,H,W], got shape={tuple(mid_map.shape)}")
        if frontier_tokens.dim() != 3:
            raise ValueError(
                "frontier_tokens must be 3D [B,K,D], "
                f"got shape={tuple(frontier_tokens.shape)}"
            )

        b_mid = mid_map.shape[0]
        b_tok = frontier_tokens.shape[0]
        if b_mid != b_tok:
            raise ValueError(
                "batch size mismatch across global-side inputs: "
                f"mid={b_mid}, token={b_tok}"
            )

        if mid_map.shape[1] != self.cfg.mid_map_channels:
            raise ValueError(
                f"mid_map channel mismatch: expected {self.cfg.mid_map_channels}, got {mid_map.shape[1]}"
            )
        if frontier_tokens.shape[2] != self.cfg.frontier_token_input_dim:
            raise ValueError(
                "frontier token dim mismatch: "
                f"expected {self.cfg.frontier_token_input_dim}, got {frontier_tokens.shape[2]}"
            )

        mid_map_vector = self.mid_encoder(mid_map)
        frontier_token_context = self.frontier_token_encoder(
            frontier_tokens,
            frontier_token_mask=frontier_token_mask,
        )
        global_context, source_gates = self.fusion(
            mid_map_vector,
            frontier_token_context,
        )

        if not return_aux:
            return global_context

        aux: Dict[str, torch.Tensor] = {
            "mid_map_vector": mid_map_vector,
            "frontier_token_context": frontier_token_context,
            "global_source_gates": source_gates,
        }
        return global_context, aux


def _smoke_test() -> None:
    cfg = GlobalSideEncoderConfig(
        mid_map_channels=MID_MAP_CHANNEL_COUNT,
        frontier_token_input_dim=FRONTIER_REGION_TOKEN_FIELD_COUNT,
        global_context_dim=256,
    )
    model = GlobalSideEncoder(cfg)

    bsz = 4
    mid_map = torch.rand(bsz, MID_MAP_CHANNEL_COUNT, 24, 24)
    frontier_tokens = torch.rand(bsz, 32, FRONTIER_REGION_TOKEN_FIELD_COUNT)
    frontier_token_mask = torch.ones(bsz, 32, dtype=torch.bool)
    frontier_token_mask[0, :] = False
    frontier_token_mask[:, 24:] = False

    global_context, aux = model(
        mid_map,
        frontier_tokens,
        frontier_token_mask=frontier_token_mask,
        return_aux=True,
    )

    assert global_context.shape == (bsz, cfg.global_context_dim)
    assert aux["mid_map_vector"].shape == (bsz, cfg.map_vec_dim)
    assert aux["frontier_token_context"].shape == (bsz, cfg.token_vec_dim)
    assert torch.isfinite(global_context).all()
    assert torch.isfinite(aux["frontier_token_context"]).all()

    print("GlobalSideEncoder smoke test passed")
    print("global_context:", tuple(global_context.shape))
    print("mid_map_vector:", tuple(aux["mid_map_vector"].shape))
    print("frontier_token_context:", tuple(aux["frontier_token_context"].shape))


if __name__ == "__main__":
    _smoke_test()
