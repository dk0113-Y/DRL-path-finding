from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from env.value_state_builder import VALUE_BLOCK_FEATURE_COUNT, VALUE_ENTRY_FEATURE_COUNT


@dataclass(frozen=True)
class ValueEncoderConfig:
    block_input_dim: int = VALUE_BLOCK_FEATURE_COUNT
    entry_input_dim: int = VALUE_ENTRY_FEATURE_COUNT
    entry_model_dim: int = 96
    block_model_dim: int = 160
    value_state_dim: int = 192
    dropout: float = 0.1


def _masked_mean(values: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    mask_f = mask.to(dtype=values.dtype)
    summed = (values * mask_f.unsqueeze(-1)).sum(dim=dim)
    denom = mask_f.sum(dim=dim, keepdim=False).clamp_min(1.0).unsqueeze(-1)
    return summed / denom


def _masked_max(values: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    minus_inf = torch.finfo(values.dtype).min
    masked = values.masked_fill(~mask.unsqueeze(-1), minus_inf)
    out = masked.max(dim=dim).values
    all_invalid = ~mask.any(dim=dim)
    if torch.any(all_invalid):
        out = out.clone()
        out[all_invalid] = 0.0
    return out


def _masked_softmax(logits: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    # Keep the masking-softmax in fp32 under autocast so fp16 paths do not overflow on large negative fills.
    logits_fp32 = logits.float()
    masked_logits = logits_fp32.masked_fill(~mask, torch.finfo(logits_fp32.dtype).min)
    weights = torch.softmax(masked_logits, dim=dim)
    weights = weights * mask.to(dtype=weights.dtype)
    denom = weights.sum(dim=dim, keepdim=True).clamp_min(1e-6)
    weights = weights / denom
    return weights.to(dtype=logits.dtype)


class ValueTreeEncoder(nn.Module):
    """
    Hierarchical value encoder.

    Tree shape:
      direct block scalars (area/count)
        -> child entry clusters are encoded and aggregated within each block
        -> block representations are aggregated into the final value state

    The block-entry correspondence is expressed by the nested tensor layout:
    entry_features[:, block_slot, ...] are the children of block_features[:, block_slot, :].
    The encoder intentionally learns from the full child entry set instead of a
    single selected child entrance copied into block-level features.
    """

    def __init__(self, cfg: Optional[ValueEncoderConfig] = None):
        super().__init__()
        self.cfg = cfg if cfg is not None else ValueEncoderConfig()
        self.entry_encoder = nn.Sequential(
            nn.LayerNorm(self.cfg.entry_input_dim),
            nn.Linear(self.cfg.entry_input_dim, self.cfg.entry_model_dim),
            nn.GELU(),
            nn.Dropout(self.cfg.dropout),
            nn.Linear(self.cfg.entry_model_dim, self.cfg.entry_model_dim),
            nn.LayerNorm(self.cfg.entry_model_dim),
        )
        self.entry_score = nn.Sequential(
            nn.Linear(self.cfg.entry_model_dim, self.cfg.entry_model_dim),
            nn.GELU(),
            nn.Linear(self.cfg.entry_model_dim, 1),
        )
        self.block_fusion = nn.Sequential(
            nn.LayerNorm(self.cfg.block_input_dim + (self.cfg.entry_model_dim * 3)),
            nn.Linear(self.cfg.block_input_dim + (self.cfg.entry_model_dim * 3), self.cfg.block_model_dim),
            nn.GELU(),
            nn.Dropout(self.cfg.dropout),
            nn.Linear(self.cfg.block_model_dim, self.cfg.block_model_dim),
            nn.LayerNorm(self.cfg.block_model_dim),
        )
        self.block_score = nn.Sequential(
            nn.Linear(self.cfg.block_model_dim, self.cfg.block_model_dim),
            nn.GELU(),
            nn.Linear(self.cfg.block_model_dim, 1),
        )
        self.state_head = nn.Sequential(
            nn.LayerNorm(self.cfg.block_model_dim * 3),
            nn.Linear(self.cfg.block_model_dim * 3, self.cfg.value_state_dim),
            nn.GELU(),
            nn.Dropout(self.cfg.dropout),
            nn.Linear(self.cfg.value_state_dim, self.cfg.value_state_dim),
            nn.LayerNorm(self.cfg.value_state_dim),
        )

    def forward(
        self,
        block_features: torch.Tensor,
        entry_features: torch.Tensor,
        block_mask: torch.Tensor,
        entry_mask: torch.Tensor,
        *,
        return_aux: bool = False,
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]] | torch.Tensor:
        if block_features.dim() != 3:
            raise ValueError(f"block_features must be [B,N,D], got {tuple(block_features.shape)}")
        if entry_features.dim() != 4:
            raise ValueError(f"entry_features must be [B,N,M,D], got {tuple(entry_features.shape)}")
        if block_mask.shape != block_features.shape[:2]:
            raise ValueError("block_mask shape mismatch")
        if entry_mask.shape != entry_features.shape[:3]:
            raise ValueError("entry_mask shape mismatch")

        entry_repr = self.entry_encoder(entry_features)
        entry_logits = self.entry_score(entry_repr).squeeze(-1)
        entry_attn = _masked_softmax(entry_logits, entry_mask, dim=2)

        entry_weighted = torch.sum(entry_repr * entry_attn.unsqueeze(-1), dim=2)
        entry_mean = _masked_mean(entry_repr, entry_mask, dim=2)
        entry_max = _masked_max(entry_repr, entry_mask, dim=2)
        block_repr = self.block_fusion(torch.cat([block_features, entry_weighted, entry_mean, entry_max], dim=-1))

        block_logits = self.block_score(block_repr).squeeze(-1)
        block_attn = _masked_softmax(block_logits, block_mask, dim=1)

        block_weighted = torch.sum(block_repr * block_attn.unsqueeze(-1), dim=1)
        block_mean = _masked_mean(block_repr, block_mask, dim=1)
        block_max = _masked_max(block_repr, block_mask, dim=1)
        value_state = self.state_head(torch.cat([block_weighted, block_mean, block_max], dim=-1))

        if not return_aux:
            return value_state

        aux: Dict[str, torch.Tensor] = {
            "value_block_attention_top1": block_attn.max(dim=1).values,
            "value_accessible_block_count": block_mask.to(dtype=torch.float32).sum(dim=1),
        }
        return value_state, aux
