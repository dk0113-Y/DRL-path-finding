from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, Optional

import torch
import torch.nn as nn

from env.value_state_builder import VALUE_BLOCK_FEATURE_COUNT, VALUE_ENTRY_FEATURE_COUNT


@dataclass(frozen=True)
class ValueEncoderConfig:
    block_input_dim: int = VALUE_BLOCK_FEATURE_COUNT
    entry_input_dim: int = VALUE_ENTRY_FEATURE_COUNT
    entry_model_dim: int = 96
    block_model_dim: int = 160
    value_state_dim: int = 192
    attention_heads: int = 4
    entry_attention_layers: int = 1
    block_attention_layers: int = 1
    dropout: float = 0.1


def _masked_softmax(logits: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    # Keep the masking-softmax in fp32 under autocast so fp16 paths do not overflow on large negative fills.
    logits_fp32 = logits.float()
    mask = mask.to(dtype=torch.bool)
    masked_logits = logits_fp32.masked_fill(~mask, torch.finfo(logits_fp32.dtype).min)
    weights = torch.softmax(masked_logits, dim=dim)
    weights = weights * mask.to(dtype=weights.dtype)
    denom = weights.sum(dim=dim, keepdim=True).clamp_min(1e-6)
    weights = weights / denom
    return weights.to(dtype=logits.dtype)


class _MaskedSelfAttentionLayer(nn.Module):
    """
    Masked set self-attention over one explicit set axis.

    The caller decides what the set axis means. ValueTreeEncoder uses this
    layer once over sibling entries inside each block and once over block
    tokens across the accessible-block set.
    """

    def __init__(self, model_dim: int, num_heads: int, dropout: float):
        super().__init__()
        if model_dim % num_heads != 0:
            raise ValueError(f"model_dim={model_dim} must be divisible by num_heads={num_heads}")
        self.model_dim = int(model_dim)
        self.num_heads = int(num_heads)
        self.head_dim = int(model_dim // num_heads)
        self.scale = 1.0 / math.sqrt(float(self.head_dim))

        self.attn_norm = nn.LayerNorm(model_dim)
        self.qkv = nn.Linear(model_dim, model_dim * 3)
        self.out_proj = nn.Linear(model_dim, model_dim)
        self.ffn_norm = nn.LayerNorm(model_dim)
        self.ffn = nn.Sequential(
            nn.Linear(model_dim, model_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim * 4, model_dim),
        )
        self.out_norm = nn.LayerNorm(model_dim)
        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, token_count, _ = x.shape
        return x.view(batch, token_count, self.num_heads, self.head_dim).transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, _, token_count, _ = x.shape
        return x.transpose(1, 2).contiguous().view(batch, token_count, self.model_dim)

    def _self_attention(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = self._split_heads(q)
        k = self._split_heads(k)
        v = self._split_heads(v)

        logits = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        weights = _masked_softmax(logits, mask[:, None, None, :], dim=-1)
        context = torch.matmul(weights, v)
        return self.out_proj(self._merge_heads(context))

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask_f = mask.to(dtype=x.dtype).unsqueeze(-1)
        x = x * mask_f

        attn_out = self._self_attention(self.attn_norm(x), mask)
        x = (x + self.dropout(attn_out)) * mask_f

        ffn_out = self.ffn(self.ffn_norm(x))
        x = self.out_norm(x + self.dropout(ffn_out))
        return x * mask_f


class _BlockConditionedChildAggregator(nn.Module):
    """
    Parent-conditioned soft aggregation from child entries to one block summary.

    The query comes from the parent block token; child tokens only provide
    keys/values after sibling relation modeling. This is not child self-scoring
    and it never selects a representative entry.
    """

    def __init__(self, block_model_dim: int, entry_model_dim: int, dropout: float):
        super().__init__()
        self.query = nn.Sequential(
            nn.LayerNorm(block_model_dim),
            nn.Linear(block_model_dim, entry_model_dim),
        )
        self.key = nn.Sequential(
            nn.LayerNorm(entry_model_dim),
            nn.Linear(entry_model_dim, entry_model_dim),
        )
        self.value = nn.Sequential(
            nn.LayerNorm(entry_model_dim),
            nn.Linear(entry_model_dim, entry_model_dim),
        )
        self.output = nn.Sequential(
            nn.Linear(entry_model_dim, entry_model_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(entry_model_dim, entry_model_dim),
            nn.LayerNorm(entry_model_dim),
        )
        self.scale = 1.0 / math.sqrt(float(entry_model_dim))

    def forward(
        self,
        block_token: torch.Tensor,
        entry_token: torch.Tensor,
        entry_mask: torch.Tensor,
        block_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        valid_child_mask = entry_mask & block_mask.unsqueeze(-1)
        query = self.query(block_token).unsqueeze(2)
        key = self.key(entry_token)
        logits = torch.sum(query * key, dim=-1) * self.scale
        weights = _masked_softmax(logits, valid_child_mask, dim=2)

        value = self.value(entry_token)
        child_summary = torch.sum(value * weights.unsqueeze(-1), dim=2)
        child_summary = self.output(child_summary)
        child_summary = child_summary * block_mask.to(dtype=child_summary.dtype).unsqueeze(-1)
        return child_summary, weights


class _ParentGroundedBlockBuilder(nn.Module):
    """
    Gated parent-grounded update from child summary to block representation.

    The block token remains the semantic trunk. The child summary contributes a
    gated residual update, so this is not an ungrounded peer-vector fusion.
    """

    def __init__(self, block_model_dim: int, entry_model_dim: int, dropout: float):
        super().__init__()
        self.child_update = nn.Sequential(
            nn.LayerNorm(entry_model_dim),
            nn.Linear(entry_model_dim, block_model_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(block_model_dim, block_model_dim),
        )
        self.update_gate = nn.Sequential(
            nn.LayerNorm(block_model_dim + entry_model_dim),
            nn.Linear(block_model_dim + entry_model_dim, block_model_dim),
            nn.GELU(),
            nn.Linear(block_model_dim, block_model_dim),
            nn.Sigmoid(),
        )
        self.out_norm = nn.LayerNorm(block_model_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        block_token: torch.Tensor,
        child_summary: torch.Tensor,
        block_mask: torch.Tensor,
    ) -> torch.Tensor:
        child_update = self.child_update(child_summary)
        gate = self.update_gate(torch.cat([block_token, child_summary], dim=-1))
        # Parent-grounded update: block semantics are the trunk, child access cues are a gated residual.
        block_repr = block_token + gate * self.dropout(child_update)
        block_repr = self.out_norm(block_repr)
        return block_repr * block_mask.to(dtype=block_repr.dtype).unsqueeze(-1)


class _StateQueryValuePooler(nn.Module):
    """
    Single-path weighted pooling from contextual block tokens to a state token.

    A learned state query attends over all valid blocks. This keeps the block
    level aligned with the child level: both use one weighted aggregation path
    and no mean/max fallback pooling.
    """

    def __init__(self, block_model_dim: int, dropout: float):
        super().__init__()
        self.state_query = nn.Parameter(torch.empty(block_model_dim))
        nn.init.normal_(self.state_query, mean=0.0, std=0.02)
        self.query = nn.Sequential(
            nn.LayerNorm(block_model_dim),
            nn.Linear(block_model_dim, block_model_dim),
        )
        self.key = nn.Sequential(
            nn.LayerNorm(block_model_dim),
            nn.Linear(block_model_dim, block_model_dim),
        )
        self.value = nn.Sequential(
            nn.LayerNorm(block_model_dim),
            nn.Linear(block_model_dim, block_model_dim),
        )
        self.out_norm = nn.LayerNorm(block_model_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = 1.0 / math.sqrt(float(block_model_dim))

    def forward(self, context_block_repr: torch.Tensor, block_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch = context_block_repr.shape[0]
        query = self.query(self.state_query.expand(batch, -1)).unsqueeze(1)
        key = self.key(context_block_repr)
        logits = torch.sum(query * key, dim=-1) * self.scale
        weights = _masked_softmax(logits, block_mask, dim=1)

        value = self.value(context_block_repr)
        pooled_state = torch.sum(value * weights.unsqueeze(-1), dim=1)
        pooled_state = self.out_norm(self.dropout(pooled_state))
        return pooled_state, weights


class ValueTreeEncoder(nn.Module):
    """
    Block-conditioned child encoder for the value branch.

    Data flow:
      block_features -> block semantic encoder -> parent block tokens
      entry_features -> entry access encoder -> child entry tokens
      sibling entries within each block -> masked intra-block self-attention
      parent block token queries child entries -> soft child summary
      parent-grounded gated update -> block representations
      block representations -> masked global block self-attention
      learned state query -> weighted block pooling -> value state head

    The block-entry correspondence is expressed by the nested tensor layout:
    entry_features[:, block_slot, ...] are the children of block_features[:, block_slot, :].
    The value branch forms state-level context; it does not make action decisions
    for the advantage branch. It no longer uses child self-scoring or mean/max
    fallback pooling at either the child or block level.
    """

    def __init__(self, cfg: Optional[ValueEncoderConfig] = None):
        super().__init__()
        self.cfg = cfg if cfg is not None else ValueEncoderConfig()
        if int(self.cfg.entry_attention_layers) not in (1, 2):
            raise ValueError("entry_attention_layers must be 1 or 2")
        if int(self.cfg.block_attention_layers) not in (1, 2):
            raise ValueError("block_attention_layers must be 1 or 2")

        self.block_encoder = nn.Sequential(
            nn.LayerNorm(self.cfg.block_input_dim),
            nn.Linear(self.cfg.block_input_dim, self.cfg.block_model_dim),
            nn.GELU(),
            nn.Dropout(self.cfg.dropout),
            nn.Linear(self.cfg.block_model_dim, self.cfg.block_model_dim),
            nn.LayerNorm(self.cfg.block_model_dim),
        )
        self.entry_encoder = nn.Sequential(
            nn.LayerNorm(self.cfg.entry_input_dim),
            nn.Linear(self.cfg.entry_input_dim, self.cfg.entry_model_dim),
            nn.GELU(),
            nn.Dropout(self.cfg.dropout),
            nn.Linear(self.cfg.entry_model_dim, self.cfg.entry_model_dim),
            nn.LayerNorm(self.cfg.entry_model_dim),
        )

        self.entry_relation_layers = nn.ModuleList(
            [
                _MaskedSelfAttentionLayer(
                    self.cfg.entry_model_dim,
                    self.cfg.attention_heads,
                    self.cfg.dropout,
                )
                for _ in range(int(self.cfg.entry_attention_layers))
            ]
        )
        self.child_aggregator = _BlockConditionedChildAggregator(
            self.cfg.block_model_dim,
            self.cfg.entry_model_dim,
            self.cfg.dropout,
        )
        self.block_builder = _ParentGroundedBlockBuilder(
            self.cfg.block_model_dim,
            self.cfg.entry_model_dim,
            self.cfg.dropout,
        )
        self.block_context_layers = nn.ModuleList(
            [
                _MaskedSelfAttentionLayer(
                    self.cfg.block_model_dim,
                    self.cfg.attention_heads,
                    self.cfg.dropout,
                )
                for _ in range(int(self.cfg.block_attention_layers))
            ]
        )
        self.state_pooler = _StateQueryValuePooler(self.cfg.block_model_dim, self.cfg.dropout)
        self.value_state_head = nn.Sequential(
            nn.LayerNorm(self.cfg.block_model_dim),
            nn.Linear(self.cfg.block_model_dim, self.cfg.value_state_dim),
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

        block_token = self.block_encoder(block_features)
        block_token = block_token * block_mask.to(dtype=block_token.dtype).unsqueeze(-1)
        entry_token = self.entry_encoder(entry_features)
        entry_token = entry_token * entry_mask.to(dtype=entry_token.dtype).unsqueeze(-1)

        # Sibling relation modeling only: attention runs across entries within the same block, not across blocks.
        batch, block_count, entry_count, entry_dim = entry_token.shape
        entry_token_flat = entry_token.reshape(batch * block_count, entry_count, entry_dim)
        entry_mask_flat = entry_mask.reshape(batch * block_count, entry_count)
        for layer in self.entry_relation_layers:
            entry_token_flat = layer(entry_token_flat, entry_mask_flat)
        entry_token = entry_token_flat.reshape(batch, block_count, entry_count, entry_dim)

        child_summary, _child_attn = self.child_aggregator(
            block_token,
            entry_token,
            entry_mask,
            block_mask,
        )
        block_repr = self.block_builder(block_token, child_summary, block_mask)

        # Block-block global context: attention runs over accessible blocks and does not replace child aggregation.
        context_block_repr = block_repr
        for layer in self.block_context_layers:
            context_block_repr = layer(context_block_repr, block_mask)

        pooled_state, block_attn = self.state_pooler(context_block_repr, block_mask)
        value_state = self.value_state_head(pooled_state)

        if not return_aux:
            return value_state

        aux: Dict[str, torch.Tensor] = {
            "value_block_attention_top1": block_attn.max(dim=1).values,
            "value_accessible_block_count": block_mask.to(dtype=torch.float32).sum(dim=1),
        }
        return value_state, aux
