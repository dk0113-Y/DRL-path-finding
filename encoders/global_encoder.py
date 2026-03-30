from __future__ import annotations

"""Legacy near/mid/token encoder path kept only as historical reference."""

import math
from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from env.core_cummap import MID_MAP_CHANNEL_COUNT
from env.frontier_token_builder import FRONTIER_REGION_TOKEN_FIELD_COUNT


@dataclass(frozen=True)
class GlobalSideEncoderConfig:
    """Configuration for the split global-side encoder."""

    mid_map_channels: int = MID_MAP_CHANNEL_COUNT
    frontier_token_input_dim: int = FRONTIER_REGION_TOKEN_FIELD_COUNT
    map_base_dim: int = 64
    map_vec_dim: int = 128
    token_model_dim: int = 128
    token_vec_dim: int = 128
    global_context_dim: int = 256
    token_pre_score_bias_init: float = 0.02
    token_pre_score_bias_max: float = 0.05
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

    token = sparse semantic-frontier-cluster-centered local potential-gain representation
            with fields [dx, dy, local_potential_gain, frontier_contact_span, obstacle_density]

    The same hand-crafted priority used for coarse top-k selection is reconstructed
    inside the encoder and injected as a weak attention-logit bias. The learned
    token-score branch still owns the attention details; the pre-score only provides
    a light prior after masked per-sample normalization. The bias-strength raw
    parameter is unconstrained, but the effective scale used in attention is
    bounded into a small non-negative range so the pre-score remains a weak prior.
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
        # Unconstrained raw parameter for the weak pre-score prior strength.
        # The effective scale used in attention is max * sigmoid(raw), so training
        # can tune it freely while the actual bias strength stays non-negative
        # and explicitly bounded in a weak-prior range.
        self.pre_score_bias_raw = nn.Parameter(
            torch.tensor(
                self._inverse_bounded_sigmoid(
                    float(cfg.token_pre_score_bias_init),
                    max_value=float(cfg.token_pre_score_bias_max),
                ),
                dtype=torch.float32,
            )
        )
        self.pre_score_bias_max = float(cfg.token_pre_score_bias_max)
        self.vec_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, cfg.token_vec_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.token_vec_dim, cfg.token_vec_dim),
        )

    @staticmethod
    def _inverse_bounded_sigmoid(value: float, *, max_value: float) -> float:
        if max_value <= 0.0:
            raise ValueError(f"token_pre_score_bias_max must be > 0, got {max_value}")
        if value <= 0.0:
            raise ValueError(f"token_pre_score_bias_init must be > 0, got {value}")
        if value >= max_value:
            raise ValueError(
                "token_pre_score_bias_init must stay below token_pre_score_bias_max "
                f"to keep sigmoid initialization finite, got init={value} max={max_value}"
            )
        ratio = value / max_value
        return float(math.log(ratio / (1.0 - ratio)))

    def _effective_pre_score_bias_scale(self) -> torch.Tensor:
        return self.pre_score_bias_max * torch.sigmoid(self.pre_score_bias_raw)

    @staticmethod
    def _build_valid_mask(
        frontier_tokens: torch.Tensor,
        frontier_token_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        bsz, k, _ = frontier_tokens.shape
        if frontier_token_mask is None:
            return torch.ones((bsz, k), dtype=torch.bool, device=frontier_tokens.device)
        if frontier_token_mask.shape != (bsz, k):
            raise ValueError(
                "frontier_token_mask shape mismatch: "
                f"expected {(bsz, k)}, got {tuple(frontier_token_mask.shape)}"
            )
        return frontier_token_mask.to(dtype=torch.bool)

    @staticmethod
    def _reconstruct_pre_score(frontier_tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        token_features = frontier_tokens.to(dtype=torch.float32)
        dx = token_features[..., 0]
        dy = token_features[..., 1]
        local_potential_gain = token_features[..., 2]
        frontier_contact_span = token_features[..., 3]
        obstacle_density = token_features[..., 4]
        geom_dist_norm = torch.clamp(torch.sqrt(dx.square() + dy.square()) / math.sqrt(2.0), 0.0, 1.0)
        pre_score = (
            0.50 * local_potential_gain
            + 0.08 * frontier_contact_span
            - 0.12 * obstacle_density
            - 0.30 * geom_dist_norm
        )
        return pre_score, geom_dist_norm

    @staticmethod
    def _masked_zscore(values: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        valid_f = valid.to(dtype=values.dtype)
        valid_count = valid_f.sum(dim=1, keepdim=True)
        denom = valid_count.clamp_min(1.0)
        masked_values = values * valid_f
        mean = masked_values.sum(dim=1, keepdim=True) / denom
        centered = (values - mean) * valid_f
        variance = centered.square().sum(dim=1, keepdim=True) / denom
        std = torch.sqrt(variance.clamp_min(1e-6))
        normalized = centered / std
        enough_tokens = valid_count >= 2.0
        return torch.where(enough_tokens, normalized, torch.zeros_like(normalized))

    @staticmethod
    def _attention_summary(
        attn_weights: torch.Tensor,
        valid: torch.Tensor,
        feature: torch.Tensor,
    ) -> torch.Tensor:
        weighted = attn_weights * valid.to(dtype=attn_weights.dtype)
        return torch.sum(weighted * feature.to(dtype=weighted.dtype), dim=1)

    @staticmethod
    def _average_ranks(values: torch.Tensor) -> torch.Tensor:
        sorted_values, sorted_indices = torch.sort(values)
        ranks = torch.zeros_like(values, dtype=torch.float32)
        start = 0
        n = int(values.numel())
        while start < n:
            end = start + 1
            while end < n and torch.isclose(sorted_values[end], sorted_values[start]):
                end += 1
            avg_rank = 0.5 * float(start + end - 1) + 1.0
            ranks[sorted_indices[start:end]] = avg_rank
            start = end
        return ranks

    @classmethod
    def _spearman_corr(
        cls,
        values: torch.Tensor,
        attn_weights: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        batch = int(values.shape[0])
        corr = torch.full((batch,), float("nan"), dtype=torch.float32, device=values.device)
        for batch_idx in range(batch):
            valid_idx = valid[batch_idx]
            valid_count = int(valid_idx.sum().item())
            if valid_count < 2:
                continue
            x = values[batch_idx, valid_idx]
            y = attn_weights[batch_idx, valid_idx]
            x_rank = cls._average_ranks(x)
            y_rank = cls._average_ranks(y)
            x_centered = x_rank - x_rank.mean()
            y_centered = y_rank - y_rank.mean()
            denom = torch.sqrt(x_centered.square().sum() * y_centered.square().sum())
            if not torch.isfinite(denom) or float(denom.item()) <= 1e-6:
                continue
            corr[batch_idx] = (x_centered * y_centered).sum() / denom
        return corr

    @staticmethod
    def _topk_weight_summaries(attn_weights: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        sorted_weights, _ = torch.sort(attn_weights, dim=1, descending=True)
        top1 = sorted_weights[:, 0]
        if sorted_weights.shape[1] <= 1:
            top2_sum = top1
        else:
            top2_sum = sorted_weights[:, :2].sum(dim=1)
        return top1, top2_sum

    @staticmethod
    def _attention_entropy(attn_weights: torch.Tensor, valid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        weights = attn_weights * valid.to(dtype=attn_weights.dtype)
        entropy = -(weights * torch.log(weights.clamp_min(1e-8))).sum(dim=1)
        valid_count = valid.sum(dim=1)
        has_valid = valid_count > 0
        entropy = torch.where(has_valid, entropy, torch.zeros_like(entropy))
        effective_count = torch.exp(entropy)
        effective_count = torch.where(has_valid, effective_count, torch.zeros_like(effective_count))
        return entropy, effective_count

    def forward(
        self,
        frontier_tokens: torch.Tensor,
        frontier_token_mask: Optional[torch.Tensor] = None,
        return_aux: bool = False,
    ):
        bsz, k, _ = frontier_tokens.shape

        token_embeddings = self.token_mlp(frontier_tokens)
        valid = self._build_valid_mask(frontier_tokens, frontier_token_mask)
        pre_score, geom_dist_norm = self._reconstruct_pre_score(frontier_tokens)
        normalized_pre_score = self._masked_zscore(pre_score, valid).to(dtype=token_embeddings.dtype)

        learned_logits = self.token_score(token_embeddings).squeeze(-1)
        effective_bias_scale = self._effective_pre_score_bias_scale().to(dtype=learned_logits.dtype)
        attn_logits = learned_logits + effective_bias_scale * normalized_pre_score
        attn_logits = attn_logits.masked_fill(~valid, torch.finfo(attn_logits.dtype).min)
        attn_weights = torch.softmax(attn_logits, dim=1)
        attn_weights = attn_weights * valid.to(dtype=attn_weights.dtype)
        attn_weights = attn_weights / attn_weights.sum(dim=1, keepdim=True).clamp_min(1e-6)

        pooled_tokens = torch.sum(token_embeddings * attn_weights.unsqueeze(-1), dim=1)
        frontier_token_context = self.vec_head(pooled_tokens)

        if not return_aux:
            return frontier_token_context

        attn_weights_f32 = attn_weights.to(dtype=torch.float32)
        entropy, effective_count = self._attention_entropy(attn_weights_f32, valid)
        top1_weight, top2_weight_sum = self._topk_weight_summaries(attn_weights_f32)
        token_pre_score_attn_corr = self._spearman_corr(pre_score, attn_weights_f32, valid)

        aux: Dict[str, torch.Tensor] = {
            "token_pre_score": pre_score,
            "token_normalized_pre_score": normalized_pre_score.to(dtype=torch.float32),
            "token_pre_score_bias_scale": effective_bias_scale.to(dtype=torch.float32),
            "token_attn_weights": attn_weights_f32,
            "token_attn_entropy": entropy,
            "token_attn_effective_count": effective_count,
            "token_attn_top1_weight": top1_weight,
            "token_attn_top2_weight_sum": top2_weight_sum,
            "token_attn_weighted_local_potential_gain": self._attention_summary(
                attn_weights_f32,
                valid,
                frontier_tokens[..., 2],
            ),
            "token_attn_weighted_frontier_contact_span": self._attention_summary(
                attn_weights_f32,
                valid,
                frontier_tokens[..., 3],
            ),
            "token_attn_weighted_obstacle_density": self._attention_summary(
                attn_weights_f32,
                valid,
                frontier_tokens[..., 4],
            ),
            "token_attn_weighted_geom_dist_norm": self._attention_summary(
                attn_weights_f32,
                valid,
                geom_dist_norm,
            ),
            "token_pre_score_attn_corr": token_pre_score_attn_corr,
        }
        return frontier_token_context, aux


class GlobalSideEncoder(nn.Module):
    """
    Split global-side encoder.

    Data flow:
      mid_map -> mid encoder -> mid_map_vector -> mid_context
      frontier_tokens -> frontier token encoder -> frontier_token_context -> token_context

    Semantics:
      - mid_context is the regional dense-background context used downstream for
        local action modulation.
      - token_context is the sparse candidate-region summary used downstream for
        broader state-value estimation.
      - token attention now uses the coarse top-k hand score as a weak prior and
        exposes attention-usage diagnostics for evaluation/probing.
    """

    def __init__(self, cfg: Optional[GlobalSideEncoderConfig] = None):
        super().__init__()
        self.cfg = cfg if cfg is not None else GlobalSideEncoderConfig()

        self.mid_encoder = MidMapEncoder(self.cfg)
        self.frontier_token_encoder = FrontierTokenEncoder(self.cfg)
        self.mid_context_proj = nn.Sequential(
            nn.Linear(self.cfg.map_vec_dim, self.cfg.global_context_dim),
            nn.GELU(),
            nn.LayerNorm(self.cfg.global_context_dim),
        )
        self.token_context_proj = nn.Sequential(
            nn.Linear(self.cfg.token_vec_dim, self.cfg.global_context_dim),
            nn.GELU(),
            nn.LayerNorm(self.cfg.global_context_dim),
        )

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
        token_aux: Dict[str, torch.Tensor] = {}
        if return_aux:
            frontier_token_context, token_aux = self.frontier_token_encoder(
                frontier_tokens,
                frontier_token_mask=frontier_token_mask,
                return_aux=True,
            )
        else:
            frontier_token_context = self.frontier_token_encoder(
                frontier_tokens,
                frontier_token_mask=frontier_token_mask,
                return_aux=False,
            )
        mid_context = self.mid_context_proj(mid_map_vector)
        token_context = self.token_context_proj(frontier_token_context)

        if not return_aux:
            return mid_context, token_context

        aux: Dict[str, torch.Tensor] = {
            "mid_map_vector": mid_map_vector,
            "frontier_token_context": frontier_token_context,
            "mid_context": mid_context,
            "token_context": token_context,
        }
        aux.update(token_aux)
        return mid_context, token_context, aux


def _smoke_test() -> None:
    cfg = GlobalSideEncoderConfig(
        mid_map_channels=MID_MAP_CHANNEL_COUNT,
        frontier_token_input_dim=FRONTIER_REGION_TOKEN_FIELD_COUNT,
        global_context_dim=256,
    )
    model = GlobalSideEncoder(cfg)
    model.eval()

    bsz = 4
    mid_map = torch.rand(bsz, MID_MAP_CHANNEL_COUNT, 24, 24)
    frontier_tokens = torch.rand(bsz, 32, FRONTIER_REGION_TOKEN_FIELD_COUNT)
    frontier_token_mask = torch.ones(bsz, 32, dtype=torch.bool)
    frontier_token_mask[0, :] = False
    frontier_token_mask[:, 24:] = False

    with torch.no_grad():
        mid_context, token_context, aux = model(
            mid_map,
            frontier_tokens,
            frontier_token_mask=frontier_token_mask,
            return_aux=True,
        )
        mid_context_no_aux, token_context_no_aux = model(
            mid_map,
            frontier_tokens,
            frontier_token_mask=frontier_token_mask,
            return_aux=False,
        )

    assert mid_context.shape == (bsz, cfg.global_context_dim)
    assert token_context.shape == (bsz, cfg.global_context_dim)
    assert aux["mid_map_vector"].shape == (bsz, cfg.map_vec_dim)
    assert aux["frontier_token_context"].shape == (bsz, cfg.token_vec_dim)
    assert aux["mid_context"].shape == (bsz, cfg.global_context_dim)
    assert aux["token_context"].shape == (bsz, cfg.global_context_dim)
    assert aux["token_attn_weights"].shape == (bsz, 32)
    assert aux["token_normalized_pre_score"].shape == (bsz, 32)
    assert aux["token_pre_score_bias_scale"].ndim == 0
    assert torch.allclose(mid_context, mid_context_no_aux)
    assert torch.allclose(token_context, token_context_no_aux)
    assert torch.isfinite(mid_context).all()
    assert torch.isfinite(token_context).all()
    assert torch.isfinite(aux["frontier_token_context"]).all()
    for key in (
        "token_attn_entropy",
        "token_attn_effective_count",
        "token_attn_top1_weight",
        "token_attn_top2_weight_sum",
        "token_attn_weighted_local_potential_gain",
        "token_attn_weighted_frontier_contact_span",
        "token_attn_weighted_obstacle_density",
        "token_attn_weighted_geom_dist_norm",
    ):
        assert torch.isfinite(aux[key]).all(), key
    corr0 = aux["token_pre_score_attn_corr"][0]
    assert bool(torch.isnan(corr0).item() or torch.isfinite(corr0).item())
    assert torch.isfinite(aux["token_pre_score_bias_scale"]).all()
    effective_scale = float(aux["token_pre_score_bias_scale"].item())
    assert 0.0 <= effective_scale <= float(cfg.token_pre_score_bias_max)
    assert abs(effective_scale - float(cfg.token_pre_score_bias_init)) < 1e-4

    print("GlobalSideEncoder smoke test passed")
    print("mid_context:", tuple(mid_context.shape))
    print("token_context:", tuple(token_context.shape))
    print("mid_map_vector:", tuple(aux["mid_map_vector"].shape))
    print("frontier_token_context:", tuple(aux["frontier_token_context"].shape))


if __name__ == "__main__":
    _smoke_test()
