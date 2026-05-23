from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import torch
import torch.nn as nn

from agents.q_value_agent import ACTION_DIM, ExplorationQConfig
from encoders.advantage_encoder import AdvantageCanvasEncoder
from encoders.value_encoder import ValueTreeEncoder


NO_SEMANTIC_DUAL_STATE_SPLIT_ABLATION_ID = "E_ablation_no_semantic_dual_state_split"
NO_SEMANTIC_DUAL_STATE_SPLIT_SHORT_ID = "E"


@dataclass(frozen=True)
class NoSemanticDualStateSplitConfig:
    base: ExplorationQConfig = field(default_factory=ExplorationQConfig)
    latent_dim: int = 192
    hidden_dim: int = 192
    dropout: float = 0.1


class NoSemanticDualStateSplitQNetwork(nn.Module):
    """
    Structural ablation for the semantic dual-state split.

    This model keeps the full method's advantage canvas encoder and value tree
    encoder, but replaces SemanticDuelingHead with one action-value pathway over
    a fused per-action latent. It tests the decision-structure contribution of
    explicit value_state / advantage_state separation, not the presence of value
    tree information.
    """

    def __init__(self, cfg: Optional[NoSemanticDualStateSplitConfig] = None):
        super().__init__()
        self.cfg = cfg if cfg is not None else NoSemanticDualStateSplitConfig()
        base = self.cfg.base
        if int(base.advantage_encoder.action_dim) != ACTION_DIM:
            raise ValueError(
                f"Advantage encoder action_dim must be {ACTION_DIM}, got {base.advantage_encoder.action_dim}"
            )
        if int(base.decision_head.action_dim) != ACTION_DIM:
            raise ValueError(f"Action dim must be {ACTION_DIM}, got {base.decision_head.action_dim}")
        if int(base.decision_head.value_state_dim) != int(base.value_encoder.value_state_dim):
            raise ValueError("Value encoder output dim must match configured value_state_dim")
        if int(base.decision_head.advantage_state_dim) != int(base.advantage_encoder.action_state_dim):
            raise ValueError("Advantage encoder output dim must match configured advantage_state_dim")

        self.advantage_encoder = AdvantageCanvasEncoder(base.advantage_encoder)
        self.value_encoder = ValueTreeEncoder(base.value_encoder)
        latent_dim = int(self.cfg.latent_dim)
        hidden_dim = int(self.cfg.hidden_dim)
        dropout = float(self.cfg.dropout)

        self.value_projection = nn.Sequential(
            nn.LayerNorm(base.value_encoder.value_state_dim),
            nn.Linear(base.value_encoder.value_state_dim, latent_dim),
            nn.GELU(),
        )
        self.advantage_projection = nn.Sequential(
            nn.LayerNorm(base.advantage_encoder.action_state_dim),
            nn.Linear(base.advantage_encoder.action_state_dim, latent_dim),
            nn.GELU(),
        )
        self.q_head = nn.Sequential(
            nn.LayerNorm(latent_dim * 2),
            nn.Linear(latent_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def _q_from_states(self, value_state: torch.Tensor, advantage_state: torch.Tensor) -> torch.Tensor:
        if value_state.dim() != 2:
            raise ValueError(f"value_state must be [B,D], got {tuple(value_state.shape)}")
        if advantage_state.dim() != 3:
            raise ValueError(f"advantage_state must be [B,A,D], got {tuple(advantage_state.shape)}")
        if value_state.shape[0] != advantage_state.shape[0]:
            raise ValueError("batch mismatch between value_state and advantage_state")
        if int(advantage_state.shape[1]) != ACTION_DIM:
            raise ValueError(f"advantage_state action dim mismatch: expected {ACTION_DIM}, got {advantage_state.shape[1]}")

        value_latent = self.value_projection(value_state).unsqueeze(1).expand(-1, ACTION_DIM, -1)
        advantage_latent = self.advantage_projection(advantage_state)
        fused = torch.cat([value_latent, advantage_latent], dim=-1)
        return self.q_head(fused).squeeze(-1)

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
            return self._q_from_states(value_state, advantage_state)

        advantage_state, advantage_aux = self.advantage_encoder(advantage_canvas, return_aux=True)
        value_state, value_aux = self.value_encoder(
            value_block_features,
            value_entry_features,
            value_block_mask,
            value_entry_mask,
            return_aux=True,
        )
        q_values = self._q_from_states(value_state, advantage_state)
        batch = int(q_values.shape[0])
        marker = torch.ones((batch,), device=q_values.device, dtype=q_values.dtype)
        aux: Dict[str, torch.Tensor] = {}
        aux.update(advantage_aux)
        aux.update(value_aux)
        aux.update(
            {
                "no_semantic_dual_state_split": marker,
                "value_tree_used_by_model": marker,
                "semantic_dual_state_split_used": torch.zeros_like(marker),
                "value_tree_input_block_mask_true_count": value_block_mask.to(dtype=q_values.dtype).sum(dim=1),
                "value_tree_input_entry_mask_true_count": value_entry_mask.to(dtype=q_values.dtype).sum(dim=(1, 2)),
            }
        )
        return q_values, aux


def count_model_parameters(model: nn.Module) -> int:
    return int(sum(param.numel() for param in model.parameters()))


def build_no_semantic_dual_state_split_manifest(*, model: nn.Module) -> dict[str, object]:
    return {
        "semantic_split_ablation_schema_version": "semantic_split_ablation/v1",
        "experiment_id": "E",
        "ablation_id": NO_SEMANTIC_DUAL_STATE_SPLIT_ABLATION_ID,
        "ablation_group": "structural",
        "ablation_name": "no_semantic_dual_state_split",
        "no_semantic_dual_state_split": True,
        "semantic_dual_state_split_used": False,
        "value_tree_enabled": True,
        "value_tree_used_by_model": True,
        "value_replacement_strategy": "none",
        "channel_ablation": "none",
        "reward_override": {},
        "model_class": type(model).__name__,
        "model_parameter_count": count_model_parameters(model),
        "scientific_interpretation": (
            "Tests whether explicitly separating value-state and action-conditioned "
            "advantage-state before dueling fusion is beneficial; it does not test "
            "whether value-tree information exists."
        ),
    }
