from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from env.grid_topology import ACTIONS_8


@dataclass(frozen=True)
class SemanticDuelingHeadConfig:
    value_state_dim: int = 192
    advantage_state_dim: int = 160
    hidden_dim: int = 192
    action_dim: int = len(ACTIONS_8)
    dropout: float = 0.1


class SemanticDuelingHead(nn.Module):
    """
    Dueling decision head with separated semantic inputs:
      value_state      -> V(s)
      advantage_state  -> A(s, a)
    """

    def __init__(self, cfg: SemanticDuelingHeadConfig | None = None):
        super().__init__()
        self.cfg = cfg if cfg is not None else SemanticDuelingHeadConfig()
        self.value_head = nn.Sequential(
            nn.LayerNorm(self.cfg.value_state_dim),
            nn.Linear(self.cfg.value_state_dim, self.cfg.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.cfg.dropout),
            nn.Linear(self.cfg.hidden_dim, 1),
        )
        self.advantage_head = nn.Sequential(
            nn.LayerNorm(self.cfg.advantage_state_dim),
            nn.Linear(self.cfg.advantage_state_dim, self.cfg.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.cfg.dropout),
            nn.Linear(self.cfg.hidden_dim, 1),
        )

    def forward(
        self,
        value_state: torch.Tensor,
        advantage_state: torch.Tensor,
    ) -> torch.Tensor:
        if value_state.dim() != 2:
            raise ValueError(f"value_state must be [B,D], got {tuple(value_state.shape)}")
        if advantage_state.dim() != 3:
            raise ValueError(f"advantage_state must be [B,A,D], got {tuple(advantage_state.shape)}")
        if value_state.shape[0] != advantage_state.shape[0]:
            raise ValueError("batch mismatch between value_state and advantage_state")
        if advantage_state.shape[1] != int(self.cfg.action_dim):
            raise ValueError(
                f"advantage_state action dim mismatch: expected {self.cfg.action_dim}, got {advantage_state.shape[1]}"
            )

        value = self.value_head(value_state)
        advantage = self.advantage_head(advantage_state).squeeze(-1)
        return value + advantage - advantage.mean(dim=1, keepdim=True)
