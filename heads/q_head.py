from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from env.grid_topology import ACTIONS_8


@dataclass(frozen=True)
class DecisionHeadConfig:
    raw_near_summary_dim: int = 128
    global_context_dim: int = 256
    hidden_dim: int = 256
    action_dim: int = len(ACTIONS_8)
    dropout: float = 0.0


class SplitDuelingDecisionHead(nn.Module):
    """
    Split-input dueling decision head.

    Inputs:
      raw_near_summary [B, D_near]
      global_context   [B, D_global]

    Output:
      q_values         [B, A]
    """

    def __init__(self, cfg: DecisionHeadConfig | None = None):
        super().__init__()
        self.cfg = cfg if cfg is not None else DecisionHeadConfig()
        hidden_dim = self.cfg.hidden_dim
        value_near_dim = max(32, hidden_dim // 4)
        advantage_global_dim = max(32, hidden_dim // 4)

        self.global_value_trunk = nn.Sequential(
            nn.LayerNorm(self.cfg.global_context_dim),
            nn.Linear(self.cfg.global_context_dim, hidden_dim),
            nn.GELU(),
        )
        self.value_near_aux = nn.Sequential(
            nn.LayerNorm(self.cfg.raw_near_summary_dim),
            nn.Linear(self.cfg.raw_near_summary_dim, value_near_dim),
            nn.GELU(),
        )
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim + value_near_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(self.cfg.dropout),
            nn.Linear(hidden_dim, 1),
        )

        self.advantage_gate = nn.Sequential(
            nn.LayerNorm(self.cfg.global_context_dim),
            nn.Linear(self.cfg.global_context_dim, self.cfg.raw_near_summary_dim),
            nn.Sigmoid(),
        )
        self.advantage_near_trunk = nn.Sequential(
            nn.LayerNorm(self.cfg.raw_near_summary_dim),
            nn.Linear(self.cfg.raw_near_summary_dim, hidden_dim),
            nn.GELU(),
        )
        self.advantage_global_aux = nn.Sequential(
            nn.LayerNorm(self.cfg.global_context_dim),
            nn.Linear(self.cfg.global_context_dim, advantage_global_dim),
            nn.GELU(),
        )
        self.advantage_head = nn.Sequential(
            nn.Linear(hidden_dim + advantage_global_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(self.cfg.dropout),
            nn.Linear(hidden_dim, self.cfg.action_dim),
        )

    def forward(
        self,
        raw_near_summary: torch.Tensor,
        global_context: torch.Tensor,
    ) -> torch.Tensor:
        if raw_near_summary.dim() != 2:
            raise ValueError(
                f"raw_near_summary must be 2D [B,D], got shape={tuple(raw_near_summary.shape)}"
            )
        if global_context.dim() != 2:
            raise ValueError(
                f"global_context must be 2D [B,D], got shape={tuple(global_context.shape)}"
            )
        if raw_near_summary.shape[0] != global_context.shape[0]:
            raise ValueError(
                "batch mismatch between raw_near_summary and global_context: "
                f"{raw_near_summary.shape[0]} vs {global_context.shape[0]}"
            )
        if raw_near_summary.shape[1] != self.cfg.raw_near_summary_dim:
            raise ValueError(
                "raw_near_summary dim mismatch: "
                f"expected {self.cfg.raw_near_summary_dim}, got {raw_near_summary.shape[1]}"
            )
        if global_context.shape[1] != self.cfg.global_context_dim:
            raise ValueError(
                f"global_context dim mismatch: expected {self.cfg.global_context_dim}, got {global_context.shape[1]}"
            )

        value_global = self.global_value_trunk(global_context)
        value_near = self.value_near_aux(raw_near_summary)
        value = self.value_head(torch.cat([value_global, value_near], dim=-1))

        near_gate = self.advantage_gate(global_context)
        conditioned_near = raw_near_summary * (1.0 + near_gate)
        advantage_near = self.advantage_near_trunk(conditioned_near)
        advantage_global = self.advantage_global_aux(global_context)
        advantage = self.advantage_head(torch.cat([advantage_near, advantage_global], dim=-1))

        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        return q_values


def _smoke_test() -> None:
    cfg = DecisionHeadConfig(
        raw_near_summary_dim=128,
        global_context_dim=256,
        hidden_dim=128,
        action_dim=8,
    )
    head = SplitDuelingDecisionHead(cfg)

    raw_near_summary = torch.rand(4, cfg.raw_near_summary_dim)
    global_context = torch.rand(4, cfg.global_context_dim)
    q_values = head(raw_near_summary, global_context)

    assert q_values.shape == (4, 8)
    assert torch.isfinite(q_values).all()
    print("SplitDuelingDecisionHead smoke test passed", tuple(q_values.shape))


if __name__ == "__main__":
    _smoke_test()
