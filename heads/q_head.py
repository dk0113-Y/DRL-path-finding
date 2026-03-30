from __future__ import annotations

"""Legacy split-input dueling head kept only as historical reference."""

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
      mid_context      [B, D_context]
      token_context    [B, D_context]

    Output:
      q_values         [B, A]

    Semantics:
      - value stream estimates a joint-state value from the explicit
        concatenation of local execution summary, regional dense context,
        and sparse candidate-region context.
      - advantage stream remains local-dominant, and only the regional dense
        context is allowed to condition the raw local summary.
    """

    def __init__(self, cfg: DecisionHeadConfig | None = None):
        super().__init__()
        self.cfg = cfg if cfg is not None else DecisionHeadConfig()
        hidden_dim = self.cfg.hidden_dim
        joint_value_dim = self.cfg.raw_near_summary_dim + (2 * self.cfg.global_context_dim)

        self.value_joint_trunk = nn.Sequential(
            nn.LayerNorm(joint_value_dim),
            nn.Linear(joint_value_dim, hidden_dim),
            nn.GELU(),
        )
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
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
        self.advantage_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(self.cfg.dropout),
            nn.Linear(hidden_dim, self.cfg.action_dim),
        )

    def forward(
        self,
        raw_near_summary: torch.Tensor,
        mid_context: torch.Tensor,
        token_context: torch.Tensor,
    ) -> torch.Tensor:
        if raw_near_summary.dim() != 2:
            raise ValueError(
                f"raw_near_summary must be 2D [B,D], got shape={tuple(raw_near_summary.shape)}"
            )
        if mid_context.dim() != 2:
            raise ValueError(
                f"mid_context must be 2D [B,D], got shape={tuple(mid_context.shape)}"
            )
        if token_context.dim() != 2:
            raise ValueError(
                f"token_context must be 2D [B,D], got shape={tuple(token_context.shape)}"
            )
        if raw_near_summary.shape[0] != mid_context.shape[0]:
            raise ValueError(
                "batch mismatch between raw_near_summary and mid_context: "
                f"{raw_near_summary.shape[0]} vs {mid_context.shape[0]}"
            )
        if raw_near_summary.shape[0] != token_context.shape[0]:
            raise ValueError(
                "batch mismatch between raw_near_summary and token_context: "
                f"{raw_near_summary.shape[0]} vs {token_context.shape[0]}"
            )
        if raw_near_summary.shape[1] != self.cfg.raw_near_summary_dim:
            raise ValueError(
                "raw_near_summary dim mismatch: "
                f"expected {self.cfg.raw_near_summary_dim}, got {raw_near_summary.shape[1]}"
            )
        if mid_context.shape[1] != self.cfg.global_context_dim:
            raise ValueError(
                "mid_context dim mismatch: "
                f"expected {self.cfg.global_context_dim}, got {mid_context.shape[1]}"
            )
        if token_context.shape[1] != self.cfg.global_context_dim:
            raise ValueError(
                "token_context dim mismatch: "
                f"expected {self.cfg.global_context_dim}, got {token_context.shape[1]}"
            )

        joint_value_input = torch.cat([raw_near_summary, mid_context, token_context], dim=-1)
        value_joint = self.value_joint_trunk(joint_value_input)
        value = self.value_head(value_joint)

        near_gate = self.advantage_gate(mid_context)
        conditioned_near = raw_near_summary * (1.0 + near_gate)
        advantage_near = self.advantage_near_trunk(conditioned_near)
        advantage = self.advantage_head(advantage_near)

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
    mid_context = torch.rand(4, cfg.global_context_dim)
    token_context = torch.rand(4, cfg.global_context_dim)
    q_values = head(raw_near_summary, mid_context, token_context)

    assert q_values.shape == (4, 8)
    assert torch.isfinite(q_values).all()
    print("SplitDuelingDecisionHead smoke test passed", tuple(q_values.shape))


if __name__ == "__main__":
    _smoke_test()
