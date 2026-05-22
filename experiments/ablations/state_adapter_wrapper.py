from __future__ import annotations

from typing import Optional, Sequence

import torch

from agents.q_value_agent import SharedStepArtifacts, StateAdapterConfig, StateTensorAdapter
from experiments.ablations.channel_ablation import (
    apply_channel_ablation_to_state_batch,
    validate_zeroed_channels,
)


class AblationStateTensorAdapter(StateTensorAdapter):
    def __init__(
        self,
        cfg: Optional[StateAdapterConfig] = None,
        device: str = "cpu",
        zeroed_channels=(),
    ):
        super().__init__(cfg=cfg, device=device)
        self.zeroed_channels = validate_zeroed_channels(zeroed_channels)

    def build_single_state_tensors(
        self,
        cum_map,
        agent_state,
        recent_trajectory_positions: Optional[Sequence[tuple[int, int]]] = None,
        shared_artifacts: Optional[SharedStepArtifacts] = None,
        target_device: Optional[torch.device | str] = None,
        return_state_meta: bool = False,
    ):
        result = super().build_single_state_tensors(
            cum_map,
            agent_state,
            recent_trajectory_positions=recent_trajectory_positions,
            shared_artifacts=shared_artifacts,
            target_device=target_device,
            return_state_meta=return_state_meta,
        )
        if not self.zeroed_channels:
            return result

        if return_state_meta:
            state_batch, state_meta = result
            state_batch = apply_channel_ablation_to_state_batch(state_batch, self.zeroed_channels)
            state_meta = dict(state_meta)
            state_meta["ablation_zeroed_advantage_channels"] = list(self.zeroed_channels)
            state_meta["advantage_channel_ablation_active"] = True
            return state_batch, state_meta

        return apply_channel_ablation_to_state_batch(result, self.zeroed_channels)
