from __future__ import annotations

from typing import Optional, Sequence

import torch

from agents.q_value_agent import SharedStepArtifacts, StateAdapterConfig, StateTensorAdapter
from experiments.ablations.channel_ablation import (
    apply_channel_ablation_to_state_batch,
    validate_zeroed_channels,
)
from experiments.ablations.value_tree_ablation import (
    VALUE_REPLACEMENT_STRATEGY_ZERO,
    apply_zero_value_state_to_state_batch,
    zero_value_state_metadata,
)


class AblationStateTensorAdapter(StateTensorAdapter):
    def __init__(
        self,
        cfg: Optional[StateAdapterConfig] = None,
        device: str = "cpu",
        zeroed_channels=(),
        value_replacement_strategy: str = "none",
    ):
        super().__init__(cfg=cfg, device=device)
        self.zeroed_channels = validate_zeroed_channels(zeroed_channels)
        self.value_replacement_strategy = str(value_replacement_strategy or "none")
        if self.value_replacement_strategy not in {"none", VALUE_REPLACEMENT_STRATEGY_ZERO}:
            raise ValueError(
                "Unsupported value_replacement_strategy: "
                f"{self.value_replacement_strategy!r}"
            )

    def _apply_ablation_to_state_batch(self, state_batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        result = dict(state_batch)
        if self.zeroed_channels:
            result = apply_channel_ablation_to_state_batch(result, self.zeroed_channels)
        if self.value_replacement_strategy == VALUE_REPLACEMENT_STRATEGY_ZERO:
            result = apply_zero_value_state_to_state_batch(result)
        return result

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
        if not self.zeroed_channels and self.value_replacement_strategy == "none":
            return result

        if return_state_meta:
            state_batch, state_meta = result
            state_batch = self._apply_ablation_to_state_batch(state_batch)
            state_meta = dict(state_meta)
            if self.zeroed_channels:
                state_meta["ablation_zeroed_advantage_channels"] = list(self.zeroed_channels)
                state_meta["advantage_channel_ablation_active"] = True
            if self.value_replacement_strategy == VALUE_REPLACEMENT_STRATEGY_ZERO:
                state_meta.update(zero_value_state_metadata(state_batch))
            return state_batch, state_meta

        return self._apply_ablation_to_state_batch(result)
