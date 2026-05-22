from __future__ import annotations

from collections.abc import Iterable

import torch


ADVANTAGE_CHANNEL_INDEX = {
    "free": 0,
    "obstacle": 1,
    "frontier_block_area_map": 2,
    "visit_count_log_norm": 3,
    "recent_trajectory_decay": 4,
}


def validate_zeroed_channels(zeroed_channels: Iterable[str]) -> tuple[str, ...]:
    normalized: list[str] = []
    seen: set[str] = set()
    for channel in zeroed_channels:
        name = str(channel).strip()
        if name not in ADVANTAGE_CHANNEL_INDEX:
            available = ", ".join(ADVANTAGE_CHANNEL_INDEX)
            raise ValueError(f"Unknown advantage channel {channel!r}. Available channels: {available}")
        if name not in seen:
            normalized.append(name)
            seen.add(name)
    return tuple(normalized)


def apply_channel_ablation_to_canvas(
    advantage_canvas: torch.Tensor,
    zeroed_channels: Iterable[str],
) -> torch.Tensor:
    channels = validate_zeroed_channels(zeroed_channels)
    if not isinstance(advantage_canvas, torch.Tensor):
        raise TypeError("advantage_canvas must be a torch.Tensor")
    if advantage_canvas.dim() != 4:
        raise ValueError(
            "advantage_canvas must have shape [B, 5, H, W], "
            f"got {tuple(advantage_canvas.shape)}"
        )
    if int(advantage_canvas.shape[1]) != len(ADVANTAGE_CHANNEL_INDEX):
        raise ValueError(
            "advantage_canvas channel count must be 5, "
            f"got {int(advantage_canvas.shape[1])}"
        )

    ablated = advantage_canvas.clone()
    for channel in channels:
        ablated[:, ADVANTAGE_CHANNEL_INDEX[channel], :, :] = 0
    return ablated


def apply_channel_ablation_to_state_batch(
    state_batch: dict[str, torch.Tensor],
    zeroed_channels: Iterable[str],
) -> dict[str, torch.Tensor]:
    if "advantage_canvas" not in state_batch:
        raise KeyError("state_batch is missing required key 'advantage_canvas'")
    result = dict(state_batch)
    result["advantage_canvas"] = apply_channel_ablation_to_canvas(
        state_batch["advantage_canvas"],
        zeroed_channels,
    )
    return result
