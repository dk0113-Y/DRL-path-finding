from __future__ import annotations

from typing import Mapping

import torch


VALUE_REPLACEMENT_STRATEGY_ZERO = "zero_value_state"


def apply_zero_value_state_to_state_batch(
    state_batch: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """
    Remove value tree information while preserving the main network interface.

    The value branch keeps the original tensor shapes, but every value feature is
    zero and every value mask is false. This removes block area, frontier cluster
    counts, entry geometry, support density, and the real block-entry structure.
    The current ValueTreeEncoder handles all-false masks with zero attention
    weights, so no dummy block or learned replacement is needed.
    """
    required = (
        "value_block_features",
        "value_entry_features",
        "value_block_mask",
        "value_entry_mask",
    )
    missing = [key for key in required if key not in state_batch]
    if missing:
        raise KeyError("state_batch is missing value keys: " + ", ".join(missing))

    result = dict(state_batch)
    result["value_block_features"] = torch.zeros_like(state_batch["value_block_features"])
    result["value_entry_features"] = torch.zeros_like(state_batch["value_entry_features"])
    result["value_block_mask"] = torch.zeros_like(state_batch["value_block_mask"], dtype=torch.bool)
    result["value_entry_mask"] = torch.zeros_like(state_batch["value_entry_mask"], dtype=torch.bool)
    return result


def zero_value_state_metadata(state_batch: Mapping[str, torch.Tensor]) -> dict[str, object]:
    return {
        "value_replacement_strategy": VALUE_REPLACEMENT_STRATEGY_ZERO,
        "value_tree_enabled": False,
        "safe_zero_dummy_value_state": False,
        "dummy_value_state_shape": None,
        "dummy_mask_rule": None,
        "zero_value_state_block_shape": list(state_batch["value_block_features"].shape),
        "zero_value_state_entry_shape": list(state_batch["value_entry_features"].shape),
        "zero_value_state_block_mask_true_count": int(state_batch["value_block_mask"].sum().item()),
        "zero_value_state_entry_mask_true_count": int(state_batch["value_entry_mask"].sum().item()),
        "zero_value_state_contains_real_unknown_block_info": False,
        "zero_value_state_contains_real_frontier_cluster_info": False,
        "zero_value_state_contains_real_entry_geometry_or_area_info": False,
    }

