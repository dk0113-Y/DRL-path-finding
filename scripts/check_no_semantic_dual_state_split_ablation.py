from __future__ import annotations

from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from agents.q_value_agent import ACTION_DIM
from experiments.ablations.semantic_split_ablation import (
    NoSemanticDualStateSplitConfig,
    NoSemanticDualStateSplitQNetwork,
    count_model_parameters,
)


def _assert_finite(name: str, tensor: torch.Tensor) -> None:
    if not torch.isfinite(tensor).all():
        raise AssertionError(f"{name} contains NaN or Inf")


def main() -> int:
    torch.manual_seed(7)
    cfg = NoSemanticDualStateSplitConfig()
    base = cfg.base
    batch = 3
    height = 11
    width = 11
    block_count = 4
    entry_count = 5

    advantage_canvas = torch.randn(
        batch,
        int(base.advantage_encoder.canvas_in_channels),
        height,
        width,
    )
    value_block_features = torch.randn(batch, block_count, int(base.value_encoder.block_input_dim))
    value_entry_features = torch.randn(batch, block_count, entry_count, int(base.value_encoder.entry_input_dim))
    value_block_mask = torch.ones(batch, block_count, dtype=torch.bool)
    value_entry_mask = torch.ones(batch, block_count, entry_count, dtype=torch.bool)

    model = NoSemanticDualStateSplitQNetwork(cfg).eval()
    with torch.no_grad():
        q_values, aux = model(
            advantage_canvas,
            value_block_features,
            value_entry_features,
            value_block_mask,
            value_entry_mask,
            return_aux=True,
        )
        q_values_no_aux = model(
            advantage_canvas,
            value_block_features,
            value_entry_features,
            value_block_mask,
            value_entry_mask,
            return_aux=False,
        )
        q_values_changed_tree = model(
            advantage_canvas,
            -value_block_features,
            -value_entry_features,
            value_block_mask,
            value_entry_mask,
            return_aux=False,
        )

    if q_values.shape != (batch, ACTION_DIM):
        raise AssertionError(f"q_values shape mismatch: expected {(batch, ACTION_DIM)}, got {tuple(q_values.shape)}")
    if q_values_no_aux.shape != (batch, ACTION_DIM):
        raise AssertionError(
            f"return_aux=False q_values shape mismatch: expected {(batch, ACTION_DIM)}, got {tuple(q_values_no_aux.shape)}"
        )
    _assert_finite("q_values", q_values)
    _assert_finite("q_values_no_aux", q_values_no_aux)
    if not torch.allclose(q_values, q_values_no_aux, atol=1e-6, rtol=1e-5):
        raise AssertionError("return_aux=True and return_aux=False disagree in eval mode")

    required_aux = {
        "no_semantic_dual_state_split",
        "value_tree_used_by_model",
        "semantic_dual_state_split_used",
        "value_accessible_block_count",
        "value_tree_input_block_mask_true_count",
        "value_tree_input_entry_mask_true_count",
    }
    missing = sorted(required_aux.difference(aux))
    if missing:
        raise AssertionError("missing aux keys: " + ", ".join(missing))
    if not torch.all(aux["no_semantic_dual_state_split"] == 1):
        raise AssertionError("aux no_semantic_dual_state_split marker must be 1")
    if not torch.all(aux["value_tree_used_by_model"] == 1):
        raise AssertionError("aux value_tree_used_by_model marker must be 1")
    if not torch.all(aux["semantic_dual_state_split_used"] == 0):
        raise AssertionError("aux semantic_dual_state_split_used marker must be 0")
    if not torch.all(aux["value_tree_input_block_mask_true_count"] > 0):
        raise AssertionError("value tree block masks were not preserved")
    if not torch.all(aux["value_tree_input_entry_mask_true_count"] > 0):
        raise AssertionError("value tree entry masks were not preserved")
    if torch.max(torch.abs(q_values - q_values_changed_tree)).item() <= 1e-7:
        raise AssertionError("q_values did not respond to changed value tree tensors")

    print(
        "E no-semantic-dual-state-split structural check passed: "
        f"q_shape={tuple(q_values.shape)} parameter_count={count_model_parameters(model)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
