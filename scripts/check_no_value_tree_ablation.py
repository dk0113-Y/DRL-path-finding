from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from agents.q_value_agent import ExplorationQNetwork, StateTensorAdapter
from env.agent_version import LocalObservationModel
from env.block_random_g import RandomMapGenerator
from env.core_cummap import CumulativeBeliefMap
from experiments.ablations.state_adapter_wrapper import AblationStateTensorAdapter
from experiments.ablations.value_tree_ablation import VALUE_REPLACEMENT_STRATEGY_ZERO


def _assert_same_shape(name: str, lhs: torch.Tensor, rhs: torch.Tensor) -> None:
    if tuple(lhs.shape) != tuple(rhs.shape):
        raise AssertionError(f"{name} shape mismatch: {tuple(lhs.shape)} != {tuple(rhs.shape)}")


def main() -> int:
    torch.manual_seed(0)
    grid, start = RandomMapGenerator(30, 40, 5, 0.20).generate_map(seed=20261323)
    obs = LocalObservationModel(grid, start)
    cum_map = CumulativeBeliefMap(grid, start, obs.local_snap)

    base_adapter = StateTensorAdapter(device="cpu")
    ablation_adapter = AblationStateTensorAdapter(
        device="cpu",
        value_replacement_strategy=VALUE_REPLACEMENT_STRATEGY_ZERO,
    )
    shared_artifacts = base_adapter.build_shared_step_artifacts(cum_map, start)
    base_state, base_meta = base_adapter.build_single_state_tensors(
        cum_map,
        start,
        shared_artifacts=shared_artifacts,
        return_state_meta=True,
    )
    ablated_state, ablated_meta = ablation_adapter.build_single_state_tensors(
        cum_map,
        start,
        shared_artifacts=shared_artifacts,
        return_state_meta=True,
    )

    for key in (
        "advantage_canvas",
        "value_block_features",
        "value_entry_features",
        "value_block_mask",
        "value_entry_mask",
    ):
        _assert_same_shape(key, base_state[key], ablated_state[key])

    if not torch.allclose(base_state["advantage_canvas"], ablated_state["advantage_canvas"]):
        raise AssertionError("D ablation must not modify advantage_canvas")
    if int(ablated_state["value_block_mask"].sum().item()) != 0:
        raise AssertionError("D ablation value_block_mask must be all false")
    if int(ablated_state["value_entry_mask"].sum().item()) != 0:
        raise AssertionError("D ablation value_entry_mask must be all false")
    if float(ablated_state["value_block_features"].abs().sum().item()) != 0.0:
        raise AssertionError("D ablation value_block_features must be all zero")
    if float(ablated_state["value_entry_features"].abs().sum().item()) != 0.0:
        raise AssertionError("D ablation value_entry_features must be all zero")
    if int(base_state["value_block_mask"].sum().item()) <= 0:
        raise AssertionError("base state unexpectedly has no real value blocks to ablate")
    if str(ablated_meta.get("value_replacement_strategy")) != VALUE_REPLACEMENT_STRATEGY_ZERO:
        raise AssertionError("ablation metadata did not record zero_value_state")
    if bool(ablated_meta.get("value_tree_enabled", True)):
        raise AssertionError("ablation metadata did not disable value_tree_enabled")

    net = ExplorationQNetwork()
    q_values, aux = net(
        ablated_state["advantage_canvas"],
        ablated_state["value_block_features"],
        ablated_state["value_entry_features"],
        ablated_state["value_block_mask"],
        ablated_state["value_entry_mask"],
        return_aux=True,
    )
    if tuple(q_values.shape) != (1, 8):
        raise AssertionError(f"unexpected q_values shape: {tuple(q_values.shape)}")
    if not torch.isfinite(q_values).all():
        raise AssertionError("model forward produced non-finite q_values")
    for name, value in aux.items():
        if isinstance(value, torch.Tensor) and not torch.isfinite(value.float()).all():
            raise AssertionError(f"model aux tensor is non-finite: {name}")

    print(
        "no-value-tree shape/forward check passed "
        f"advantage={tuple(ablated_state['advantage_canvas'].shape)} "
        f"block={tuple(ablated_state['value_block_features'].shape)} "
        f"entry={tuple(ablated_state['value_entry_features'].shape)} "
        f"base_real_blocks={int(base_meta['value_packed_block_count'])}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

