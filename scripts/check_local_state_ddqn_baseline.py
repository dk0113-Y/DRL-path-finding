from __future__ import annotations

from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from baselines.local_state_ddqn import (
    LOCAL_STATE_BASELINE_ID,
    LOCAL_STATE_CHANNELS,
    LocalStateQNetwork,
    LocalStateTensorAdapter,
)
from env.agent_version import LocalObservationModel
from env.block_random_g import RandomMapGenerator
from env.core_cummap import CumulativeBeliefMap
from env.grid_topology import ACTIONS_8, GridTopology


def main() -> int:
    torch.manual_seed(0)
    grid, start = RandomMapGenerator(30, 40, 5, 0.2).generate_map()
    obs = LocalObservationModel(grid, start)
    cum_map = CumulativeBeliefMap(grid, start, obs.local_snap)

    adapter = LocalStateTensorAdapter(device="cpu")
    artifacts = adapter.build_shared_step_artifacts(cum_map, start)
    assert artifacts.semantic_snapshot is None
    state_batch, state_meta = adapter.build_single_state_tensors(
        cum_map,
        start,
        shared_artifacts=artifacts,
        return_state_meta=True,
    )

    canvas = state_batch["advantage_canvas"]
    block_features = state_batch["value_block_features"]
    entry_features = state_batch["value_entry_features"]
    block_mask = state_batch["value_block_mask"]
    entry_mask = state_batch["value_entry_mask"]

    expected_patch = int(cum_map.local_shape[0])
    assert canvas.shape == (1, len(LOCAL_STATE_CHANNELS), expected_patch, expected_patch), tuple(canvas.shape)
    assert block_features.dim() == 3 and block_features.shape[0] == 1, tuple(block_features.shape)
    assert entry_features.dim() == 4 and entry_features.shape[0] == 1, tuple(entry_features.shape)
    assert block_mask.shape == block_features.shape[:2], tuple(block_mask.shape)
    assert entry_mask.shape == entry_features.shape[:3], tuple(entry_mask.shape)
    assert torch.count_nonzero(block_features).item() == 0
    assert torch.count_nonzero(entry_features).item() == 0
    assert torch.count_nonzero(block_mask).item() == 0
    assert torch.count_nonzero(entry_mask).item() == 0

    assert state_meta["baseline_id"] == LOCAL_STATE_BASELINE_ID
    assert state_meta["no_shared_semantic_dual_state"] is True
    assert state_meta["no_value_tree"] is True
    assert state_meta["no_frontier_cluster_input"] is True
    assert state_meta["no_accessible_unknown_block_input"] is True
    assert state_meta["no_ground_truth_map_for_decision"] is True
    assert state_meta["value_tensors_used_by_model"] is False

    model = LocalStateQNetwork()
    model.eval()
    with torch.inference_mode():
        q_values = model(
            canvas,
            block_features,
            entry_features,
            block_mask,
            entry_mask,
            return_aux=False,
        )
        perturbed_blocks = torch.randn_like(block_features)
        perturbed_entries = torch.randn_like(entry_features)
        perturbed_block_mask = torch.ones_like(block_mask, dtype=torch.bool)
        perturbed_entry_mask = torch.ones_like(entry_mask, dtype=torch.bool)
        q_values_perturbed = model(
            canvas,
            perturbed_blocks,
            perturbed_entries,
            perturbed_block_mask,
            perturbed_entry_mask,
            return_aux=False,
        )

    assert q_values.shape == (1, len(ACTIONS_8)), tuple(q_values.shape)
    assert torch.isfinite(q_values).all()
    assert torch.allclose(q_values, q_values_perturbed, atol=0.0, rtol=0.0)

    valid_actions = GridTopology.valid_action_indices_fast(GridTopology.free_mask(grid), start)
    assert len(valid_actions) > 0

    print(
        "C local-state DDQN check passed: "
        f"canvas={tuple(canvas.shape)} "
        f"q_values={tuple(q_values.shape)} "
        f"dummy_block={tuple(block_features.shape)} "
        f"dummy_entry={tuple(entry_features.shape)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
