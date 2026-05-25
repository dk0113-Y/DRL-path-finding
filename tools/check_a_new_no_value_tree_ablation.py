from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agents.q_value_agent import ACTION_DIM, ExplorationQNetwork  # noqa: E402
from env.advantage_state_builder import (  # noqa: E402
    ADVANTAGE_CANVAS_SCHEMA_FINAL_4CH_NO_FRONTIER_RASTER,
    FINAL_4CH_ADVANTAGE_CANVAS_CHANNELS,
)
from env.core_cummap import AnalysisBox  # noqa: E402
from env.shared_semantic_layer import SharedSemanticSnapshot  # noqa: E402
from env.value_state_builder import (  # noqa: E402
    VALUE_REPLACEMENT_STRATEGY_ZERO_VALUE_STATE,
    ValueStateBuilder,
    ValueStateConfig,
)


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _load_json_from_output(output: str) -> dict[str, object]:
    start = output.find("{")
    if start < 0:
        raise AssertionError(f"No JSON object found in output:\n{output}")
    return json.loads(output[start:])


def _run_json(command: list[str]) -> dict[str, object]:
    result = subprocess.run(
        command,
        cwd=str(REPO_ROOT),
        text=True,
        capture_output=True,
        check=True,
    )
    return _load_json_from_output(result.stdout)


def _empty_snapshot() -> SharedSemanticSnapshot:
    analysis_box = AnalysisBox(
        r0=0,
        r1=5,
        c0=0,
        c1=5,
        margin=0,
        known_r0=0,
        known_r1=5,
        known_c0=0,
        known_c1=5,
    )
    return SharedSemanticSnapshot(
        analysis_box=analysis_box,
        accessible_blocks=tuple(),
        total_accessible_unknown_area=0,
    )


def _check_d_dry_run() -> dict[str, object]:
    payload = _run_json(
        [
            sys.executable,
            "experiments/final_method/run_a_new_no_value_tree_ablation.py",
            "--run-stage",
            "formal",
            "--device",
            "cpu",
            "--dry-run",
        ]
    )
    _assert(payload["method_id"] == "Anew_D_no_value_tree", "D dry-run method_id mismatch")
    _assert(payload["method_name"] == "no_value_tree", "D dry-run method_name mismatch")
    _assert(payload["ablation_group"] == "structural", "D dry-run ablation_group mismatch")
    _assert(payload["ablation_name"] == "no_value_tree", "D dry-run ablation_name mismatch")
    _assert(payload["run_stage"] == "formal", "D dry-run run_stage mismatch")
    _assert(
        payload["advantage_canvas_schema"] == ADVANTAGE_CANVAS_SCHEMA_FINAL_4CH_NO_FRONTIER_RASTER,
        "D dry-run advantage schema mismatch",
    )
    _assert(payload["advantage_canvas_channels"] == list(FINAL_4CH_ADVANTAGE_CANVAS_CHANNELS), "D channel list mismatch")
    _assert(int(payload["advantage_canvas_channel_count"]) == 4, "D dry-run channel count must be 4")
    _assert(payload["frontier_raster_used"] is False, "D dry-run frontier_raster_used must be false")
    _assert(payload["value_tree_enabled"] is False, "D dry-run value_tree_enabled must be false")
    _assert(payload["value_tree_unchanged"] is False, "D dry-run value_tree_unchanged must be false")
    _assert(
        str(payload["value_replacement_strategy"]) == VALUE_REPLACEMENT_STRATEGY_ZERO_VALUE_STATE,
        "D dry-run replacement strategy must be zero_value_state",
    )
    _assert(payload["dummy_value_tensors_for_interface"] is True, "D must use dummy value tensors")
    _assert(payload["dummy_value_mask_rule"] == "all_false", "D value masks must be all false")
    _assert(payload["reward_override"] == {}, "D reward_override must be empty")
    _assert(payload["model_class"] == "ExplorationQNetwork", "D model class mismatch")
    _assert(int(payload["advantage_encoder.canvas_in_channels"]) == 4, "D encoder channels must stay 4")
    _assert(float(payload["reward_info_scale"]) == 3.1, "D reward_info_scale default changed")
    _assert(float(payload["reward_obstacle_weight"]) == 0.2, "D reward_obstacle_weight default changed")
    _assert(int(payload["learner_updates_per_iter"]) == 1, "D learner_updates_per_iter default changed")
    _assert(int(payload["min_replay_size"]) == 8000, "D min_replay_size default changed")
    _assert(float(payload["epsilon_end"]) == 0.04, "D epsilon_end default changed")
    _assert(int(payload["epsilon_decay_steps"]) == 240000, "D epsilon_decay_steps default changed")
    _assert(payload["train_side_only_tuning"] is True, "D train_side_only_tuning must default true")
    _assert(
        "frontier_block_area_map" not in json.dumps(payload, ensure_ascii=False),
        "D dry-run must not use frontier_block_area_map",
    )
    return payload


def _check_zero_value_state_and_forward() -> None:
    builder = ValueStateBuilder(
        ValueStateConfig(value_replacement_strategy=VALUE_REPLACEMENT_STRATEGY_ZERO_VALUE_STATE)
    )
    block, entry, block_mask, entry_mask, meta = builder.build(_empty_snapshot())
    _assert(block.shape == (16, 2), f"unexpected zero block shape: {block.shape}")
    _assert(entry.shape == (16, 8, 4), f"unexpected zero entry shape: {entry.shape}")
    _assert(not bool(block_mask.any()), "zero value block mask must be all false")
    _assert(not bool(entry_mask.any()), "zero value entry mask must be all false")
    _assert(float(meta["value_packed_block_count"]) == 0.0, "zero value metadata should not report packed blocks")
    _assert(float(meta["value_packed_entry_count"]) == 0.0, "zero value metadata should not report packed entries")

    net = ExplorationQNetwork()
    net.eval()
    batch = 2
    advantage_canvas = torch.zeros((batch, 4, 5, 5), dtype=torch.float32)
    value_block_features = torch.from_numpy(block).unsqueeze(0).repeat(batch, 1, 1)
    value_entry_features = torch.from_numpy(entry).unsqueeze(0).repeat(batch, 1, 1, 1)
    value_block_mask = torch.from_numpy(block_mask).unsqueeze(0).repeat(batch, 1)
    value_entry_mask = torch.from_numpy(entry_mask).unsqueeze(0).repeat(batch, 1, 1)
    with torch.no_grad():
        q_values = net(
            advantage_canvas,
            value_block_features,
            value_entry_features,
            value_block_mask,
            value_entry_mask,
            return_aux=False,
        )
    _assert(q_values.shape == (batch, ACTION_DIM), f"Q shape mismatch: {tuple(q_values.shape)}")
    _assert(torch.isfinite(q_values).all().item(), "Q values must be finite")


def _check_a_new_contract_unchanged() -> None:
    payload = _run_json(
        [
            sys.executable,
            "experiments/final_method/run_a_new_final_method.py",
            "--run-stage",
            "formal",
            "--device",
            "cpu",
            "--dry-run",
        ]
    )
    train_config = payload["train_config"]
    _assert(payload["method_id"] == "A_new", "A_new dry-run method_id changed")
    _assert(payload["method_name"] == "final_4ch_no_frontier_raster", "A_new dry-run method_name changed")
    _assert(int(payload["advantage_canvas_channel_count"]) == 4, "A_new channel count changed")
    _assert(payload["frontier_raster_used"] is False, "A_new frontier_raster_used changed")
    _assert(payload["value_tree_enabled"] is True, "A_new value_tree_enabled changed")
    _assert(payload["reward_override"] == {}, "A_new reward_override changed")
    _assert(train_config["value_replacement_strategy"] == "none", "A_new replacement strategy changed")
    _assert(train_config["no_value_tree"] is False, "A_new no_value_tree changed")
    _assert(train_config["train_side_only_tuning"] is True, "A_new train_side_only_tuning changed")
    _assert("frontier_block_area_map" not in payload["advantage_canvas_channels"], "A_new restored frontier raster channel")


def main() -> int:
    _check_d_dry_run()
    _check_zero_value_state_and_forward()
    _check_a_new_contract_unchanged()
    print("A_new no-value-tree ablation checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
