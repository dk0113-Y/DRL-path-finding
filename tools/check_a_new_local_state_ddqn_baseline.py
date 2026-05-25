from __future__ import annotations

import inspect
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agents.local_state_q_network import (  # noqa: E402
    ACTION_DIM,
    LOCAL_STATE_CHANNELS,
    LocalStateQConfig,
    LocalStateQNetwork,
)
from env.grid_topology import EMPTY, INVISIBLE, OBSTACLE  # noqa: E402
from experiments.final_method.a_new_local_state_ddqn import (  # noqa: E402
    LOCAL_STATE_CANVAS_SCHEMA,
    METHOD_ID,
    METHOD_NAME,
    LocalStateTensorAdapter,
    build_local_state_patch,
)


class DummyCumMap:
    def __init__(self) -> None:
        self.map = np.full((25, 25), INVISIBLE, dtype=np.int8)
        self.map[12, 12] = EMPTY
        self.map[12, 13] = OBSTACLE
        self.map[13, 12] = EMPTY

    def world_to_array(self, world_rc: tuple[int, int]) -> tuple[int, int]:
        return int(world_rc[0]), int(world_rc[1])


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


def _check_c_dry_run() -> dict[str, object]:
    payload = _run_json(
        [
            sys.executable,
            "experiments/final_method/run_a_new_local_state_ddqn_baseline.py",
            "--run-stage",
            "formal",
            "--device",
            "cuda",
            "--dry-run",
        ]
    )
    train_config = payload["train_config"]
    _assert(payload["experiment_id"] == "Anew_C", "C experiment_id mismatch")
    _assert(payload["method_id"] == METHOD_ID, "C method_id mismatch")
    _assert(payload["method_name"] == METHOD_NAME, "C method_name mismatch")
    _assert(payload["baseline_id"] == METHOD_ID, "C baseline_id mismatch")
    _assert(payload["baseline_group"] == "learning", "C baseline_group mismatch")
    _assert(payload["baseline_name"] == METHOD_NAME, "C baseline_name mismatch")
    _assert(payload["baseline_type"] == "learning_ddqn", "C baseline_type mismatch")
    _assert(payload["is_learning_baseline"] is True, "C must be a learning baseline")
    _assert(payload["is_ablation"] is False, "C must not be an ablation")
    _assert(payload["model_class"] == "LocalStateQNetwork", "C model class mismatch")
    _assert(int(payload["model_parameter_count"]) > 0, "C model_parameter_count must be positive")
    _assert(payload["local_state_channels"] == list(LOCAL_STATE_CHANNELS), "C local channels mismatch")
    _assert(int(payload["local_state_patch_size"]) == 21, "C patch size must be 21")
    _assert(payload["local_state_source"] == "cumulative_belief_patch", "C local state source mismatch")
    _assert(payload["no_ground_truth_map_for_decision"] is True, "C no-ground-truth contract mismatch")
    _assert(payload["uses_structured_value_tree"] is False, "C must not use structured value tree")
    _assert(payload["value_tree_enabled"] is False, "C value_tree_enabled must be false")
    _assert(payload["value_tensors_used_by_model"] is False, "C model must ignore value tensors")
    _assert(payload["dummy_value_tensors_for_interface"] is True, "C must declare dummy value tensors")
    _assert(payload["dummy_value_mask_rule"] == "all_false", "C dummy masks must be all false")
    _assert(payload["advantage_canvas_schema"] == LOCAL_STATE_CANVAS_SCHEMA, "C local-state schema mismatch")
    _assert(payload["local_state_canvas_role"] == "baseline_local_state_input", "C canvas role mismatch")
    _assert(payload["frontier_raster_used"] is False, "C frontier_raster_used must be false")
    _assert(payload["behavior_memory_channels_used"] is False, "C behavior-memory channels must be false")
    _assert(payload["reward_override"] == {}, "C reward_override must be empty")
    _assert(payload["checkpoint_source"] == "trained_from_scratch", "C checkpoint source mismatch")
    _assert(float(payload["reward_info_scale"]) == 3.1, "C reward_info_scale default changed")
    _assert(float(payload["reward_obstacle_weight"]) == 0.2, "C reward_obstacle_weight default changed")
    _assert(int(payload["learner_updates_per_iter"]) == 1, "C learner_updates_per_iter default changed")
    _assert(int(payload["min_replay_size"]) == 8000, "C min_replay_size default changed")
    _assert(float(payload["epsilon_end"]) == 0.04, "C epsilon_end default changed")
    _assert(int(payload["epsilon_decay_steps"]) == 240000, "C epsilon_decay_steps default changed")
    _assert(payload["train_side_only_tuning"] is True, "C train_side_only_tuning must default true")
    _assert(train_config["baseline_type"] == "learning_ddqn", "C train_config baseline_type mismatch")
    _assert(train_config["is_learning_baseline"] is True, "C train_config is_learning_baseline mismatch")
    _assert(train_config["value_tensors_used_by_model"] is False, "C train_config value tensor contract mismatch")
    _assert(train_config["reward_override"] == {}, "C train_config reward_override changed")
    dumped = json.dumps(payload, ensure_ascii=False)
    _assert("Anew_F4" not in dumped, "C payload must not create an Anew_F4 row")
    _assert("legacy C" not in dumped, "C payload must not emit a legacy C row")
    _assert("experiments/ablations" not in dumped, "C payload must not restore old ablation framework")
    return payload


def _check_network_forward_and_value_independence() -> None:
    torch.manual_seed(7)
    net = LocalStateQNetwork(LocalStateQConfig(local_state_patch_size=21))
    net.eval()
    local_state = torch.rand((2, 3, 21, 21), dtype=torch.float32)
    zeros_block = torch.zeros((2, 16, 2), dtype=torch.float32)
    zeros_entry = torch.zeros((2, 16, 8, 4), dtype=torch.float32)
    zeros_block_mask = torch.zeros((2, 16), dtype=torch.bool)
    zeros_entry_mask = torch.zeros((2, 16, 8), dtype=torch.bool)
    with torch.no_grad():
        q1 = net(local_state, zeros_block, zeros_entry, zeros_block_mask, zeros_entry_mask, return_aux=False)
        q2 = net(
            local_state,
            torch.randn_like(zeros_block),
            torch.randn_like(zeros_entry),
            torch.ones_like(zeros_block_mask),
            torch.ones_like(zeros_entry_mask),
            return_aux=False,
        )
    _assert(q1.shape == (2, ACTION_DIM), f"C Q shape mismatch: {tuple(q1.shape)}")
    _assert(torch.isfinite(q1).all().item(), "C Q values must be finite")
    _assert(torch.allclose(q1, q2, atol=0.0, rtol=0.0), "C Q output changed when dummy value tensors changed")


def _check_local_state_tensor_semantics() -> None:
    cum_map = DummyCumMap()
    patch = build_local_state_patch(cum_map, (12, 12), patch_size=21)
    _assert(patch.shape == (3, 21, 21), f"C local patch shape mismatch: {patch.shape}")
    center = 10
    _assert(float(patch[0, center, center]) == 1.0, "center cell should be known_free")
    _assert(float(patch[1, center, center + 1]) == 1.0, "east cell should be known_obstacle")
    _assert(float(patch[2, center - 1, center - 1]) == 1.0, "unseen cell should be unknown")
    one_hot = patch.sum(axis=0)
    _assert(np.allclose(one_hot, 1.0), "known_free/known_obstacle/unknown channels must be one-hot")

    adapter = LocalStateTensorAdapter(patch_size=21)
    state_batch, state_meta = adapter.build_single_state_tensors(
        cum_map,
        (12, 12),
        return_state_meta=True,
    )
    _assert(tuple(state_batch["advantage_canvas"].shape) == (1, 3, 21, 21), "adapter tensor shape mismatch")
    _assert(not bool(state_batch["value_block_mask"].any()), "C dummy value block mask must be all false")
    _assert(not bool(state_batch["value_entry_mask"].any()), "C dummy value entry mask must be all false")
    _assert(float(state_meta["local_state_patch_size"]) == 21.0, "C state meta patch size mismatch")


def _check_leak_audit() -> None:
    path_source = inspect.getsource(build_local_state_patch)
    adapter_source = inspect.getsource(LocalStateTensorAdapter)
    model_source = inspect.getsource(LocalStateQNetwork)
    decision_source = "\n".join((path_source, adapter_source, model_source))
    forbidden = (
        "true_grid",
        "full_map",
        "ground_truth",
        "RandomMapGenerator",
        "LocalObservationModel",
        "generate_map",
        "oracle",
        "future sensor",
    )
    for token in forbidden:
        _assert(token not in decision_source, f"C decision/model path contains forbidden token: {token}")
    _assert("get_frontier" not in path_source, "C local patch builder must not request frontier state")
    _assert("frontier_bool" not in adapter_source, "C adapter must not build frontier raster inputs")
    _assert("visit_count" not in path_source, "C local patch builder must not use visit-count memory")


def _check_existing_contracts_still_parse() -> None:
    subprocess.run(
        [sys.executable, "tools/check_a_new_final_4ch.py"],
        cwd=str(REPO_ROOT),
        text=True,
        capture_output=True,
        check=True,
    )
    d_payload = _run_json(
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
    _assert(d_payload["method_id"] == "Anew_D_no_value_tree", "D dry-run no longer parses")
    f_payload = _run_json(
        [
            sys.executable,
            "experiments/final_method/run_a_new_no_behavior_memory_ablation.py",
            "--run-stage",
            "formal",
            "--device",
            "cpu",
            "--dry-run",
        ]
    )
    _assert(f_payload["method_id"] == "Anew_F3_no_behavior_memory", "F_key dry-run no longer parses")
    b_payload = _run_json(
        [
            sys.executable,
            "experiments/final_method/run_a_new_classical_frontier_baseline.py",
            "--run-stage",
            "formal",
            "--device",
            "cpu",
            "--dry-run",
        ]
    )
    _assert(b_payload["method_id"] == "Anew_B_classical_frontier_greedy", "B dry-run no longer parses")
    r_payload = _run_json(
        [
            sys.executable,
            "experiments/final_method/run_a_new_reward_ablation_batch.py",
            "--reward-ablation-ids",
            "R1,R2,R3,R4,R5",
            "--run-stage",
            "smoke",
            "--device",
            "cpu",
            "--dry-run",
        ]
    )
    methods = r_payload.get("methods", [])
    _assert(isinstance(methods, list) and len(methods) == 5, "R dry-run no longer parses all five rows")


def main() -> int:
    _check_c_dry_run()
    _check_network_forward_and_value_independence()
    _check_local_state_tensor_semantics()
    _check_leak_audit()
    _check_existing_contracts_still_parse()
    print("A_new local-state DDQN baseline checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
