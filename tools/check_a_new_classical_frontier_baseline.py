from __future__ import annotations

import inspect
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from env.grid_topology import EMPTY, INVISIBLE, OBSTACLE  # noqa: E402
from experiments.final_method.a_new_classical_frontier_greedy_policy import (  # noqa: E402
    ClassicalFrontierGreedyPolicy,
)


METHOD_ID = "Anew_B_classical_frontier_greedy"


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


def _check_b_dry_run() -> None:
    formal = _run_json(
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
    smoke = _run_json(
        [
            sys.executable,
            "experiments/final_method/run_a_new_classical_frontier_baseline.py",
            "--run-stage",
            "smoke",
            "--device",
            "cpu",
            "--dry-run",
        ]
    )
    _assert(formal["experiment_id"] == "Anew_B", "B experiment_id mismatch")
    _assert(formal["method_id"] == METHOD_ID, "B method_id mismatch")
    _assert(formal["method_name"] == "classical_frontier_greedy", "B method_name mismatch")
    _assert(formal["baseline_group"] == "classical", "B baseline_group mismatch")
    _assert(formal["baseline_type"] == "traditional_non_learning", "B baseline_type mismatch")
    _assert(formal["trainable_parameters"] == 0, "B trainable parameter count must be zero")
    _assert(formal["checkpoint_used"] is False, "B must not use checkpoints")
    _assert(formal["no_ground_truth_map_for_decision"] is True, "B decision leak contract mismatch")
    _assert(formal["reward_override"] == {}, "B reward_override must be empty")
    _assert(formal["episodes"] == 100, "B formal default episodes must be 100")
    _assert(smoke["episodes"] == 2, "B smoke default episodes must be 2")
    _assert(formal["seed_base"] == 20261323, "B seed_base must match current A_new final probe seed base")
    _assert(float(formal["reward_info_scale"]) == 3.1, "B reward_info_scale default changed")
    _assert(float(formal["reward_obstacle_weight"]) == 0.2, "B reward_obstacle_weight default changed")
    _assert(int(formal["max_episode_steps"]) == 600, "B max_episode_steps default changed")
    _assert(float(formal["coverage_stop_threshold"]) == 0.95, "B coverage threshold default changed")
    summary = formal["policy_summary"]
    for key in ("target_selection_rule", "fallback_rule", "tie_break_rule", "path_cost_rule"):
        _assert(key in summary and str(summary[key]).strip(), f"B policy summary missing {key}")
    planned = set(formal["planned_artifacts"])
    for artifact in (
        "logs/config_snapshot.json",
        "logs/baseline_manifest.json",
        "logs/baseline_policy_summary.json",
        "logs/benchmark_summary.json",
        "logs/reproducibility_contract.json",
        "logs/artifact_index.json",
    ):
        _assert(artifact in planned, f"B dry-run missing planned artifact {artifact}")


def _check_policy_leak_audit() -> None:
    policy_path = REPO_ROOT / "experiments" / "final_method" / "a_new_classical_frontier_greedy_policy.py"
    text = policy_path.read_text(encoding="utf-8")
    forbidden = (
        "true_grid",
        "full_map",
        "ground_truth",
        "RandomMapGenerator",
        "LocalObservationModel",
        "generate_map",
        "oracle",
        "ExplorationQNetwork",
        "torch",
        "free_mask",
        "obstacle_mask",
    )
    for token in forbidden:
        _assert(token not in text, f"B policy file contains forbidden decision-path token: {token}")
    signature = inspect.signature(ClassicalFrontierGreedyPolicy.decide)
    forbidden_params = {"true_grid", "full_map", "ground_truth", "oracle"}
    for param_name in signature.parameters:
        _assert(param_name not in forbidden_params, f"B policy decide parameter leaks {param_name}")


def _check_policy_synthetic_cases() -> None:
    policy = ClassicalFrontierGreedyPolicy()

    target_case = np.full((7, 7), OBSTACLE, dtype=np.int8)
    target_case[3, 1:5] = EMPTY
    target_case[3, 5] = INVISIBLE
    decision = policy.decide(
        belief_map=target_case,
        agent_array_rc=(3, 1),
        valid_action_indices=[2],
        semantic_snapshot=None,
        scan_radius=3,
    )
    _assert(decision.action_idx == 2, f"frontier target case returned invalid action {decision}")
    _assert(decision.decision_mode == "frontier_greedy", "target case should use frontier_greedy")

    fallback_case = np.full((7, 7), OBSTACLE, dtype=np.int8)
    fallback_case[3, 3] = EMPTY
    fallback_case[3, 4] = EMPTY
    fallback_case[3, 6] = INVISIBLE
    fallback = policy.decide(
        belief_map=fallback_case,
        agent_array_rc=(3, 3),
        valid_action_indices=[2],
        semantic_snapshot=None,
        scan_radius=3,
    )
    _assert(fallback.action_idx == 2, f"fallback case returned invalid action {fallback}")
    _assert(fallback.decision_mode == "immediate_info_gain", "fallback case should use immediate_info_gain")

    tie_case = np.full((5, 5), EMPTY, dtype=np.int8)
    first = policy.decide(
        belief_map=tie_case,
        agent_array_rc=(2, 2),
        valid_action_indices=[2, 0],
        semantic_snapshot=None,
        scan_radius=2,
    )
    second = policy.decide(
        belief_map=tie_case,
        agent_array_rc=(2, 2),
        valid_action_indices=[2, 0],
        semantic_snapshot=None,
        scan_radius=2,
    )
    _assert(first.action_idx == 0, f"tie-break should choose action 0, got {first}")
    _assert(second.action_idx == first.action_idx, "tie-break must be deterministic")


def _check_existing_contracts_still_parse() -> None:
    final_payload = _run_json(
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
    _assert(final_payload["method_id"] == "A_new", "A_new dry-run no longer parses")

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
    _check_b_dry_run()
    _check_policy_leak_audit()
    _check_policy_synthetic_cases()
    _check_existing_contracts_still_parse()
    print("A_new classical frontier baseline checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
