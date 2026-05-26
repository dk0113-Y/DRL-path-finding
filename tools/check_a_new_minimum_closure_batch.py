from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DEFAULT_INCLUDED = [
    "Anew_C_local_state_ddqn",
    "Anew_D_no_value_tree",
    "Anew_E_no_dual_state_split",
    "Anew_F3_no_behavior_memory",
    "Anew_R5",
]
FULL_REWARD = ["Anew_R1", "Anew_R2", "Anew_R3", "Anew_R4", "Anew_R5"]
FORBIDDEN_BATCH_OPTIONS = [
    "--reward-info-scale",
    "--reward-obstacle-weight",
    "--learner-updates-per-iter",
    "--min-replay-size",
    "--epsilon-end",
    "--epsilon-decay-steps",
    "--reward-step-penalty",
    "--reward-terminal-bonus",
    "--reward-revisit-penalty",
    "--reward-turn-penalty-scale",
    "--reward-timeout-penalty",
]
EXISTING_CHECKS = [
    "tools/check_a_new_final_4ch.py",
    "tools/check_a_new_no_value_tree_ablation.py",
    "tools/check_a_new_no_behavior_memory_ablation.py",
    "tools/check_a_new_classical_frontier_baseline.py",
    "tools/check_a_new_local_state_ddqn_baseline.py",
    "tools/check_a_new_no_dual_state_split_ablation.py",
]


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _load_json_from_output(output: str) -> dict[str, Any]:
    start = output.find("{")
    if start < 0:
        raise AssertionError(f"No JSON object found in output:\n{output}")
    return json.loads(output[start:])


def _run_plan(*extra: str) -> dict[str, Any]:
    result = subprocess.run(
        [
            sys.executable,
            "experiments/final_method/run_a_new_minimum_closure_batch.py",
            "--dry-run",
            *extra,
        ],
        cwd=str(REPO_ROOT),
        text=True,
        capture_output=True,
        check=True,
    )
    return _load_json_from_output(result.stdout)


def _commands(plan: dict[str, Any]) -> list[dict[str, Any]]:
    commands = plan.get("commands")
    _assert(isinstance(commands, list), "batch plan commands must be a list")
    for command in commands:
        _assert(isinstance(command, dict), "batch command must be a JSON object")
    return commands


def _command_texts(plan: dict[str, Any]) -> list[str]:
    texts = []
    for command in _commands(plan):
        text = str(command.get("command_text", ""))
        texts.append(text)
        raw = command.get("command", [])
        _assert(isinstance(raw, list), "command must be a list")
        texts.append(" ".join(str(item) for item in raw))
    return texts


def _check_default_plan() -> None:
    plan = _run_plan("--run-stage", "formal", "--device", "cuda")
    _assert(plan["batch_id"] == "A_new_minimum_closure_batch", "batch_id mismatch")
    _assert(plan["run_stage"] == "formal", "default formal dry-run run_stage mismatch")
    _assert(plan["device"] == "cuda", "default device mismatch")
    _assert(plan["included_methods"] == DEFAULT_INCLUDED, "default included_methods mismatch")
    _assert("A_new" not in plan["included_methods"], "default batch must not include A_new")
    _assert("Anew_B_classical_frontier_greedy" not in plan["included_methods"], "default batch must not include B")
    for method_id in FULL_REWARD[:-1]:
        _assert(method_id not in plan["included_methods"], f"default batch must not include {method_id}")
    _assert(plan["uses_current_a_new_defaults"] is True, "uses_current_a_new_defaults must be true")
    _assert(plan["parameters_hardcoded_in_batch"] is False, "parameters_hardcoded_in_batch must be false")
    _assert(plan["a_new_parameter_status"] == "candidate_not_frozen", "A_new parameter status mismatch")
    _assert(plan["extra_args"] == [], "default plan must not include extra args")
    for text in _command_texts(plan):
        for option in FORBIDDEN_BATCH_OPTIONS:
            _assert(option not in text, f"batch command hardcodes forbidden option {option}")


def _check_include_b() -> None:
    plan = _run_plan("--include-b")
    _assert("Anew_B_classical_frontier_greedy" in plan["included_methods"], "IncludeB must include B")
    b_commands = [command for command in _commands(plan) if command.get("label") == "B"]
    _assert(len(b_commands) == 1, "IncludeB must add exactly one B command")
    b_command = b_commands[0].get("command", [])
    _assert("--device" in b_command and "cpu" in b_command, "B command must force device cpu")


def _check_include_all_reward_ablations() -> None:
    plan = _run_plan("--include-all-reward-ablations")
    for method_id in FULL_REWARD:
        _assert(method_id in plan["included_methods"], f"IncludeAllRewardAblations missing {method_id}")
    reward_commands = [command for command in _commands(plan) if command.get("label") == "R_full"]
    _assert(len(reward_commands) == 1, "full reward plan must have one R_full command")
    text = str(reward_commands[0].get("command_text", ""))
    _assert("R1,R2,R3,R4,R5" in text, "full reward command must select R1-R5")


def _check_paths_exist() -> None:
    plan = _run_plan()
    for command in _commands(plan):
        runner = REPO_ROOT / str(command["runner_entrypoint"])
        launcher = REPO_ROOT / str(command["launcher_script"])
        _assert(runner.exists(), f"runner path missing: {runner}")
        _assert(launcher.exists(), f"launcher script missing: {launcher}")
    for path in [
        "scripts/run_a_new_minimum_closure_batch.ps1",
        "experiments/final_method/run_a_new_minimum_closure_batch.py",
    ]:
        _assert((REPO_ROOT / path).exists(), f"minimum closure path missing: {path}")


def _check_dry_run_no_outputs() -> None:
    temp_dir = Path(tempfile.mkdtemp(prefix="a_new_minimum_closure_check_"))
    output_root = temp_dir / "outputs_should_not_exist"
    try:
        plan = _run_plan("--output-root", str(output_root))
        _assert(plan["output_root"] == str(output_root), "dry-run output root mismatch")
        _assert(not output_root.exists(), "dry-run must not create output root")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def _check_extra_args_passthrough() -> None:
    default_plan = _run_plan()
    for text in _command_texts(default_plan):
        _assert("--episode-print-interval" not in text, "default plan unexpectedly contains extra arg")
    plan = _run_plan("--extra-args", "--episode-print-interval", "0")
    _assert(plan["extra_args"] == ["--episode-print-interval", "0"], "extra args were not recorded")
    for command in _commands(plan):
        label = command.get("label")
        text = str(command.get("command_text", ""))
        if label in {"C", "D", "E", "F_key", "R_key"}:
            _assert("--episode-print-interval" in text, f"{label} command missing passthrough arg")
        if label == "B":
            _assert("--episode-print-interval" not in text, "B must not receive training passthrough args")


def _check_existing_checks() -> None:
    for check_path in EXISTING_CHECKS:
        subprocess.run(
            [sys.executable, check_path],
            cwd=str(REPO_ROOT),
            text=True,
            capture_output=True,
            check=True,
        )


def main() -> int:
    _check_default_plan()
    _check_include_b()
    _check_include_all_reward_ablations()
    _check_paths_exist()
    _check_dry_run_no_outputs()
    _check_extra_args_passthrough()
    _check_existing_checks()
    print("A_new minimum-closure batch checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
