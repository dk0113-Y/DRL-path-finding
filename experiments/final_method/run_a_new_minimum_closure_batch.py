from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_STAGES = ("smoke", "pilot", "formal")
DEVICES = ("cuda", "cpu")
RUN_SET_MINIMUM_CLOSURE = "minimum_closure"


@dataclass(frozen=True)
class BatchCommandSpec:
    label: str
    runner_entrypoint: str
    launcher_script: str
    method_ids: tuple[str, ...]
    command: tuple[str, ...]
    notes: tuple[str, ...] = ()


def _normalize_extra_args(args: list[str]) -> list[str]:
    normalized = list(args)
    while normalized and normalized[0] == "--":
        normalized = normalized[1:]
    return normalized


def _command_text(command: list[str] | tuple[str, ...]) -> str:
    try:
        return subprocess.list2cmdline([str(item) for item in command])
    except Exception:
        return " ".join(str(item) for item in command)


def _training_runner_command(
    *,
    runner_entrypoint: str,
    run_stage: str,
    device: str,
    output_root: str,
    extra_args: list[str],
) -> list[str]:
    command = [
        sys.executable,
        runner_entrypoint,
        "--run-stage",
        run_stage,
        "--device",
        device,
        "--output-root",
        output_root,
    ]
    if extra_args:
        command.extend(["--", *extra_args])
    return command


def _reward_runner_command(
    *,
    reward_ids: str,
    run_stage: str,
    device: str,
    output_root: str,
    extra_args: list[str],
) -> list[str]:
    command = [
        sys.executable,
        "experiments/final_method/run_a_new_reward_ablation_batch.py",
        "--reward-ablation-ids",
        reward_ids,
        "--run-stage",
        run_stage,
        "--device",
        device,
        "--output-root",
        output_root,
    ]
    if extra_args:
        command.extend(["--", *extra_args])
    return command


def _classical_b_command(*, run_stage: str, output_root: str) -> list[str]:
    return [
        sys.executable,
        "experiments/final_method/run_a_new_classical_frontier_baseline.py",
        "--run-stage",
        run_stage,
        "--device",
        "cpu",
        "--output-root",
        output_root,
    ]


def build_batch_commands(
    *,
    run_stage: str,
    device: str,
    output_root: str,
    include_b: bool,
    include_all_reward_ablations: bool,
    extra_args: list[str],
) -> list[BatchCommandSpec]:
    reward_ids = "R1,R2,R3,R4,R5" if include_all_reward_ablations else "R5"
    reward_method_ids = (
        ("Anew_R1", "Anew_R2", "Anew_R3", "Anew_R4", "Anew_R5")
        if include_all_reward_ablations
        else ("Anew_R5",)
    )
    commands = [
        BatchCommandSpec(
            label="C",
            runner_entrypoint="experiments/final_method/run_a_new_local_state_ddqn_baseline.py",
            launcher_script="scripts/run_a_new_local_state_ddqn_baseline.ps1",
            method_ids=("Anew_C_local_state_ddqn",),
            command=tuple(
                _training_runner_command(
                    runner_entrypoint="experiments/final_method/run_a_new_local_state_ddqn_baseline.py",
                    run_stage=run_stage,
                    device=device,
                    output_root=output_root,
                    extra_args=extra_args,
                )
            ),
        ),
        BatchCommandSpec(
            label="D",
            runner_entrypoint="experiments/final_method/run_a_new_no_value_tree_ablation.py",
            launcher_script="scripts/run_a_new_no_value_tree_ablation.ps1",
            method_ids=("Anew_D_no_value_tree",),
            command=tuple(
                _training_runner_command(
                    runner_entrypoint="experiments/final_method/run_a_new_no_value_tree_ablation.py",
                    run_stage=run_stage,
                    device=device,
                    output_root=output_root,
                    extra_args=extra_args,
                )
            ),
        ),
        BatchCommandSpec(
            label="E",
            runner_entrypoint="experiments/final_method/run_a_new_no_dual_state_split_ablation.py",
            launcher_script="scripts/run_a_new_no_dual_state_split_ablation.ps1",
            method_ids=("Anew_E_no_dual_state_split",),
            command=tuple(
                _training_runner_command(
                    runner_entrypoint="experiments/final_method/run_a_new_no_dual_state_split_ablation.py",
                    run_stage=run_stage,
                    device=device,
                    output_root=output_root,
                    extra_args=extra_args,
                )
            ),
        ),
        BatchCommandSpec(
            label="F_key",
            runner_entrypoint="experiments/final_method/run_a_new_no_behavior_memory_ablation.py",
            launcher_script="scripts/run_a_new_no_behavior_memory_ablation.ps1",
            method_ids=("Anew_F3_no_behavior_memory",),
            command=tuple(
                _training_runner_command(
                    runner_entrypoint="experiments/final_method/run_a_new_no_behavior_memory_ablation.py",
                    run_stage=run_stage,
                    device=device,
                    output_root=output_root,
                    extra_args=extra_args,
                )
            ),
        ),
        BatchCommandSpec(
            label="R_key" if not include_all_reward_ablations else "R_full",
            runner_entrypoint="experiments/final_method/run_a_new_reward_ablation_batch.py",
            launcher_script="scripts/run_a_new_reward_ablations.ps1",
            method_ids=reward_method_ids,
            command=tuple(
                _reward_runner_command(
                    reward_ids=reward_ids,
                    run_stage=run_stage,
                    device=device,
                    output_root=output_root,
                    extra_args=extra_args,
                )
            ),
            notes=(
                "Default R_key is Anew_R5/no_efficiency_penalties.",
                "Use include_all_reward_ablations to run Anew_R1 through Anew_R5.",
            ),
        ),
    ]
    if include_b:
        commands.append(
            BatchCommandSpec(
                label="B",
                runner_entrypoint="experiments/final_method/run_a_new_classical_frontier_baseline.py",
                launcher_script="scripts/run_a_new_classical_frontier_baseline.ps1",
                method_ids=("Anew_B_classical_frontier_greedy",),
                command=tuple(_classical_b_command(run_stage=run_stage, output_root=output_root)),
                notes=("B is a CPU non-learning benchmark; batch forces device=cpu for B.",),
            )
        )
    return commands


def included_methods(commands: list[BatchCommandSpec]) -> list[str]:
    return [method_id for command in commands for method_id in command.method_ids]


def excluded_methods(*, include_b: bool, include_all_reward_ablations: bool) -> list[str]:
    excluded = [
        "A_new",
        "Anew_F1_no_visit_count",
        "Anew_F2_no_recent_trajectory",
    ]
    if not include_b:
        excluded.append("Anew_B_classical_frontier_greedy")
    if not include_all_reward_ablations:
        excluded.extend(["Anew_R1", "Anew_R2", "Anew_R3", "Anew_R4"])
    return excluded


def build_batch_plan(
    *,
    run_stage: str,
    device: str,
    output_root: str,
    include_b: bool,
    include_all_reward_ablations: bool,
    continue_on_failure: bool,
    extra_args: list[str],
) -> dict[str, object]:
    commands = build_batch_commands(
        run_stage=run_stage,
        device=device,
        output_root=output_root,
        include_b=include_b,
        include_all_reward_ablations=include_all_reward_ablations,
        extra_args=extra_args,
    )
    return {
        "batch_id": "A_new_minimum_closure_batch",
        "schema_version": "a_new_minimum_closure_batch_plan/v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_set": RUN_SET_MINIMUM_CLOSURE,
        "run_stage": run_stage,
        "device": device,
        "output_root": output_root,
        "included_methods": included_methods(commands),
        "excluded_methods": excluded_methods(
            include_b=include_b,
            include_all_reward_ablations=include_all_reward_ablations,
        ),
        "uses_current_a_new_defaults": True,
        "parameters_hardcoded_in_batch": False,
        "a_new_parameter_status": "frozen_v1_last_pt_default",
        "default_includes_a_new": False,
        "default_includes_b": False,
        "include_b": bool(include_b),
        "include_all_reward_ablations": bool(include_all_reward_ablations),
        "r_key_method_id": "Anew_R5",
        "r_key_name": "no_efficiency_penalties",
        "continue_on_failure": bool(continue_on_failure),
        "extra_args": list(extra_args),
        "extra_args_policy": (
            "Extra args are appended as runner passthrough for C/D/E/F_key/R training-side runners. "
            "B is non-learning and does not receive training passthrough args."
        ),
        "commands": [
            {
                "label": command.label,
                "method_ids": list(command.method_ids),
                "runner_entrypoint": command.runner_entrypoint,
                "launcher_script": command.launcher_script,
                "command": list(command.command),
                "command_text": _command_text(command.command),
                "notes": list(command.notes),
            }
            for command in commands
        ],
        "notes": [
            "This batch launcher follows the current A_new default training configuration at execution time.",
            "A_new training parameters are frozen to the AN_tuned_v1 last.pt-oriented formal training contract.",
            "Final-probe evaluation is intentionally deferred to a later unified last.pt evaluation pass.",
            "The batch does not run A_new by default because A_new is tuned separately.",
            "The batch does not run B by default; use include_b for the optional CPU non-learning benchmark.",
            "Successful C/D/E/F_key runs archive logs and last.pt under the A_new_minimum_closure records/checkpoint roots.",
            "R_key keeps the existing A_new_reward_ablations records/checkpoint archive roots.",
            "Smoke and pilot stages are local checks only, not paper Results.",
            "Formal train-side-only outputs require artifact audit before they are imported into paper_work.",
            "Do not commit outputs, checkpoint stores, checkpoints, or model checkpoint files.",
        ],
    }


def _print_plan(plan: dict[str, object]) -> None:
    print(json.dumps(plan, indent=2, ensure_ascii=False), flush=True)


def run_batch(plan: dict[str, object]) -> int:
    failures: list[dict[str, object]] = []
    commands = plan.get("commands", [])
    if not isinstance(commands, list):
        raise TypeError("batch plan commands must be a list")
    for index, item in enumerate(commands, start=1):
        if not isinstance(item, dict):
            raise TypeError("batch command item must be a JSON object")
        command = item.get("command")
        if not isinstance(command, list) or not all(isinstance(part, str) for part in command):
            raise TypeError("batch command must be a list of strings")
        label = str(item.get("label", f"command_{index}"))
        method_ids = item.get("method_ids", [])
        print(f"[A_new_minimum_closure] start {index}/{len(commands)} {label}: {method_ids}", flush=True)
        print(f"[A_new_minimum_closure] command: {_command_text(command)}", flush=True)
        result = subprocess.run(command, cwd=str(REPO_ROOT), text=True)
        print(
            f"[A_new_minimum_closure] end {index}/{len(commands)} {label}: exit_code={result.returncode}",
            flush=True,
        )
        if result.returncode != 0:
            failure = {
                "label": label,
                "method_ids": method_ids,
                "exit_code": int(result.returncode),
                "command": command,
            }
            failures.append(failure)
            if not bool(plan.get("continue_on_failure", False)):
                print("[A_new_minimum_closure] stopping after first failure", flush=True)
                print(json.dumps({"failures": failures}, indent=2, ensure_ascii=False), flush=True)
                return int(result.returncode) if int(result.returncode) != 0 else 1
    if failures:
        print("[A_new_minimum_closure] completed with failures", flush=True)
        print(json.dumps({"failures": failures}, indent=2, ensure_ascii=False), flush=True)
        return 1
    print("[A_new_minimum_closure] completed successfully", flush=True)
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the A_new minimum-closure train-side batch.")
    parser.add_argument("--run-stage", choices=RUN_STAGES, default="formal")
    parser.add_argument("--device", choices=DEVICES, default="cuda")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--include-b", action="store_true")
    parser.add_argument("--include-all-reward-ablations", action="store_true")
    parser.add_argument("--run-set", choices=(RUN_SET_MINIMUM_CLOSURE,), default=RUN_SET_MINIMUM_CLOSURE)
    parser.add_argument("--output-root", type=str, default="outputs")
    parser.add_argument("--continue-on-failure", action="store_true")
    parser.add_argument("--stop-on-failure", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--extra-args", nargs=argparse.REMAINDER, default=[])
    args, passthrough = parser.parse_known_args(argv)
    extra_args = _normalize_extra_args(list(args.extra_args or []))
    passthrough = _normalize_extra_args(list(passthrough or []))
    if passthrough:
        extra_args.extend(passthrough)
    args.extra_args = extra_args
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    continue_on_failure = bool(args.continue_on_failure) or not bool(args.stop_on_failure)
    plan = build_batch_plan(
        run_stage=str(args.run_stage),
        device=str(args.device),
        output_root=str(args.output_root),
        include_b=bool(args.include_b),
        include_all_reward_ablations=bool(args.include_all_reward_ablations),
        continue_on_failure=continue_on_failure,
        extra_args=list(args.extra_args),
    )
    if bool(args.dry_run):
        _print_plan(plan)
        return 0
    _print_plan(plan)
    return run_batch(plan)


if __name__ == "__main__":
    raise SystemExit(main())
