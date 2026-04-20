from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.tuning.decision_rules import ComparisonOutcome, compare_candidate_to_reference
from scripts.tuning.recipes import TrialPlan, get_recipe
from scripts.tuning.result_reader import COMPLETE_STATUSES, RunResult, find_latest_run_dir, read_run_result
from scripts.tuning.scheduler_core import SessionLogger, TrialExecution, build_train_command, execute_trial


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local sequential experiment scheduler.")
    parser.add_argument("--python-executable", default="python")
    parser.add_argument("--train-script", default="train_q_agent.py")
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--baseline-run-dir")
    parser.add_argument("--baseline-run-prefix")
    parser.add_argument("--entry-cap", type=int, default=8)
    parser.add_argument("--total-env-steps", type=int, default=500000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--generate-plots-on-finish",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--session-name")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--recipe", default="turn_revisit_tree_v1")
    args = parser.parse_args()

    if bool(args.baseline_run_dir) == bool(args.baseline_run_prefix):
        parser.error("Provide exactly one of --baseline-run-dir or --baseline-run-prefix.")
    return args


def _resolve_repo_path(path_text: str, *, base_dir: Path) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def _resolve_baseline_run_dir(args: argparse.Namespace, output_root: Path) -> tuple[Path, str]:
    if args.baseline_run_dir:
        baseline_dir = _resolve_repo_path(args.baseline_run_dir, base_dir=REPO_ROOT)
        if not baseline_dir.exists():
            raise FileNotFoundError(f"Baseline run directory does not exist: {baseline_dir}")
        return baseline_dir, "baseline_run_dir"
    baseline_dir = find_latest_run_dir(output_root, args.baseline_run_prefix)
    return baseline_dir, "baseline_run_prefix"


def _json_dump(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _format_value(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4f}".rstrip("0").rstrip(".")
    return str(value)


def _metrics_lines(title: str, metrics: dict[str, Any], keys: list[str]) -> list[str]:
    lines = [f"### {title}", ""]
    for key in keys:
        lines.append(f"- {key}: {_format_value(metrics.get(key))}")
    lines.append("")
    return lines


def _trial_markdown(record: dict[str, Any]) -> str:
    result = record.get("result") or {}
    reference = record.get("comparison_reference") or {}
    lines = [
        f"# {record['trial_id']}",
        "",
        f"- Status: {record['status']}",
        f"- Reason: {record['status_reason']}",
        f"- Branch position: {record.get('branch_position') or 'n/a'}",
        f"- Decision note: {record.get('decision_note') or 'n/a'}",
        f"- Comparison reference: {reference.get('label') or 'n/a'}",
        f"- Run dir: {record.get('run_dir') or 'n/a'}",
        f"- Return code: {_format_value(record.get('return_code'))}",
        f"- Params: turn={record['params']['reward_turn_penalty_scale']}, revisit={record['params']['reward_revisit_penalty']}, entry={record['params']['max_entries_per_block']}",
        "",
    ]

    if record.get("comparison"):
        comparison = record["comparison"]
        lines.extend(
            [
                "## Baseline Comparison",
                "",
                f"- Verdict: {comparison['verdict']}",
            ]
        )
        for reason in comparison.get("reasons", []):
            lines.append(f"- Reason: {reason}")
        lines.append("")

    if result:
        lines.extend(
            _metrics_lines(
                "Final Probe",
                result.get("final_probe") or {},
                [
                    "eval_mean_reward",
                    "eval_mean_coverage",
                    "eval_success_rate",
                    "eval_mean_episode_length",
                    "eval_mean_repeat_visit_ratio",
                    "eval_mean_timeout_flag",
                    "eval_mean_turn_ge_90_count",
                    "eval_mean_turn_180_count",
                    "eval_mean_weighted_info_gain_sum",
                ],
            )
        )
        lines.extend(
            _metrics_lines(
                "Best Eval",
                result.get("best_eval") or {},
                [
                    "env_steps",
                    "learner_steps",
                    "eval_mean_reward",
                    "eval_mean_coverage",
                    "eval_success_rate",
                    "eval_mean_episode_length",
                    "eval_mean_repeat_visit_ratio",
                    "eval_mean_timeout_flag",
                ],
            )
        )
        lines.extend(
            _metrics_lines(
                "Last Eval",
                result.get("last_eval") or {},
                [
                    "env_steps",
                    "learner_steps",
                    "eval_mean_reward",
                    "eval_mean_coverage",
                    "eval_success_rate",
                    "eval_mean_episode_length",
                    "eval_mean_repeat_visit_ratio",
                    "eval_mean_timeout_flag",
                ],
            )
        )
        lines.extend(
            _metrics_lines(
                "Train Recent",
                result.get("train_recent") or {},
                [
                    "env_steps",
                    "learner_steps",
                    "recent_mean_reward",
                    "recent_mean_coverage",
                    "recent_success_rate",
                    "recent_mean_episode_length",
                    "recent_mean_repeat_visit_ratio",
                ],
            )
        )

    return "\n".join(lines).strip() + "\n"


def _session_markdown(summary: dict[str, Any]) -> str:
    baseline = summary["baseline"]
    lines = [
        "# Scheduler Session",
        "",
        f"- Session: {summary['session_name']}",
        f"- Recipe: {summary['recipe']}",
        f"- Dry run: {summary['dry_run']}",
        f"- Baseline source: {baseline['source_type']}",
        f"- Baseline run dir: {baseline['run_dir']}",
        f"- Final recommendation: {summary['final_recommendation']['source']}",
        f"- Recommendation reason: {summary['final_recommendation']['reason']}",
        f"- Recommendation basis: {summary['final_recommendation'].get('recommendation_basis', 'n/a')}",
        "",
        "## Baseline Final Probe",
        "",
    ]
    baseline_probe = (baseline.get("result") or {}).get("final_probe") or {}
    for key in [
        "eval_mean_reward",
        "eval_mean_coverage",
        "eval_success_rate",
        "eval_mean_episode_length",
        "eval_mean_repeat_visit_ratio",
        "eval_mean_timeout_flag",
        "eval_mean_turn_ge_90_count",
        "eval_mean_turn_180_count",
        "eval_mean_weighted_info_gain_sum",
    ]:
        lines.append(f"- {key}: {_format_value(baseline_probe.get(key))}")
    lines.append("")

    if summary.get("planned_commands"):
        lines.extend(["## Planned Commands", ""])
        for label, plan in summary["planned_commands"].items():
            lines.append(
                f"- {label}: {plan['command']} | compare_to={plan['compare_to']} | branch={plan['branch_position']}"
            )
        lines.append("")

    if summary.get("trials"):
        lines.extend(["## Trials", ""])
        for record in summary["trials"]:
            reference_label = ((record.get("comparison_reference") or {}).get("label")) or "n/a"
            lines.append(
                f"- {record['trial_id']}: {record['status']} / "
                f"{(record.get('comparison') or {}).get('verdict', 'n/a')} / "
                f"reference={reference_label} / branch={record.get('branch_position') or 'n/a'}"
            )
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def _build_trial_record(
    execution: TrialExecution,
    trial_plan: TrialPlan,
    reference_info: dict[str, Any],
    comparison: ComparisonOutcome | None,
    entry_cap: int,
) -> dict[str, Any]:
    return {
        "trial_id": execution.trial_id,
        "note": execution.trial_spec.note,
        "branch_position": trial_plan.branch_position,
        "decision_note": trial_plan.decision_note,
        "params": {
            "reward_turn_penalty_scale": execution.trial_spec.turn_penalty_scale,
            "reward_revisit_penalty": execution.trial_spec.revisit_penalty,
            "max_entries_per_block": entry_cap,
            "run_name": execution.trial_spec.run_name(entry_cap),
        },
        "comparison_reference": reference_info,
        "command": execution.command,
        "started_at": execution.started_at,
        "ended_at": execution.ended_at,
        "process_log_path": str(execution.process_log_path),
        "return_code": execution.return_code,
        "run_dir": str(execution.run_dir) if execution.run_dir else None,
        "status": execution.status,
        "status_reason": execution.status_reason,
        "result": execution.result.to_dict() if execution.result else None,
        "comparison": comparison.to_dict() if comparison else None,
    }


def _write_trial_summary(session_dir: Path, record: dict[str, Any]) -> None:
    json_path = session_dir / f"{record['trial_id']}_summary.json"
    md_path = session_dir / f"{record['trial_id']}_summary.md"
    _json_dump(json_path, record)
    md_path.write_text(_trial_markdown(record), encoding="utf-8")


def _planned_commands(
    *,
    args: argparse.Namespace,
    output_root: Path,
    recipe: Any,
) -> dict[str, dict[str, str]]:
    commands: dict[str, dict[str, str]] = {}
    for label, plan in recipe.possible_trial_plans(args.entry_cap).items():
        command = build_train_command(
            python_executable=args.python_executable,
            train_script=args.train_script,
            output_root=output_root,
            device=args.device,
            seed=args.seed,
            total_env_steps=args.total_env_steps,
            entry_cap=args.entry_cap,
            generate_plots_on_finish=args.generate_plots_on_finish,
            trial_spec=plan.trial_spec,
            run_name=plan.trial_spec.run_name(args.entry_cap),
        )
        commands[label] = {
            "command": subprocess.list2cmdline(command),
            "compare_to": plan.compare_to,
            "branch_position": plan.branch_position,
            "decision_note": plan.decision_note,
        }
    return commands


def _default_recommendation(baseline_reference: dict[str, Any], reason: str) -> dict[str, Any]:
    return {
        "source": "baseline",
        "parameters": baseline_reference,
        "comparison_reference": "baseline",
        "reason": reason,
        "verdict": "baseline",
        "recommendation_basis": "decision_tree_internal_comparisons",
    }


def _resolve_reference_context(
    reference_label: str,
    *,
    baseline_result: RunResult,
    baseline_reference: dict[str, Any],
    baseline_run_dir: Path,
    trial_records: list[dict[str, Any]],
) -> tuple[Any, dict[str, Any]]:
    if reference_label == "baseline":
        return baseline_result, {
            "label": "baseline",
            "source_type": "baseline",
            "run_dir": str(baseline_run_dir),
            "params": baseline_reference,
        }

    for record in trial_records:
        if record.get("trial_id") != reference_label:
            continue
        if not record.get("result"):
            raise RuntimeError(f"Reference trial {reference_label} has no readable result payload.")
        return record["result"], {
            "label": reference_label,
            "source_type": "trial",
            "run_dir": record.get("run_dir"),
            "params": record.get("params"),
        }

    raise RuntimeError(f"Reference '{reference_label}' is not available in the executed trial history.")


def main() -> int:
    args = parse_args()
    output_root = _resolve_repo_path(args.output_root, base_dir=REPO_ROOT)
    recipe = get_recipe(args.recipe)
    baseline_reference = recipe.baseline_reference(args.entry_cap)
    session_name = args.session_name or f"scheduler_{args.recipe}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    session_dir = output_root / "scheduler_runs" / session_name
    session_dir.mkdir(parents=True, exist_ok=True)
    session_logger = SessionLogger(session_dir / "scheduler.log")

    session_logger.write(f"Session dir: {session_dir}")
    session_logger.write(f"Using recipe: {args.recipe}")

    baseline_run_dir, baseline_source_type = _resolve_baseline_run_dir(args, output_root)
    baseline_result = read_run_result(baseline_run_dir, return_code=None)
    session_logger.write(f"Resolved baseline run dir: {baseline_run_dir}")
    session_logger.write(f"Baseline status: {baseline_result.status}")

    if baseline_result.status not in COMPLETE_STATUSES:
        raise RuntimeError(
            f"Baseline artifacts are incomplete: {baseline_run_dir} ({baseline_result.status_reason})"
        )

    planned_commands = _planned_commands(args=args, output_root=output_root, recipe=recipe)
    for label, command in planned_commands.items():
        session_logger.write(f"Planned {label}: {command}")

    summary: dict[str, Any] = {
        "session_name": session_name,
        "session_dir": str(session_dir),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "recipe": args.recipe,
        "dry_run": args.dry_run,
        "decision_tree": recipe.branch_preview(args.entry_cap),
        "max_new_trials": recipe.max_new_trials(),
        "baseline": {
            "source_type": baseline_source_type,
            "run_dir": str(baseline_run_dir),
            "reference_parameters": baseline_reference,
            "result": baseline_result.to_dict(),
        },
        "planned_commands": planned_commands,
        "trials": [],
        "final_recommendation": _default_recommendation(
            baseline_reference,
            "Dry run or no successful trials yet.",
        ),
    }

    if args.dry_run:
        session_logger.write("Dry-run mode enabled. No training process will be launched.")
        _json_dump(session_dir / "session_summary.json", summary)
        (session_dir / "session_summary.md").write_text(_session_markdown(summary), encoding="utf-8")
        return 0

    trial_records: list[dict[str, Any]] = []
    next_plan = recipe.initial_trial_plan(args.entry_cap)
    failed_trial = False
    while next_plan is not None and len(trial_records) < recipe.max_new_trials():
        trial_index = len(trial_records) + 1
        session_logger.write(
            f"Planned trial_{trial_index:02d}: compare_to={next_plan.compare_to}, "
            f"branch={next_plan.branch_position}"
        )
        reference_summary, reference_info = _resolve_reference_context(
            next_plan.compare_to,
            baseline_result=baseline_result,
            baseline_reference=baseline_reference,
            baseline_run_dir=baseline_run_dir,
            trial_records=trial_records,
        )
        execution = execute_trial(
            repo_root=REPO_ROOT,
            session_logger=session_logger,
            session_dir=session_dir,
            python_executable=args.python_executable,
            train_script=args.train_script,
            output_root=output_root,
            device=args.device,
            seed=args.seed,
            total_env_steps=args.total_env_steps,
            entry_cap=args.entry_cap,
            generate_plots_on_finish=args.generate_plots_on_finish,
            trial_spec=next_plan.trial_spec,
            trial_index=trial_index,
            dry_run=False,
        )

        comparison: ComparisonOutcome | None = None
        if execution.result and execution.status in COMPLETE_STATUSES:
            comparison = compare_candidate_to_reference(execution.result, reference_summary)
            session_logger.write(
                f"{execution.trial_id} comparison verdict vs {reference_info['label']}: {comparison.verdict}"
            )

        record = _build_trial_record(
            execution,
            next_plan,
            reference_info,
            comparison,
            args.entry_cap,
        )
        trial_records.append(record)
        _write_trial_summary(session_dir, record)

        if execution.status not in COMPLETE_STATUSES:
            session_logger.write(
                f"Stopping after {execution.trial_id} because core artifacts were incomplete."
            )
            failed_trial = True
            break

        next_plan = recipe.next_trial_plan(args.entry_cap, trial_records)

    summary["trials"] = trial_records
    if failed_trial:
        last_trial_id = trial_records[-1]["trial_id"] if trial_records else "trial_00"
        summary["final_recommendation"] = _default_recommendation(
            baseline_reference,
            f"{last_trial_id} failed to produce complete core artifacts.",
        )
    else:
        summary["final_recommendation"] = recipe.finalize_recommendation(
            baseline_result=baseline_result,
            trial_records=trial_records,
            entry_cap=args.entry_cap,
        )
    _json_dump(session_dir / "session_summary.json", summary)
    (session_dir / "session_summary.md").write_text(_session_markdown(summary), encoding="utf-8")
    session_logger.write(
        "Final recommendation: "
        f"{summary['final_recommendation']['source']} -> {summary['final_recommendation']['parameters']}"
    )
    if failed_trial:
        session_logger.write("Stopping with non-zero exit because a trial had incomplete core artifacts.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
