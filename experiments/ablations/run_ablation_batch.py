from __future__ import annotations

import argparse
import json
import shlex
import shutil
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.ablations.ablation_specs import AblationSpec, get_ablation_spec
from experiments.ablations.batch_presets import get_batch_preset, list_batch_presets
from train_q_agent import TrainConfig


DEFAULT_BASE_CONFIG = Path("experiment_records/full_method_main/logs/config_snapshot.json")
DEFAULT_RECORDS_ROOT = Path("experiment_records/ablations")

COPIED_LOG_FILES = (
    "final_probe.csv",
    "final_probe_summary.json",
    "metric_snapshot.json",
    "config_snapshot.json",
    "reproducibility_contract.json",
    "posthoc_selection_summary.json",
    "formal_selection_manifest.json",
    "artifact_index.json",
    "ablation_manifest.json",
    "training_summary.txt",
)

FINAL_REQUIRED_FILES = (
    "final_probe.csv",
    "final_probe_summary.json",
    "metric_snapshot.json",
    "config_snapshot.json",
    "reproducibility_contract.json",
    "posthoc_selection_summary.json",
    "formal_selection_manifest.json",
    "artifact_index.json",
    "ablation_manifest.json",
)

PARAMETER_FIELDS: tuple[tuple[str, str], ...] = (
    ("rows", "--rows"),
    ("cols", "--cols"),
    ("obs_size", "--obs-size"),
    ("scan_radius", "--scan-radius"),
    ("obstacle_ratio", "--obstacle-ratio"),
    ("max_accessible_blocks", "--max-accessible-blocks"),
    ("max_entries_per_block", "--max-entries-per-block"),
    ("max_episode_steps", "--max-episode-steps"),
    ("coverage_stop_threshold", "--coverage-stop-threshold"),
    ("budget_mode", "--budget-mode"),
    ("total_env_steps", "--total-env-steps"),
    ("total_train_episodes", "--total-train-episodes"),
    ("warmup_steps", "--warmup-steps"),
    ("warmup_episodes", "--warmup-episodes"),
    ("collect_steps_per_iter", "--collect-steps-per-iter"),
    ("learner_updates_per_iter", "--learner-updates-per-iter"),
    ("train_every_env_steps", "--train-every-env-steps"),
    ("batch_size", "--batch-size"),
    ("min_replay_size", "--min-replay-size"),
    ("replay_capacity", "--replay-capacity"),
    ("gamma", "--gamma"),
    ("n_step", "--n-step"),
    ("learning_rate", "--learning-rate"),
    ("weight_decay", "--weight-decay"),
    ("grad_clip_norm", "--grad-clip-norm"),
    ("target_update_interval", "--target-update-interval"),
    ("epsilon_start", "--epsilon-start"),
    ("epsilon_end", "--epsilon-end"),
    ("epsilon_decay_steps", "--epsilon-decay-steps"),
    ("formal_protocol", "--formal-protocol"),
    ("final_greedy_episodes", "--final-greedy-episodes"),
    ("fixed_train_episode_seed_base", "--fixed-train-episode-seed-base"),
    ("fixed_final_probe_seed_base", "--fixed-final-probe-seed-base"),
    ("periodic_checkpoint_interval_env_steps", "--periodic-checkpoint-interval-env-steps"),
    ("posthoc_candidate_start_env_steps", "--posthoc-candidate-start-env-steps"),
    ("posthoc_candidate_end_env_steps", "--posthoc-candidate-end-env-steps"),
    ("posthoc_selection_window_env_steps", "--posthoc-selection-window-env-steps"),
    ("posthoc_final_probe_topk", "--posthoc-final-probe-topk"),
    ("fixed_model_select_seed_base", "--fixed-model-select-seed-base"),
    ("reward_info_scale", "--reward-info-scale"),
    ("reward_obstacle_weight", "--reward-obstacle-weight"),
    ("reward_step_penalty", "--reward-step-penalty"),
    ("reward_terminal_bonus", "--reward-terminal-bonus"),
    ("reward_revisit_penalty", "--reward-revisit-penalty"),
    ("reward_turn_penalty_scale", "--reward-turn-penalty-scale"),
    ("reward_turn_weight_45", "--reward-turn-weight-45"),
    ("reward_turn_weight_90", "--reward-turn-weight-90"),
    ("reward_turn_weight_135", "--reward-turn-weight-135"),
    ("reward_turn_weight_180", "--reward-turn-weight-180"),
    ("reward_timeout_penalty", "--reward-timeout-penalty"),
)

BOOLEAN_FIELDS: tuple[tuple[str, str], ...] = (
    ("use_fixed_train_episode_seeds", "--use-fixed-train-episode-seeds"),
    ("use_fixed_eval_seeds", "--use-fixed-eval-seeds"),
    ("use_fixed_model_select_seeds", "--use-fixed-model-select-seeds"),
    ("train_side_only_tuning", "--train-side-only-tuning"),
)


def _load_base_train_config(base_config_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    if not base_config_path.exists():
        raise FileNotFoundError(
            f"Base config not found: {base_config_path}. "
            "Expected experiment_records/full_method_main/logs/config_snapshot.json."
        )
    try:
        payload = json.loads(base_config_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"Failed to parse base config JSON at {base_config_path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Base config root must be a JSON object: {base_config_path}")
    train_config = payload.get("full_train_config")
    if not isinstance(train_config, dict):
        raise ValueError(f"Base config is missing object field full_train_config: {base_config_path}")
    return payload, train_config


def _normalize_ablation_ids(raw: str | None, preset: str) -> list[AblationSpec]:
    if raw:
        ids = [item.strip() for item in raw.split(",") if item.strip()]
    else:
        ids = get_batch_preset(preset)
    if not ids:
        raise ValueError("No ablation IDs were selected.")
    return [get_ablation_spec(ablation_id) for ablation_id in ids]


def _append_value_arg(args: list[str], option_name: str, value: Any) -> None:
    if value is None:
        raise ValueError(f"Base config field for {option_name} is null or missing.")
    args.extend([option_name, str(value)])


def _append_boolean_arg(args: list[str], option_name: str, value: Any) -> None:
    if bool(value):
        args.append(option_name)
    else:
        args.append(f"--no-{option_name[2:]}")


def _build_aligned_train_args(
    train_config: Mapping[str, Any],
    *,
    spec: AblationSpec,
    run_stage: str,
    device: str,
    output_root: str,
    extra_train_args: list[str],
) -> list[str]:
    args: list[str] = []
    for field_name, option_name in PARAMETER_FIELDS:
        _append_value_arg(args, option_name, train_config.get(field_name))
    for field_name, option_name in BOOLEAN_FIELDS:
        _append_boolean_arg(args, option_name, train_config.get(field_name))

    args.extend(extra_train_args)
    args.extend(["--device", device])
    args.extend(["--output-root", output_root])
    args.extend(["--run-name", f"ablation_{spec.ablation_id}_{run_stage}"])
    return args


def _command_for_ablation(spec: AblationSpec, run_stage: str, train_args: list[str]) -> list[str]:
    return [
        sys.executable,
        "experiments/ablations/run_ablation_train.py",
        "--ablation-id",
        spec.short_id,
        "--run-stage",
        run_stage,
        "--",
        *train_args,
    ]


def _records_logs_dir(records_root: Path, spec: AblationSpec) -> Path:
    return records_root / f"ablation_{spec.ablation_id}" / "logs"


def _has_existing_curated_logs(logs_dir: Path) -> bool:
    if not logs_dir.exists():
        return False
    return any(path.is_file() and path.name != ".gitkeep" for path in logs_dir.iterdir())


def _resolve_run_dir_from_output(output_lines: list[str], cwd: Path) -> Path:
    run_dir: Path | None = None
    manifest_path: Path | None = None
    for line in output_lines:
        stripped = line.strip()
        if stripped.startswith("run_dir:"):
            run_dir = Path(stripped.split(":", 1)[1].strip())
        elif stripped.startswith("ablation_manifest_json:"):
            manifest_path = Path(stripped.split(":", 1)[1].strip())
    if manifest_path is not None:
        run_dir = manifest_path.parent.parent
    if run_dir is None:
        raise RuntimeError("Could not determine run_dir from ablation child process output.")
    if not run_dir.is_absolute():
        run_dir = cwd / run_dir
    return run_dir.resolve()


def _run_child_command(command: list[str], cwd: Path) -> Path:
    print(f"[batch] running: {' '.join(command)}")
    process = subprocess.Popen(
        command,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    output_lines: list[str] = []
    assert process.stdout is not None
    for line in process.stdout:
        output_lines.append(line)
        print(line, end="")
    return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"Ablation command failed with exit code {return_code}: {' '.join(command)}")
    return _resolve_run_dir_from_output(output_lines, cwd)


def _read_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _copy_curated_logs(run_dir: Path, records_root: Path, spec: AblationSpec) -> tuple[Path, list[str], list[str]]:
    source_logs = run_dir / "logs"
    target_logs = _records_logs_dir(records_root, spec)
    target_logs.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    missing: list[str] = []
    for file_name in COPIED_LOG_FILES:
        source_path = source_logs / file_name
        if source_path.exists() and source_path.is_file():
            shutil.copy2(source_path, target_logs / file_name)
            copied.append(file_name)
        else:
            missing.append(file_name)
    return target_logs, copied, missing


def _eligibility_verdict(run_stage: str, train_side_only_tuning: bool, missing_artifacts: list[str]) -> str:
    if run_stage != "formal":
        return "unable_to_judge"
    if train_side_only_tuning:
        return "unable_to_judge_for_final_results"
    missing_required = [name for name in FINAL_REQUIRED_FILES if name in missing_artifacts]
    return "fail" if missing_required else "pass"


def _write_run_record(
    *,
    target_logs: Path,
    spec: AblationSpec,
    run_stage: str,
    run_dir: Path,
    base_config_path: Path,
    copied: list[str],
    missing: list[str],
) -> Path:
    config_snapshot = _read_json_if_exists(target_logs / "config_snapshot.json") or {}
    train_config = config_snapshot.get("full_train_config", {})
    if not isinstance(train_config, dict):
        train_config = {}
    train_side_only = bool(train_config.get("train_side_only_tuning"))
    source_commit = config_snapshot.get("git_commit_sha") or "unknown"
    verdict = _eligibility_verdict(run_stage, train_side_only, missing)
    record = [
        "# Run Record",
        "",
        f"- ablation_id: {spec.ablation_id}",
        f"- run_stage: {run_stage}",
        f"- source run_dir: {run_dir}",
        f"- source commit: {source_commit}",
        f"- base_config path: {base_config_path}",
        f"- copied artifact list: {', '.join(copied) if copied else 'none'}",
        f"- missing artifact list: {', '.join(missing) if missing else 'none'}",
        f"- train_side_only_tuning: {str(train_side_only).lower()}",
        f"- eligibility verdict: {verdict}",
        "",
        "## Notes",
        "",
        "- Curated logs only; checkpoints and raw outputs are intentionally not copied.",
    ]
    record_path = target_logs.parent / "run_record.md"
    record_path.write_text("\n".join(record) + "\n", encoding="utf-8")
    return record_path


def _parse_extra_train_args(raw: str | None) -> list[str]:
    if raw is None or str(raw).strip() == "":
        return []
    return shlex.split(raw)


def _load_base_config_for_mode(base_config_path: Path, *, dry_run: bool) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        return _load_base_train_config(base_config_path)
    except FileNotFoundError as exc:
        if not dry_run:
            raise
        print(
            f"[batch:dry-run] warning: {exc} "
            "Using TrainConfig defaults for command preview only; non-dry-run still requires the base config."
        )
        return {}, asdict(TrainConfig())


def _print_dry_run(
    *,
    specs: list[AblationSpec],
    run_stage: str,
    train_config: Mapping[str, Any],
    args: argparse.Namespace,
    extra_train_args: list[str],
) -> None:
    print(f"[batch:dry-run] preset={args.preset}")
    print(f"[batch:dry-run] ablation_ids={', '.join(spec.short_id for spec in specs)}")
    print(f"[batch:dry-run] base_config={args.base_config}")
    print(f"[batch:dry-run] train_side_only_tuning={bool(train_config.get('train_side_only_tuning'))}")
    for spec in specs:
        train_args = _build_aligned_train_args(
            train_config,
            spec=spec,
            run_stage=run_stage,
            device=args.device,
            output_root=args.output_root,
            extra_train_args=extra_train_args,
        )
        command = _command_for_ablation(spec, run_stage, train_args)
        print(f"[batch:dry-run] {spec.short_id}/{spec.ablation_id}")
        print(f"  command: {' '.join(command)}")
        print(f"  records: {_records_logs_dir(Path(args.records_root), spec)}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Batch runner for F/R ablation experiments")
    parser.add_argument("--preset", type=str, default="recommended_first_batch")
    parser.add_argument("--ablation-ids", type=str, default=None)
    parser.add_argument("--base-config", type=Path, default=DEFAULT_BASE_CONFIG)
    parser.add_argument("--run-stage", choices=("smoke", "pilot", "formal"), default="formal")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output-root", type=str, default="outputs")
    parser.add_argument("--records-root", type=Path, default=DEFAULT_RECORDS_ROOT)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--stop-on-failure", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skip-existing-records", action="store_true", default=False)
    parser.add_argument("--extra-train-args", type=str, default=None)
    parser.add_argument("--list-presets", action="store_true")
    args = parser.parse_args(argv)

    if args.list_presets:
        for name, ablation_ids in list_batch_presets().items():
            print(f"{name}: {', '.join(ablation_ids)}")
        return 0

    base_payload, train_config = _load_base_config_for_mode(args.base_config, dry_run=bool(args.dry_run))
    _ = base_payload
    specs = _normalize_ablation_ids(args.ablation_ids, args.preset)
    extra_train_args = _parse_extra_train_args(args.extra_train_args)

    if args.dry_run:
        _print_dry_run(
            specs=specs,
            run_stage=args.run_stage,
            train_config=train_config,
            args=args,
            extra_train_args=extra_train_args,
        )
        return 0

    repo_root = Path.cwd()
    failures: list[str] = []
    for spec in specs:
        target_logs = _records_logs_dir(args.records_root, spec)
        if args.skip_existing_records and _has_existing_curated_logs(target_logs):
            print(f"[batch] skipping {spec.short_id}/{spec.ablation_id}: curated logs already exist at {target_logs}")
            continue

        train_args = _build_aligned_train_args(
            train_config,
            spec=spec,
            run_stage=args.run_stage,
            device=args.device,
            output_root=args.output_root,
            extra_train_args=extra_train_args,
        )
        command = _command_for_ablation(spec, args.run_stage, train_args)
        try:
            run_dir = _run_child_command(command, repo_root)
            target_logs, copied, missing = _copy_curated_logs(run_dir, args.records_root, spec)
            record_path = _write_run_record(
                target_logs=target_logs,
                spec=spec,
                run_stage=args.run_stage,
                run_dir=run_dir,
                base_config_path=args.base_config,
                copied=copied,
                missing=missing,
            )
            print(f"[batch] archived {spec.short_id}/{spec.ablation_id} to {target_logs}")
            print(f"[batch] run_record: {record_path}")
        except Exception as exc:
            message = f"{spec.short_id}/{spec.ablation_id}: {exc}"
            failures.append(message)
            print(f"[batch] failure: {message}")
            if bool(args.stop_on_failure):
                raise

    if failures:
        print("[batch] completed with failures:")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
