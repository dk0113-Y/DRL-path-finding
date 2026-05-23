from __future__ import annotations

import argparse
import json
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from baselines.local_state_ddqn import LOCAL_STATE_BASELINE_ID


DEFAULT_RECORDS_ROOT = Path("experiment_records/baselines")
DEFAULT_CHECKPOINT_STORE_ROOT = Path("checkpoint_store/baselines")

COPIED_LOG_FILES = (
    "final_probe.csv",
    "final_probe_summary.json",
    "metric_snapshot.json",
    "config_snapshot.json",
    "reproducibility_contract.json",
    "posthoc_selection_summary.json",
    "formal_selection_manifest.json",
    "artifact_index.json",
    "baseline_manifest.json",
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
    "baseline_manifest.json",
)


def _has_option(args: list[str], option_name: str) -> bool:
    prefix = f"{option_name}="
    return any(arg == option_name or arg.startswith(prefix) for arg in args)


def _parse_extra_train_args(raw: str | None) -> list[str]:
    if raw is None or str(raw).strip() == "":
        return []
    return shlex.split(raw)


def _validate_baseline_id(baseline_id: str) -> None:
    if baseline_id != LOCAL_STATE_BASELINE_ID:
        raise ValueError(
            f"Only {LOCAL_STATE_BASELINE_ID!r} is supported by this baseline batch runner; "
            f"got {baseline_id!r}."
        )


def _build_train_args(
    *,
    baseline_id: str,
    run_stage: str,
    device: str,
    output_root: str,
    extra_train_args: list[str],
) -> list[str]:
    _validate_baseline_id(baseline_id)
    train_args = list(extra_train_args)
    train_args.extend(["--device", device])
    train_args.extend(["--output-root", output_root])
    if not _has_option(train_args, "--run-name"):
        train_args.extend(["--run-name", f"{LOCAL_STATE_BASELINE_ID}_{run_stage}"])
    return train_args


def _command_for_baseline(*, baseline_id: str, run_stage: str, train_args: list[str]) -> list[str]:
    _validate_baseline_id(baseline_id)
    return [
        sys.executable,
        "experiments/baselines/run_local_state_ddqn_train.py",
        "--baseline-id",
        baseline_id,
        "--run-stage",
        run_stage,
        "--",
        *train_args,
    ]


def _records_logs_dir(records_root: Path, baseline_id: str) -> Path:
    _validate_baseline_id(baseline_id)
    return records_root / LOCAL_STATE_BASELINE_ID / "logs"


def _checkpoint_target_path(checkpoint_store_root: Path, baseline_id: str) -> Path:
    _validate_baseline_id(baseline_id)
    return checkpoint_store_root / f"{LOCAL_STATE_BASELINE_ID}.pt"


def _format_command(command: list[str]) -> str:
    return subprocess.list2cmdline(command)


def _display_path(path: Path) -> str:
    return path.as_posix()


def _resolve_run_dir_from_output(output_lines: list[str], cwd: Path) -> Path:
    manifest_path: Path | None = None
    for line in output_lines:
        stripped = line.strip()
        if stripped.startswith("baseline_manifest_json:"):
            manifest_path = Path(stripped.split(":", 1)[1].strip())
    if manifest_path is None:
        raise RuntimeError("Could not determine run_dir from baseline_manifest_json in child process output.")
    run_dir = manifest_path.parent.parent
    if not run_dir.is_absolute():
        run_dir = cwd / run_dir
    return run_dir.resolve()


def _run_child_command(command: list[str], cwd: Path) -> Path:
    print(f"[baseline-batch] running: {_format_command(command)}")
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
        raise RuntimeError(f"Baseline command failed with exit code {return_code}: {_format_command(command)}")
    return _resolve_run_dir_from_output(output_lines, cwd)


def _copy_curated_logs(run_dir: Path, records_root: Path, baseline_id: str) -> tuple[Path, list[str], list[str]]:
    source_logs = run_dir / "logs"
    target_logs = _records_logs_dir(records_root, baseline_id)
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


def _copy_last_checkpoint(
    run_dir: Path,
    checkpoint_store_root: Path,
    baseline_id: str,
    *,
    copy_checkpoints: bool,
) -> tuple[Path, Path, bool, str | None]:
    source_path = run_dir / "checkpoints" / "last.pt"
    target_path = _checkpoint_target_path(checkpoint_store_root, baseline_id)
    if not copy_checkpoints:
        return source_path, target_path, False, "disabled_by_user"
    if not source_path.exists():
        print(f"[baseline-batch] warning: last checkpoint not found at {source_path}")
        return source_path, target_path, False, "missing_last_checkpoint"
    if target_path.exists():
        print(f"[baseline-batch] warning: overwriting existing checkpoint target {target_path}")
    try:
        checkpoint_store_root.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, target_path)
    except Exception as exc:
        print(f"[baseline-batch] warning: failed to copy last checkpoint to {target_path}: {exc}")
        return source_path, target_path, False, f"copy_failed:{type(exc).__name__}"
    return source_path, target_path, True, None


def _read_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


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
    baseline_id: str,
    run_stage: str,
    run_dir: Path,
    copied: list[str],
    missing: list[str],
    checkpoint_source: Path,
    checkpoint_store_path: Path,
    checkpoint_copied: bool,
    checkpoint_copy_reason: str | None,
) -> Path:
    config_snapshot = _read_json_if_exists(target_logs / "config_snapshot.json") or {}
    train_config = config_snapshot.get("full_train_config", {})
    if not isinstance(train_config, dict):
        train_config = {}
    baseline_manifest = _read_json_if_exists(target_logs / "baseline_manifest.json") or {}
    train_side_only = bool(train_config.get("train_side_only_tuning"))
    source_commit = (
        config_snapshot.get("git_commit_sha")
        or config_snapshot.get("source_commit")
        or baseline_manifest.get("git_sha")
        or "unknown"
    )
    verdict = _eligibility_verdict(run_stage, train_side_only, missing)
    record = [
        "# Run Record",
        "",
        f"- baseline_id: {baseline_id}",
        f"- run_stage: {run_stage}",
        f"- source run_dir: {run_dir}",
        f"- source commit: {source_commit}",
        f"- copied artifact list: {', '.join(copied) if copied else 'none'}",
        f"- missing artifact list: {', '.join(missing) if missing else 'none'}",
        f"- checkpoint_source: {checkpoint_source}",
        f"- checkpoint_store_path: {checkpoint_store_path}",
        f"- checkpoint_copied: {str(checkpoint_copied).lower()}",
        f"- checkpoint_copy_reason: {checkpoint_copy_reason if checkpoint_copy_reason is not None else 'none'}",
        f"- train_side_only_tuning: {str(train_side_only).lower()}",
        f"- eligibility verdict: {verdict}",
        "",
        "## Notes",
        "",
        "- Curated logs are copied into experiment_records/baselines. last.pt may be copied into checkpoint_store/baselines when --copy-checkpoints is enabled. Raw outputs remain under outputs/.",
    ]
    record_path = target_logs.parent / "run_record.md"
    record_path.write_text("\n".join(record) + "\n", encoding="utf-8")
    return record_path


def _print_dry_run(
    *,
    baseline_id: str,
    run_stage: str,
    output_root: str,
    records_root: Path,
    checkpoint_store_root: Path,
    copy_checkpoints: bool,
    command: list[str],
) -> None:
    print(f"[baseline-batch:dry-run] baseline_id={baseline_id}")
    print(f"[baseline-batch:dry-run] run_stage={run_stage}")
    print(f"[baseline-batch:dry-run] output_root={output_root}")
    print(f"[baseline-batch:dry-run] command: {_format_command(command)}")
    print(f"[baseline-batch:dry-run] records={_display_path(_records_logs_dir(records_root, baseline_id))}")
    print("[baseline-batch:dry-run] checkpoint source=outputs/<run_name_timestamp>/checkpoints/last.pt cannot be known before run")
    print(
        "[baseline-batch:dry-run] "
        f"checkpoint target={_display_path(_checkpoint_target_path(checkpoint_store_root, baseline_id))}"
    )
    print(f"[baseline-batch:dry-run] checkpoint copying {'enabled' if copy_checkpoints else 'disabled'}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Batch/archive runner for learning baselines")
    parser.add_argument("--baseline-id", type=str, default=LOCAL_STATE_BASELINE_ID)
    parser.add_argument("--run-stage", choices=("smoke", "pilot", "formal"), default="pilot")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output-root", type=str, default="outputs")
    parser.add_argument("--records-root", type=Path, default=DEFAULT_RECORDS_ROOT)
    parser.add_argument("--checkpoint-store-root", type=Path, default=DEFAULT_CHECKPOINT_STORE_ROOT)
    parser.add_argument("--copy-checkpoints", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--extra-train-args", type=str, default=None)
    parser.add_argument("--stop-on-failure", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args(argv)

    try:
        _validate_baseline_id(args.baseline_id)
        extra_train_args = _parse_extra_train_args(args.extra_train_args)
        train_args = _build_train_args(
            baseline_id=args.baseline_id,
            run_stage=args.run_stage,
            device=args.device,
            output_root=args.output_root,
            extra_train_args=extra_train_args,
        )
        command = _command_for_baseline(
            baseline_id=args.baseline_id,
            run_stage=args.run_stage,
            train_args=train_args,
        )
        if args.dry_run:
            _print_dry_run(
                baseline_id=args.baseline_id,
                run_stage=args.run_stage,
                output_root=args.output_root,
                records_root=args.records_root,
                checkpoint_store_root=args.checkpoint_store_root,
                copy_checkpoints=bool(args.copy_checkpoints),
                command=command,
            )
            return 0

        repo_root = Path.cwd()
        run_dir = _run_child_command(command, repo_root)
        target_logs, copied, missing = _copy_curated_logs(run_dir, args.records_root, args.baseline_id)
        checkpoint_source, checkpoint_store_path, checkpoint_copied, checkpoint_copy_reason = _copy_last_checkpoint(
            run_dir,
            args.checkpoint_store_root,
            args.baseline_id,
            copy_checkpoints=bool(args.copy_checkpoints),
        )
        record_path = _write_run_record(
            target_logs=target_logs,
            baseline_id=args.baseline_id,
            run_stage=args.run_stage,
            run_dir=run_dir,
            copied=copied,
            missing=missing,
            checkpoint_source=checkpoint_source,
            checkpoint_store_path=checkpoint_store_path,
            checkpoint_copied=checkpoint_copied,
            checkpoint_copy_reason=checkpoint_copy_reason,
        )
        print(f"[baseline-batch] archived {args.baseline_id} to {target_logs}")
        print(f"[baseline-batch] run_record: {record_path}")
        return 0
    except Exception as exc:
        print(f"[baseline-batch] failure: {exc}")
        if bool(args.stop_on_failure):
            raise
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
