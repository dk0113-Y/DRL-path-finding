from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any


DEFAULT_RECORDS_ROOT = Path("experiment_records/final_method/A_new_minimum_closure")
DEFAULT_CHECKPOINT_STORE_ROOT = Path("checkpoint_store/final_method/A_new_minimum_closure")


def checkpoint_target_path(checkpoint_store_root: Path, method_id: str) -> Path:
    return Path(checkpoint_store_root) / f"{method_id}.pt"


def records_logs_dir(records_root: Path, method_id: str) -> Path:
    return Path(records_root) / method_id / "logs"


def archive_dry_run_payload(
    *,
    method_id: str,
    records_root: Path,
    checkpoint_store_root: Path,
    copy_checkpoints: bool,
) -> dict[str, Any]:
    return {
        "records_logs_dir": str(records_logs_dir(records_root, method_id)),
        "checkpoint_store_path": str(checkpoint_target_path(checkpoint_store_root, method_id)),
        "checkpoint_copying": "enabled" if copy_checkpoints else "disabled",
        "log_copying": "enabled_all_top_level_log_files",
    }


def _copy_log_files(run_dir: Path, target_logs: Path) -> tuple[list[str], list[str]]:
    source_logs = run_dir / "logs"
    copied: list[str] = []
    skipped: list[str] = []
    if not source_logs.exists():
        return copied, ["logs_dir_missing"]

    target_logs.mkdir(parents=True, exist_ok=True)
    for source_path in sorted(source_logs.iterdir()):
        if not source_path.is_file():
            skipped.append(source_path.name)
            continue
        shutil.copy2(source_path, target_logs / source_path.name)
        copied.append(source_path.name)
    return copied, skipped


def _copy_last_checkpoint(
    *,
    run_dir: Path,
    checkpoint_store_root: Path,
    method_id: str,
    copy_checkpoints: bool,
) -> tuple[Path, Path, bool, str | None]:
    source_path = run_dir / "checkpoints" / "last.pt"
    target_path = checkpoint_target_path(checkpoint_store_root, method_id)
    if not copy_checkpoints:
        return source_path, target_path, False, "disabled_by_user"
    if not source_path.exists():
        return source_path, target_path, False, "missing_last_checkpoint"

    target_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, target_path)
    return source_path, target_path, True, None


def archive_training_run(
    *,
    run_dir: Path,
    method_id: str,
    method_name: str,
    run_stage: str,
    records_root: Path = DEFAULT_RECORDS_ROOT,
    checkpoint_store_root: Path = DEFAULT_CHECKPOINT_STORE_ROOT,
    copy_checkpoints: bool = True,
) -> dict[str, Any]:
    run_dir = Path(run_dir).resolve()
    records_root = Path(records_root)
    checkpoint_store_root = Path(checkpoint_store_root)
    target_logs = records_logs_dir(records_root, method_id)

    copied_logs, skipped_logs = _copy_log_files(run_dir, target_logs)
    checkpoint_source, checkpoint_store_path, checkpoint_copied, checkpoint_copy_reason = _copy_last_checkpoint(
        run_dir=run_dir,
        checkpoint_store_root=checkpoint_store_root,
        method_id=method_id,
        copy_checkpoints=copy_checkpoints,
    )

    record = {
        "schema_version": "a_new_minimum_closure_archive_record/v1",
        "method_id": method_id,
        "method_name": method_name,
        "run_stage": run_stage,
        "source_run_dir": str(run_dir),
        "records_logs_dir": str(target_logs),
        "copied_logs": copied_logs,
        "skipped_logs": skipped_logs,
        "checkpoint_source": str(checkpoint_source),
        "checkpoint_store_path": str(checkpoint_store_path),
        "checkpoint_copied": checkpoint_copied,
        "checkpoint_copy_reason": checkpoint_copy_reason,
    }
    record_path = target_logs.parent / "run_record.json"
    record_path.parent.mkdir(parents=True, exist_ok=True)
    record_path.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    return {
        **record,
        "run_record_path": str(record_path),
    }
