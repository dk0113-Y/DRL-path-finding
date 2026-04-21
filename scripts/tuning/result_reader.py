from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any


FINAL_PROBE_KEYS = [
    "eval_mean_reward",
    "eval_mean_coverage",
    "eval_success_rate",
    "eval_mean_episode_length",
    "eval_mean_repeat_visit_ratio",
    "eval_mean_recent_revisit_trigger_count",
    "eval_mean_stall_trigger_count",
    "eval_mean_zero_info_step_count",
    "eval_mean_timeout_flag",
    "eval_mean_turn_ge_90_count",
    "eval_mean_turn_135_count",
    "eval_mean_turn_180_count",
    "eval_mean_turn_penalty_weight_sum",
    "eval_mean_weighted_info_gain_sum",
]

EVAL_KEYS = [
    "env_steps",
    "learner_steps",
    "eval_mean_reward",
    "eval_mean_coverage",
    "eval_success_rate",
    "eval_mean_episode_length",
    "eval_mean_repeat_visit_ratio",
    "eval_mean_recent_revisit_trigger_count",
    "eval_mean_timeout_flag",
    "eval_mean_zero_info_step_count",
    "eval_mean_turn_ge_90_count",
    "eval_mean_turn_180_count",
    "eval_mean_weighted_info_gain_sum",
]

TRAIN_RECENT_KEYS = [
    "env_steps",
    "learner_steps",
    "recent_mean_reward",
    "recent_mean_coverage",
    "recent_success_rate",
    "recent_mean_episode_length",
    "recent_mean_repeat_visit_ratio",
]

CHECKPOINT_CONFIG_KEYS = [
    "run_name",
    "seed",
    "device",
    "total_env_steps",
    "reward_turn_penalty_scale",
    "reward_revisit_penalty",
    "max_accessible_blocks",
    "max_entries_per_block",
]

COMPLETE_STATUSES = {"completed", "completed_with_postprocess_error"}


@dataclass
class RunResult:
    run_dir: Path
    status: str
    status_reason: str
    return_code: int | None
    final_probe: dict[str, Any]
    best_eval: dict[str, Any]
    last_eval: dict[str, Any]
    train_recent: dict[str, Any]
    checkpoint: dict[str, Any]
    file_status: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_dir": str(self.run_dir),
            "status": self.status,
            "status_reason": self.status_reason,
            "return_code": self.return_code,
            "final_probe": self.final_probe,
            "best_eval": self.best_eval,
            "last_eval": self.last_eval,
            "train_recent": self.train_recent,
            "checkpoint": self.checkpoint,
            "file_status": self.file_status,
        }


def _to_scalar(value: Any) -> Any:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return text


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return [dict(row) for row in reader]


def _has_valid_final_probe(rows: list[dict[str, str]]) -> bool:
    for row in rows:
        for key in FINAL_PROBE_KEYS:
            if _to_scalar(row.get(key)) is not None:
                return True
    return False


def _select_formal_final_probe_row(rows: list[dict[str, str]]) -> dict[str, str] | None:
    for row in rows:
        if str(row.get("formal_winner", "")).strip().lower() == "true":
            return row
    for row in rows:
        if str(row.get("source", "")).strip() == "posthoc_final_winner":
            return row
    return rows[-1] if rows else None


def _extract_metrics(row: dict[str, str] | None, keys: list[str]) -> dict[str, Any]:
    row = row or {}
    alias_map = {
        "eval_mean_recent_revisit_trigger_count": "eval_mean_recent_revisit_count",
    }
    extracted: dict[str, Any] = {}
    for key in keys:
        value = row.get(key)
        if value is None and key in alias_map:
            value = row.get(alias_map[key])
        extracted[key] = _to_scalar(value)
    return extracted


def _eval_score(row: dict[str, str]) -> tuple[float, float, float, float]:
    reward = _to_scalar(row.get("eval_mean_reward"))
    success = _to_scalar(row.get("eval_success_rate"))
    coverage = _to_scalar(row.get("eval_mean_coverage"))
    length = _to_scalar(row.get("eval_mean_episode_length"))
    reward_score = float(reward) if isinstance(reward, (int, float)) else float("-inf")
    success_score = float(success) if isinstance(success, (int, float)) else float("-inf")
    coverage_score = float(coverage) if isinstance(coverage, (int, float)) else float("-inf")
    length_score = -float(length) if isinstance(length, (int, float)) else float("-inf")
    return (success_score, coverage_score, reward_score, length_score)


def _select_best_eval_row(rows: list[dict[str, str]]) -> dict[str, str] | None:
    if not rows:
        return None
    return max(rows, key=_eval_score)


def _load_checkpoint_payload(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    try:
        import torch
    except Exception as exc:  # pragma: no cover - environment dependent
        return None, f"{type(exc).__name__}: {exc}"

    try:
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:  # pragma: no cover - older torch versions
            payload = torch.load(path, map_location="cpu")
        if not isinstance(payload, dict):
            return None, "checkpoint payload is not a dict"
        return payload, None
    except Exception as exc:  # pragma: no cover - checkpoint compatibility dependent
        return None, f"{type(exc).__name__}: {exc}"


def _extract_checkpoint_train_config(payload: dict[str, Any]) -> dict[str, Any]:
    train_config = payload.get("train_config")
    if not isinstance(train_config, dict):
        return {}
    return {key: train_config.get(key) for key in CHECKPOINT_CONFIG_KEYS if key in train_config}


def _checkpoint_summary(run_dir: Path) -> dict[str, Any]:
    checkpoints_dir = run_dir / "checkpoints"
    best_path = checkpoints_dir / "best.pt"
    last_path = checkpoints_dir / "last.pt"
    summary: dict[str, Any] = {
        "best_checkpoint_path": str(best_path) if best_path.exists() else None,
        "last_checkpoint_path": str(last_path) if last_path.exists() else None,
        "best_checkpoint_eval_metrics": None,
        "best_checkpoint_selection_metadata": None,
        "best_checkpoint_train_config": None,
        "last_checkpoint_train_config": None,
        "checkpoint_read_error": None,
    }

    errors: list[str] = []
    if best_path.exists():
        payload, error = _load_checkpoint_payload(best_path)
        if error:
            errors.append(f"best.pt: {error}")
        elif payload is not None:
            summary["best_checkpoint_eval_metrics"] = payload.get("eval_metrics")
            summary["best_checkpoint_selection_metadata"] = payload.get("selection_metadata")
            summary["best_checkpoint_train_config"] = _extract_checkpoint_train_config(payload)

    if last_path.exists():
        payload, error = _load_checkpoint_payload(last_path)
        if error:
            errors.append(f"last.pt: {error}")
        elif payload is not None:
            summary["last_checkpoint_train_config"] = _extract_checkpoint_train_config(payload)

    if errors:
        summary["checkpoint_read_error"] = "; ".join(errors)
    return summary


def find_latest_run_dir(
    output_root: Path | str,
    run_name_prefix: str,
    started_after_epoch: float | None = None,
) -> Path:
    output_root = Path(output_root)
    if not output_root.exists():
        raise FileNotFoundError(f"Output root does not exist: {output_root}")

    candidates: list[tuple[float, Path]] = []
    for child in output_root.iterdir():
        if not child.is_dir():
            continue
        if child.name == "scheduler_runs":
            continue
        if not child.name.startswith(f"{run_name_prefix}_"):
            continue
        stat = child.stat()
        created_epoch = max(stat.st_ctime, stat.st_mtime)
        if started_after_epoch is not None and created_epoch < started_after_epoch - 2.0:
            continue
        candidates.append((created_epoch, child))

    if not candidates:
        threshold_text = (
            f" after epoch {started_after_epoch:.3f}" if started_after_epoch is not None else ""
        )
        raise FileNotFoundError(
            f"No run directory found under {output_root} for prefix '{run_name_prefix}'{threshold_text}."
        )

    candidates.sort(key=lambda item: (item[0], item[1].name), reverse=True)
    return candidates[0][1]


def read_run_result(run_dir: Path | str, return_code: int | None) -> RunResult:
    run_dir = Path(run_dir)
    logs_dir = run_dir / "logs"
    checkpoints_dir = run_dir / "checkpoints"
    final_probe_path = logs_dir / "final_probe.csv"
    eval_metrics_path = logs_dir / "eval_metrics.csv"
    model_select_eval_path = logs_dir / "model_select_eval.csv"
    best_recheck_eval_path = logs_dir / "best_recheck_eval.csv"
    posthoc_candidate_scores_path = logs_dir / "posthoc_candidate_scores.csv"
    posthoc_selection_summary_path = logs_dir / "posthoc_selection_summary.json"
    formal_selection_manifest_path = logs_dir / "formal_selection_manifest.json"
    train_steps_path = logs_dir / "train_steps.csv"
    train_episodes_path = logs_dir / "train_episodes.csv"
    best_checkpoint_path = checkpoints_dir / "best.pt"
    last_checkpoint_path = checkpoints_dir / "last.pt"

    final_probe_rows = _read_csv_rows(final_probe_path)
    eval_rows = _read_csv_rows(eval_metrics_path)
    model_select_rows = _read_csv_rows(model_select_eval_path)
    best_recheck_rows = _read_csv_rows(best_recheck_eval_path)
    posthoc_candidate_rows = _read_csv_rows(posthoc_candidate_scores_path)
    train_step_rows = _read_csv_rows(train_steps_path)
    train_episode_rows = _read_csv_rows(train_episodes_path)
    formal_final_probe_row = _select_formal_final_probe_row(final_probe_rows)

    has_valid_probe = _has_valid_final_probe(final_probe_rows)
    has_eval = bool(eval_rows)
    has_model_select = bool(model_select_rows)
    has_recheck = bool(best_recheck_rows)
    has_posthoc_selection = bool(posthoc_candidate_rows) or formal_selection_manifest_path.exists()
    has_best_checkpoint = best_checkpoint_path.exists()
    has_last_checkpoint = last_checkpoint_path.exists()
    artifacts_complete = run_dir.exists() and has_valid_probe and has_best_checkpoint and has_last_checkpoint

    missing_bits: list[str] = []
    if not run_dir.exists():
        missing_bits.append("run_dir_missing")
    if not has_valid_probe:
        missing_bits.append("final_probe_missing_or_invalid")
    if not has_best_checkpoint:
        missing_bits.append("best_checkpoint_missing")
    if not has_last_checkpoint:
        missing_bits.append("last_checkpoint_missing")

    if artifacts_complete:
        status = "completed" if return_code in (None, 0) else "completed_with_postprocess_error"
        status_reason = "core artifacts complete"
    else:
        status = "failed"
        status_reason = ", ".join(missing_bits) if missing_bits else "core artifacts incomplete"

    checkpoint_summary = _checkpoint_summary(run_dir)
    file_status = {
        "final_probe_csv": str(final_probe_path),
        "final_probe_row_count": len(final_probe_rows),
        "eval_metrics_csv": str(eval_metrics_path),
        "eval_metrics_row_count": len(eval_rows),
        "model_select_eval_csv": str(model_select_eval_path),
        "model_select_eval_row_count": len(model_select_rows),
        "best_recheck_eval_csv": str(best_recheck_eval_path),
        "best_recheck_eval_row_count": len(best_recheck_rows),
        "posthoc_candidate_scores_csv": str(posthoc_candidate_scores_path),
        "posthoc_candidate_scores_row_count": len(posthoc_candidate_rows),
        "posthoc_selection_summary_json": str(posthoc_selection_summary_path),
        "formal_selection_manifest_json": str(formal_selection_manifest_path),
        "train_steps_csv": str(train_steps_path),
        "train_steps_row_count": len(train_step_rows),
        "train_episodes_csv": str(train_episodes_path),
        "train_episodes_row_count": len(train_episode_rows),
        "best_checkpoint_exists": best_checkpoint_path.exists(),
        "last_checkpoint_exists": last_checkpoint_path.exists(),
        "periodic_eval_available": has_eval,
        "model_select_eval_available": has_model_select,
        "best_recheck_eval_available": has_recheck,
        "posthoc_selection_available": has_posthoc_selection,
    }
    best_eval_source_rows = best_recheck_rows if best_recheck_rows else (
        model_select_rows if model_select_rows else (eval_rows if eval_rows else final_probe_rows)
    )
    final_probe_env_steps = _to_scalar(formal_final_probe_row.get("env_steps")) if formal_final_probe_row else None
    last_eval_source_rows = (
        [
            row for row in best_eval_source_rows
            if _to_scalar(row.get("env_steps")) == final_probe_env_steps
        ]
        if final_probe_env_steps is not None else []
    )

    return RunResult(
        run_dir=run_dir,
        status=status,
        status_reason=status_reason,
        return_code=return_code,
        final_probe=_extract_metrics(formal_final_probe_row, FINAL_PROBE_KEYS),
        best_eval=_extract_metrics(_select_best_eval_row(best_eval_source_rows), EVAL_KEYS),
        last_eval=_extract_metrics((last_eval_source_rows[-1] if last_eval_source_rows else (eval_rows[-1] if eval_rows else None)), EVAL_KEYS),
        train_recent=_extract_metrics(train_step_rows[-1] if train_step_rows else None, TRAIN_RECENT_KEYS),
        checkpoint=checkpoint_summary,
        file_status=file_status,
    )
