from __future__ import annotations

import csv
import json
import math
import re
import statistics
from pathlib import Path
from typing import Any, Mapping


PROTOCOL_NAME = "formal_posthoc_trainselect_v1"

DEFAULT_SELECTION_WEIGHTS = {
    "reward": 0.35,
    "coverage": 0.25,
    "success_rate": 0.20,
    "episode_length": -0.10,
    "repeat_visit_ratio": -0.10,
}

TRAIN_WINDOW_FIELDS = {
    "reward": "recent_mean_reward",
    "coverage": "recent_mean_coverage",
    "success_rate": "recent_success_rate",
    "episode_length": "recent_mean_episode_length",
    "repeat_visit_ratio": "recent_mean_repeat_visit_ratio",
}

EVAL_FIELDS = {
    "reward": "eval_mean_reward",
    "coverage": "eval_mean_coverage",
    "success_rate": "eval_success_rate",
    "episode_length": "eval_mean_episode_length",
    "repeat_visit_ratio": "eval_mean_repeat_visit_ratio",
}

RECENT_TRAIN_FIELDS = {
    "reward": "recent_mean_reward",
    "coverage": "recent_mean_coverage",
    "success_rate": "recent_success_rate",
    "episode_length": "recent_mean_episode_length",
    "repeat_visit_ratio": "recent_mean_repeat_visit_ratio",
}

_CHECKPOINT_STEP_RE = re.compile(r"^ckpt_step_(\d+)\.pt$")


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    text = str(value).strip()
    if not text:
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, float):
        if value != value or value in {float("inf"), float("-inf")}:
            return None
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(dict(payload)), ensure_ascii=False, indent=2), encoding="utf-8")


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _write_csv(path: Path, rows: list[Mapping[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})


def _relative_path(run_dir: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(run_dir.resolve()).as_posix()
    except Exception:
        return path.as_posix()


def discover_posthoc_checkpoints(run_dir: Path) -> list[dict[str, Any]]:
    checkpoint_dir = Path(run_dir) / "checkpoints"
    discovered = []
    for path in sorted(checkpoint_dir.glob("ckpt_step_*.pt")):
        match = _CHECKPOINT_STEP_RE.match(path.name)
        if not match:
            continue
        discovered.append(
            {
                "checkpoint_step": int(match.group(1)),
                "checkpoint_path": _relative_path(Path(run_dir), path),
                "absolute_path": path,
            }
        )
    return sorted(discovered, key=lambda item: int(item["checkpoint_step"]))


def _mean_metric(rows: list[Mapping[str, Any]], field_name: str) -> float | None:
    values = [
        value
        for row in rows
        for value in [_to_float(row.get(field_name))]
        if value is not None
    ]
    if not values:
        return None
    return float(statistics.fmean(values))


def _window_rows(train_step_rows: list[Mapping[str, Any]], start: int, end: int) -> list[Mapping[str, Any]]:
    selected = []
    for row in train_step_rows:
        env_steps = _to_float(row.get("env_steps"))
        if env_steps is None:
            continue
        if float(start) <= env_steps <= float(end):
            selected.append(row)
    return selected


def _z_scores(values: list[float | None]) -> list[float]:
    numeric = [float(value) for value in values if value is not None]
    if not numeric:
        return [0.0 for _ in values]
    mean_value = statistics.fmean(numeric)
    std_value = statistics.pstdev(numeric)
    if std_value <= 0.0:
        return [0.0 for _ in values]
    return [0.0 if value is None else float((float(value) - mean_value) / std_value) for value in values]


def select_posthoc_candidates(
    *,
    run_dir: Path,
    candidate_start_step: int,
    candidate_end_step: int,
    checkpoint_interval: int,
    window_env_steps: int,
    topk: int,
    weights: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    run_dir = Path(run_dir)
    logs_dir = run_dir / "logs"
    train_step_rows = _read_csv_rows(logs_dir / "train_steps.csv")
    train_episode_rows = _read_csv_rows(logs_dir / "train_episodes.csv")
    checkpoints = discover_posthoc_checkpoints(run_dir)
    score_weights = dict(DEFAULT_SELECTION_WEIGHTS)
    if weights:
        score_weights.update({str(key): float(value) for key, value in weights.items()})

    candidates: list[dict[str, Any]] = []
    for checkpoint in checkpoints:
        step = int(checkpoint["checkpoint_step"])
        if step < int(candidate_start_step) or step > int(candidate_end_step):
            continue
        window_start = max(0, step - int(window_env_steps))
        rows = _window_rows(train_step_rows, window_start, step)
        row: dict[str, Any] = {
            "candidate_step": step,
            "checkpoint_path": checkpoint["checkpoint_path"],
            "window_start_env_steps": window_start,
            "window_end_env_steps": step,
            "window_row_count": len(rows),
        }
        for metric_name, field_name in TRAIN_WINDOW_FIELDS.items():
            row[metric_name] = _mean_metric(rows, field_name)
        row["valid_candidate"] = bool(rows) and all(row.get(metric_name) is not None for metric_name in TRAIN_WINDOW_FIELDS)
        candidates.append(row)

    valid_candidates = [row for row in candidates if row.get("valid_candidate")]
    for metric_name in TRAIN_WINDOW_FIELDS:
        values = [_to_float(row.get(metric_name)) for row in valid_candidates]
        for row, z_value in zip(valid_candidates, _z_scores(values)):
            row[f"{metric_name}_z"] = z_value
    for row in candidates:
        if not row.get("valid_candidate"):
            for metric_name in TRAIN_WINDOW_FIELDS:
                row[f"{metric_name}_z"] = 0.0
            row["selection_score"] = None
            row["selected"] = False
            row["selection_rank"] = None
            continue
        row["selection_score"] = float(
            sum(
                float(score_weights[metric_name]) * float(row.get(f"{metric_name}_z", 0.0))
                for metric_name in TRAIN_WINDOW_FIELDS
            )
        )
        row["selected"] = False
        row["selection_rank"] = None

    ranked = sorted(
        valid_candidates,
        key=lambda row: (
            float(row.get("selection_score") or float("-inf")),
            float(row.get("success_rate_z") or 0.0),
            float(row.get("coverage_z") or 0.0),
            float(row.get("reward_z") or 0.0),
            -int(row.get("candidate_step") or 0),
        ),
        reverse=True,
    )
    selected = ranked[: max(0, int(topk))]
    selected_steps = {int(row["candidate_step"]): rank for rank, row in enumerate(selected, start=1)}
    for row in candidates:
        rank = selected_steps.get(int(row["candidate_step"]))
        if rank is not None:
            row["selected"] = True
            row["selection_rank"] = rank

    output_rows = sorted(
        candidates,
        key=lambda row: (
            1 if row.get("selection_score") is None else 0,
            -(float(row.get("selection_score") or 0.0)),
            int(row.get("candidate_step") or 0),
        ),
    )
    fieldnames = [
        "selection_rank",
        "selected",
        "candidate_step",
        "checkpoint_path",
        "window_start_env_steps",
        "window_end_env_steps",
        "window_row_count",
        "valid_candidate",
        "reward",
        "coverage",
        "success_rate",
        "episode_length",
        "repeat_visit_ratio",
        "reward_z",
        "coverage_z",
        "success_rate_z",
        "episode_length_z",
        "repeat_visit_ratio_z",
        "selection_score",
    ]
    scores_path = logs_dir / "posthoc_candidate_scores.csv"
    _write_csv(scores_path, output_rows, fieldnames)

    summary = {
        "protocol_name": PROTOCOL_NAME,
        "artifact_type": "posthoc_selection_summary",
        "candidate_score_csv": "logs/posthoc_candidate_scores.csv",
        "train_steps_source": "logs/train_steps.csv",
        "train_episodes_source": "logs/train_episodes.csv",
        "train_step_row_count": len(train_step_rows),
        "train_episode_row_count": len(train_episode_rows),
        "candidate_start_step": int(candidate_start_step),
        "candidate_end_step": int(candidate_end_step),
        "checkpoint_interval": int(checkpoint_interval),
        "selection_window_env_steps": int(window_env_steps),
        "selection_weights": score_weights,
        "ranking_tie_break": [
            "selection_score_desc",
            "success_rate_z_desc",
            "coverage_z_desc",
            "reward_z_desc",
            "candidate_step_asc",
        ],
        "candidate_count": len(candidates),
        "valid_candidate_count": len(valid_candidates),
        "selected_candidate_count": len(selected),
        "selected_candidate_steps": [int(row["candidate_step"]) for row in selected],
        "top_candidates": [_json_safe(dict(row)) for row in selected],
        "all_candidates": [_json_safe(dict(row)) for row in output_rows],
    }
    summary_path = logs_dir / "posthoc_selection_summary.json"
    _write_json(summary_path, summary)
    return {
        "scores_path": scores_path,
        "summary_path": summary_path,
        "summary": summary,
        "selected_candidates": [dict(row) for row in selected],
        "all_candidates": [dict(row) for row in output_rows],
    }


def final_probe_rank_key(row: Mapping[str, Any]) -> tuple[float, float, float]:
    values = []
    for field_name in ("eval_success_rate", "eval_mean_coverage", "eval_mean_reward"):
        value = _to_float(row.get(field_name))
        values.append(float("-inf") if value is None else float(value))
    return tuple(values)  # type: ignore[return-value]


def build_best_vs_last_gap_summary(
    *,
    winner_probe_row: Mapping[str, Any],
    recent_train_row: Mapping[str, Any] | None,
    last_probe_row: Mapping[str, Any] | None,
    last_checkpoint_path: str,
    last_env_steps: int,
) -> dict[str, Any]:
    if last_probe_row:
        comparison_mode = "heldout_winner_vs_heldout_last_checkpoint"
        last_source = "final_probe.csv::last_checkpoint_candidate"
        last_getter = lambda metric: _to_float(last_probe_row.get(EVAL_FIELDS[metric]))
    else:
        comparison_mode = "heldout_winner_vs_train_endpoint_recent_window"
        last_source = "logs/train_steps.csv::recent_train_endpoint"
        last_getter = lambda metric: _to_float((recent_train_row or {}).get(RECENT_TRAIN_FIELDS[metric]))

    details: dict[str, Any] = {}
    for metric_name, field_name in EVAL_FIELDS.items():
        best_value = _to_float(winner_probe_row.get(field_name))
        last_value = last_getter(metric_name)
        details[metric_name] = {
            "best": best_value,
            "last": last_value,
            "best_minus_last": (
                float(best_value) - float(last_value)
                if best_value is not None and last_value is not None
                else None
            ),
        }
    return {
        "protocol_name": PROTOCOL_NAME,
        "role": "diagnostic_best_vs_last",
        "comparison_mode": comparison_mode,
        "direction_note": "Positive best_minus_last is better for reward/coverage/success; negative is better for episode_length/repeat_visit_ratio.",
        "best_source": "logs/final_probe.csv::formal_winner",
        "last_source": last_source,
        "last_checkpoint_path": last_checkpoint_path,
        "last_env_steps": int(last_env_steps),
        "details": details,
    }


def write_posthoc_final_artifacts(
    *,
    run_dir: Path,
    total_env_steps: int,
    candidate_start_step: int,
    candidate_end_step: int,
    checkpoint_interval: int,
    selected_candidates: list[Mapping[str, Any]],
    final_probe_rows: list[Mapping[str, Any]],
    winner_probe_row: Mapping[str, Any],
    recent_train_row: Mapping[str, Any] | None,
    last_checkpoint_path: str,
    best_pt_path: str,
    final_probe_episode_count: int,
    seed_base: int | None,
) -> dict[str, Path]:
    run_dir = Path(run_dir)
    logs_dir = run_dir / "logs"
    winner_step = int(winner_probe_row["env_steps"])
    last_probe_row = next(
        (row for row in final_probe_rows if int(row.get("env_steps", -1)) == int(total_env_steps)),
        None,
    )
    best_vs_last = build_best_vs_last_gap_summary(
        winner_probe_row=winner_probe_row,
        recent_train_row=recent_train_row,
        last_probe_row=last_probe_row,
        last_checkpoint_path=last_checkpoint_path,
        last_env_steps=int(total_env_steps),
    )
    final_probe_summary = {
        "protocol_name": PROTOCOL_NAME,
        "artifact_type": "final_probe_summary",
        "final_probe_csv": "logs/final_probe.csv",
        "ranking_order": ["success_rate", "coverage", "reward"],
        "seed_base": seed_base,
        "final_probe_episode_count": int(final_probe_episode_count),
        "selected_candidate_steps": [int(row["candidate_step"]) for row in selected_candidates],
        "winner_step": winner_step,
        "winner_checkpoint_path": winner_probe_row.get("checkpoint_path"),
        "best_pt_path": best_pt_path,
        "last_pt_path": last_checkpoint_path,
        "final_probe_rows": [_json_safe(dict(row)) for row in final_probe_rows],
        "winner_row": _json_safe(dict(winner_probe_row)),
    }
    manifest = {
        "protocol_name": PROTOCOL_NAME,
        "total_env_steps": int(total_env_steps),
        "candidate_start_step": int(candidate_start_step),
        "candidate_end_step": int(candidate_end_step),
        "checkpoint_interval": int(checkpoint_interval),
        "selected_candidate_steps": [int(row["candidate_step"]) for row in selected_candidates],
        "final_probe_episode_count": int(final_probe_episode_count),
        "final_probe_seed_base": seed_base,
        "winner_step": winner_step,
        "winner_checkpoint_path": str(winner_probe_row.get("checkpoint_path")),
        "best_pt_path": best_pt_path,
        "last_pt_path": last_checkpoint_path,
        "posthoc_candidate_scores_csv": "logs/posthoc_candidate_scores.csv",
        "posthoc_selection_summary_json": "logs/posthoc_selection_summary.json",
        "final_probe_csv": "logs/final_probe.csv",
        "final_probe_summary_json": "logs/final_probe_summary.json",
        "best_vs_last_gap_summary_json": "logs/best_vs_last_gap_summary.json",
    }
    final_probe_summary_path = logs_dir / "final_probe_summary.json"
    best_vs_last_path = logs_dir / "best_vs_last_gap_summary.json"
    manifest_path = logs_dir / "formal_selection_manifest.json"
    _write_json(final_probe_summary_path, final_probe_summary)
    _write_json(best_vs_last_path, best_vs_last)
    _write_json(manifest_path, manifest)
    return {
        "final_probe_summary": final_probe_summary_path,
        "best_vs_last_gap_summary": best_vs_last_path,
        "formal_selection_manifest": manifest_path,
    }
