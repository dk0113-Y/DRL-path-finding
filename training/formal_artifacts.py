from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
import subprocess
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

from training.rewarding import STALL_DIAGNOSTIC_WINDOW

SCHEMA_VERSION = "formal_train_artifacts/v4"
DEFAULT_MAIN_BASELINE_IDENTIFIER = "4.9_30万轮基线"
CURRENT_FORMAL_PROTOCOL_REVISION = "formal_posthoc_trainselect_v1"
LEGACY_BEST_CHECKPOINT_PROTOCOL_REVISION = "formal_best_checkpoint_v3"
CURRENT_DEFAULT_FORMAL_FINAL_PROBE_EPISODES = 100
HISTORICAL_FORMAL_FINAL_PROBE_EPISODES = 16

RUNTIME_ONLY_FIELDS = (
    "enable_amp",
    "enable_inference_amp",
    "amp_dtype",
    "enable_torch_compile",
    "compile_mode",
    "enable_cudnn_benchmark",
    "enable_tf32",
    "strict_reproducibility",
    "enable_channels_last",
    "generate_plots_on_finish",
    "save_train_representative_trajectories",
    "save_train_special_trajectories",
    "save_final_probe_trajectories",
)

TIMING_FLAG_FIELDS = (
    "enable_collector_timing",
    "enable_learner_timing",
    "enable_replay_timing",
    "enable_state_adapter_timing",
    "enable_cummap_timing",
    "enable_shared_semantic_timing",
    "enable_advantage_state_timing",
    "enable_value_state_timing",
)

ALLOWED_TUNING_FIELDS = (
    "reward_turn_penalty_scale",
    "reward_turn_weight_45",
    "reward_turn_weight_90",
    "reward_turn_weight_135",
    "reward_turn_weight_180",
    "reward_revisit_penalty",
)

MANUAL_REVIEW_FIELDS = (
    "max_entries_per_block",
)

FROZEN_COMPARABILITY_FIELDS = (
    "rows",
    "cols",
    "obs_size",
    "scan_radius",
    "trajectory_history_steps",
    "obstacle_ratio",
    "budget_mode",
    "total_env_steps",
    "total_train_episodes",
    "warmup_steps",
    "warmup_episodes",
    "collect_steps_per_iter",
    "learner_updates_per_iter",
    "train_every_env_steps",
    "formal_protocol",
    "final_greedy_episodes",
    "use_fixed_train_episode_seeds",
    "fixed_train_episode_seed_base",
    "use_fixed_eval_seeds",
    "fixed_final_probe_seed_base",
    "periodic_checkpoint_interval_env_steps",
    "posthoc_candidate_start_env_steps",
    "posthoc_candidate_end_env_steps",
    "posthoc_selection_window_env_steps",
    "posthoc_final_probe_topk",
    "enable_best_checkpoint_selection",
    "best_checkpoint_selection_start_env_steps",
    "best_checkpoint_selection_interval_env_steps",
    "best_checkpoint_validation_episodes",
    "best_checkpoint_topk_recheck",
    "best_checkpoint_recheck_episodes",
    "use_fixed_model_select_seeds",
    "fixed_model_select_seed_base",
    "replay_capacity",
    "batch_size",
    "min_replay_size",
    "gamma",
    "n_step",
    "learning_rate",
    "weight_decay",
    "grad_clip_norm",
    "target_update_interval",
    "epsilon_start",
    "epsilon_end",
    "epsilon_decay_steps",
    "max_episode_steps",
    "coverage_stop_threshold",
    "reward_info_scale",
    "reward_obstacle_weight",
    "reward_step_penalty",
    "reward_terminal_bonus",
    "reward_timeout_penalty",
    "max_accessible_blocks",
)

REWARD_BREAKDOWN_FIELDS = (
    "info_reward_sum",
    "step_penalty_sum",
    "recent_revisit_penalty_sum",
    "turn_penalty_sum",
    "timeout_penalty_sum",
    "terminal_bonus_sum",
)

REWARD_EVENT_FIELDS = (
    "delta_empty_sum",
    "delta_obstacle_sum",
    "empty_info_gain_sum",
    "obstacle_info_gain_sum",
    "weighted_obstacle_info_gain_sum",
    "weighted_info_gain_sum",
    "empty_info_reward_sum",
    "obstacle_info_reward_sum",
    "obstacle_info_contribution_ratio",
    "recent_revisit_trigger_count",
    "stall_trigger_count",
    "zero_info_step_count",
    "turn_ge_90_count",
    "turn_135_count",
    "turn_180_count",
    "turn_penalty_weight_sum",
    "timeout_flag",
)

PRIMARY_UNIFIED_METRICS = {
    "reward": "higher_is_better",
    "coverage": "higher_is_better",
    "success_rate": "higher_is_better",
}

SECONDARY_UNIFIED_METRICS = {
    "episode_length": "lower_is_better",
    "repeat_visit_ratio": "lower_is_better",
}

STABILITY_UNIFIED_METRICS = {
    "timeout_flag": "lower_is_better",
    "stall_trigger_count": "lower_is_better",
    "zero_info_step_count": "lower_is_better",
    "recent_revisit_trigger_count": "lower_is_better",
}

RECENT_CORE_FIELD_MAP = {
    "reward": "recent_mean_reward",
    "coverage": "recent_mean_coverage",
    "success_rate": "recent_success_rate",
    "episode_length": "recent_mean_episode_length",
    "repeat_visit_ratio": "recent_mean_repeat_visit_ratio",
}

TRAIN_DYNAMICS_SLOPE_FIELDS = {
    "reward": "recent_mean_reward",
    "coverage": "recent_mean_coverage",
    "success_rate": "recent_success_rate",
    "episode_length": "recent_mean_episode_length",
    "repeat_visit_ratio": "recent_mean_repeat_visit_ratio",
    "accessible_block_count": "recent_accessible_block_count",
    "total_frontier_cluster_count": "recent_total_frontier_cluster_count",
    "stall_trigger_count": "recent_stall_trigger_count",
    "zero_info_step_count": "recent_zero_info_step_count",
    "turn_180_count": "recent_turn_180_count",
    "recent_revisit_penalty_sum": "recent_recent_revisit_penalty_sum",
    "terminal_bonus_sum": "recent_terminal_bonus_sum",
}

EVAL_CORE_FIELD_MAP = {
    "reward": "eval_mean_reward",
    "coverage": "eval_mean_coverage",
    "success_rate": "eval_success_rate",
    "episode_length": "eval_mean_episode_length",
    "repeat_visit_ratio": "eval_mean_repeat_visit_ratio",
}

TRAIN_FINAL_CONSISTENCY_TOLERANCES = {
    "reward": 20.0,
    "coverage": 0.05,
    "success_rate": 0.10,
    "episode_length": 50.0,
    "repeat_visit_ratio": 0.08,
}

TRAIN_FINAL_CONSISTENCY_DIRECTIONS = {
    "reward": True,
    "coverage": True,
    "success_rate": True,
    "episode_length": False,
    "repeat_visit_ratio": False,
}

ROW_ALIAS_FIELDS = {
    "recent_revisit_count": "recent_revisit_trigger_count",
    "recent_recent_revisit_count": "recent_recent_revisit_trigger_count",
    "eval_mean_recent_revisit_count": "eval_mean_recent_revisit_trigger_count",
}


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _read_csv_header(path: Path) -> list[str]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration:
            return []
    return [str(item) for item in header]


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _to_scalar(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            return None
        return value
    text = str(value).strip()
    if not text:
        return None
    lowered = text.lower()
    if lowered in {"nan", "none", "null", "n/a"}:
        return None
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    try:
        number = float(text)
    except ValueError:
        return text
    if number.is_integer():
        return int(number)
    return number


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    return _to_scalar(value)


def _serialize_config(cfg: Any) -> dict[str, Any] | None:
    if cfg is None:
        return None
    if is_dataclass(cfg):
        return _json_safe(asdict(cfg))
    if isinstance(cfg, Mapping):
        return _json_safe(dict(cfg))
    return None


def _formal_protocol_revision(config_dict: Mapping[str, Any] | None = None) -> str:
    configured = None if config_dict is None else config_dict.get("formal_protocol")
    protocol = str(configured or CURRENT_FORMAL_PROTOCOL_REVISION).strip()
    return protocol or CURRENT_FORMAL_PROTOCOL_REVISION


def _is_posthoc_protocol(protocol_revision: str) -> bool:
    return str(protocol_revision) == CURRENT_FORMAL_PROTOCOL_REVISION


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), ensure_ascii=False, indent=2), encoding="utf-8")


def _git_output(repo_dir: Path, args: list[str]) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=str(repo_dir),
            capture_output=True,
            text=True,
            check=True,
        )
    except Exception:
        return None
    text = result.stdout.strip()
    return text or None


def _apply_row_aliases(row: Mapping[str, Any] | None) -> dict[str, Any]:
    if not row:
        return {}
    normalized = dict(row)
    for old_name, new_name in ROW_ALIAS_FIELDS.items():
        if old_name in normalized and new_name not in normalized:
            normalized[new_name] = normalized[old_name]
    return normalized


def _extract_fields(row: Mapping[str, Any] | None, field_names: tuple[str, ...], prefix: str) -> dict[str, Any]:
    normalized = _apply_row_aliases(row)
    extracted: dict[str, Any] = {}
    for field_name in field_names:
        key = f"{prefix}{field_name}"
        extracted[field_name] = _to_scalar(normalized.get(key))
    return extracted


def _extract_dynamic_metric_fields(
    row: Mapping[str, Any] | None,
    *,
    prefix: str,
    reserved_fields: set[str],
) -> dict[str, Any]:
    normalized = _apply_row_aliases(row)
    extra: dict[str, Any] = {}
    for key, value in normalized.items():
        if not key.startswith(prefix):
            continue
        suffix = key[len(prefix):]
        if suffix in reserved_fields:
            continue
        extra[suffix] = _to_scalar(value)
    return extra


def _normalize_recent_train(row: Mapping[str, Any] | None) -> dict[str, Any]:
    normalized = _apply_row_aliases(row)
    if not normalized:
        return {}
    reserved = {
        "mean_reward",
        "mean_coverage",
        "success_rate",
        "mean_episode_length",
        "mean_repeat_visit_ratio",
    }
    reserved.update(REWARD_BREAKDOWN_FIELDS)
    reserved.update(REWARD_EVENT_FIELDS)
    return {
        "source": "logs/train_steps.csv",
        "env_steps": _to_scalar(normalized.get("env_steps")),
        "learner_steps": _to_scalar(normalized.get("learner_steps")),
        "optimizer_monitoring": {
            "loss": _to_scalar(normalized.get("loss")),
            "q_mean": _to_scalar(normalized.get("q_mean")),
            "target_q_mean": _to_scalar(normalized.get("target_q_mean")),
            "td_abs_mean": _to_scalar(normalized.get("td_abs_mean")),
            "grad_norm": _to_scalar(normalized.get("grad_norm")),
            "replay_size": _to_scalar(normalized.get("replay_size")),
            "epsilon": _to_scalar(normalized.get("epsilon")),
        },
        "metrics": {name: _to_scalar(normalized.get(field_name)) for name, field_name in RECENT_CORE_FIELD_MAP.items()},
        "reward_breakdown": _extract_fields(normalized, REWARD_BREAKDOWN_FIELDS, "recent_"),
        "reward_events": _extract_fields(normalized, REWARD_EVENT_FIELDS, "recent_"),
        "semantic_monitoring": _extract_dynamic_metric_fields(normalized, prefix="recent_", reserved_fields=reserved),
        "raw_row": _json_safe(normalized),
    }


def _normalize_eval_like(row: Mapping[str, Any] | None, *, source_name: str) -> dict[str, Any]:
    normalized = _apply_row_aliases(row)
    if not normalized:
        return {}
    raw_reserved = {
        "eval_mean_reward",
        "eval_mean_coverage",
        "eval_success_rate",
        "eval_mean_episode_length",
        "eval_mean_repeat_visit_ratio",
        "eval_episodes",
    }
    raw_reserved.update({f"eval_mean_{field_name}" for field_name in REWARD_BREAKDOWN_FIELDS})
    raw_reserved.update({f"eval_mean_{field_name}" for field_name in REWARD_EVENT_FIELDS})
    row_source = _to_scalar(normalized.get("source"))
    return {
        "source": row_source or source_name,
        "artifact_source": source_name,
        "tag": _to_scalar(normalized.get("tag")),
        "env_steps": _to_scalar(normalized.get("env_steps")),
        "learner_steps": _to_scalar(normalized.get("learner_steps")),
        "episodes": _to_scalar(normalized.get("eval_episodes")),
        "metrics": {name: _to_scalar(normalized.get(field_name)) for name, field_name in EVAL_CORE_FIELD_MAP.items()},
        "reward_breakdown": _extract_fields(normalized, REWARD_BREAKDOWN_FIELDS, "eval_mean_"),
        "reward_events": _extract_fields(normalized, REWARD_EVENT_FIELDS, "eval_mean_"),
        "semantic_monitoring": _extract_dynamic_metric_fields(
            normalized,
            prefix="eval_mean_",
            reserved_fields={field_name.removeprefix("eval_mean_") for field_name in raw_reserved},
        ),
        "raw_row": _json_safe(normalized),
    }


def _best_eval_score(row: Mapping[str, Any]) -> tuple[float, float, float]:
    normalized = _apply_row_aliases(row)
    success = _to_scalar(normalized.get("eval_success_rate"))
    coverage = _to_scalar(normalized.get("eval_mean_coverage"))
    reward = _to_scalar(normalized.get("eval_mean_reward"))
    success_value = float(success) if isinstance(success, (int, float)) else float("-inf")
    coverage_value = float(coverage) if isinstance(coverage, (int, float)) else float("-inf")
    reward_value = float(reward) if isinstance(reward, (int, float)) else float("-inf")
    return (success_value, coverage_value, reward_value)


def select_best_eval_row(eval_rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not eval_rows:
        return None
    return max(eval_rows, key=_best_eval_score)


def _metric_from_block(block: Mapping[str, Any], metric_name: str) -> Any:
    metrics = block.get("metrics", {})
    if not isinstance(metrics, Mapping):
        return None
    return _to_scalar(metrics.get(metric_name))


def _reward_event_from_block(block: Mapping[str, Any], metric_name: str) -> Any:
    reward_events = block.get("reward_events", {})
    if not isinstance(reward_events, Mapping):
        return None
    return _to_scalar(reward_events.get(metric_name))


def _build_unified_metric_table(metric_blocks: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    def metric_table(metric_names: Mapping[str, str], getter) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        for metric_name, direction in metric_names.items():
            values = {
                source_name: getter(block, metric_name)
                for source_name, block in metric_blocks.items()
            }
            payload[metric_name] = {
                "direction": direction,
                **values,
            }
        return payload

    semantic_keys: set[str] = set()
    for block in metric_blocks.values():
        semantic_monitoring = block.get("semantic_monitoring", {})
        if isinstance(semantic_monitoring, Mapping):
            semantic_keys.update(str(key) for key in semantic_monitoring.keys())

    semantic_payload = {}
    for metric_name in sorted(semantic_keys):
        values = {
            source_name: _to_scalar(block.get("semantic_monitoring", {}).get(metric_name))
            for source_name, block in metric_blocks.items()
        }
        semantic_payload[metric_name] = {
            "direction": "context_dependent_monitoring",
            **values,
        }

    return {
        "primary_metrics": metric_table(PRIMARY_UNIFIED_METRICS, lambda block, name: _metric_from_block(block, name)),
        "secondary_metrics": metric_table(SECONDARY_UNIFIED_METRICS, lambda block, name: _metric_from_block(block, name)),
        "stability_metrics": metric_table(STABILITY_UNIFIED_METRICS, lambda block, name: _reward_event_from_block(block, name)),
        "semantic_monitoring": semantic_payload,
    }


def _late_stage_window(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        return []
    count = max(4, int(math.ceil(len(rows) * 0.25)))
    return rows[-min(len(rows), count):]


def _linear_slope_per_1k_env_steps(rows: list[dict[str, Any]], metric_key: str) -> float | None:
    points: list[tuple[float, float]] = []
    for row in rows:
        normalized = _apply_row_aliases(row)
        env_steps = _to_scalar(normalized.get("env_steps"))
        metric_value = _to_scalar(normalized.get(metric_key))
        if isinstance(env_steps, (int, float)) and isinstance(metric_value, (int, float)):
            points.append((float(env_steps), float(metric_value)))
    if len(points) < 2:
        return None
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    x_mean = statistics.fmean(xs)
    y_mean = statistics.fmean(ys)
    denominator = sum((x - x_mean) ** 2 for x in xs)
    if denominator <= 0.0:
        return None
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in points)
    return float((numerator / denominator) * 1000.0)


def _late_stage_variance(rows: list[dict[str, Any]], metric_key: str) -> float | None:
    values: list[float] = []
    for row in rows:
        normalized = _apply_row_aliases(row)
        metric_value = _to_scalar(normalized.get(metric_key))
        if isinstance(metric_value, (int, float)):
            values.append(float(metric_value))
    if not values:
        return None
    if len(values) == 1:
        return 0.0
    return float(statistics.pvariance(values))


def _env_step_range(rows: list[dict[str, Any]]) -> tuple[float | None, float | None]:
    values = [
        float(value)
        for row in rows
        for value in [_to_scalar(_apply_row_aliases(row).get("env_steps"))]
        if isinstance(value, (int, float))
    ]
    if not values:
        return None, None
    return min(values), max(values)


def _rows_between_env_steps(rows: list[dict[str, Any]], start: float, end: float) -> list[dict[str, Any]]:
    selected = []
    for row in rows:
        env_steps = _to_scalar(_apply_row_aliases(row).get("env_steps"))
        if isinstance(env_steps, (int, float)) and float(start) <= float(env_steps) <= float(end):
            selected.append(row)
    return selected


def _train_dynamics_phase_windows(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    if not rows:
        return {"full": [], "early": [], "mid": [], "late": [], "last_window": []}
    start, end = _env_step_range(rows)
    if start is None or end is None or end <= start:
        return {"full": rows, "early": rows, "mid": [], "late": [], "last_window": _late_stage_window(rows)}
    width = float(end) - float(start)
    early_end = float(start) + width / 3.0
    mid_end = float(start) + (2.0 * width / 3.0)
    last_window_start = max(float(start), float(end) - max(50_000.0, width * 0.20))
    return {
        "full": rows,
        "early": _rows_between_env_steps(rows, float(start), early_end),
        "mid": _rows_between_env_steps(rows, early_end, mid_end),
        "late": _rows_between_env_steps(rows, mid_end, float(end)),
        "last_window": _rows_between_env_steps(rows, last_window_start, float(end)),
    }


def _build_train_slope_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    windows = _train_dynamics_phase_windows(rows)
    payload: dict[str, Any] = {}
    for window_name, window_rows in windows.items():
        start, end = _env_step_range(window_rows)
        payload[window_name] = {
            "row_count": len(window_rows),
            "start_env_steps": start,
            "end_env_steps": end,
            "slope_per_1000_env_steps": {
                metric_name: _linear_slope_per_1k_env_steps(window_rows, field_name)
                for metric_name, field_name in TRAIN_DYNAMICS_SLOPE_FIELDS.items()
            },
        }
    return payload


def _threshold_reach_steps(
    rows: list[dict[str, Any]],
    *,
    metric_key: str,
    threshold: float,
    higher_is_better: bool,
) -> int | None:
    for row in rows:
        normalized = _apply_row_aliases(row)
        env_steps = _to_scalar(normalized.get("env_steps"))
        metric_value = _to_scalar(normalized.get(metric_key))
        if not isinstance(env_steps, (int, float)) or not isinstance(metric_value, (int, float)):
            continue
        meets_threshold = float(metric_value) >= float(threshold) if higher_is_better else float(metric_value) <= float(threshold)
        if meets_threshold:
            return int(env_steps)
    return None


def _build_train_final_consistency_summary(
    recent_train: Mapping[str, Any],
    final_probe: Mapping[str, Any],
) -> dict[str, Any]:
    recent_metrics = recent_train.get("metrics", {}) if isinstance(recent_train, Mapping) else {}
    final_metrics = final_probe.get("metrics", {}) if isinstance(final_probe, Mapping) else {}
    details: dict[str, Any] = {}
    counts = {
        "aligned": 0,
        "final_probe_stronger": 0,
        "final_probe_weaker": 0,
        "insufficient_evidence": 0,
    }
    for metric_name, tolerance in TRAIN_FINAL_CONSISTENCY_TOLERANCES.items():
        recent_value = _to_scalar(recent_metrics.get(metric_name))
        final_value = _to_scalar(final_metrics.get(metric_name))
        if not isinstance(recent_value, (int, float)) or not isinstance(final_value, (int, float)):
            counts["insufficient_evidence"] += 1
            details[metric_name] = {
                "recent_train": recent_value,
                "final_probe": final_value,
                "delta": None,
                "verdict": "insufficient_evidence",
            }
            continue
        delta = float(final_value) - float(recent_value)
        if abs(delta) <= float(tolerance):
            verdict = "aligned"
        else:
            higher_is_better = TRAIN_FINAL_CONSISTENCY_DIRECTIONS[metric_name]
            strengthened = delta > 0.0 if higher_is_better else delta < 0.0
            verdict = "final_probe_stronger" if strengthened else "final_probe_weaker"
        counts[verdict] += 1
        details[metric_name] = {
            "recent_train": float(recent_value),
            "final_probe": float(final_value),
            "delta": float(delta),
            "verdict": verdict,
        }

    if counts["insufficient_evidence"] == len(TRAIN_FINAL_CONSISTENCY_TOLERANCES):
        verdict = "insufficient_evidence"
    elif counts["final_probe_weaker"] >= 2:
        verdict = "diverges_from_final_probe"
    elif counts["final_probe_weaker"] == 0 and (counts["aligned"] + counts["final_probe_stronger"]) >= 4:
        verdict = "supports_final_probe"
    else:
        verdict = "mixed"

    notes: list[str] = []
    weaker_metrics = [name for name, payload in details.items() if payload.get("verdict") == "final_probe_weaker"]
    stronger_metrics = [name for name, payload in details.items() if payload.get("verdict") == "final_probe_stronger"]
    if verdict == "supports_final_probe":
        notes.append("recent_train and final_probe are directionally consistent on the tracked training-quality metrics.")
    elif verdict == "diverges_from_final_probe":
        notes.append("recent_train is materially stronger than held-out final_probe on multiple metrics.")
    elif verdict == "mixed":
        notes.append("recent_train and final_probe are mixed across the tracked quality metrics.")
    else:
        notes.append("recent_train versus final_probe consistency is unavailable because one or more metrics are missing.")
    if weaker_metrics:
        notes.append(f"final_probe underperformed recent_train on: {', '.join(sorted(weaker_metrics))}.")
    if stronger_metrics:
        notes.append(f"final_probe outperformed recent_train on: {', '.join(sorted(stronger_metrics))}.")
    return {
        "verdict": verdict,
        "details": details,
        "counts": counts,
        "notes": notes,
    }


def _build_eval_gap_summary(
    best_eval: Mapping[str, Any],
    last_eval: Mapping[str, Any],
) -> dict[str, Any]:
    best_metrics = best_eval.get("metrics", {}) if isinstance(best_eval, Mapping) else {}
    last_metrics = last_eval.get("metrics", {}) if isinstance(last_eval, Mapping) else {}
    details: dict[str, Any] = {}
    for metric_name in TRAIN_FINAL_CONSISTENCY_DIRECTIONS:
        best_value = _to_scalar(best_metrics.get(metric_name))
        last_value = _to_scalar(last_metrics.get(metric_name))
        details[metric_name] = {
            "best": best_value,
            "last": last_value,
            "best_minus_last": (
                float(best_value) - float(last_value)
                if isinstance(best_value, (int, float)) and isinstance(last_value, (int, float))
                else None
            ),
        }
    return {
        "role": "diagnostic_training_endpoint_drift_only",
        "direction_note": "Positive best_minus_last is better for reward/coverage/success; negative is better for episode_length/repeat_visit_ratio.",
        "details": details,
    }


def _build_training_dynamics_summary(
    *,
    train_step_rows: list[dict[str, Any]],
    recent_train: Mapping[str, Any],
    final_probe: Mapping[str, Any],
) -> dict[str, Any]:
    late_stage_rows = _late_stage_window(train_step_rows)
    consistency_summary = _build_train_final_consistency_summary(recent_train, final_probe)
    phase_slope_summary = _build_train_slope_summary(train_step_rows)
    return {
        "final_window_stats": {
            "recent_mean_reward": _metric_from_block(recent_train, "reward"),
            "recent_mean_coverage": _metric_from_block(recent_train, "coverage"),
            "recent_success_rate": _metric_from_block(recent_train, "success_rate"),
            "recent_mean_episode_length": _metric_from_block(recent_train, "episode_length"),
            "recent_mean_repeat_visit_ratio": _metric_from_block(recent_train, "repeat_visit_ratio"),
        },
        "growth_rates": {
            "growth_rate_reward": _linear_slope_per_1k_env_steps(late_stage_rows, "recent_mean_reward"),
            "growth_rate_coverage": _linear_slope_per_1k_env_steps(late_stage_rows, "recent_mean_coverage"),
            "growth_rate_success_rate": _linear_slope_per_1k_env_steps(late_stage_rows, "recent_success_rate"),
            "growth_rate_repeat_visit_ratio": _linear_slope_per_1k_env_steps(late_stage_rows, "recent_mean_repeat_visit_ratio"),
        },
        "threshold_reach_steps": {
            "threshold_reach_steps_success_050": _threshold_reach_steps(
                train_step_rows,
                metric_key="recent_success_rate",
                threshold=0.50,
                higher_is_better=True,
            ),
            "threshold_reach_steps_coverage_090": _threshold_reach_steps(
                train_step_rows,
                metric_key="recent_mean_coverage",
                threshold=0.90,
                higher_is_better=True,
            ),
            "threshold_reach_steps_reward_custom": None,
        },
        "late_stage_variance": {
            "late_stage_variance_reward": _late_stage_variance(late_stage_rows, "recent_mean_reward"),
            "late_stage_variance_coverage": _late_stage_variance(late_stage_rows, "recent_mean_coverage"),
            "late_stage_variance_success_rate": _late_stage_variance(late_stage_rows, "recent_success_rate"),
            "late_stage_variance_repeat_visit_ratio": _late_stage_variance(late_stage_rows, "recent_mean_repeat_visit_ratio"),
        },
        "train_final_consistency": consistency_summary,
        "phase_slope_summary": phase_slope_summary,
        "late_stage_window": {
            "definition": "last_25_percent_of_train_step_logging_rows_min_4_rows",
            "row_count": len(late_stage_rows),
            "fraction_of_logging_points": 0.25 if train_step_rows else None,
            "start_env_steps": _to_scalar((late_stage_rows[0] if late_stage_rows else {}).get("env_steps")),
            "end_env_steps": _to_scalar((late_stage_rows[-1] if late_stage_rows else {}).get("env_steps")),
        },
        "definitions": {
            "growth_rate_unit": "least_squares_slope_per_1000_env_steps_over_late_stage_window",
            "phase_slope_summary_unit": "least_squares_slope_per_1000_env_steps; includes full/early/mid/late/last_window for future training-first comparisons",
            "reward_threshold_note": "threshold_reach_steps_reward_custom is intentionally left null in v2 because a stable formal reward threshold has not been fixed.",
            "late_stage_variance_window": "last_25_percent_of_train_step_logging_rows_min_4_rows",
            "formal_acceptance_note": "future comparisons should prioritize training dynamics first; final formal acceptance still comes from best.pt held-out final test.",
        },
        "notes": consistency_summary.get("notes", []),
    }


def build_observed_run_contract(
    *,
    run_dir: Path,
    recent_train_row: Mapping[str, Any] | None = None,
    final_probe_row: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    logs_dir = run_dir / "logs"
    final_env_steps = _to_scalar((final_probe_row or {}).get("env_steps")) or _to_scalar((recent_train_row or {}).get("env_steps"))
    final_train_episode_idx = _to_scalar((final_probe_row or {}).get("completed_train_episodes")) or _to_scalar((recent_train_row or {}).get("completed_train_episodes"))
    observed_budget_mode = _to_scalar((recent_train_row or {}).get("budget_mode")) or _to_scalar((final_probe_row or {}).get("budget_mode"))

    if final_env_steps is None:
        final_probe_rows = _read_csv_rows(logs_dir / "final_probe.csv")
        train_steps_rows = _read_csv_rows(logs_dir / "train_steps.csv")
        if final_probe_rows:
            final_env_steps = _to_scalar(_apply_row_aliases(final_probe_rows[-1]).get("env_steps"))
        elif train_steps_rows:
            final_env_steps = _to_scalar(_apply_row_aliases(train_steps_rows[-1]).get("env_steps"))

    return {
        "budget_mode": observed_budget_mode,
        "final_env_steps": final_env_steps,
        "final_train_episode_idx": final_train_episode_idx,
        "train_episodes_header": _read_csv_header(logs_dir / "train_episodes.csv"),
        "train_steps_header": _read_csv_header(logs_dir / "train_steps.csv"),
        "model_select_eval_header": _read_csv_header(logs_dir / "model_select_eval.csv"),
        "best_recheck_eval_header": _read_csv_header(logs_dir / "best_recheck_eval.csv"),
        "final_probe_header": _read_csv_header(logs_dir / "final_probe.csv"),
    }


def build_metric_snapshot(
    *,
    run_dir: Path,
    recent_train_row: Mapping[str, Any] | None,
    final_probe_row: Mapping[str, Any] | None,
    last_checkpoint_env_steps: int | None,
    last_checkpoint_train_episode_idx: int | None,
    best_checkpoint_env_steps: int | None,
    best_checkpoint_train_episode_idx: int | None,
    final_probe_source: str,
    source_of_truth_repo: str,
    formal_protocol_revision: str | None = None,
    last_eval_row: Mapping[str, Any] | None = None,
    best_eval_row: Mapping[str, Any] | None = None,
    model_select_rows: list[Mapping[str, Any]] | None = None,
    best_recheck_rows: list[Mapping[str, Any]] | None = None,
    best_checkpoint_selection_row: Mapping[str, Any] | None = None,
    last_checkpoint_diagnostic_row: Mapping[str, Any] | None = None,
    insufficient_evidence_flags: list[str] | None = None,
) -> dict[str, Any]:
    logs_dir = run_dir / "logs"
    protocol_revision = str(formal_protocol_revision or CURRENT_FORMAL_PROTOCOL_REVISION)
    posthoc_protocol = _is_posthoc_protocol(protocol_revision)
    model_select_csv_rows = _read_csv_rows(logs_dir / "model_select_eval.csv")
    best_recheck_csv_rows = _read_csv_rows(logs_dir / "best_recheck_eval.csv")
    posthoc_selection_summary = _read_json(logs_dir / "posthoc_selection_summary.json")
    posthoc_final_probe_summary = _read_json(logs_dir / "final_probe_summary.json")
    posthoc_best_vs_last_summary = _read_json(logs_dir / "best_vs_last_gap_summary.json")
    formal_selection_manifest = _read_json(logs_dir / "formal_selection_manifest.json")
    model_select_rows = list(model_select_rows or model_select_csv_rows)
    best_recheck_rows = list(best_recheck_rows or best_recheck_csv_rows)
    if best_checkpoint_selection_row is None and best_recheck_rows:
        best_checkpoint_selection_row = select_best_eval_row([dict(row) for row in best_recheck_rows])
    if best_checkpoint_selection_row is None and model_select_rows:
        best_checkpoint_selection_row = select_best_eval_row([dict(row) for row in model_select_rows])
    if last_checkpoint_diagnostic_row is None:
        last_candidates = [
            row for row in best_recheck_rows
            if _to_scalar(_apply_row_aliases(row).get("env_steps")) == last_checkpoint_env_steps
        ]
        if not last_candidates:
            last_candidates = [
                row for row in model_select_rows
                if _to_scalar(_apply_row_aliases(row).get("env_steps")) == last_checkpoint_env_steps
            ]
        last_checkpoint_diagnostic_row = last_candidates[-1] if last_candidates else None

    eval_rows = _read_csv_rows(logs_dir / "eval_metrics.csv")
    if last_eval_row is None and eval_rows:
        last_eval_row = eval_rows[-1]
    if best_eval_row is None and eval_rows:
        best_eval_row = select_best_eval_row(eval_rows)

    recent_train = _normalize_recent_train(recent_train_row)
    final_probe = _normalize_eval_like(final_probe_row, source_name="logs/final_probe.csv")
    model_select_best = (
        _normalize_eval_like(select_best_eval_row([dict(row) for row in model_select_rows]), source_name="logs/model_select_eval.csv::best")
        if model_select_rows else {}
    )
    best_recheck_best = (
        _normalize_eval_like(best_checkpoint_selection_row, source_name="logs/best_recheck_eval.csv::selected_best")
        if best_checkpoint_selection_row else {}
    )
    diagnostic_last_checkpoint = (
        _normalize_eval_like(last_checkpoint_diagnostic_row, source_name="logs/model_select_or_recheck.csv::last_checkpoint")
        if last_checkpoint_diagnostic_row else {}
    )
    legacy_last_eval = _normalize_eval_like(last_eval_row, source_name="logs/eval_metrics.csv") if last_eval_row else {}
    legacy_best_eval = _normalize_eval_like(best_eval_row, source_name="logs/eval_metrics.csv::best_eval") if best_eval_row else {}
    best_checkpoint_exists = (run_dir / "checkpoints" / "best.pt").exists()
    legacy_context_available = bool(eval_rows) or bool(legacy_last_eval) or bool(legacy_best_eval)
    train_step_rows = _read_csv_rows(logs_dir / "train_steps.csv")
    training_dynamics_summary = _build_training_dynamics_summary(
        train_step_rows=train_step_rows,
        recent_train=recent_train,
        final_probe=final_probe,
    )
    consistency_summary = training_dynamics_summary.get("train_final_consistency", {})

    metric_blocks = {
        "recent_train": recent_train,
        "final_probe": final_probe,
        "model_select_best": model_select_best,
        "best_recheck_best": best_recheck_best,
        "diagnostic_last_checkpoint": diagnostic_last_checkpoint,
    }

    if posthoc_protocol:
        checkpoint_validation_summary = {
            "role": "disabled_in_formal_posthoc_trainselect_v1",
            "artifact_source": None,
            "row_count": 0,
            "training_during_validation_episodes": 0,
        }
        best_checkpoint_selection_summary = {
            "artifact_source": "logs/final_probe_summary.json",
            "row_count": len(_read_csv_rows(logs_dir / "final_probe.csv")),
            "selection_rule": "train_side_smoothed_topk_then_heldout_success_rate_coverage_reward",
            "posthoc_candidate_selection": posthoc_selection_summary,
            "posthoc_final_probe_summary": posthoc_final_probe_summary,
            "selected_best": final_probe,
            "best_checkpoint_env_steps": best_checkpoint_env_steps,
            "best_checkpoint_train_episode_idx": best_checkpoint_train_episode_idx,
            "best_checkpoint_path": "checkpoints/best.pt",
        }
        diagnostic_eval_rule = (
            "last.pt final_probe row is present only if the last checkpoint was selected into post-hoc top-k"
        )
        best_vs_last_summary = posthoc_best_vs_last_summary or _build_eval_gap_summary(final_probe, diagnostic_last_checkpoint)
    else:
        checkpoint_validation_summary = {
            "artifact_source": "logs/model_select_eval.csv",
            "row_count": len(model_select_rows),
            "selection_rule": "success_rate_then_coverage_then_reward",
            "seed_base": _to_scalar((model_select_rows[0] if model_select_rows else {}).get("seed_base")),
            "episodes_per_eval": _to_scalar((model_select_rows[0] if model_select_rows else {}).get("eval_episodes")),
            "best_periodic_validation": model_select_best,
        }
        best_checkpoint_selection_summary = {
            "artifact_source": "logs/best_recheck_eval.csv",
            "row_count": len(best_recheck_rows),
            "selection_rule": "success_rate_then_coverage_then_reward",
            "seed_base": _to_scalar((best_recheck_rows[0] if best_recheck_rows else {}).get("seed_base")),
            "episodes_per_recheck": _to_scalar((best_recheck_rows[0] if best_recheck_rows else {}).get("eval_episodes")),
            "selected_best": best_recheck_best,
            "best_checkpoint_env_steps": best_checkpoint_env_steps,
            "best_checkpoint_train_episode_idx": best_checkpoint_train_episode_idx,
            "best_checkpoint_path": "checkpoints/best.pt",
        }
        diagnostic_eval_rule = "reuse_500k_checkpoint_validation_or_topk_recheck_when_available; no separate last.pt final test"
        best_vs_last_summary = _build_eval_gap_summary(best_recheck_best, diagnostic_last_checkpoint)

    payload = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "metric_snapshot",
        "experiment_mode": "formal_train",
        "formal_protocol_revision": protocol_revision,
        "source_of_truth_repo": source_of_truth_repo,
        "run_dir": str(run_dir.resolve()),
        "generated_at": _now_iso(),
        "recent_train": recent_train,
        "checkpoint_validation_summary": checkpoint_validation_summary,
        "best_checkpoint_selection_summary": best_checkpoint_selection_summary,
        "posthoc_candidate_selection_summary": posthoc_selection_summary if posthoc_protocol else {},
        "posthoc_final_probe_summary": posthoc_final_probe_summary if posthoc_protocol else {},
        "formal_selection_manifest": formal_selection_manifest if posthoc_protocol else {},
        "final_probe": final_probe,
        "diagnostic_last_checkpoint_summary": {
            "role": "diagnostic_only_training_endpoint",
            "checkpoint_path": "checkpoints/last.pt",
            "env_steps": last_checkpoint_env_steps,
            "train_episode_idx": last_checkpoint_train_episode_idx,
            "evaluation_reuse_rule": diagnostic_eval_rule,
            "diagnostic_eval": diagnostic_last_checkpoint,
        },
        "best_vs_last_gap_summary": best_vs_last_summary,
        "training_dynamics_summary": training_dynamics_summary,
        "train_final_consistency_summary": consistency_summary,
        "recent_train_support_summary": consistency_summary,
        "last_checkpoint_env_steps": last_checkpoint_env_steps,
        "last_checkpoint_train_episode_idx": last_checkpoint_train_episode_idx,
        "best_checkpoint_env_steps": best_checkpoint_env_steps,
        "best_checkpoint_train_episode_idx": best_checkpoint_train_episode_idx,
        "final_probe_source": final_probe_source,
        "formal_final_object": {
            "checkpoint_path": "checkpoints/best.pt",
            "network_source": final_probe_source,
            "env_steps": best_checkpoint_env_steps,
            "train_episode_idx": best_checkpoint_train_episode_idx,
            "evaluation_artifact": "logs/final_probe.csv",
            "role": "formal_acceptance_object",
        },
        "automatic_tuning_ranking_basis": {
            "final_network_outcome": [
                "success_rate",
                "coverage",
                "reward",
                "episode_length",
                "repeat_visit_ratio",
            ],
            "training_dynamics_quality": [
                "phase_slope_summary",
                "recent_mean_reward",
                "recent_mean_coverage",
                "recent_success_rate",
                "recent_mean_episode_length",
                "recent_mean_repeat_visit_ratio",
                "growth_rate",
                "threshold_reach_steps",
                "late_stage_variance",
                "train_final_consistency",
            ],
        },
        "unified_metrics": _build_unified_metric_table(metric_blocks),
        "insufficient_evidence_flags": sorted(set(insufficient_evidence_flags or [])),
    }
    if legacy_context_available:
        payload["legacy_diagnostic_context"] = {
            "periodic_eval_available": bool(eval_rows),
            "legacy_best_checkpoint_exists": best_checkpoint_exists,
            "role": "legacy_diagnostic_only",
        }
        if legacy_last_eval:
            payload["legacy_last_eval"] = legacy_last_eval
        if legacy_best_eval:
            payload["legacy_best_eval"] = legacy_best_eval
    return payload


def _structured_timing_stats(component_name: str, stats: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if not stats:
        return None
    numeric_stats = {
        str(key): float(value)
        for key, value in stats.items()
        if isinstance(value, (int, float)) and float(value) > 0.0
    }
    if not numeric_stats:
        return None
    total = numeric_stats.get("total_time_sec")
    if total is None:
        total = sum(value for key, value in numeric_stats.items() if key != "total_time_sec")
    return _json_safe(
        {
            "component": component_name,
            "total_time_sec": total,
            "breakdown_sec": numeric_stats,
        }
    )


def _collect_timing_stats(
    *,
    collector: Any | None = None,
    learner: Any | None = None,
    replay: Any | None = None,
    state_adapter: Any | None = None,
) -> dict[str, Any]:
    timing_summary: dict[str, Any] = {}
    if collector is not None and hasattr(collector, "get_timing_stats"):
        timing_summary["collector"] = _structured_timing_stats("collector", collector.get_timing_stats())
        cum_map = getattr(collector, "cum_map", None)
        if cum_map is not None and hasattr(cum_map, "get_timing_stats"):
            timing_summary["cummap"] = _structured_timing_stats("cummap", cum_map.get_timing_stats())
    if learner is not None and hasattr(learner, "get_timing_stats"):
        timing_summary["learner"] = _structured_timing_stats("learner", learner.get_timing_stats())
    if replay is not None and hasattr(replay, "get_timing_stats"):
        timing_summary["replay"] = _structured_timing_stats("replay", replay.get_timing_stats())
    if state_adapter is not None and hasattr(state_adapter, "get_timing_stats"):
        timing_summary["adapter"] = _structured_timing_stats("adapter", state_adapter.get_timing_stats())
        semantic_builder = getattr(state_adapter, "shared_semantic_layer", None)
        if semantic_builder is not None and hasattr(semantic_builder, "get_timing_stats"):
            timing_summary["semantic"] = _structured_timing_stats("semantic", semantic_builder.get_timing_stats())
        advantage_builder = getattr(state_adapter, "advantage_builder", None)
        if advantage_builder is not None and hasattr(advantage_builder, "get_timing_stats"):
            timing_summary["advantage"] = _structured_timing_stats("advantage", advantage_builder.get_timing_stats())
        value_builder = getattr(state_adapter, "value_builder", None)
        if value_builder is not None and hasattr(value_builder, "get_timing_stats"):
            timing_summary["value"] = _structured_timing_stats("value", value_builder.get_timing_stats())
    return timing_summary


def build_benchmark_summary(
    *,
    cfg: Any | None,
    run_dir: Path,
    run_mode: str,
    total_runtime_sec: float | None,
    total_runtime_hms: str | None,
    total_train_episodes_completed: int | None,
    source_of_truth_repo: str,
    best_checkpoint_env_steps: int | None = None,
    last_checkpoint_env_steps: int | None = None,
    model_selection_eval_count: int | None = None,
    recheck_eval_count: int | None = None,
    collector: Any | None = None,
    learner: Any | None = None,
    replay: Any | None = None,
    state_adapter: Any | None = None,
    insufficient_evidence_flags: list[str] | None = None,
) -> dict[str, Any]:
    config_dict = _serialize_config(cfg) or {}
    protocol_revision = _formal_protocol_revision(config_dict)
    runtime_flags = {field_name: config_dict.get(field_name) for field_name in RUNTIME_ONLY_FIELDS if field_name in config_dict}
    timing_flags = {field_name: config_dict.get(field_name) for field_name in TIMING_FLAG_FIELDS if field_name in config_dict}
    timing_summary = _collect_timing_stats(
        collector=collector,
        learner=learner,
        replay=replay,
        state_adapter=state_adapter,
    )
    flags = list(insufficient_evidence_flags or [])
    if not timing_summary:
        flags.append("timing_summary_unavailable")
    if total_runtime_sec is None:
        flags.append("total_runtime_unavailable")

    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "benchmark_summary",
        "experiment_mode": "formal_train",
        "source_of_truth_repo": source_of_truth_repo,
        "run_dir": str(run_dir.resolve()),
        "generated_at": _now_iso(),
        "run_mode": run_mode,
        "total_runtime_sec": total_runtime_sec,
        "total_runtime_hms": total_runtime_hms,
        "runtime_performance_switches": runtime_flags,
        "timing_switches": timing_flags,
        "timing_summary": timing_summary,
        "formal_protocol_revision": protocol_revision,
        "budget_mode": config_dict.get("budget_mode"),
        "best_checkpoint_env_steps": best_checkpoint_env_steps,
        "last_checkpoint_env_steps": last_checkpoint_env_steps,
        "model_selection_eval_count": model_selection_eval_count,
        "recheck_eval_count": recheck_eval_count,
        "diagnostic_env_steps_to_best": best_checkpoint_env_steps,
        "diagnostic_train_episodes_to_best": None,
        "env_steps_to_best": best_checkpoint_env_steps,
        "train_episodes_to_best": None,
        "total_train_episodes_completed": total_train_episodes_completed,
        "insufficient_evidence_flags": sorted(set(flags)),
    }


def _comparability_sections(config_dict: Mapping[str, Any], protocol_revision: str) -> dict[str, Any]:
    frozen_fields = {
        field_name: config_dict.get(field_name)
        for field_name in FROZEN_COMPARABILITY_FIELDS
        if field_name in config_dict
    }
    allowed_tuning = {
        field_name: config_dict.get(field_name)
        for field_name in ALLOWED_TUNING_FIELDS
        if field_name in config_dict
    }
    manual_review = {
        field_name: config_dict.get(field_name)
        for field_name in MANUAL_REVIEW_FIELDS
        if field_name in config_dict
    }
    group_seed = json.dumps(_json_safe(frozen_fields), ensure_ascii=False, sort_keys=True).encode("utf-8")
    group_hash = hashlib.sha1(group_seed).hexdigest()[:12]
    return {
        "comparability_group": f"{protocol_revision}__{group_hash}",
        "frozen_fields": frozen_fields,
        "allowed_tuning_fields": allowed_tuning,
        "manual_review_fields": manual_review,
    }


def build_config_snapshot(
    *,
    cfg: Any | None,
    run_dir: Path,
    run_mode: str,
    source_of_truth_repo: str,
    observed_run_contract: Mapping[str, Any] | None = None,
    baseline_identifier: str = DEFAULT_MAIN_BASELINE_IDENTIFIER,
    insufficient_evidence_flags: list[str] | None = None,
) -> dict[str, Any]:
    repo_dir = Path(source_of_truth_repo)
    config_dict = _serialize_config(cfg) or {}
    protocol_revision = _formal_protocol_revision(config_dict)
    posthoc_protocol = _is_posthoc_protocol(protocol_revision)
    comparability_sections = _comparability_sections(config_dict, protocol_revision)
    flags = list(insufficient_evidence_flags or [])
    if not config_dict:
        flags.append("train_config_unavailable")

    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "config_snapshot",
        "experiment_mode": "formal_train",
        "source_of_truth_repo": source_of_truth_repo,
        "run_dir": str(run_dir.resolve()),
        "generated_at": _now_iso(),
        "run_mode": run_mode,
        "git_commit_sha": _git_output(repo_dir, ["rev-parse", "HEAD"]),
        "git_branch": _git_output(repo_dir, ["rev-parse", "--abbrev-ref", "HEAD"]),
        "full_train_config": config_dict or None,
        "default_main_baseline": {
            "identifier": baseline_identifier,
            "evidence_status": "bootstrap_inference_from_local_run_inventory",
            "note": (
                "Historical outputs predate formal snapshots. "
                "This baseline identifier is a bootstrap pointer until a fully snapshotted formal baseline round is published."
            ),
        },
        "comparability": comparability_sections,
        "runtime_only_fields": {field_name: config_dict.get(field_name) for field_name in RUNTIME_ONLY_FIELDS if field_name in config_dict},
        "observed_run_contract": _json_safe(dict(observed_run_contract or {})),
        "evaluation_contract": {
            "protocol_revision": protocol_revision,
            "formal_final_object": {
                "checkpoint_path": "checkpoints/best.pt",
                "acceptance_target": (
                    "held_out_final_probe_of_posthoc_train_side_selected_winner"
                    if posthoc_protocol
                    else "held_out_final_probe_of_validation_selected_best_checkpoint"
                ),
                "role": "formal_acceptance_object",
            },
            "training_first_ranking_context": {
                "role": "first_class_ranking_and_diagnostic_evidence",
                "note": (
                    "post-hoc protocol uses train-side smoothed candidate scoring before a single held-out final probe"
                    if posthoc_protocol
                    else "legacy protocol inspects train dynamics, checkpoint validation, recheck, and best-vs-last drift"
                ),
            },
            "training_during_eval_rule": (
                {
                    "validation_during_training": False,
                    "recheck_during_training": False,
                    "final_probe_during_training": False,
                    "collector_pause_for_eval": False,
                    "checkpoint_behavior": "save_periodic_train_only_checkpoints",
                }
                if posthoc_protocol
                else {
                    "validation_during_training": bool(config_dict.get("enable_best_checkpoint_selection")),
                    "compatibility_role": "legacy_best_checkpoint_v3",
                }
            ),
            "posthoc_candidate_selection_rule": (
                {
                    "candidate_score_csv": "logs/posthoc_candidate_scores.csv",
                    "selection_summary_json": "logs/posthoc_selection_summary.json",
                    "candidate_checkpoint_pattern": "checkpoints/ckpt_step_<env_steps>.pt",
                    "checkpoint_interval_env_steps": config_dict.get("periodic_checkpoint_interval_env_steps"),
                    "candidate_start_env_steps": config_dict.get("posthoc_candidate_start_env_steps"),
                    "candidate_end_env_steps": (
                        config_dict.get("total_env_steps")
                        if not config_dict.get("posthoc_candidate_end_env_steps")
                        else config_dict.get("posthoc_candidate_end_env_steps")
                    ),
                    "selection_window_env_steps": config_dict.get("posthoc_selection_window_env_steps"),
                    "topk": config_dict.get("posthoc_final_probe_topk"),
                    "train_side_only": True,
                    "weights": {
                        "reward_z": 0.35,
                        "coverage_z": 0.25,
                        "success_rate_z": 0.20,
                        "episode_length_z": -0.10,
                        "repeat_visit_ratio_z": -0.10,
                    },
                }
                if posthoc_protocol
                else None
            ),
            "model_selection_rule": (
                None
                if posthoc_protocol
                else {
                    "checkpoint_path": "checkpoints/best.pt",
                    "candidate_eval_csv": "logs/model_select_eval.csv",
                    "topk_recheck_csv": "logs/best_recheck_eval.csv",
                    "selection_start_env_steps": config_dict.get("best_checkpoint_selection_start_env_steps"),
                    "selection_interval_env_steps": config_dict.get("best_checkpoint_selection_interval_env_steps"),
                    "validation_episodes": config_dict.get("best_checkpoint_validation_episodes"),
                    "topk_recheck": config_dict.get("best_checkpoint_topk_recheck"),
                    "recheck_episodes": config_dict.get("best_checkpoint_recheck_episodes"),
                    "ranking_order": ["success_rate", "coverage", "reward"],
                    "seed_toggle_field": "use_fixed_model_select_seeds",
                    "seed_base_field": "fixed_model_select_seed_base",
                    "seed_base": config_dict.get("fixed_model_select_seed_base"),
                    "seed_set_role": "checkpoint_validation_only",
                }
            ),
            "final_probe_rule": {
                "source": (
                    "posthoc_topk_candidates_only"
                    if posthoc_protocol
                    else "best_checkpoint_only"
                ),
                "held_out_seed_rule": "fixed_final_probe_seed_base_when_use_fixed_eval_seeds_final_probe_toggle_else_runtime_seed_stream",
                "seed_toggle_field": "use_fixed_eval_seeds",
                "seed_toggle_note": "legacy field name retained; it controls final held-out test seeds only",
                "seed_base_field": "fixed_final_probe_seed_base",
                "seed_base": config_dict.get("fixed_final_probe_seed_base"),
                "seed_set_role": "final_formal_test_only",
                "run_final_greedy_episodes": config_dict.get("final_greedy_episodes"),
                "posthoc_final_probe_topk": config_dict.get("posthoc_final_probe_topk") if posthoc_protocol else None,
                "current_default_formal_final_probe_episodes": CURRENT_DEFAULT_FORMAL_FINAL_PROBE_EPISODES,
                "historical_formal_final_probe_episodes": HISTORICAL_FORMAL_FINAL_PROBE_EPISODES,
                "strict_comparability_field": "final_greedy_episodes",
                "protocol_upgrade_note": (
                    "This protocol does not evaluate during training. It selects up to top-k candidates from smoothed train-side metrics, "
                    "then runs one held-out final_probe pass over those candidates and promotes the winner to best.pt."
                    if posthoc_protocol
                    else "Legacy protocol selects best.pt with dedicated model-selection seeds, then runs final_probe only on best.pt."
                ),
                "csv_file": "logs/final_probe.csv",
                "summary_json": "logs/final_probe_summary.json" if posthoc_protocol else None,
            },
            "diagnostic_last_checkpoint": {
                "checkpoint_path": "checkpoints/last.pt",
                "role": "diagnostic_only_training_endpoint",
                "evaluation_reuse_rule": (
                    "last.pt is not separately held-out probed unless it is one of the post-hoc top-k candidates"
                    if posthoc_protocol
                    else "reuse the 500k checkpoint validation or top-k recheck result when present; do not run a separate final test for last.pt"
                ),
                "gap_summary_role": "last-vs-best drift diagnostics only",
            },
            "reward_semantics": {
                "info_norm_rule": {
                    "mode": "fixed_half_perimeter",
                    "formula": "pi * scan_radius",
                    "derived_info_norm": (
                        float(math.pi * float(config_dict["scan_radius"]))
                        if config_dict.get("scan_radius") is not None else None
                    ),
                },
                "recent_revisit_horizon_rule": {
                    "source_field": "trajectory_history_steps",
                    "note": "recent revisit penalty horizon is fixed to trajectory_history_steps in phase-1 reward cleanup",
                },
                "turn_penalty_rule": {
                    "total_scale_field": "reward_turn_penalty_scale",
                    "angle_weight_fields": [
                        "reward_turn_weight_45",
                        "reward_turn_weight_90",
                        "reward_turn_weight_135",
                        "reward_turn_weight_180",
                    ],
                    "per_step_formula": "reward_turn_penalty_scale * selected_turn_weight",
                },
                "stall_penalty_status": "removed_from_formal_reward_mainline",
                "stall_diagnostic_rule": {
                    "threshold": int(STALL_DIAGNOSTIC_WINDOW),
                    "threshold_role": "internal_fixed_diagnostic_constant",
                    "trigger_definition": "count one stall_trigger_count event when consecutive zero-info steps reach the threshold",
                    "reward_effect": "no_formal_reward_effect",
                },
                "zero_info_step_count_role": "raw diagnostic event count; not a reward term",
            },
            "recent_train_role": "training_first_ranking_context",
            "automatic_tuning_ranking_basis": {
                "training_dynamics_quality": [
                    "phase_slope_summary",
                    "recent_mean_reward",
                    "recent_mean_coverage",
                    "recent_success_rate",
                    "recent_mean_episode_length",
                    "recent_mean_repeat_visit_ratio",
                    "growth_rate",
                    "threshold_reach_steps",
                    "late_stage_variance",
                    "best_vs_last_gap_summary",
                ],
                "final_best_checkpoint_outcome": {
                    "primary": ["success_rate", "coverage", "reward"],
                    "secondary": ["episode_length", "repeat_visit_ratio"],
                },
            },
            "structured_artifacts": (
                {
                    "posthoc_candidate_scores": "logs/posthoc_candidate_scores.csv",
                    "posthoc_selection_summary": "logs/posthoc_selection_summary.json",
                    "final_probe": "logs/final_probe.csv",
                    "final_probe_summary": "logs/final_probe_summary.json",
                    "best_vs_last_gap_summary": "logs/best_vs_last_gap_summary.json",
                    "formal_selection_manifest": "logs/formal_selection_manifest.json",
                }
                if posthoc_protocol
                else {
                    "model_select_eval": "logs/model_select_eval.csv",
                    "best_recheck_eval": "logs/best_recheck_eval.csv",
                    "final_probe_best": "logs/final_probe.csv",
                }
            ),
        },
        "insufficient_evidence_flags": sorted(set(flags)),
    }


def _artifact_record(run_dir: Path, relative_path: str, *, required: bool, category: str) -> dict[str, Any]:
    absolute_path = run_dir / relative_path
    return {
        "path": relative_path.replace("\\", "/"),
        "exists": absolute_path.exists(),
        "required": required,
        "category": category,
    }


def build_artifact_index(run_dir: Path) -> dict[str, Any]:
    plots_dir = run_dir / "plots"
    trajectories_dir = run_dir / "trajectories"
    plot_records = []
    if plots_dir.exists():
        plot_records.extend(
            {
                "path": path.relative_to(run_dir).as_posix(),
                "exists": True,
                "required": False,
                "category": "plots",
            }
            for path in sorted(plots_dir.rglob("*"))
            if path.is_file()
        )
    trajectory_records = []
    if trajectories_dir.exists():
        trajectory_records.extend(
            {
                "path": path.relative_to(run_dir).as_posix(),
                "exists": True,
                "required": False,
                "category": "trajectories",
            }
            for path in sorted(trajectories_dir.rglob("*"))
            if path.is_file()
        )
    model_select_checkpoint_records = []
    model_select_dir = run_dir / "checkpoints" / "model_select"
    if model_select_dir.exists():
        model_select_checkpoint_records.extend(
            {
                "path": path.relative_to(run_dir).as_posix(),
                "exists": True,
                "required": False,
                "category": "model_selection_candidate_checkpoint",
            }
            for path in sorted(model_select_dir.glob("*.pt"))
            if path.is_file()
        )
    posthoc_checkpoint_records = [
        {
            "path": path.relative_to(run_dir).as_posix(),
            "exists": True,
            "required": False,
            "category": "posthoc_candidate_checkpoint",
        }
        for path in sorted((run_dir / "checkpoints").glob("ckpt_step_*.pt"))
        if path.is_file()
    ]
    posthoc_required = (run_dir / "logs" / "formal_selection_manifest.json").exists()

    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "artifact_index",
        "experiment_mode": "formal_train",
        "run_dir": str(run_dir.resolve()),
        "generated_at": _now_iso(),
        "csv": [
            _artifact_record(run_dir, "logs/train_steps.csv", required=True, category="csv"),
            _artifact_record(run_dir, "logs/train_episodes.csv", required=True, category="csv"),
            _artifact_record(run_dir, "logs/model_select_eval.csv", required=False, category="legacy_model_selection_csv"),
            _artifact_record(run_dir, "logs/best_recheck_eval.csv", required=False, category="legacy_model_selection_csv"),
            _artifact_record(run_dir, "logs/posthoc_candidate_scores.csv", required=posthoc_required, category="posthoc_selection_csv"),
            _artifact_record(run_dir, "logs/final_probe.csv", required=True, category="csv"),
            _artifact_record(run_dir, "logs/eval_metrics.csv", required=False, category="legacy_diagnostic_csv"),
        ],
        "checkpoints": [
            _artifact_record(run_dir, "checkpoints/best.pt", required=True, category="formal_primary_checkpoint"),
            _artifact_record(run_dir, "checkpoints/last.pt", required=True, category="diagnostic_endpoint_checkpoint"),
            *model_select_checkpoint_records,
            *posthoc_checkpoint_records,
        ],
        "structured_summaries": [
            _artifact_record(run_dir, "logs/metric_snapshot.json", required=True, category="structured_summary"),
            _artifact_record(run_dir, "logs/benchmark_summary.json", required=True, category="structured_summary"),
            _artifact_record(run_dir, "logs/config_snapshot.json", required=True, category="structured_summary"),
            _artifact_record(run_dir, "logs/artifact_index.json", required=True, category="structured_summary"),
            _artifact_record(run_dir, "logs/posthoc_selection_summary.json", required=posthoc_required, category="posthoc_selection_summary"),
            _artifact_record(run_dir, "logs/final_probe_summary.json", required=posthoc_required, category="posthoc_final_summary"),
            _artifact_record(run_dir, "logs/best_vs_last_gap_summary.json", required=posthoc_required, category="posthoc_final_summary"),
            _artifact_record(run_dir, "logs/formal_selection_manifest.json", required=posthoc_required, category="posthoc_manifest"),
        ],
        "stdout_summaries": [
            _artifact_record(run_dir, "logs/training_summary.txt", required=False, category="stdout_summary"),
        ],
        "plots": plot_records,
        "trajectories": trajectory_records,
    }


def build_training_summary_text(
    *,
    run_dir: Path,
    metric_snapshot: Mapping[str, Any],
    benchmark_summary: Mapping[str, Any],
) -> str:
    recent_train = metric_snapshot.get("recent_train", {})
    final_probe = metric_snapshot.get("final_probe", {})
    training_dynamics_summary = metric_snapshot.get("training_dynamics_summary", {})
    consistency_summary = metric_snapshot.get("train_final_consistency_summary", {})
    best_selection = metric_snapshot.get("best_checkpoint_selection_summary", {})
    last_diag = metric_snapshot.get("diagnostic_last_checkpoint_summary", {})

    def line_for(block: Mapping[str, Any], label: str) -> str:
        metrics = block.get("metrics", {})
        return (
            f"{label}: "
            f"reward={metrics.get('reward')} "
            f"coverage={metrics.get('coverage')} "
            f"success_rate={metrics.get('success_rate')} "
            f"episode_length={metrics.get('episode_length')} "
            f"repeat_visit_ratio={metrics.get('repeat_visit_ratio')}"
        )

    lines = [
        "Training Summary",
        f"run_dir: {run_dir.resolve()}",
        f"run_mode: {benchmark_summary.get('run_mode')}",
        f"budget_mode: {benchmark_summary.get('budget_mode')}",
        f"total_runtime_sec: {benchmark_summary.get('total_runtime_sec')}",
        f"total_runtime_hms: {benchmark_summary.get('total_runtime_hms')}",
        f"total_train_episodes_completed: {benchmark_summary.get('total_train_episodes_completed')}",
        line_for(recent_train, "recent_train"),
        line_for(final_probe, "final_probe_best"),
        f"train_final_consistency_verdict: {consistency_summary.get('verdict')}",
        f"training_dynamics_final_window: {training_dynamics_summary.get('final_window_stats')}",
        f"training_dynamics_growth_rates: {training_dynamics_summary.get('growth_rates')}",
        f"best_checkpoint_env_steps: {metric_snapshot.get('best_checkpoint_env_steps')}",
        f"best_checkpoint_selection_summary: {best_selection}",
        f"last_checkpoint_env_steps: {metric_snapshot.get('last_checkpoint_env_steps')}",
        f"last_checkpoint_train_episode_idx: {metric_snapshot.get('last_checkpoint_train_episode_idx')}",
        f"diagnostic_last_checkpoint_summary: {last_diag}",
        f"final_probe_source: {metric_snapshot.get('final_probe_source')}",
    ]
    return "\n".join(lines) + "\n"


def write_formal_run_artifacts(
    *,
    run_dir: Path,
    cfg: Any | None,
    run_mode: str,
    recent_train_row: Mapping[str, Any] | None,
    final_probe_row: Mapping[str, Any] | None,
    last_checkpoint_env_steps: int | None,
    last_checkpoint_train_episode_idx: int | None,
    final_probe_source: str,
    best_checkpoint_env_steps: int | None = None,
    best_checkpoint_train_episode_idx: int | None = None,
    total_runtime_sec: float | None = None,
    total_runtime_hms: str | None = None,
    collector: Any | None = None,
    learner: Any | None = None,
    replay: Any | None = None,
    state_adapter: Any | None = None,
    source_of_truth_repo: str | None = None,
    extra_insufficient_evidence_flags: list[str] | None = None,
    last_eval_row: Mapping[str, Any] | None = None,
    best_eval_row: Mapping[str, Any] | None = None,
    model_select_rows: list[Mapping[str, Any]] | None = None,
    best_recheck_rows: list[Mapping[str, Any]] | None = None,
    best_checkpoint_selection_row: Mapping[str, Any] | None = None,
    last_checkpoint_diagnostic_row: Mapping[str, Any] | None = None,
) -> dict[str, Path]:
    run_dir = run_dir.resolve()
    source_repo = source_of_truth_repo or str(run_dir.parents[1])
    flags = list(extra_insufficient_evidence_flags or [])
    config_dict = _serialize_config(cfg) or {}
    protocol_revision = _formal_protocol_revision(config_dict)

    metric_snapshot = build_metric_snapshot(
        run_dir=run_dir,
        recent_train_row=recent_train_row,
        final_probe_row=final_probe_row,
        last_checkpoint_env_steps=last_checkpoint_env_steps,
        last_checkpoint_train_episode_idx=last_checkpoint_train_episode_idx,
        best_checkpoint_env_steps=best_checkpoint_env_steps,
        best_checkpoint_train_episode_idx=best_checkpoint_train_episode_idx,
        final_probe_source=final_probe_source,
        source_of_truth_repo=source_repo,
        formal_protocol_revision=protocol_revision,
        last_eval_row=last_eval_row,
        best_eval_row=best_eval_row,
        model_select_rows=model_select_rows,
        best_recheck_rows=best_recheck_rows,
        best_checkpoint_selection_row=best_checkpoint_selection_row,
        last_checkpoint_diagnostic_row=last_checkpoint_diagnostic_row,
        insufficient_evidence_flags=flags,
    )
    benchmark_summary = build_benchmark_summary(
        cfg=cfg,
        run_dir=run_dir,
        run_mode=run_mode,
        total_runtime_sec=total_runtime_sec,
        total_runtime_hms=total_runtime_hms,
        total_train_episodes_completed=_to_scalar((recent_train_row or {}).get("completed_train_episodes")),
        best_checkpoint_env_steps=best_checkpoint_env_steps,
        last_checkpoint_env_steps=last_checkpoint_env_steps,
        model_selection_eval_count=len(model_select_rows or []),
        recheck_eval_count=len(best_recheck_rows or []),
        collector=collector,
        learner=learner,
        replay=replay,
        state_adapter=state_adapter,
        source_of_truth_repo=source_repo,
        insufficient_evidence_flags=flags,
    )
    observed_run_contract = build_observed_run_contract(
        run_dir=run_dir,
        recent_train_row=recent_train_row,
        final_probe_row=final_probe_row,
    )
    config_snapshot = build_config_snapshot(
        cfg=cfg,
        run_dir=run_dir,
        run_mode=run_mode,
        source_of_truth_repo=source_repo,
        observed_run_contract=observed_run_contract,
        insufficient_evidence_flags=flags,
    )

    metric_path = run_dir / "logs" / "metric_snapshot.json"
    benchmark_path = run_dir / "logs" / "benchmark_summary.json"
    config_path = run_dir / "logs" / "config_snapshot.json"
    training_summary_path = run_dir / "logs" / "training_summary.txt"

    _write_json(metric_path, metric_snapshot)
    _write_json(benchmark_path, benchmark_summary)
    _write_json(config_path, config_snapshot)
    training_summary_path.write_text(
        build_training_summary_text(
            run_dir=run_dir,
            metric_snapshot=metric_snapshot,
            benchmark_summary=benchmark_summary,
        ),
        encoding="utf-8",
    )

    artifact_index = build_artifact_index(run_dir)
    structured_summaries = artifact_index.get("structured_summaries", [])
    if isinstance(structured_summaries, list):
        for record in structured_summaries:
            if isinstance(record, dict) and record.get("path") == "logs/artifact_index.json":
                record["exists"] = True
    artifact_index_path = run_dir / "logs" / "artifact_index.json"
    _write_json(artifact_index_path, artifact_index)

    return {
        "metric_snapshot": metric_path,
        "benchmark_summary": benchmark_path,
        "config_snapshot": config_path,
        "artifact_index": artifact_index_path,
        "training_summary": training_summary_path,
    }


def build_run_record_from_artifacts(run_dir: Path) -> dict[str, Any] | None:
    logs_dir = run_dir / "logs"
    if not logs_dir.exists():
        return None

    train_steps_rows = _read_csv_rows(logs_dir / "train_steps.csv")
    final_probe_rows = _read_csv_rows(logs_dir / "final_probe.csv")
    eval_rows = _read_csv_rows(logs_dir / "eval_metrics.csv")
    model_select_rows = _read_csv_rows(logs_dir / "model_select_eval.csv")
    best_recheck_rows = _read_csv_rows(logs_dir / "best_recheck_eval.csv")
    if not train_steps_rows or not final_probe_rows:
        return None

    recent_train_row = train_steps_rows[-1]
    final_probe_row = final_probe_rows[-1]
    last_eval_row = eval_rows[-1] if eval_rows else None
    best_eval_row = select_best_eval_row(best_recheck_rows) if best_recheck_rows else (
        select_best_eval_row(model_select_rows) if model_select_rows else (
            select_best_eval_row(eval_rows) if eval_rows else None
        )
    )
    checkpoint_dir = run_dir / "checkpoints"

    insufficient_flags: list[str] = []
    config_snapshot = benchmark_summary = metric_snapshot = None
    for name in ("config_snapshot.json", "benchmark_summary.json", "metric_snapshot.json"):
        file_path = logs_dir / name
        if not file_path.exists():
            continue
        payload = json.loads(file_path.read_text(encoding="utf-8"))
        if name == "config_snapshot.json":
            config_snapshot = payload
        elif name == "benchmark_summary.json":
            benchmark_summary = payload
        else:
            metric_snapshot = payload

    if config_snapshot is None:
        insufficient_flags.append("config_snapshot_missing")
    if benchmark_summary is None:
        insufficient_flags.append("benchmark_summary_missing")
    if metric_snapshot is None:
        insufficient_flags.append("metric_snapshot_missing")

    record = {
        "run_id": run_dir.name,
        "run_dir": str(run_dir.resolve()),
        "recent_train": _normalize_recent_train(recent_train_row),
        "final_probe": _normalize_eval_like(final_probe_row, source_name="logs/final_probe.csv"),
        "last_eval": _normalize_eval_like(last_eval_row, source_name="logs/eval_metrics.csv") if last_eval_row else {},
        "best_eval": _normalize_eval_like(best_eval_row, source_name="logs/best_recheck_or_model_select.csv::best_eval") if best_eval_row else {},
        "model_select_eval_count": len(model_select_rows),
        "best_recheck_eval_count": len(best_recheck_rows),
        "best_checkpoint_exists": (checkpoint_dir / "best.pt").exists(),
        "last_checkpoint_exists": (checkpoint_dir / "last.pt").exists(),
        "plots_present": (run_dir / "plots").exists(),
        "trajectories_present": (run_dir / "trajectories").exists(),
        "config_snapshot": config_snapshot,
        "benchmark_summary": benchmark_summary,
        "metric_snapshot": metric_snapshot,
        "insufficient_evidence_flags": insufficient_flags,
    }

    use_exact_config = bool(config_snapshot and config_snapshot.get("full_train_config"))
    if use_exact_config:
        comparability = config_snapshot.get("comparability", {})
        record["comparability_group"] = comparability.get("comparability_group")
    else:
        signature_seed = json.dumps(
            {
                "env_steps": _to_scalar(_apply_row_aliases(final_probe_row).get("env_steps")),
                "train_steps_header": _read_csv_header(logs_dir / "train_steps.csv"),
                "final_probe_header": _read_csv_header(logs_dir / "final_probe.csv"),
            },
            ensure_ascii=False,
        ).encode("utf-8")
        signature_hash = hashlib.sha1(signature_seed).hexdigest()[:12]
        record["comparability_group"] = f"bootstrap_header_signature__{signature_hash}"
        record["insufficient_evidence_flags"].append("comparability_group_bootstrap_from_csv_headers")
    return _json_safe(record)


def _quantiles(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "mean": None, "median": None, "min": None, "max": None, "p25": None, "p75": None}
    ordered = sorted(values)
    mean_value = statistics.fmean(values)
    median_value = statistics.median(ordered)
    p25 = ordered[max(0, int(round((len(ordered) - 1) * 0.25)))]
    p75 = ordered[max(0, int(round((len(ordered) - 1) * 0.75)))]
    return {
        "count": len(values),
        "mean": mean_value,
        "median": median_value,
        "min": ordered[0],
        "max": ordered[-1],
        "p25": p25,
        "p75": p75,
    }


def build_historical_baseline_summary(
    *,
    output_root: Path,
    source_of_truth_repo: str,
) -> dict[str, Any]:
    run_records = []
    if output_root.exists():
        for child in sorted(output_root.iterdir()):
            if not child.is_dir() or child.name == "scheduler_runs":
                continue
            record = build_run_record_from_artifacts(child)
            if record is not None:
                run_records.append(record)

    grouped: dict[str, list[dict[str, Any]]] = {}
    exact_group_count = 0
    for record in run_records:
        group = str(record.get("comparability_group") or "unknown")
        grouped.setdefault(group, []).append(record)
        if "comparability_group_bootstrap_from_csv_headers" not in record.get("insufficient_evidence_flags", []):
            exact_group_count += 1

    group_summaries = []
    for group_name, records in sorted(grouped.items()):
        success_values = [
            float(record["final_probe"]["metrics"]["success_rate"])
            for record in records
            if isinstance(record["final_probe"].get("metrics", {}).get("success_rate"), (int, float))
        ]
        coverage_values = [
            float(record["final_probe"]["metrics"]["coverage"])
            for record in records
            if isinstance(record["final_probe"].get("metrics", {}).get("coverage"), (int, float))
        ]
        reward_values = [
            float(record["final_probe"]["metrics"]["reward"])
            for record in records
            if isinstance(record["final_probe"].get("metrics", {}).get("reward"), (int, float))
        ]
        runtime_values = [
            float((record.get("benchmark_summary") or {}).get("total_runtime_sec"))
            for record in records
            if isinstance((record.get("benchmark_summary") or {}).get("total_runtime_sec"), (int, float))
        ]
        group_summaries.append(
            {
                "comparability_group": group_name,
                "run_count": len(records),
                "evidence_status": (
                    "formal_exact_group"
                    if all(
                        "comparability_group_bootstrap_from_csv_headers" not in record.get("insufficient_evidence_flags", [])
                        for record in records
                    )
                    else "bootstrap_grouped_from_csv_headers"
                ),
                "run_ids": [record["run_id"] for record in records],
                "final_probe_distributions": {
                    "success_rate": _quantiles(success_values),
                    "coverage": _quantiles(coverage_values),
                    "reward": _quantiles(reward_values),
                    "runtime_sec": _quantiles(runtime_values),
                },
            }
        )

    insufficient_history = exact_group_count < 3
    notes = [
        "Historical runs before formal snapshots may lack config_snapshot.json and benchmark_summary.json.",
        "Bootstrap grouping falls back to final env_steps plus train/final CSV header signatures when exact comparability metadata is unavailable.",
        "Current formal_train selects checkpoints with a dedicated model-selection seed set, promotes best.pt, and runs held-out final_probe only on best.pt.",
        (
            "Historical formal lanes used a 16-episode final_probe; current default formal lanes use a "
            "100-episode final_probe and enter a separate strict comparability protocol lane."
        ),
    ]
    if insufficient_history:
        notes.append(
            "Exact comparability metadata is insufficient for calibrated stop thresholds. "
            "Use bootstrap thresholds until new formal_train rounds accumulate."
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "historical_baseline_summary",
        "experiment_mode": "formal_train",
        "source_of_truth_repo": source_of_truth_repo,
        "generated_at": _now_iso(),
        "run_count_total": len(run_records),
        "exact_comparability_run_count": exact_group_count,
        "insufficient_history_for_calibration": insufficient_history,
        "default_main_baseline_identifier": DEFAULT_MAIN_BASELINE_IDENTIFIER,
        "notes": notes,
        "group_summaries": group_summaries,
    }


def write_historical_baseline_summary(
    *,
    output_root: Path,
    output_path: Path,
    source_of_truth_repo: str,
) -> Path:
    summary = build_historical_baseline_summary(
        output_root=output_root,
        source_of_truth_repo=source_of_truth_repo,
    )
    _write_json(output_path, summary)
    return output_path
