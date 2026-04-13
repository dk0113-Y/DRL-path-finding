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


SCHEMA_VERSION = "formal_train_artifacts/v1"
DEFAULT_MAIN_BASELINE_IDENTIFIER = "4.9_30万轮基线"

RUNTIME_ONLY_FIELDS = (
    "enable_amp",
    "enable_inference_amp",
    "amp_dtype",
    "enable_torch_compile",
    "compile_mode",
    "enable_cudnn_benchmark",
    "enable_tf32",
    "enable_channels_last",
    "generate_plots_on_finish",
    "save_eval_trajectories",
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
    "reward_revisit_penalty",
)

MANUAL_REVIEW_FIELDS = (
    "reward_stall_penalty",
    "max_entries_per_block",
)

FROZEN_COMPARABILITY_FIELDS = (
    "rows",
    "cols",
    "obs_size",
    "scan_radius",
    "trajectory_history_steps",
    "obstacle_ratio",
    "total_env_steps",
    "warmup_steps",
    "collect_steps_per_iter",
    "learner_updates_per_iter",
    "train_every_env_steps",
    "eval_interval_env_steps",
    "eval_episodes",
    "final_greedy_episodes",
    "use_fixed_eval_seeds",
    "fixed_eval_seed_base",
    "fixed_final_probe_seed_base",
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
    "reward_info_norm",
    "reward_recent_revisit_window",
    "reward_stall_window",
    "reward_step_penalty",
    "reward_terminal_bonus",
    "reward_timeout_penalty",
    "max_accessible_blocks",
)

REWARD_BREAKDOWN_FIELDS = (
    "info_reward_sum",
    "step_penalty_sum",
    "recent_revisit_penalty_sum",
    "stall_penalty_sum",
    "turn_penalty_sum",
    "timeout_penalty_sum",
    "terminal_bonus_sum",
)

REWARD_EVENT_FIELDS = (
    "delta_empty_sum",
    "delta_obstacle_sum",
    "weighted_info_gain_sum",
    "recent_revisit_count",
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
    "recent_revisit_count": "lower_is_better",
}

RECENT_CORE_FIELD_MAP = {
    "reward": "recent_mean_reward",
    "coverage": "recent_mean_coverage",
    "success_rate": "recent_success_rate",
    "episode_length": "recent_mean_episode_length",
    "repeat_visit_ratio": "recent_mean_repeat_visit_ratio",
}

EVAL_CORE_FIELD_MAP = {
    "reward": "eval_mean_reward",
    "coverage": "eval_mean_coverage",
    "success_rate": "eval_success_rate",
    "episode_length": "eval_mean_episode_length",
    "repeat_visit_ratio": "eval_mean_repeat_visit_ratio",
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


def _best_eval_score(row: Mapping[str, Any]) -> tuple[float, float]:
    success = _to_scalar(row.get("eval_success_rate"))
    coverage = _to_scalar(row.get("eval_mean_coverage"))
    success_value = float(success) if isinstance(success, (int, float)) else float("-inf")
    coverage_value = float(coverage) if isinstance(coverage, (int, float)) else float("-inf")
    return (success_value, coverage_value)


def select_best_eval_row(eval_rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not eval_rows:
        return None
    return max(eval_rows, key=_best_eval_score)


def _extract_fields(row: Mapping[str, Any] | None, field_names: tuple[str, ...], prefix: str) -> dict[str, Any]:
    row = row or {}
    extracted: dict[str, Any] = {}
    for field_name in field_names:
        key = f"{prefix}{field_name}"
        if key in row:
            extracted[field_name] = _to_scalar(row.get(key))
    return extracted


def _extract_dynamic_metric_fields(
    row: Mapping[str, Any] | None,
    *,
    prefix: str,
    reserved_fields: set[str],
) -> dict[str, Any]:
    row = row or {}
    extra: dict[str, Any] = {}
    for key, value in row.items():
        if not key.startswith(prefix):
            continue
        suffix = key[len(prefix):]
        if suffix in reserved_fields:
            continue
        extra[suffix] = _to_scalar(value)
    return extra


def _normalize_recent_train(row: Mapping[str, Any] | None) -> dict[str, Any]:
    row = row or {}
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
        "env_steps": _to_scalar(row.get("env_steps")),
        "learner_steps": _to_scalar(row.get("learner_steps")),
        "optimizer_monitoring": {
            "loss": _to_scalar(row.get("loss")),
            "q_mean": _to_scalar(row.get("q_mean")),
            "target_q_mean": _to_scalar(row.get("target_q_mean")),
            "td_abs_mean": _to_scalar(row.get("td_abs_mean")),
            "grad_norm": _to_scalar(row.get("grad_norm")),
            "replay_size": _to_scalar(row.get("replay_size")),
            "epsilon": _to_scalar(row.get("epsilon")),
        },
        "metrics": {name: _to_scalar(row.get(field_name)) for name, field_name in RECENT_CORE_FIELD_MAP.items()},
        "reward_breakdown": _extract_fields(row, REWARD_BREAKDOWN_FIELDS, "recent_"),
        "reward_events": _extract_fields(row, REWARD_EVENT_FIELDS, "recent_"),
        "semantic_monitoring": _extract_dynamic_metric_fields(row, prefix="recent_", reserved_fields=reserved),
        "raw_row": _json_safe(dict(row)),
    }


def _normalize_eval_like(row: Mapping[str, Any] | None, *, source_name: str) -> dict[str, Any]:
    row = row or {}
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
    return {
        "source": source_name,
        "tag": _to_scalar(row.get("tag")),
        "env_steps": _to_scalar(row.get("env_steps")),
        "learner_steps": _to_scalar(row.get("learner_steps")),
        "episodes": _to_scalar(row.get("eval_episodes")),
        "metrics": {name: _to_scalar(row.get(field_name)) for name, field_name in EVAL_CORE_FIELD_MAP.items()},
        "reward_breakdown": _extract_fields(row, REWARD_BREAKDOWN_FIELDS, "eval_mean_"),
        "reward_events": _extract_fields(row, REWARD_EVENT_FIELDS, "eval_mean_"),
        "semantic_monitoring": _extract_dynamic_metric_fields(
            row,
            prefix="eval_mean_",
            reserved_fields={field_name.removeprefix("eval_mean_") for field_name in raw_reserved},
        ),
        "raw_row": _json_safe(dict(row)),
    }


def _build_unified_metric_table(metric_blocks: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    def metric_table(metric_names: Mapping[str, str], source_key: str) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        for metric_name, direction in metric_names.items():
            payload[metric_name] = {
                "direction": direction,
                "recent_train": _to_scalar(metric_blocks["recent_train"][source_key].get(metric_name)),
                "last_eval": _to_scalar(metric_blocks["last_eval"][source_key].get(metric_name)),
                "best_eval": _to_scalar(metric_blocks["best_eval"][source_key].get(metric_name)),
                "final_probe": _to_scalar(metric_blocks["final_probe"][source_key].get(metric_name)),
            }
        return payload

    semantic_keys: set[str] = set()
    for block in metric_blocks.values():
        semantic_keys.update(str(key) for key in block.get("semantic_monitoring", {}).keys())

    semantic_payload = {}
    for metric_name in sorted(semantic_keys):
        semantic_payload[metric_name] = {
            "direction": "context_dependent_monitoring",
            "recent_train": _to_scalar(metric_blocks["recent_train"]["semantic_monitoring"].get(metric_name)),
            "last_eval": _to_scalar(metric_blocks["last_eval"]["semantic_monitoring"].get(metric_name)),
            "best_eval": _to_scalar(metric_blocks["best_eval"]["semantic_monitoring"].get(metric_name)),
            "final_probe": _to_scalar(metric_blocks["final_probe"]["semantic_monitoring"].get(metric_name)),
        }

    return {
        "primary_metrics": metric_table(PRIMARY_UNIFIED_METRICS, "metrics"),
        "secondary_metrics": metric_table(SECONDARY_UNIFIED_METRICS, "metrics"),
        "stability_metrics": metric_table(STABILITY_UNIFIED_METRICS, "reward_events"),
        "semantic_monitoring": semantic_payload,
    }


def build_observed_run_contract(
    *,
    run_dir: Path,
    recent_train_row: Mapping[str, Any] | None = None,
    last_eval_row: Mapping[str, Any] | None = None,
    final_probe_row: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    logs_dir = run_dir / "logs"
    final_env_steps = (
        _to_scalar((final_probe_row or {}).get("env_steps"))
        or _to_scalar((last_eval_row or {}).get("env_steps"))
        or _to_scalar((recent_train_row or {}).get("env_steps"))
    )
    if final_env_steps is None:
        final_probe_rows = _read_csv_rows(logs_dir / "final_probe.csv")
        eval_rows = _read_csv_rows(logs_dir / "eval_metrics.csv")
        train_steps_rows = _read_csv_rows(logs_dir / "train_steps.csv")
        if final_probe_rows:
            final_env_steps = _to_scalar(final_probe_rows[-1].get("env_steps"))
        elif eval_rows:
            final_env_steps = _to_scalar(eval_rows[-1].get("env_steps"))
        elif train_steps_rows:
            final_env_steps = _to_scalar(train_steps_rows[-1].get("env_steps"))

    return {
        "final_env_steps": final_env_steps,
        "train_steps_header": _read_csv_header(logs_dir / "train_steps.csv"),
        "eval_metrics_header": _read_csv_header(logs_dir / "eval_metrics.csv"),
        "final_probe_header": _read_csv_header(logs_dir / "final_probe.csv"),
    }


def build_metric_snapshot(
    *,
    run_dir: Path,
    recent_train_row: Mapping[str, Any] | None,
    last_eval_row: Mapping[str, Any] | None,
    best_eval_row: Mapping[str, Any] | None,
    final_probe_row: Mapping[str, Any] | None,
    best_checkpoint_source: str,
    best_checkpoint_env_steps: int | None,
    last_checkpoint_env_steps: int | None,
    final_probe_source: str,
    source_of_truth_repo: str,
    insufficient_evidence_flags: list[str] | None = None,
) -> dict[str, Any]:
    recent_train = _normalize_recent_train(recent_train_row)
    last_eval = _normalize_eval_like(last_eval_row, source_name="logs/eval_metrics.csv")
    best_eval = _normalize_eval_like(best_eval_row, source_name="logs/eval_metrics.csv::best_eval")
    final_probe = _normalize_eval_like(final_probe_row, source_name="logs/final_probe.csv")

    metric_blocks = {
        "recent_train": recent_train,
        "last_eval": last_eval,
        "best_eval": best_eval,
        "final_probe": final_probe,
    }

    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "metric_snapshot",
        "experiment_mode": "formal_train",
        "source_of_truth_repo": source_of_truth_repo,
        "run_dir": str(run_dir.resolve()),
        "generated_at": _now_iso(),
        "recent_train": recent_train,
        "last_eval": last_eval,
        "best_eval": best_eval,
        "final_probe": final_probe,
        "best_checkpoint_source": best_checkpoint_source,
        "best_checkpoint_env_steps": best_checkpoint_env_steps,
        "last_checkpoint_env_steps": last_checkpoint_env_steps,
        "final_probe_source": final_probe_source,
        "unified_metrics": _build_unified_metric_table(metric_blocks),
        "insufficient_evidence_flags": sorted(set(insufficient_evidence_flags or [])),
    }


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
    payload = {
        "component": component_name,
        "total_time_sec": total,
        "breakdown_sec": numeric_stats,
    }
    return _json_safe(payload)


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
    env_steps_to_best: int | None,
    collector: Any | None = None,
    learner: Any | None = None,
    replay: Any | None = None,
    state_adapter: Any | None = None,
    source_of_truth_repo: str,
    insufficient_evidence_flags: list[str] | None = None,
) -> dict[str, Any]:
    config_dict = _serialize_config(cfg) or {}
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
        "env_steps_to_best": env_steps_to_best,
        "insufficient_evidence_flags": sorted(set(flags)),
    }


def _comparability_sections(config_dict: Mapping[str, Any]) -> dict[str, Any]:
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
        "comparability_group": f"formal_mainline_v1__{group_hash}",
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
    comparability_sections = _comparability_sections(config_dict)
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
            "best_checkpoint_rule": {
                "primary_metric": "eval_success_rate",
                "tie_breaker": "eval_mean_coverage",
                "source": "training/checkpointing.py",
            },
            "final_probe_rule": {
                "source": "best_checkpoint_if_available_else_online_last",
                "csv_file": "logs/final_probe.csv",
            },
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

    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "artifact_index",
        "experiment_mode": "formal_train",
        "run_dir": str(run_dir.resolve()),
        "generated_at": _now_iso(),
        "csv": [
            _artifact_record(run_dir, "logs/train_steps.csv", required=True, category="csv"),
            _artifact_record(run_dir, "logs/train_episodes.csv", required=True, category="csv"),
            _artifact_record(run_dir, "logs/eval_metrics.csv", required=True, category="csv"),
            _artifact_record(run_dir, "logs/final_probe.csv", required=True, category="csv"),
        ],
        "checkpoints": [
            _artifact_record(run_dir, "checkpoints/best.pt", required=True, category="checkpoint"),
            _artifact_record(run_dir, "checkpoints/last.pt", required=True, category="checkpoint"),
        ],
        "structured_summaries": [
            _artifact_record(run_dir, "logs/metric_snapshot.json", required=True, category="structured_summary"),
            _artifact_record(run_dir, "logs/benchmark_summary.json", required=True, category="structured_summary"),
            _artifact_record(run_dir, "logs/config_snapshot.json", required=True, category="structured_summary"),
            _artifact_record(run_dir, "logs/artifact_index.json", required=True, category="structured_summary"),
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
    last_eval = metric_snapshot.get("last_eval", {})
    best_eval = metric_snapshot.get("best_eval", {})
    final_probe = metric_snapshot.get("final_probe", {})

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
        f"total_runtime_sec: {benchmark_summary.get('total_runtime_sec')}",
        f"total_runtime_hms: {benchmark_summary.get('total_runtime_hms')}",
        line_for(recent_train, "recent_train"),
        line_for(last_eval, "last_eval"),
        line_for(best_eval, "best_eval"),
        line_for(final_probe, "final_probe"),
        f"best_checkpoint_source: {metric_snapshot.get('best_checkpoint_source')}",
        f"best_checkpoint_env_steps: {metric_snapshot.get('best_checkpoint_env_steps')}",
        f"last_checkpoint_env_steps: {metric_snapshot.get('last_checkpoint_env_steps')}",
        f"final_probe_source: {metric_snapshot.get('final_probe_source')}",
    ]
    return "\n".join(lines) + "\n"


def write_formal_run_artifacts(
    *,
    run_dir: Path,
    cfg: Any | None,
    run_mode: str,
    recent_train_row: Mapping[str, Any] | None,
    last_eval_row: Mapping[str, Any] | None,
    best_eval_row: Mapping[str, Any] | None,
    final_probe_row: Mapping[str, Any] | None,
    best_checkpoint_source: str,
    best_checkpoint_env_steps: int | None,
    last_checkpoint_env_steps: int | None,
    final_probe_source: str,
    total_runtime_sec: float | None,
    total_runtime_hms: str | None,
    collector: Any | None = None,
    learner: Any | None = None,
    replay: Any | None = None,
    state_adapter: Any | None = None,
    source_of_truth_repo: str | None = None,
    extra_insufficient_evidence_flags: list[str] | None = None,
) -> dict[str, Path]:
    run_dir = run_dir.resolve()
    source_repo = source_of_truth_repo or str(run_dir.parents[1])
    flags = list(extra_insufficient_evidence_flags or [])
    metric_snapshot = build_metric_snapshot(
        run_dir=run_dir,
        recent_train_row=recent_train_row,
        last_eval_row=last_eval_row,
        best_eval_row=best_eval_row,
        final_probe_row=final_probe_row,
        best_checkpoint_source=best_checkpoint_source,
        best_checkpoint_env_steps=best_checkpoint_env_steps,
        last_checkpoint_env_steps=last_checkpoint_env_steps,
        final_probe_source=final_probe_source,
        source_of_truth_repo=source_repo,
        insufficient_evidence_flags=flags,
    )
    benchmark_summary = build_benchmark_summary(
        cfg=cfg,
        run_dir=run_dir,
        run_mode=run_mode,
        total_runtime_sec=total_runtime_sec,
        total_runtime_hms=total_runtime_hms,
        env_steps_to_best=best_checkpoint_env_steps,
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
        last_eval_row=last_eval_row,
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
            if (
                isinstance(record, dict)
                and record.get("path") == "logs/artifact_index.json"
            ):
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
    eval_rows = _read_csv_rows(logs_dir / "eval_metrics.csv")
    final_probe_rows = _read_csv_rows(logs_dir / "final_probe.csv")
    if not train_steps_rows or not eval_rows or not final_probe_rows:
        return None

    recent_train_row = train_steps_rows[-1]
    last_eval_row = eval_rows[-1]
    best_eval_row = select_best_eval_row(eval_rows)
    final_probe_row = final_probe_rows[-1]
    checkpoint_dir = run_dir / "checkpoints"

    insufficient_flags: list[str] = []
    config_snapshot = None
    benchmark_summary = None
    metric_snapshot = None
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
        "last_eval": _normalize_eval_like(last_eval_row, source_name="logs/eval_metrics.csv"),
        "best_eval": _normalize_eval_like(best_eval_row, source_name="logs/eval_metrics.csv::best_eval"),
        "final_probe": _normalize_eval_like(final_probe_row, source_name="logs/final_probe.csv"),
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
        bootstrap_signature = ((config_snapshot or {}).get("comparability") or {}).get("bootstrap_signature", {})
        total_env_steps = bootstrap_signature.get("env_steps", _to_scalar(recent_train_row.get("env_steps")))
        eval_header = bootstrap_signature.get("eval_columns", sorted(eval_rows[-1].keys()) if eval_rows else [])
        signature_seed = json.dumps({"env_steps": total_env_steps, "eval_header": eval_header}, ensure_ascii=False).encode("utf-8")
        signature_hash = hashlib.sha1(signature_seed).hexdigest()[:12]
        record["comparability_group"] = f"bootstrap_header_signature__{signature_hash}"
        record["insufficient_evidence_flags"].append("comparability_group_bootstrap_from_csv_header")
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
        if not any(
            flag == "comparability_group_bootstrap_from_csv_header"
            for flag in record.get("insufficient_evidence_flags", [])
        ):
            exact_group_count += 1

    group_summaries = []
    for group_name, records in sorted(grouped.items()):
        success_values = [
            float(record["final_probe"]["metrics"]["success_rate"])
            for record in records
            if isinstance(record["final_probe"]["metrics"].get("success_rate"), (int, float))
        ]
        coverage_values = [
            float(record["final_probe"]["metrics"]["coverage"])
            for record in records
            if isinstance(record["final_probe"]["metrics"].get("coverage"), (int, float))
        ]
        reward_values = [
            float(record["final_probe"]["metrics"]["reward"])
            for record in records
            if isinstance(record["final_probe"]["metrics"].get("reward"), (int, float))
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
                    "formal_exact_group" if all(
                        "comparability_group_bootstrap_from_csv_header" not in record.get("insufficient_evidence_flags", [])
                        for record in records
                    ) else "bootstrap_grouped_from_csv_headers"
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
        "Bootstrap grouping falls back to final env_steps plus eval CSV header signatures when exact comparability metadata is unavailable.",
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
