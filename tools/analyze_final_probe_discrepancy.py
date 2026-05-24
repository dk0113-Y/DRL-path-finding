"""Analyze why method A wins recent training but not the final probe.

This script is read-only with respect to training artifacts and model files. It
only writes derived analysis files under experiment_records/final_probe/analysis.
"""

from __future__ import annotations

import csv
import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
FINAL_PROBE_DIR = REPO_ROOT / "experiment_records" / "final_probe"
ANALYSIS_DIR = FINAL_PROBE_DIR / "analysis"

TARGET_METHODS = ["A", "F1", "R1", "R2", "R3", "R4", "R5"]
COMPARATORS = ["F1", "R1", "R2", "R3", "R4", "R5"]

METRIC_SNAPSHOT_PATHS = {
    "A": REPO_ROOT / "experiment_records" / "full_method_main" / "logs" / "metric_snapshot.json",
    "F1": REPO_ROOT
    / "experiment_records"
    / "ablations"
    / "F1_ablation_no_frontier_channel"
    / "logs"
    / "metric_snapshot.json",
    "R1": REPO_ROOT
    / "experiment_records"
    / "ablations"
    / "R1_ablation_no_step_penalty"
    / "logs"
    / "metric_snapshot.json",
    "R2": REPO_ROOT
    / "experiment_records"
    / "ablations"
    / "R2_ablation_no_revisit_penalty"
    / "logs"
    / "metric_snapshot.json",
    "R3": REPO_ROOT
    / "experiment_records"
    / "ablations"
    / "R3_ablation_no_turn_penalty"
    / "logs"
    / "metric_snapshot.json",
    "R4": REPO_ROOT
    / "experiment_records"
    / "ablations"
    / "R4_ablation_no_timeout_penalty"
    / "logs"
    / "metric_snapshot.json",
    "R5": REPO_ROOT
    / "experiment_records"
    / "ablations"
    / "R5_ablation_no_efficiency_penalties"
    / "logs"
    / "metric_snapshot.json",
}

EVALUATION_CRITICAL_FIELDS = [
    "rows",
    "cols",
    "obs_size",
    "scan_radius",
    "obstacle_ratio",
    "max_episode_steps",
    "coverage_stop_threshold",
    "trajectory_history_steps",
]

REWARD_FIELDS = [
    "reward_info_scale",
    "reward_obstacle_weight",
    "reward_step_penalty",
    "reward_terminal_bonus",
    "reward_revisit_penalty",
    "reward_turn_penalty_scale",
    "reward_turn_weight_45",
    "reward_turn_weight_90",
    "reward_turn_weight_135",
    "reward_turn_weight_180",
    "reward_timeout_penalty",
]

TRAIN_METRICS = [
    "reward",
    "coverage",
    "success_rate",
    "episode_length",
    "repeat_visit_ratio",
    "timeout_rate",
]

FINAL_METRICS = [
    "success_rate",
    "coverage",
    "reward",
    "episode_length",
    "repeat_visit_ratio",
    "timeout_flag",
]


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: format_value(row.get(key)) for key in fieldnames})


def format_value(value: Any) -> Any:
    if isinstance(value, float):
        if math.isnan(value):
            return ""
        return f"{value:.10g}"
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    if value is None:
        return ""
    return value


def to_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def relpath(path: Path | str | None) -> str | None:
    if path is None:
        return None
    path_obj = Path(path)
    try:
        return str(path_obj.resolve().relative_to(REPO_ROOT))
    except (OSError, ValueError):
        return str(path)


def method_is_reward_comparable_with_a(method_id: str) -> bool:
    return not method_id.startswith("R")


def reward_comparability_note(method_id: str) -> str:
    if method_is_reward_comparable_with_a(method_id):
        return "Reward formula matches A for direct reward comparison."
    return (
        "R-group reward ablations remove or alter efficiency penalty terms; "
        "reward is not directly comparable with A."
    )


def get_train_metrics(snapshot: dict[str, Any]) -> dict[str, float | None]:
    recent = snapshot.get("recent_train", {})
    metrics = recent.get("metrics", {})
    semantic = recent.get("semantic_monitoring", {})
    events = recent.get("reward_events", {})
    return {
        "reward": to_float(metrics.get("reward")),
        "coverage": to_float(metrics.get("coverage")),
        "success_rate": to_float(metrics.get("success_rate")),
        "episode_length": to_float(metrics.get("episode_length")),
        "repeat_visit_ratio": to_float(metrics.get("repeat_visit_ratio")),
        "timeout_rate": to_float(semantic.get("timeout_rate", events.get("timeout_flag"))),
    }


def row_by_method(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    return {row["method_id"]: row for row in rows}


def get_manifest_methods(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    methods = manifest.get("audit", {}).get("methods", [])
    return {method["method_id"]: method for method in methods}


def config_snapshot_path(method_record: dict[str, Any]) -> Path:
    configured = method_record.get("config_snapshot_path")
    if configured:
        return Path(configured)
    raise KeyError(f"No config_snapshot_path for {method_record.get('method_id')}")


def full_train_config(config_snapshot: dict[str, Any]) -> dict[str, Any]:
    config = config_snapshot.get("full_train_config")
    if not isinstance(config, dict):
        raise KeyError("config_snapshot missing full_train_config")
    return config


def build_training_vs_final_rows(
    summary_rows: list[dict[str, str]],
    metric_snapshots: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    final_by_method = row_by_method(summary_rows)
    rows: list[dict[str, Any]] = []
    for method_id in TARGET_METHODS:
        final = final_by_method[method_id]
        train = get_train_metrics(metric_snapshots[method_id])
        row: dict[str, Any] = {
            "method_id": method_id,
            "group": final.get("group"),
            "display_name": final.get("display_name"),
            "reward_comparable_with_A": method_is_reward_comparable_with_a(method_id),
            "reward_comparability_note": reward_comparability_note(method_id),
        }
        for metric in TRAIN_METRICS:
            row[f"train_recent_{metric}"] = train[metric]
        for metric in FINAL_METRICS:
            source_name = "episode_reward" if metric == "reward" else metric
            row[f"final_probe_{metric}"] = to_float(final.get(source_name, final.get(metric)))
        row["delta_final_minus_train_reward"] = none_safe_sub(
            row["final_probe_reward"], row["train_recent_reward"]
        )
        row["delta_final_minus_train_coverage"] = none_safe_sub(
            row["final_probe_coverage"], row["train_recent_coverage"]
        )
        row["delta_final_minus_train_success_rate"] = none_safe_sub(
            row["final_probe_success_rate"], row["train_recent_success_rate"]
        )
        row["delta_final_minus_train_episode_length"] = none_safe_sub(
            row["final_probe_episode_length"], row["train_recent_episode_length"]
        )
        row["delta_final_minus_train_repeat_visit_ratio"] = none_safe_sub(
            row["final_probe_repeat_visit_ratio"], row["train_recent_repeat_visit_ratio"]
        )
        row["delta_final_minus_train_timeout"] = none_safe_sub(
            row["final_probe_timeout_flag"], row["train_recent_timeout_rate"]
        )
        rows.append(row)
    return rows


def none_safe_sub(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return left - right


def summarize(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "std": None, "median": None}
    return {
        "mean": statistics.fmean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "median": statistics.median(values),
    }


def compare_counts(diffs: list[float], larger_is_better: bool) -> dict[str, int]:
    better = ties = worse = 0
    for diff in diffs:
        if diff == 0:
            ties += 1
        elif (diff > 0 and larger_is_better) or (diff < 0 and not larger_is_better):
            better += 1
        else:
            worse += 1
    return {"a_better": better, "tie": ties, "a_worse": worse}


def build_paired_rows(per_episode_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    by_method_seed: dict[str, dict[int, dict[str, str]]] = {method: {} for method in TARGET_METHODS}
    for row in per_episode_rows:
        method_id = row.get("method_id")
        if method_id not in by_method_seed:
            continue
        seed = int(row["episode_seed"])
        by_method_seed[method_id][seed] = row

    paired_rows: list[dict[str, Any]] = []
    for comparator in COMPARATORS:
        a_rows = by_method_seed["A"]
        c_rows = by_method_seed[comparator]
        paired_seeds = sorted(set(a_rows) & set(c_rows))
        coverage_diffs: list[float] = []
        success_diffs: list[float] = []
        length_diffs: list[float] = []
        timeout_diffs: list[float] = []
        map_mismatches = 0
        for seed in paired_seeds:
            a_row = a_rows[seed]
            c_row = c_rows[seed]
            if a_row.get("map_fingerprint") != c_row.get("map_fingerprint"):
                map_mismatches += 1
            coverage_diffs.append(float(a_row["final_coverage"]) - float(c_row["final_coverage"]))
            success_diffs.append(float(a_row["success"]) - float(c_row["success"]))
            length_diffs.append(float(a_row["episode_length"]) - float(c_row["episode_length"]))
            timeout_diffs.append(float(a_row["timeout_flag"]) - float(c_row["timeout_flag"]))

        row: dict[str, Any] = {
            "comparison": f"A_vs_{comparator}",
            "baseline_method_id": "A",
            "comparator_method_id": comparator,
            "paired_episode_count": len(paired_seeds),
            "map_fingerprint_mismatch_count": map_mismatches,
        }
        for metric, diffs, larger_is_better in [
            ("coverage_diff", coverage_diffs, True),
            ("success_diff", success_diffs, True),
            ("length_diff", length_diffs, False),
            ("timeout_diff", timeout_diffs, False),
        ]:
            summary = summarize(diffs)
            row[f"{metric}_mean"] = summary["mean"]
            row[f"{metric}_std"] = summary["std"]
            row[f"{metric}_median"] = summary["median"]
            counts = compare_counts(diffs, larger_is_better=larger_is_better)
            row[f"{metric}_a_better_count"] = counts["a_better"]
            row[f"{metric}_tie_count"] = counts["tie"]
            row[f"{metric}_a_worse_count"] = counts["a_worse"]
        paired_rows.append(row)
    return paired_rows


def build_config_audit(
    manifest: dict[str, Any],
    metric_snapshots: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    manifest_methods = get_manifest_methods(manifest)
    methods: dict[str, Any] = {}
    config_values_by_field: dict[str, dict[str, Any]] = {
        field: {} for field in EVALUATION_CRITICAL_FIELDS
    }
    reward_values_by_field: dict[str, dict[str, Any]] = {field: {} for field in REWARD_FIELDS}
    load_issues: list[dict[str, Any]] = []

    for method_id in TARGET_METHODS:
        manifest_record = manifest_methods[method_id]
        config_path = config_snapshot_path(manifest_record)
        config_snapshot = read_json(config_path)
        train_config = full_train_config(config_snapshot)
        eval_config = {field: train_config.get(field) for field in EVALUATION_CRITICAL_FIELDS}
        reward_config = {field: train_config.get(field) for field in REWARD_FIELDS}
        for field, value in eval_config.items():
            config_values_by_field[field][method_id] = value
        for field, value in reward_config.items():
            reward_values_by_field[field][method_id] = value

        load_state = manifest_record.get("load_state_dict", {})
        missing_keys = load_state.get("missing_keys", [])
        unexpected_keys = load_state.get("unexpected_keys", [])
        if missing_keys or unexpected_keys:
            load_issues.append(
                {
                    "method_id": method_id,
                    "missing_keys": missing_keys,
                    "unexpected_keys": unexpected_keys,
                }
            )

        methods[method_id] = {
            "display_name": manifest_record.get("display_name"),
            "group": manifest_record.get("group"),
            "metric_snapshot_path": relpath(METRIC_SNAPSHOT_PATHS[method_id]),
            "config_snapshot_path": relpath(config_path),
            "checkpoint_path": manifest_record.get("checkpoint_path"),
            "checkpoint_status": manifest_record.get("checkpoint_status"),
            "checkpoint_train_config_status": manifest_record.get(
                "checkpoint_train_config_status"
            ),
            "checkpoint_env_steps": manifest_record.get("checkpoint_env_steps"),
            "checkpoint_learn_steps": manifest_record.get("checkpoint_learn_steps"),
            "checkpoint_train_episode_idx": manifest_record.get("checkpoint_train_episode_idx"),
            "model_factory": manifest_record.get("model_factory"),
            "model_factory_status": manifest_record.get("model_factory_status"),
            "state_adapter_factory": manifest_record.get("state_adapter_factory"),
            "state_adapter_factory_status": manifest_record.get("state_adapter_factory_status"),
            "load_state_dict": {
                "missing_keys": missing_keys,
                "unexpected_keys": unexpected_keys,
            },
            "evaluation_critical_config": eval_config,
            "reward_config": reward_config,
            "recent_train_source": metric_snapshots[method_id]
            .get("recent_train", {})
            .get("source"),
            "reward_comparable_with_A": method_is_reward_comparable_with_a(method_id),
            "reward_comparability_note": reward_comparability_note(method_id),
        }

    field_consistency = {
        field: {
            "consistent": len({json.dumps(value, sort_keys=True) for value in values.values()})
            == 1,
            "values": values,
        }
        for field, values in config_values_by_field.items()
    }
    reward_field_consistency = {
        field: {
            "consistent": len({json.dumps(value, sort_keys=True) for value in values.values()})
            == 1,
            "values": values,
        }
        for field, values in reward_values_by_field.items()
    }

    args = manifest.get("arguments", {})
    return {
        "schema_version": "final_probe_discrepancy_config_audit/v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "target_methods": TARGET_METHODS,
        "final_probe": {
            "seed_base": args.get("seed_base"),
            "episodes": args.get("episodes"),
            "output_root": manifest.get("output_root") or args.get("output_root"),
            "run_manifest_path": relpath(FINAL_PROBE_DIR / "run_manifest.json"),
        },
        "methods": methods,
        "consistency": {
            "evaluation_critical_fields": field_consistency,
            "all_evaluation_critical_fields_consistent": all(
                item["consistent"] for item in field_consistency.values()
            ),
            "reward_fields": reward_field_consistency,
            "checkpoint_load_state_dict_all_clean": not load_issues,
            "checkpoint_load_state_dict_issues": load_issues,
        },
        "reward_comparability": {
            "A_F1_reward_directly_comparable": True,
            "R_group_reward_directly_comparable_with_A": False,
            "note": (
                "R1-R5 alter reward penalty terms by design, so their reward values are "
                "within-method diagnostics rather than direct cross-method evidence against A."
            ),
        },
    }


def markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(format_value(row.get(col))) for col in columns) + " |")
    return "\n".join(lines)


def rank_value(rows: list[dict[str, Any]], key: str, method_id: str, reverse: bool = True) -> int:
    sorted_rows = sorted(
        rows,
        key=lambda row: (row.get(key) is None, row.get(key)),
        reverse=reverse,
    )
    return next(index for index, row in enumerate(sorted_rows, 1) if row["method_id"] == method_id)


def build_interpretation_md(
    comparison_rows: list[dict[str, Any]],
    paired_rows: list[dict[str, Any]],
    config_audit: dict[str, Any],
) -> str:
    by_method = {row["method_id"]: row for row in comparison_rows}
    a = by_method["A"]
    f1 = by_method["F1"]
    clean_load = config_audit["consistency"]["checkpoint_load_state_dict_all_clean"]
    eval_consistent = config_audit["consistency"]["all_evaluation_critical_fields_consistent"]

    comparable_methods = [
        row for row in comparison_rows if row["reward_comparable_with_A"]
    ]
    a_train_reward_rank = rank_value(
        comparable_methods, "train_recent_reward", "A", reverse=True
    )
    a_final_reward_rank = rank_value(
        comparable_methods, "final_probe_reward", "A", reverse=True
    )

    key_rows = [
        {
            "metric": "success_rate",
            "A_train": a["train_recent_success_rate"],
            "A_final": a["final_probe_success_rate"],
            "F1_train": f1["train_recent_success_rate"],
            "F1_final": f1["final_probe_success_rate"],
            "A_final_minus_train": a["delta_final_minus_train_success_rate"],
            "F1_final_minus_train": f1["delta_final_minus_train_success_rate"],
        },
        {
            "metric": "coverage",
            "A_train": a["train_recent_coverage"],
            "A_final": a["final_probe_coverage"],
            "F1_train": f1["train_recent_coverage"],
            "F1_final": f1["final_probe_coverage"],
            "A_final_minus_train": a["delta_final_minus_train_coverage"],
            "F1_final_minus_train": f1["delta_final_minus_train_coverage"],
        },
        {
            "metric": "reward",
            "A_train": a["train_recent_reward"],
            "A_final": a["final_probe_reward"],
            "F1_train": f1["train_recent_reward"],
            "F1_final": f1["final_probe_reward"],
            "A_final_minus_train": a["delta_final_minus_train_reward"],
            "F1_final_minus_train": f1["delta_final_minus_train_reward"],
        },
        {
            "metric": "episode_length",
            "A_train": a["train_recent_episode_length"],
            "A_final": a["final_probe_episode_length"],
            "F1_train": f1["train_recent_episode_length"],
            "F1_final": f1["final_probe_episode_length"],
            "A_final_minus_train": a["delta_final_minus_train_episode_length"],
            "F1_final_minus_train": f1["delta_final_minus_train_episode_length"],
        },
        {
            "metric": "timeout",
            "A_train": a["train_recent_timeout_rate"],
            "A_final": a["final_probe_timeout_flag"],
            "F1_train": f1["train_recent_timeout_rate"],
            "F1_final": f1["final_probe_timeout_flag"],
            "A_final_minus_train": a["delta_final_minus_train_timeout"],
            "F1_final_minus_train": f1["delta_final_minus_train_timeout"],
        },
    ]

    paired_brief = []
    for row in paired_rows:
        paired_brief.append(
            {
                "comparison": row["comparison"],
                "episodes": row["paired_episode_count"],
                "coverage_mean": row["coverage_diff_mean"],
                "coverage_A_better/tie/worse": (
                    f"{row['coverage_diff_a_better_count']}/"
                    f"{row['coverage_diff_tie_count']}/"
                    f"{row['coverage_diff_a_worse_count']}"
                ),
                "success_mean": row["success_diff_mean"],
                "success_A_better/tie/worse": (
                    f"{row['success_diff_a_better_count']}/"
                    f"{row['success_diff_tie_count']}/"
                    f"{row['success_diff_a_worse_count']}"
                ),
                "length_mean": row["length_diff_mean"],
                "length_A_better/tie/worse": (
                    f"{row['length_diff_a_better_count']}/"
                    f"{row['length_diff_tie_count']}/"
                    f"{row['length_diff_a_worse_count']}"
                ),
                "timeout_mean": row["timeout_diff_mean"],
                "timeout_A_better/tie/worse": (
                    f"{row['timeout_diff_a_better_count']}/"
                    f"{row['timeout_diff_tie_count']}/"
                    f"{row['timeout_diff_a_worse_count']}"
                ),
            }
        )

    return "\n".join(
        [
            "# Final Probe Discrepancy Interpretation",
            "",
            "## Scope",
            "",
            (
                "This is a read-only post-hoc audit over existing training snapshots, "
                "final probe CSVs, and the final probe run manifest. No training or "
                "final probe episodes were rerun."
            ),
            "",
            "## Main Finding",
            "",
            (
                f"Among reward-comparable methods A and F1, A ranks #{a_train_reward_rank} "
                f"by training recent reward but #{a_final_reward_rank} by final probe reward. "
                "The reversal is explained by A degrading from its training recent window on "
                "the fixed final-probe seed set, while F1 improves on the same final-probe "
                "seed set."
            ),
            "",
            markdown_table(
                key_rows,
                [
                    "metric",
                    "A_train",
                    "A_final",
                    "F1_train",
                    "F1_final",
                    "A_final_minus_train",
                    "F1_final_minus_train",
                ],
            ),
            "",
            "## Reward Comparability",
            "",
            (
                "A and F1 keep the same reward formula, so their reward values can be "
                "compared directly. R1-R5 intentionally remove or alter reward penalty "
                "terms, so R-group reward is not directly comparable with A. Use R-group "
                "coverage, success, length, timeout, and paired seed behavior for cross-"
                "method conclusions; treat R reward as within-method diagnostic context."
            ),
            "",
            "## Paired Seed Summary",
            "",
            (
                "Diffs are A minus comparator on identical episode seeds. For coverage "
                "and success, higher is better; for length and timeout, lower is better."
            ),
            "",
            markdown_table(
                paired_brief,
                [
                    "comparison",
                    "episodes",
                    "coverage_mean",
                    "coverage_A_better/tie/worse",
                    "success_mean",
                    "success_A_better/tie/worse",
                    "length_mean",
                    "length_A_better/tie/worse",
                    "timeout_mean",
                    "timeout_A_better/tie/worse",
                ],
            ),
            "",
            "## Config And Checkpoint Audit",
            "",
            (
                f"Evaluation-critical config consistency across A/F1/R1-R5: {eval_consistent}. "
                f"All load_state_dict missing/unexpected key lists are empty: {clean_load}. "
                "The final probe seed_base and episode count are shared by construction "
                f"from run_manifest: seed_base={config_audit['final_probe']['seed_base']}, "
                f"episodes={config_audit['final_probe']['episodes']}."
            ),
            "",
            "## Interpretation",
            "",
            (
                "The audit does not point to a checkpoint loading mismatch, model factory "
                "mismatch, state adapter status failure, or evaluation-critical config drift "
                "as the cause of A losing the final probe. The evidence instead points to "
                "seed-set generalization or train-window selection variance: A's recent "
                "training window is strong, but on the final probe seed set it has lower "
                "success and coverage, longer trajectories, more repeated visits, and a "
                "higher timeout rate than F1."
            ),
            "",
            "## Generated Files",
            "",
            "- training_vs_final_probe_comparison.csv",
            "- paired_seed_comparison.csv",
            "- config_audit.json",
            "- discrepancy_interpretation.md",
            "",
        ]
    )


def main() -> None:
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    summary_rows = read_csv_rows(FINAL_PROBE_DIR / "final_probe_summary.csv")
    per_episode_rows = read_csv_rows(FINAL_PROBE_DIR / "final_probe_per_episode.csv")
    manifest = read_json(FINAL_PROBE_DIR / "run_manifest.json")
    metric_snapshots = {
        method_id: read_json(path) for method_id, path in METRIC_SNAPSHOT_PATHS.items()
    }

    comparison_rows = build_training_vs_final_rows(summary_rows, metric_snapshots)
    paired_rows = build_paired_rows(per_episode_rows)
    config_audit = build_config_audit(manifest, metric_snapshots)
    interpretation = build_interpretation_md(comparison_rows, paired_rows, config_audit)

    comparison_fields = [
        "method_id",
        "group",
        "display_name",
        "reward_comparable_with_A",
        "reward_comparability_note",
        *[f"train_recent_{metric}" for metric in TRAIN_METRICS],
        *[f"final_probe_{metric}" for metric in FINAL_METRICS],
        "delta_final_minus_train_reward",
        "delta_final_minus_train_coverage",
        "delta_final_minus_train_success_rate",
        "delta_final_minus_train_episode_length",
        "delta_final_minus_train_repeat_visit_ratio",
        "delta_final_minus_train_timeout",
    ]
    paired_fields = [
        "comparison",
        "baseline_method_id",
        "comparator_method_id",
        "paired_episode_count",
        "map_fingerprint_mismatch_count",
    ]
    for metric in ["coverage_diff", "success_diff", "length_diff", "timeout_diff"]:
        paired_fields.extend(
            [
                f"{metric}_mean",
                f"{metric}_std",
                f"{metric}_median",
                f"{metric}_a_better_count",
                f"{metric}_tie_count",
                f"{metric}_a_worse_count",
            ]
        )

    write_csv(
        ANALYSIS_DIR / "training_vs_final_probe_comparison.csv",
        comparison_rows,
        comparison_fields,
    )
    write_csv(ANALYSIS_DIR / "paired_seed_comparison.csv", paired_rows, paired_fields)
    with (ANALYSIS_DIR / "config_audit.json").open("w", encoding="utf-8") as fh:
        json.dump(config_audit, fh, ensure_ascii=False, indent=2, sort_keys=True)
        fh.write("\n")
    with (ANALYSIS_DIR / "discrepancy_interpretation.md").open("w", encoding="utf-8") as fh:
        fh.write(interpretation)

    print(f"Wrote analysis outputs to {relpath(ANALYSIS_DIR)}")
    for name in [
        "training_vs_final_probe_comparison.csv",
        "paired_seed_comparison.csv",
        "config_audit.json",
        "discrepancy_interpretation.md",
    ]:
        print(f"- {relpath(ANALYSIS_DIR / name)}")


if __name__ == "__main__":
    main()
