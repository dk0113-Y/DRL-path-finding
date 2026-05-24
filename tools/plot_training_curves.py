"""Plot formal training curves and audit train-side checkpoint candidates.

The script reads existing experiment records and local output logs only. It does
not rerun training or final probes and does not copy model checkpoint files.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_RECORDS = REPO_ROOT / "experiment_records"
OUTPUTS_DIR = REPO_ROOT / "outputs"
CHECKPOINT_STORE = REPO_ROOT / "checkpoint_store"
ANALYSIS_DIR = EXPERIMENT_RECORDS / "training_curve_analysis"
FIGURE_DIR = ANALYSIS_DIR / "figures"

METHOD_ORDER = ["A", "C", "D", "E", "F1", "F2", "F3", "F4", "F5", "R1", "R2", "R3", "R4", "R5"]
CORE_METHODS = ["A", "C", "D", "E"]
SUSPICIOUS_METHODS = ["A", "F1", "R1", "R2", "R3", "R4", "R5"]

METRICS = [
    ("recent_mean_coverage", "Recent mean coverage", "higher"),
    ("recent_success_rate", "Recent success rate", "higher"),
    ("recent_mean_reward", "Recent mean reward", "higher"),
    ("recent_mean_episode_length", "Recent mean episode length", "lower"),
    ("recent_mean_repeat_visit_ratio", "Recent repeat visit ratio", "lower"),
    ("recent_timeout_rate", "Recent timeout rate", "lower"),
    ("loss", "Loss", "lower"),
    ("td_abs_mean", "TD abs mean", "lower"),
    ("q_mean", "Q mean", "context"),
    ("grad_norm", "Grad norm", "lower"),
]

METHODS = {
    "A": {
        "record_dir": EXPERIMENT_RECORDS / "full_method_main" / "logs",
        "output_glob": None,
        "checkpoint_store": CHECKPOINT_STORE / "full_method_main" / "A_full_method.pt",
    },
    "C": {
        "record_dir": EXPERIMENT_RECORDS / "baselines" / "C_baseline_local_state_ddqn" / "logs",
        "output_glob": "C_baseline_local_state_ddqn_formal_*",
        "checkpoint_store": CHECKPOINT_STORE / "baselines" / "C_baseline_local_state_ddqn.pt",
    },
    "D": {
        "record_dir": EXPERIMENT_RECORDS / "ablations" / "D_ablation_no_value_tree" / "logs",
        "output_glob": "D_ablation_no_value_tree_formal_*",
        "checkpoint_store": CHECKPOINT_STORE / "ablations" / "D_ablation_no_value_tree.pt",
    },
    "E": {
        "record_dir": EXPERIMENT_RECORDS / "ablations" / "E_ablation_no_semantic_dual_state_split" / "logs",
        "output_glob": "E_ablation_no_semantic_dual_state_split_formal_*",
        "checkpoint_store": CHECKPOINT_STORE
        / "ablations"
        / "E_ablation_no_semantic_dual_state_split.pt",
    },
    "F1": {
        "record_dir": EXPERIMENT_RECORDS / "ablations" / "F1_ablation_no_frontier_channel" / "logs",
        "output_glob": "F1_ablation_no_frontier_channel_formal_*",
        "checkpoint_store": CHECKPOINT_STORE / "ablations" / "F1_ablation_no_frontier_channel.pt",
    },
    "F2": {
        "record_dir": EXPERIMENT_RECORDS / "ablations" / "F2_ablation_no_visit_count_channel" / "logs",
        "output_glob": "F2_ablation_no_visit_count_channel_formal_*",
        "checkpoint_store": CHECKPOINT_STORE / "ablations" / "F2_ablation_no_visit_count_channel.pt",
    },
    "F3": {
        "record_dir": EXPERIMENT_RECORDS
        / "ablations"
        / "F3_ablation_no_recent_trajectory_channel"
        / "logs",
        "output_glob": "F3_ablation_no_recent_trajectory_channel_formal_*",
        "checkpoint_store": CHECKPOINT_STORE
        / "ablations"
        / "F3_ablation_no_recent_trajectory_channel.pt",
    },
    "F4": {
        "record_dir": EXPERIMENT_RECORDS / "ablations" / "F4_ablation_no_visit_traj_channels" / "logs",
        "output_glob": "F4_ablation_no_visit_traj_channels_formal_*",
        "checkpoint_store": CHECKPOINT_STORE / "ablations" / "F4_ablation_no_visit_traj_channels.pt",
    },
    "F5": {
        "record_dir": EXPERIMENT_RECORDS / "ablations" / "F5_ablation_occupancy_only_canvas" / "logs",
        "output_glob": "F5_ablation_occupancy_only_canvas_formal_*",
        "checkpoint_store": CHECKPOINT_STORE / "ablations" / "F5_ablation_occupancy_only_canvas.pt",
    },
    "R1": {
        "record_dir": EXPERIMENT_RECORDS / "ablations" / "R1_ablation_no_step_penalty" / "logs",
        "output_glob": "R1_ablation_no_step_penalty_formal_*",
        "checkpoint_store": CHECKPOINT_STORE / "ablations" / "R1_ablation_no_step_penalty.pt",
    },
    "R2": {
        "record_dir": EXPERIMENT_RECORDS / "ablations" / "R2_ablation_no_revisit_penalty" / "logs",
        "output_glob": "R2_ablation_no_revisit_penalty_formal_*",
        "checkpoint_store": CHECKPOINT_STORE / "ablations" / "R2_ablation_no_revisit_penalty.pt",
    },
    "R3": {
        "record_dir": EXPERIMENT_RECORDS / "ablations" / "R3_ablation_no_turn_penalty" / "logs",
        "output_glob": "R3_ablation_no_turn_penalty_formal_*",
        "checkpoint_store": CHECKPOINT_STORE / "ablations" / "R3_ablation_no_turn_penalty.pt",
    },
    "R4": {
        "record_dir": EXPERIMENT_RECORDS / "ablations" / "R4_ablation_no_timeout_penalty" / "logs",
        "output_glob": "R4_ablation_no_timeout_penalty_formal_*",
        "checkpoint_store": CHECKPOINT_STORE / "ablations" / "R4_ablation_no_timeout_penalty.pt",
    },
    "R5": {
        "record_dir": EXPERIMENT_RECORDS / "ablations" / "R5_ablation_no_efficiency_penalties" / "logs",
        "output_glob": "R5_ablation_no_efficiency_penalties_formal_*",
        "checkpoint_store": CHECKPOINT_STORE / "ablations" / "R5_ablation_no_efficiency_penalties.pt",
    },
}


@dataclass
class MethodData:
    method_id: str
    train_steps_path: Path | None
    train_episodes_path: Path | None
    metric_snapshot_path: Path | None
    output_run_dir: Path | None
    rows: list[dict[str, Any]]


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def parse_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(parsed) or math.isinf(parsed):
        return None
    return parsed


def format_value(value: Any) -> Any:
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return ""
        return f"{value:.10g}"
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    if value is None:
        return ""
    return value


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({name: format_value(row.get(name)) for name in fieldnames})


def relpath(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def latest_matching_output_dir(pattern: str | None) -> Path | None:
    if not pattern:
        return None
    matches = [path for path in OUTPUTS_DIR.glob(pattern) if path.is_dir()]
    formal_matches = [path for path in matches if "_formal_" in path.name]
    candidates = formal_matches or matches
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def resolve_existing_path(
    method_id: str,
    filename: str,
    output_run_dir: Path | None,
) -> Path | None:
    record_path = METHODS[method_id]["record_dir"] / filename
    if record_path.exists():
        return record_path
    if output_run_dir:
        output_path = output_run_dir / "logs" / filename
        if output_path.exists():
            return output_path
    return None


def load_method_data(method_id: str) -> MethodData:
    output_run_dir = latest_matching_output_dir(METHODS[method_id]["output_glob"])
    train_steps_path = resolve_existing_path(method_id, "train_steps.csv", output_run_dir)
    train_episodes_path = resolve_existing_path(method_id, "train_episodes.csv", output_run_dir)
    metric_snapshot_path = resolve_existing_path(method_id, "metric_snapshot.json", output_run_dir)
    rows: list[dict[str, Any]] = []
    if train_steps_path is not None:
        for row in read_csv_rows(train_steps_path):
            parsed_row: dict[str, Any] = {"method_id": method_id}
            for key, value in row.items():
                parsed_row[key] = parse_float(value)
            rows.append(parsed_row)
        rows = [row for row in rows if row.get("env_steps") is not None]
        rows.sort(key=lambda row: row["env_steps"])
    return MethodData(
        method_id=method_id,
        train_steps_path=train_steps_path,
        train_episodes_path=train_episodes_path,
        metric_snapshot_path=metric_snapshot_path,
        output_run_dir=output_run_dir,
        rows=rows,
    )


def max_row(rows: list[dict[str, Any]], metric: str) -> dict[str, Any] | None:
    valid = [row for row in rows if row.get(metric) is not None]
    if not valid:
        return None
    return max(valid, key=lambda row: row[metric])


def last_row(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    return rows[-1] if rows else None


def build_peak_summary(method_data: dict[str, MethodData]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for method_id in METHOD_ORDER:
        data = method_data[method_id]
        last = last_row(data.rows)
        best_cov = max_row(data.rows, "recent_mean_coverage")
        best_success = max_row(data.rows, "recent_success_rate")
        best_reward = max_row(data.rows, "recent_mean_reward")

        row = {
            "method_id": method_id,
            "train_steps_path": relpath(data.train_steps_path),
            "train_episodes_path": relpath(data.train_episodes_path),
            "metric_snapshot_path": relpath(data.metric_snapshot_path),
            "row_count": len(data.rows),
            "best_train_coverage_env_steps": best_cov.get("env_steps") if best_cov else None,
            "best_train_coverage": best_cov.get("recent_mean_coverage") if best_cov else None,
            "best_train_success_env_steps": best_success.get("env_steps") if best_success else None,
            "best_train_success_rate": best_success.get("recent_success_rate") if best_success else None,
            "best_train_reward_env_steps": best_reward.get("env_steps") if best_reward else None,
            "best_train_reward": best_reward.get("recent_mean_reward") if best_reward else None,
            "final_logged_env_steps": last.get("env_steps") if last else None,
            "final_logged_coverage": last.get("recent_mean_coverage") if last else None,
            "final_logged_success_rate": last.get("recent_success_rate") if last else None,
            "final_logged_reward": last.get("recent_mean_reward") if last else None,
        }
        row["peak_minus_last_coverage"] = subtract(
            row["best_train_coverage"], row["final_logged_coverage"]
        )
        row["peak_minus_last_success_rate"] = subtract(
            row["best_train_success_rate"], row["final_logged_success_rate"]
        )
        row["peak_minus_last_reward"] = subtract(
            row["best_train_reward"], row["final_logged_reward"]
        )
        row["last_is_train_side_coverage_peak"] = near_zero(row["peak_minus_last_coverage"])
        row["last_is_train_side_success_peak"] = near_zero(row["peak_minus_last_success_rate"])
        row["last_is_train_side_reward_peak"] = near_zero(row["peak_minus_last_reward"])
        row["last_not_train_side_peak_any_primary"] = any(
            not row[key]
            for key in [
                "last_is_train_side_coverage_peak",
                "last_is_train_side_success_peak",
                "last_is_train_side_reward_peak",
            ]
        )
        rows.append(row)
    return rows


def subtract(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return left - right


def near_zero(value: float | None, tolerance: float = 1e-9) -> bool:
    return value is not None and abs(value) <= tolerance


def late_window(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if row.get("env_steps") is not None and 400_000 <= row["env_steps"] <= 500_000
    ]


def mean_metric(rows: list[dict[str, Any]], metric: str) -> float | None:
    values = [row[metric] for row in rows if row.get(metric) is not None]
    if not values:
        return None
    return sum(values) / len(values)


def min_metric(rows: list[dict[str, Any]], metric: str) -> float | None:
    values = [row[metric] for row in rows if row.get(metric) is not None]
    return min(values) if values else None


def max_metric(rows: list[dict[str, Any]], metric: str) -> float | None:
    values = [row[metric] for row in rows if row.get(metric) is not None]
    return max(values) if values else None


def build_late_stage_summary(method_data: dict[str, MethodData]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for method_id in SUSPICIOUS_METHODS:
        data = method_data[method_id]
        window = late_window(data.rows)
        last = last_row(data.rows)
        row = {
            "method_id": method_id,
            "late_window_row_count": len(window),
            "final_logged_env_steps": last.get("env_steps") if last else None,
            "late_window_mean_coverage": mean_metric(window, "recent_mean_coverage"),
            "late_window_peak_coverage": max_metric(window, "recent_mean_coverage"),
            "final_logged_coverage": last.get("recent_mean_coverage") if last else None,
            "late_window_mean_success_rate": mean_metric(window, "recent_success_rate"),
            "late_window_peak_success_rate": max_metric(window, "recent_success_rate"),
            "final_logged_success_rate": last.get("recent_success_rate") if last else None,
            "late_window_mean_timeout_rate": mean_metric(window, "recent_timeout_rate"),
            "late_window_min_timeout_rate": min_metric(window, "recent_timeout_rate"),
            "final_logged_timeout_rate": last.get("recent_timeout_rate") if last else None,
            "late_window_mean_episode_length": mean_metric(window, "recent_mean_episode_length"),
            "late_window_min_episode_length": min_metric(window, "recent_mean_episode_length"),
            "final_logged_episode_length": last.get("recent_mean_episode_length") if last else None,
        }
        row["coverage_drop_from_late_peak"] = subtract(
            row["late_window_peak_coverage"], row["final_logged_coverage"]
        )
        row["success_drop_from_late_peak"] = subtract(
            row["late_window_peak_success_rate"], row["final_logged_success_rate"]
        )
        row["timeout_increase_from_late_min"] = subtract(
            row["final_logged_timeout_rate"], row["late_window_min_timeout_rate"]
        )
        row["episode_length_increase_from_late_min"] = subtract(
            row["final_logged_episode_length"], row["late_window_min_episode_length"]
        )
        row["coverage_declined_late_stage"] = positive(row["coverage_drop_from_late_peak"])
        row["success_declined_late_stage"] = positive(row["success_drop_from_late_peak"])
        row["timeout_increased_late_stage"] = positive(row["timeout_increase_from_late_min"])
        row["episode_length_increased_late_stage"] = positive(
            row["episode_length_increase_from_late_min"]
        )
        row["late_stage_degradation_signal"] = (
            row["coverage_declined_late_stage"]
            or row["success_declined_late_stage"]
            or row["timeout_increased_late_stage"]
            or row["episode_length_increased_late_stage"]
        )
        rows.append(row)
    return rows


def positive(value: float | None, tolerance: float = 1e-9) -> bool:
    return value is not None and value > tolerance


def checkpoint_files_under(path: Path | None) -> list[Path]:
    if path is None:
        return []
    checkpoint_dir = path / "checkpoints"
    if not checkpoint_dir.exists():
        return []
    return sorted(
        [
            file
            for file in checkpoint_dir.rglob("*")
            if file.is_file() and file.suffix.lower() in {".pt", ".pth", ".ckpt"}
        ]
    )


def is_periodic_checkpoint(path: Path) -> bool:
    name = path.name.lower()
    if name in {"last.pt", "best.pt", "last.pth", "best.pth", "last.ckpt", "best.ckpt"}:
        return False
    periodic_tokens = ["periodic", "checkpoint", "env", "step", "iter", "episode"]
    return any(token in name for token in periodic_tokens)


def build_checkpoint_audit(method_data: dict[str, MethodData]) -> dict[str, Any]:
    methods: dict[str, Any] = {}
    for method_id in METHOD_ORDER:
        data = method_data[method_id]
        store_path = METHODS[method_id]["checkpoint_store"]
        output_files = checkpoint_files_under(data.output_run_dir)
        best_files = [path for path in output_files if path.name.lower() == "best.pt"]
        last_files = [path for path in output_files if path.name.lower() == "last.pt"]
        periodic_files = [path for path in output_files if is_periodic_checkpoint(path)]
        methods[method_id] = {
            "output_run_dir": relpath(data.output_run_dir),
            "checkpoint_store_export_path": relpath(store_path),
            "checkpoint_store_export_exists": store_path.exists(),
            "output_checkpoint_files": [relpath(path) for path in output_files],
            "output_checkpoint_count": len(output_files),
            "output_last_pt_exists": bool(last_files),
            "output_best_pt_exists": bool(best_files),
            "output_periodic_checkpoint_exists": bool(periodic_files),
            "output_periodic_checkpoint_files": [relpath(path) for path in periodic_files],
            "only_last_endpoint_checkpoint": bool(last_files)
            and len(output_files) == len(last_files)
            and not best_files
            and not periodic_files,
        }

    return {
        "schema_version": "training_curve_checkpoint_candidate_audit/v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "note": "Checkpoint files were only inspected for paths and existence; no weights were copied.",
        "methods": methods,
        "any_periodic_checkpoint": any(
            method["output_periodic_checkpoint_exists"] for method in methods.values()
        ),
        "any_best_pt": any(method["output_best_pt_exists"] for method in methods.values()),
        "all_methods_only_last_endpoint_checkpoint": all(
            method["only_last_endpoint_checkpoint"] for method in methods.values()
        ),
    }


def method_color(method_id: str) -> str:
    colors = {
        "A": "#1f77b4",
        "C": "#7f7f7f",
        "D": "#2ca02c",
        "E": "#9467bd",
        "F1": "#ff7f0e",
        "F2": "#ffbb78",
        "F3": "#d62728",
        "F4": "#8c564b",
        "F5": "#e377c2",
        "R1": "#17becf",
        "R2": "#bcbd22",
        "R3": "#aec7e8",
        "R4": "#98df8a",
        "R5": "#c5b0d5",
    }
    return colors.get(method_id, "#333333")


def plot_category(
    method_data: dict[str, MethodData],
    method_ids: list[str],
    title: str,
    output_path: Path,
    reward_note: bool,
) -> None:
    fig, axes = plt.subplots(5, 2, figsize=(11, 15), sharex=False)
    axes_flat = list(axes.flat)
    for axis, (metric, label, _direction) in zip(axes_flat, METRICS):
        for method_id in method_ids:
            rows = method_data[method_id].rows
            points = [
                (row["env_steps"], row[metric])
                for row in rows
                if row.get("env_steps") is not None and row.get(metric) is not None
            ]
            if not points:
                continue
            x_values, y_values = zip(*points)
            axis.plot(
                x_values,
                y_values,
                label=method_id,
                linewidth=1.35 if method_id == "A" else 0.95,
                color=method_color(method_id),
                alpha=0.95,
            )
        axis.set_title(label, fontsize=10)
        axis.set_xlabel("env_steps")
        axis.grid(True, alpha=0.25, linewidth=0.5)
        axis.ticklabel_format(axis="x", style="sci", scilimits=(5, 5))
    axes_flat[0].legend(ncol=min(7, len(method_ids)), fontsize=8, loc="best")
    note = ""
    if reward_note:
        note = (
            "\nR-group rewards are not directly comparable with A because reward penalty "
            "terms are ablated."
        )
    fig.suptitle(title + note, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.savefig(output_path, dpi=300, facecolor="white", pil_kwargs={"optimize": True})
    plt.close(fig)
    optimize_png(output_path)


def optimize_png(path: Path) -> None:
    image = Image.open(path).convert("RGB")
    quantized = image.convert("P", palette=Image.Palette.ADAPTIVE, colors=128)
    quantized.save(path, optimize=True, dpi=(300, 300))


def markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(format_value(row.get(col))) for col in columns) + " |")
    return "\n".join(lines)


def build_interpretation(
    peak_rows: list[dict[str, Any]],
    late_rows: list[dict[str, Any]],
    checkpoint_audit: dict[str, Any],
) -> str:
    peak_by_method = {row["method_id"]: row for row in peak_rows}
    late_by_method = {row["method_id"]: row for row in late_rows}
    key_methods = ["A", "F1", "R1", "R2", "R3", "R4", "R5"]
    peak_brief = [
        {
            "method_id": method,
            "peak-last coverage": peak_by_method[method]["peak_minus_last_coverage"],
            "peak-last success": peak_by_method[method]["peak_minus_last_success_rate"],
            "peak-last reward": peak_by_method[method]["peak_minus_last_reward"],
            "last not peak": peak_by_method[method]["last_not_train_side_peak_any_primary"],
        }
        for method in key_methods
    ]
    late_brief = [
        {
            "method_id": method,
            "coverage drop": late_by_method[method]["coverage_drop_from_late_peak"],
            "success drop": late_by_method[method]["success_drop_from_late_peak"],
            "timeout increase": late_by_method[method]["timeout_increase_from_late_min"],
            "length increase": late_by_method[method]["episode_length_increase_from_late_min"],
            "signal": late_by_method[method]["late_stage_degradation_signal"],
        }
        for method in key_methods
    ]
    checkpoint_methods = checkpoint_audit["methods"]
    checkpoint_brief = [
        {
            "method_id": method,
            "only last.pt": checkpoint_methods[method]["only_last_endpoint_checkpoint"],
            "periodic": checkpoint_methods[method]["output_periodic_checkpoint_exists"],
            "best.pt": checkpoint_methods[method]["output_best_pt_exists"],
            "store export": checkpoint_methods[method]["checkpoint_store_export_exists"],
        }
        for method in METHOD_ORDER
    ]
    a_late = late_by_method["A"]
    a_degraded = a_late["late_stage_degradation_signal"]

    return "\n".join(
        [
            "# Training Curve Interpretation",
            "",
            "## Scope",
            "",
            (
                "This analysis reads existing formal train logs, metric snapshots, final "
                "probe summaries, and checkpoint path inventories. It does not rerun "
                "training or final probe evaluation, and it does not copy checkpoint files."
            ),
            "",
            "B, the classical frontier greedy baseline, is excluded from training curves.",
            "",
            "## Reward Comparability",
            "",
            (
                "R-group rewards are not directly comparable with A because reward penalty "
                "terms are ablated. Use R-group reward curves as within-method diagnostics; "
                "compare R-group methods against A primarily through coverage, success, "
                "episode length, timeout, and repeat-visit trends."
            ),
            "",
            "## Peak Vs Last Summary",
            "",
            markdown_table(
                peak_brief,
                [
                    "method_id",
                    "peak-last coverage",
                    "peak-last success",
                    "peak-last reward",
                    "last not peak",
                ],
            ),
            "",
            "## Late-Stage Degradation Signals",
            "",
            (
                "Late-stage rows compare the 400k-500k env-step window with the final "
                "logged point. Positive coverage/success drops and positive timeout/length "
                "increases indicate that the final logged point is worse than a late-window "
                "reference point."
            ),
            "",
            markdown_table(
                late_brief,
                [
                    "method_id",
                    "coverage drop",
                    "success drop",
                    "timeout increase",
                    "length increase",
                    "signal",
                ],
            ),
            "",
            "## A-Specific Read",
            "",
            (
                f"A late-stage degradation signal: {a_degraded}. "
                f"Coverage drop from late-window peak is {format_value(a_late['coverage_drop_from_late_peak'])}, "
                f"success drop is {format_value(a_late['success_drop_from_late_peak'])}, "
                f"timeout increase from late-window minimum is {format_value(a_late['timeout_increase_from_late_min'])}, "
                f"and episode-length increase from late-window minimum is "
                f"{format_value(a_late['episode_length_increase_from_late_min'])}."
            ),
            "",
            "## Checkpoint Candidate Audit",
            "",
            markdown_table(
                checkpoint_brief,
                ["method_id", "only last.pt", "periodic", "best.pt", "store export"],
            ),
            "",
            (
                f"Any periodic checkpoint found: {checkpoint_audit['any_periodic_checkpoint']}. "
                f"Any best.pt found: {checkpoint_audit['any_best_pt']}. "
                f"All methods only have last endpoint checkpoints in outputs: "
                f"{checkpoint_audit['all_methods_only_last_endpoint_checkpoint']}."
            ),
            "",
            "## Figures",
            "",
            "- figures/all_methods_overview.png",
            "- figures/core_methods.png",
            "- figures/suspicious_groups.png",
            "",
        ]
    )


def build_readme(
    method_data: dict[str, MethodData],
    checkpoint_audit: dict[str, Any],
) -> str:
    source_rows = [
        {
            "method_id": method_id,
            "train_steps": relpath(data.train_steps_path),
            "train_episodes": relpath(data.train_episodes_path),
            "metric_snapshot": relpath(data.metric_snapshot_path),
        }
        for method_id, data in method_data.items()
    ]
    return "\n".join(
        [
            "# Training Curve Analysis",
            "",
            "Generated by `python tools/plot_training_curves.py`.",
            "",
            "R-group rewards are not directly comparable with A because reward penalty terms are ablated.",
            "",
            "## Inputs",
            "",
            markdown_table(source_rows, ["method_id", "train_steps", "train_episodes", "metric_snapshot"]),
            "",
            "## Outputs",
            "",
            "- training_curve_peak_summary.csv",
            "- late_stage_degradation_summary.csv",
            "- checkpoint_candidate_audit.json",
            "- training_curve_interpretation.md",
            "- figures/all_methods_overview.png",
            "- figures/core_methods.png",
            "- figures/suspicious_groups.png",
            "",
            "## Checkpoint Scope",
            "",
            (
                "Checkpoint files were inspected only for path and existence metadata. "
                f"Any periodic checkpoint found: {checkpoint_audit['any_periodic_checkpoint']}. "
                f"Any best.pt found: {checkpoint_audit['any_best_pt']}."
            ),
            "",
        ]
    )


def main() -> None:
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    method_data = {method_id: load_method_data(method_id) for method_id in METHOD_ORDER}
    missing = [method_id for method_id, data in method_data.items() if not data.rows]
    if missing:
        raise FileNotFoundError(f"Missing train_steps rows for methods: {', '.join(missing)}")

    # Read these context files so missing or malformed records fail visibly.
    read_csv_rows(EXPERIMENT_RECORDS / "final_probe" / "final_probe_summary.csv")
    analysis_comparison = (
        EXPERIMENT_RECORDS
        / "final_probe"
        / "analysis"
        / "training_vs_final_probe_comparison.csv"
    )
    if analysis_comparison.exists():
        read_csv_rows(analysis_comparison)
    for method_id, data in method_data.items():
        if data.metric_snapshot_path:
            read_json(data.metric_snapshot_path)

    peak_rows = build_peak_summary(method_data)
    late_rows = build_late_stage_summary(method_data)
    checkpoint_audit = build_checkpoint_audit(method_data)

    peak_fields = [
        "method_id",
        "train_steps_path",
        "train_episodes_path",
        "metric_snapshot_path",
        "row_count",
        "best_train_coverage_env_steps",
        "best_train_coverage",
        "best_train_success_env_steps",
        "best_train_success_rate",
        "best_train_reward_env_steps",
        "best_train_reward",
        "final_logged_env_steps",
        "final_logged_coverage",
        "final_logged_success_rate",
        "final_logged_reward",
        "peak_minus_last_coverage",
        "peak_minus_last_success_rate",
        "peak_minus_last_reward",
        "last_is_train_side_coverage_peak",
        "last_is_train_side_success_peak",
        "last_is_train_side_reward_peak",
        "last_not_train_side_peak_any_primary",
    ]
    late_fields = [
        "method_id",
        "late_window_row_count",
        "final_logged_env_steps",
        "late_window_mean_coverage",
        "late_window_peak_coverage",
        "final_logged_coverage",
        "coverage_drop_from_late_peak",
        "coverage_declined_late_stage",
        "late_window_mean_success_rate",
        "late_window_peak_success_rate",
        "final_logged_success_rate",
        "success_drop_from_late_peak",
        "success_declined_late_stage",
        "late_window_mean_timeout_rate",
        "late_window_min_timeout_rate",
        "final_logged_timeout_rate",
        "timeout_increase_from_late_min",
        "timeout_increased_late_stage",
        "late_window_mean_episode_length",
        "late_window_min_episode_length",
        "final_logged_episode_length",
        "episode_length_increase_from_late_min",
        "episode_length_increased_late_stage",
        "late_stage_degradation_signal",
    ]

    write_csv(ANALYSIS_DIR / "training_curve_peak_summary.csv", peak_rows, peak_fields)
    write_csv(ANALYSIS_DIR / "late_stage_degradation_summary.csv", late_rows, late_fields)
    with (ANALYSIS_DIR / "checkpoint_candidate_audit.json").open("w", encoding="utf-8") as fh:
        json.dump(checkpoint_audit, fh, ensure_ascii=False, indent=2, sort_keys=True)
        fh.write("\n")

    plot_category(
        method_data,
        METHOD_ORDER,
        "Formal training curves: all methods except B",
        FIGURE_DIR / "all_methods_overview.png",
        reward_note=True,
    )
    plot_category(
        method_data,
        CORE_METHODS,
        "Formal training curves: core methods A/C/D/E",
        FIGURE_DIR / "core_methods.png",
        reward_note=False,
    )
    plot_category(
        method_data,
        SUSPICIOUS_METHODS,
        "Formal training curves: A vs F1 vs R1-R5",
        FIGURE_DIR / "suspicious_groups.png",
        reward_note=True,
    )

    interpretation = build_interpretation(peak_rows, late_rows, checkpoint_audit)
    with (ANALYSIS_DIR / "training_curve_interpretation.md").open("w", encoding="utf-8") as fh:
        fh.write(interpretation)

    readme = build_readme(method_data, checkpoint_audit)
    with (ANALYSIS_DIR / "README.md").open("w", encoding="utf-8") as fh:
        fh.write(readme)

    print(f"Wrote training curve analysis to {relpath(ANALYSIS_DIR)}")
    for output in [
        "training_curve_peak_summary.csv",
        "late_stage_degradation_summary.csv",
        "checkpoint_candidate_audit.json",
        "training_curve_interpretation.md",
        "README.md",
        "figures/all_methods_overview.png",
        "figures/core_methods.png",
        "figures/suspicious_groups.png",
    ]:
        print(f"- {relpath(ANALYSIS_DIR / output)}")


if __name__ == "__main__":
    main()
