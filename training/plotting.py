from __future__ import annotations

"""
Offline monitoring-plot generation.

This module stays out of the training hot path. It only reads CSV logs and
writes PNG dashboards when generate_all_plots(run_dir) is called explicitly by
the training script or by a post-run utility.
"""

import csv
import math
from pathlib import Path


_TRAIN_STEPS_CSV = "train_steps.csv"
_TRAIN_EPISODES_CSV = "train_episodes.csv"
_EVAL_METRICS_CSV = "eval_metrics.csv"


TRAIN_LEARNING_STATE_PANELS = (
    {"title": "Replay size", "y_label": "replay size", "series": (("replay_size", "replay size"),)},
    {"title": "Exploration epsilon", "y_label": "epsilon", "series": (("epsilon", "epsilon"),)},
    {"title": "Learner steps", "y_label": "learner steps", "series": (("learner_steps", "learner steps"),)},
)
TRAIN_VALUE_LEARNING_PANELS = (
    {"title": "Value loss", "y_label": "loss", "series": (("loss", "loss"),)},
    {"title": "TD magnitude", "y_label": "td abs mean", "series": (("td_abs_mean", "td abs mean"),)},
    {"title": "Gradient norm", "y_label": "grad norm", "series": (("grad_norm", "grad norm"),)},
    {
        "title": "Q vs target Q",
        "y_label": "q value",
        "series": (("q_mean", "q mean"), ("target_q_mean", "target q mean")),
    },
)
TRAIN_POLICY_PERFORMANCE_PANELS = (
    {"title": "Mean reward (recent)", "y_label": "reward", "series": (("recent_mean_reward", "mean reward"),)},
    {"title": "Coverage (recent)", "y_label": "coverage", "series": (("recent_mean_coverage", "coverage"),)},
    {"title": "Success rate (recent)", "y_label": "success rate", "series": (("recent_success_rate", "success rate"),)},
    {"title": "Episode length (recent)", "y_label": "steps", "series": (("recent_mean_episode_length", "episode length"),)},
    {
        "title": "Repeat-visit ratio (recent)",
        "y_label": "ratio",
        "series": (("recent_mean_repeat_visit_ratio", "repeat-visit ratio"),),
    },
)
TRAIN_SEMANTIC_SUMMARY_PANELS = (
    {
        "title": "Accessible block count (recent)",
        "y_label": "blocks",
        "series": (("recent_accessible_block_count", "accessible blocks"),),
    },
    {
        "title": "Accessible unknown area (recent)",
        "y_label": "cells",
        "series": (("recent_total_accessible_unknown_area", "unknown area"),),
    },
    {
        "title": "Frontier cluster count (recent)",
        "y_label": "clusters",
        "series": (("recent_total_frontier_cluster_count", "frontier clusters"),),
    },
    {
        "title": "Mean block area (recent)",
        "y_label": "cells",
        "series": (("recent_mean_block_area", "mean block area"),),
    },
    {
        "title": "Local frontier coverage (recent)",
        "y_label": "coverage",
        "series": (("recent_local_frontier_coverage", "local frontier coverage"),),
    },
    {
        "title": "Local frontier block-area mean (recent)",
        "y_label": "area",
        "series": (("recent_local_frontier_block_area_mean", "local frontier block-area mean"),),
    },
)
TRAIN_REWARD_BREAKDOWN_PANELS = (
    {"title": "Information reward", "y_label": "reward", "series": (("info_reward_sum", "info reward"),)},
    {"title": "Step penalty", "y_label": "penalty", "series": (("step_penalty_sum", "step penalty"),)},
    {
        "title": "Revisit penalty",
        "y_label": "penalty",
        "series": (("recent_revisit_penalty_sum", "revisit penalty"),),
    },
    {"title": "Stall penalty", "y_label": "penalty", "series": (("stall_penalty_sum", "stall penalty"),)},
    {"title": "Timeout penalty", "y_label": "penalty", "series": (("timeout_penalty_sum", "timeout penalty"),)},
    {"title": "Terminal bonus", "y_label": "reward", "series": (("terminal_bonus_sum", "terminal bonus"),)},
)
TRAIN_REWARD_EVENT_PANELS = (
    {"title": "Revealed free cells", "y_label": "count", "series": (("delta_empty_sum", "free cells"),)},
    {
        "title": "Revealed obstacle cells",
        "y_label": "count",
        "series": (("delta_obstacle_sum", "obstacle cells"),),
    },
    {
        "title": "Weighted information gain",
        "y_label": "gain",
        "series": (("weighted_info_gain_sum", "weighted info gain"),),
    },
    {"title": "Recent revisit count", "y_label": "count", "series": (("recent_revisit_count", "recent revisit"),)},
    {"title": "Stall trigger count", "y_label": "count", "series": (("stall_trigger_count", "stall trigger"),)},
    {"title": "Zero-info step count", "y_label": "count", "series": (("zero_info_step_count", "zero-info steps"),)},
    {"title": "Timeout indicator", "y_label": "flag", "series": (("timeout_flag", "timeout flag"),)},
)

EVAL_POLICY_PERFORMANCE_PANELS = (
    {"title": "Mean reward", "y_label": "reward", "series": (("eval_mean_reward", "mean reward"),)},
    {"title": "Mean coverage", "y_label": "coverage", "series": (("eval_mean_coverage", "coverage"),)},
    {"title": "Success rate", "y_label": "success rate", "series": (("eval_success_rate", "success rate"),)},
    {"title": "Mean episode length", "y_label": "steps", "series": (("eval_mean_episode_length", "episode length"),)},
    {
        "title": "Mean repeat-visit ratio",
        "y_label": "ratio",
        "series": (("eval_mean_repeat_visit_ratio", "repeat-visit ratio"),),
    },
)
EVAL_SEMANTIC_SUMMARY_PANELS = (
    {
        "title": "Accessible block count",
        "y_label": "blocks",
        "series": (("eval_mean_accessible_block_count", "accessible blocks"),),
    },
    {
        "title": "Accessible unknown area",
        "y_label": "cells",
        "series": (("eval_mean_total_accessible_unknown_area", "unknown area"),),
    },
    {
        "title": "Frontier cluster count",
        "y_label": "clusters",
        "series": (("eval_mean_total_frontier_cluster_count", "frontier clusters"),),
    },
    {
        "title": "Mean block area",
        "y_label": "cells",
        "series": (("eval_mean_mean_block_area", "mean block area"),),
    },
    {
        "title": "Local frontier coverage",
        "y_label": "coverage",
        "series": (("eval_mean_local_frontier_coverage", "local frontier coverage"),),
    },
    {
        "title": "Local frontier block-area mean",
        "y_label": "area",
        "series": (("eval_mean_local_frontier_block_area_mean", "local frontier block-area mean"),),
    },
)
EVAL_REWARD_BREAKDOWN_PANELS = (
    {"title": "Information reward", "y_label": "reward", "series": (("eval_mean_info_reward_sum", "info reward"),)},
    {"title": "Step penalty", "y_label": "penalty", "series": (("eval_mean_step_penalty_sum", "step penalty"),)},
    {
        "title": "Revisit penalty",
        "y_label": "penalty",
        "series": (("eval_mean_recent_revisit_penalty_sum", "revisit penalty"),),
    },
    {"title": "Stall penalty", "y_label": "penalty", "series": (("eval_mean_stall_penalty_sum", "stall penalty"),)},
    {"title": "Timeout penalty", "y_label": "penalty", "series": (("eval_mean_timeout_penalty_sum", "timeout penalty"),)},
    {"title": "Terminal bonus", "y_label": "reward", "series": (("eval_mean_terminal_bonus_sum", "terminal bonus"),)},
)
EVAL_REWARD_EVENT_PANELS = (
    {"title": "Revealed free cells", "y_label": "count", "series": (("eval_mean_delta_empty_sum", "free cells"),)},
    {
        "title": "Revealed obstacle cells",
        "y_label": "count",
        "series": (("eval_mean_delta_obstacle_sum", "obstacle cells"),),
    },
    {
        "title": "Weighted information gain",
        "y_label": "gain",
        "series": (("eval_mean_weighted_info_gain_sum", "weighted info gain"),),
    },
    {"title": "Recent revisit count", "y_label": "count", "series": (("eval_mean_recent_revisit_count", "recent revisit"),)},
    {"title": "Stall trigger count", "y_label": "count", "series": (("eval_mean_stall_trigger_count", "stall trigger"),)},
    {"title": "Zero-info step count", "y_label": "count", "series": (("eval_mean_zero_info_step_count", "zero-info steps"),)},
    {"title": "Timeout indicator", "y_label": "flag", "series": (("eval_mean_timeout_flag", "timeout flag"),)},
)

_DASHBOARD_SPECS = (
    {
        "csv_name": _TRAIN_STEPS_CSV,
        "filename": "train_learning_state.png",
        "title": "Train learning state",
        "panels": TRAIN_LEARNING_STATE_PANELS,
        "ncols": 2,
    },
    {
        "csv_name": _TRAIN_STEPS_CSV,
        "filename": "train_value_learning.png",
        "title": "Train value learning",
        "panels": TRAIN_VALUE_LEARNING_PANELS,
        "ncols": 2,
    },
    {
        "csv_name": _TRAIN_STEPS_CSV,
        "filename": "train_policy_performance.png",
        "title": "Train policy performance",
        "panels": TRAIN_POLICY_PERFORMANCE_PANELS,
        "ncols": 2,
    },
    {
        "csv_name": _TRAIN_STEPS_CSV,
        "filename": "train_semantic_summary.png",
        "title": "Train semantic summary",
        "panels": TRAIN_SEMANTIC_SUMMARY_PANELS,
        "ncols": 2,
    },
    {
        "csv_name": _TRAIN_EPISODES_CSV,
        "filename": "train_reward_breakdown.png",
        "title": "Train reward breakdown",
        "panels": TRAIN_REWARD_BREAKDOWN_PANELS,
        "ncols": 3,
    },
    {
        "csv_name": _TRAIN_EPISODES_CSV,
        "filename": "train_reward_event_summary.png",
        "title": "Train reward event summary",
        "panels": TRAIN_REWARD_EVENT_PANELS,
        "ncols": 3,
    },
    {
        "csv_name": _EVAL_METRICS_CSV,
        "filename": "eval_policy_performance.png",
        "title": "Eval policy performance",
        "panels": EVAL_POLICY_PERFORMANCE_PANELS,
        "ncols": 2,
    },
    {
        "csv_name": _EVAL_METRICS_CSV,
        "filename": "eval_semantic_summary.png",
        "title": "Eval semantic summary",
        "panels": EVAL_SEMANTIC_SUMMARY_PANELS,
        "ncols": 2,
    },
    {
        "csv_name": _EVAL_METRICS_CSV,
        "filename": "eval_reward_breakdown.png",
        "title": "Eval reward breakdown",
        "panels": EVAL_REWARD_BREAKDOWN_PANELS,
        "ncols": 3,
    },
    {
        "csv_name": _EVAL_METRICS_CSV,
        "filename": "eval_reward_event_summary.png",
        "title": "Eval reward event summary",
        "panels": EVAL_REWARD_EVENT_PANELS,
        "ncols": 3,
    },
)


def _warn(message: str) -> None:
    print(f"[plot] warning: {message}")


def _load_matplotlib_pyplot():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        _warn(f"matplotlib unavailable: {exc}")
        return None
    return plt


def _load_csv_rows(csv_path: Path) -> list[dict[str, str]]:
    if not csv_path.exists():
        _warn(f"missing csv, skip: {csv_path}")
        return []
    if csv_path.stat().st_size == 0:
        _warn(f"empty csv, skip: {csv_path}")
        return []

    try:
        with csv_path.open("r", newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    except Exception as exc:
        _warn(f"failed to read {csv_path}: {exc}")
        return []

    if csv_path.name == _EVAL_METRICS_CSV:
        rows = [row for row in rows if str(row.get("tag", "")).strip() != "final_probe"]

    if len(rows) <= 0:
        _warn(f"no usable rows, skip: {csv_path}")
        return []
    return rows


def _try_float(value: object) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _extract_xy(rows: list[dict[str, str]], y_column: str) -> tuple[list[float], list[float], str] | None:
    if len(rows) <= 0 or y_column not in rows[0]:
        return None

    if "env_steps" in rows[0]:
        x_column = "env_steps"
    elif "episode_idx" in rows[0]:
        x_column = "episode_idx"
    else:
        x_column = ""

    xs: list[float] = []
    ys: list[float] = []
    for row_idx, row in enumerate(rows):
        y = _try_float(row.get(y_column))
        if y is None or not math.isfinite(y):
            continue
        if x_column == "":
            x = float(row_idx)
        else:
            x_val = _try_float(row.get(x_column))
            if x_val is None or not math.isfinite(x_val):
                continue
            x = x_val
        xs.append(float(x))
        ys.append(float(y))

    if len(xs) <= 0:
        return None
    return xs, ys, (x_column if x_column else "index")


def _adaptive_smoothing_window(series_len: int, *, fraction: float, min_window: int, max_window: int) -> int:
    if series_len <= 1:
        return 1
    window = int(round(series_len * float(fraction)))
    window = max(int(min_window), min(int(max_window), window))
    window = min(window, int(series_len))
    if window % 2 == 0:
        window = max(1, window - 1)
    return max(1, window)


def _metric_smoothing_window(csv_path: Path, y_column: str, series_len: int) -> int:
    if csv_path.name == _TRAIN_STEPS_CSV:
        if y_column == "loss":
            return _adaptive_smoothing_window(series_len, fraction=0.12, min_window=21, max_window=61)
        return _adaptive_smoothing_window(series_len, fraction=0.08, min_window=15, max_window=41)
    if csv_path.name == _EVAL_METRICS_CSV:
        return _adaptive_smoothing_window(series_len, fraction=0.25, min_window=3, max_window=9)
    if csv_path.name == _TRAIN_EPISODES_CSV:
        return _adaptive_smoothing_window(series_len, fraction=0.08, min_window=15, max_window=51)
    return _adaptive_smoothing_window(series_len, fraction=0.10, min_window=5, max_window=31)


def _raw_marker_style(csv_path: Path) -> tuple[float, float]:
    if csv_path.name == _TRAIN_STEPS_CSV:
        return 10.0, 0.24
    if csv_path.name == _EVAL_METRICS_CSV:
        return 22.0, 0.34
    if csv_path.name == _TRAIN_EPISODES_CSV:
        return 10.0, 0.22
    return 12.0, 0.25


def _centered_rolling_mean(values: list[float], *, window: int) -> list[float]:
    n = len(values)
    if n <= 0:
        return []
    if window <= 1 or n <= 2:
        return list(values)

    half = window // 2
    out: list[float] = [0.0] * n
    prefix = [0.0]
    for value in values:
        prefix.append(prefix[-1] + float(value))

    for idx in range(n):
        lo = max(0, idx - half)
        hi = min(n, idx + half + 1)
        denom = max(1, hi - lo)
        out[idx] = (prefix[hi] - prefix[lo]) / float(denom)
    return out


def _plot_raw_and_smooth(
    ax,
    xs: list[float],
    ys: list[float],
    *,
    smooth_window: int,
    raw_marker_size: float,
    raw_alpha: float,
    label: str | None = None,
) -> None:
    smooth = _centered_rolling_mean(ys, window=smooth_window)
    smooth_line = ax.plot(
        xs,
        smooth,
        linewidth=2.3,
        alpha=0.95,
        label=label,
        zorder=3,
    )[0]
    ax.scatter(
        xs,
        ys,
        s=float(raw_marker_size),
        alpha=float(raw_alpha),
        color=smooth_line.get_color(),
        edgecolors="none",
        zorder=2,
    )


def _plot_dashboard_csv(
    *,
    csv_path: Path,
    plots_dir: Path,
    filename: str,
    title: str,
    panels: tuple[dict[str, object], ...],
    ncols: int,
) -> list[Path]:
    rows = _load_csv_rows(csv_path)
    if len(rows) <= 0:
        return []

    plt = _load_matplotlib_pyplot()
    if plt is None:
        return []

    panel_payloads: list[dict[str, object]] = []
    for panel in panels:
        series_payloads = []
        for column, label in tuple(panel.get("series", ())):
            extracted = _extract_xy(rows, str(column))
            if extracted is None:
                continue
            xs, ys, x_label = extracted
            if len(xs) <= 0:
                continue
            series_payloads.append(
                {
                    "column": str(column),
                    "label": str(label),
                    "xs": xs,
                    "ys": ys,
                    "x_label": x_label,
                }
            )
        if len(series_payloads) > 0:
            panel_payloads.append(
                {
                    "title": str(panel.get("title", "")),
                    "y_label": str(panel.get("y_label", "")),
                    "series": series_payloads,
                }
            )

    if len(panel_payloads) <= 0:
        _warn(f"no plottable panels in {csv_path.name} for {filename}")
        return []

    ncols = max(1, min(int(ncols), len(panel_payloads)))
    nrows = int(math.ceil(len(panel_payloads) / float(ncols)))
    fig_w = max(7.0, float(ncols) * 4.8)
    fig_h = max(4.2, float(nrows) * 3.6)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(fig_w, fig_h), squeeze=False)
    raw_marker_size, raw_alpha = _raw_marker_style(csv_path)

    for idx, panel_payload in enumerate(panel_payloads):
        ax = axes[idx // ncols][idx % ncols]
        series_payloads = panel_payload["series"]
        x_label = str(series_payloads[0]["x_label"])
        for series_payload in series_payloads:
            _plot_raw_and_smooth(
                ax,
                list(series_payload["xs"]),
                list(series_payload["ys"]),
                smooth_window=_metric_smoothing_window(
                    csv_path,
                    str(series_payload["column"]),
                    len(series_payload["ys"]),
                ),
                raw_marker_size=raw_marker_size,
                raw_alpha=raw_alpha,
                label=(str(series_payload["label"]) if len(series_payloads) > 1 else None),
            )
        ax.set_title(str(panel_payload["title"]))
        ax.set_xlabel(x_label)
        ax.set_ylabel(str(panel_payload["y_label"]))
        ax.grid(True, linestyle=":", linewidth=0.8, alpha=0.6)
        if len(series_payloads) > 1:
            ax.legend(loc="best")

    for idx in range(len(panel_payloads), nrows * ncols):
        axes[idx // ncols][idx % ncols].axis("off")

    fig.suptitle(title)
    fig.subplots_adjust(top=0.90)
    fig.tight_layout()

    out_path = plots_dir / filename
    try:
        fig.savefig(out_path, dpi=150)
    except Exception as exc:
        plt.close(fig)
        _warn(f"failed to plot dashboard {filename} from {csv_path.name}: {exc}")
        return []

    plt.close(fig)
    return [out_path]


def generate_all_plots(run_dir: Path) -> list[Path]:
    run_dir = Path(run_dir)
    log_dir = run_dir / "logs"
    plots_dir = run_dir / "plots"

    try:
        plots_dir.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        _warn(f"failed to create plot directory {plots_dir}: {exc}")
        return []

    generated: list[Path] = []
    for spec in _DASHBOARD_SPECS:
        generated.extend(
            _plot_dashboard_csv(
                csv_path=log_dir / str(spec["csv_name"]),
                plots_dir=plots_dir,
                filename=str(spec["filename"]),
                title=str(spec["title"]),
                panels=tuple(spec["panels"]),
                ncols=int(spec["ncols"]),
            )
        )
    return generated
