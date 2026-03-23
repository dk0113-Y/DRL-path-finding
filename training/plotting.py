from __future__ import annotations

"""
Offline monitoring-plot generation.

This module intentionally stays out of the training hot path. It only reads CSV
logs and writes PNGs when generate_all_plots(run_dir) is called explicitly by
the training script or by a post-run utility.
"""

import csv
import math
from pathlib import Path


TRAIN_PLOT_COLUMNS: dict[str, str] = {
    "replay_size": "train_replay_size.png",
    "epsilon": "train_epsilon.png",
    "loss": "train_loss.png",
    "q_mean": "train_q_mean.png",
    "target_q_mean": "train_target_q_mean.png",
    "td_abs_mean": "train_td_abs_mean.png",
    "grad_norm": "train_grad_norm.png",
    "learner_steps": "train_learner_steps.png",
    "recent_mean_reward": "train_recent_mean_reward.png",
    "recent_mean_coverage": "train_recent_mean_coverage.png",
    "recent_success_rate": "train_recent_success_rate.png",
    "recent_mean_episode_length": "train_recent_mean_episode_length.png",
    "recent_mean_repeat_visit_ratio": "train_recent_mean_repeat_visit_ratio.png",
}

TRAIN_REWARD_BREAKDOWN_COLUMNS: tuple[str, ...] = (
    "info_reward_sum",
    "step_penalty_sum",
    "recent_revisit_penalty_sum",
    "stall_penalty_sum",
    "timeout_penalty_sum",
    "terminal_bonus_sum",
)
TRAIN_REWARD_BREAKDOWN_FILENAME = "train_reward_breakdown.png"
TRAIN_EPISODE_LENGTH_FILENAME = "train_episode_length_vs_episode_idx.png"

EVAL_PLOT_COLUMNS: dict[str, str] = {
    "eval_mean_reward": "eval_mean_reward.png",
    "eval_mean_coverage": "eval_mean_coverage.png",
    "eval_success_rate": "eval_success_rate.png",
    "eval_mean_repeat_visit_ratio": "eval_mean_repeat_visit_ratio.png",
    "eval_mean_episode_length": "eval_mean_episode_length.png",
}

EVAL_REWARD_BREAKDOWN_COLUMNS: tuple[str, ...] = (
    "eval_mean_info_reward_sum",
    "eval_mean_step_penalty_sum",
    "eval_mean_recent_revisit_penalty_sum",
    "eval_mean_stall_penalty_sum",
    "eval_mean_timeout_penalty_sum",
    "eval_mean_terminal_bonus_sum",
)
EVAL_REWARD_BREAKDOWN_FILENAME = "eval_reward_breakdown.png"


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

    if csv_path.name == "eval_metrics.csv":
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

    use_env_steps = "env_steps" in rows[0]
    xs: list[float] = []
    ys: list[float] = []

    for idx, row in enumerate(rows):
        y = _try_float(row.get(y_column))
        if y is None:
            continue

        if use_env_steps:
            x = _try_float(row.get("env_steps"))
            if x is None:
                x = float(idx)
        else:
            x = float(idx)

        xs.append(x)
        ys.append(y)

    if len(xs) <= 0:
        return None
    if not any(math.isfinite(y) for y in ys):
        return None
    return xs, ys, ("env_steps" if use_env_steps else "index")


def _rolling_mean(values: list[float], window: int) -> list[float]:
    if window <= 1 or len(values) <= 0:
        return list(values)

    smoothed: list[float] = []
    running_sum = 0.0
    for idx, value in enumerate(values):
        running_sum += value
        if idx >= window:
            running_sum -= values[idx - window]
        count = min(idx + 1, window)
        smoothed.append(running_sum / float(count))
    return smoothed


def _plot_metric_csv(
    *,
    csv_path: Path,
    plots_dir: Path,
    columns_to_filenames: dict[str, str],
) -> list[Path]:
    rows = _load_csv_rows(csv_path)
    if len(rows) <= 0:
        return []

    plt = _load_matplotlib_pyplot()
    if plt is None:
        return []

    generated: list[Path] = []
    for y_column, filename in columns_to_filenames.items():
        xy = _extract_xy(rows, y_column)
        if xy is None:
            if y_column not in rows[0]:
                _warn(f"missing column '{y_column}' in {csv_path.name}, skip")
            continue

        xs, ys, x_label = xy
        fig = None
        try:
            fig, ax = plt.subplots(figsize=(7.0, 4.0))
            if y_column == "loss":
                ax.plot(xs, ys, linewidth=1.2, alpha=0.65, label="loss_raw")
                ax.plot(xs, _rolling_mean(ys, window=9), linewidth=2.0, label="loss_smooth")
                ax.legend()
            else:
                ax.plot(xs, ys, linewidth=1.8)
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_column)
            ax.set_title(y_column)
            ax.grid(True, alpha=0.3)
            fig.tight_layout()

            out_path = plots_dir / filename
            fig.savefig(out_path, dpi=150)
            generated.append(out_path)
        except Exception as exc:
            _warn(f"failed to plot {y_column} from {csv_path.name}: {exc}")
        finally:
            if fig is not None:
                plt.close(fig)

    return generated


def _plot_multi_series_csv(
    *,
    csv_path: Path,
    plots_dir: Path,
    columns: tuple[str, ...],
    filename: str,
    title: str,
) -> list[Path]:
    rows = _load_csv_rows(csv_path)
    if len(rows) <= 0:
        return []

    plt = _load_matplotlib_pyplot()
    if plt is None:
        return []

    fig = None
    generated: list[Path] = []
    try:
        fig, ax = plt.subplots(figsize=(8.0, 4.8))
        plotted = 0
        x_label = "env_steps"

        for y_column in columns:
            xy = _extract_xy(rows, y_column)
            if xy is None:
                if y_column not in rows[0]:
                    _warn(f"missing column '{y_column}' in {csv_path.name}, skip")
                continue

            xs, ys, x_label = xy
            ax.plot(xs, ys, linewidth=1.8, label=y_column)
            plotted += 1

        if plotted <= 0:
            _warn(f"no plottable series in {csv_path.name} for {title}")
            return []

        ax.set_xlabel(x_label)
        ax.set_ylabel("value")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()

        out_path = plots_dir / filename
        fig.savefig(out_path, dpi=150)
        generated.append(out_path)
    except Exception as exc:
        _warn(f"failed to plot multi-series chart from {csv_path.name}: {exc}")
    finally:
        if fig is not None:
            plt.close(fig)

    return generated


def _plot_xy_csv(
    *,
    csv_path: Path,
    plots_dir: Path,
    x_column: str,
    y_column: str,
    filename: str,
    title: str,
) -> list[Path]:
    rows = _load_csv_rows(csv_path)
    if len(rows) <= 0:
        return []
    if x_column not in rows[0] or y_column not in rows[0]:
        missing = x_column if x_column not in rows[0] else y_column
        _warn(f"missing column '{missing}' in {csv_path.name}, skip")
        return []

    xs: list[float] = []
    ys: list[float] = []
    for row in rows:
        x = _try_float(row.get(x_column))
        y = _try_float(row.get(y_column))
        if x is None or y is None:
            continue
        xs.append(x)
        ys.append(y)

    if len(xs) <= 0 or not any(math.isfinite(y) for y in ys):
        _warn(f"no plottable rows for {y_column} vs {x_column} in {csv_path.name}")
        return []

    plt = _load_matplotlib_pyplot()
    if plt is None:
        return []

    fig = None
    generated: list[Path] = []
    try:
        fig, ax = plt.subplots(figsize=(7.0, 4.0))
        ax.plot(xs, ys, linewidth=1.8)
        ax.set_xlabel(x_column)
        ax.set_ylabel(y_column)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        out_path = plots_dir / filename
        fig.savefig(out_path, dpi=150)
        generated.append(out_path)
    except Exception as exc:
        _warn(f"failed to plot {y_column} vs {x_column} from {csv_path.name}: {exc}")
    finally:
        if fig is not None:
            plt.close(fig)

    return generated


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
    generated.extend(
        _plot_metric_csv(
            csv_path=log_dir / "train_steps.csv",
            plots_dir=plots_dir,
            columns_to_filenames=TRAIN_PLOT_COLUMNS,
        )
    )
    generated.extend(
        _plot_metric_csv(
            csv_path=log_dir / "eval_metrics.csv",
            plots_dir=plots_dir,
            columns_to_filenames=EVAL_PLOT_COLUMNS,
        )
    )
    generated.extend(
        _plot_xy_csv(
            csv_path=log_dir / "train_episodes.csv",
            plots_dir=plots_dir,
            x_column="episode_idx",
            y_column="episode_length",
            filename=TRAIN_EPISODE_LENGTH_FILENAME,
            title="train_episode_length_vs_episode_idx",
        )
    )
    generated.extend(
        _plot_multi_series_csv(
            csv_path=log_dir / "train_episodes.csv",
            plots_dir=plots_dir,
            columns=TRAIN_REWARD_BREAKDOWN_COLUMNS,
            filename=TRAIN_REWARD_BREAKDOWN_FILENAME,
            title="train reward breakdown",
        )
    )
    generated.extend(
        _plot_multi_series_csv(
            csv_path=log_dir / "eval_metrics.csv",
            plots_dir=plots_dir,
            columns=EVAL_REWARD_BREAKDOWN_COLUMNS,
            filename=EVAL_REWARD_BREAKDOWN_FILENAME,
            title="eval reward breakdown",
        )
    )
    return generated
