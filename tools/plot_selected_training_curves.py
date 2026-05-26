r"""Plot selected training curves from an outputs directory.

Examples:
    python tools/plot_selected_training_curves.py --methods A,F1,F6
    python tools/plot_selected_training_curves.py --outputs-root C:\Users\Dk\Desktop\SCI\New_A\outputs --methods A,AB,AN,F1,F6,F7
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUTS_ROOT = REPO_ROOT / "outputs"
DEFAULT_FIGURE_DIR = REPO_ROOT / "run_picture"

METRICS = [
    ("recent_mean_coverage", "Recent mean coverage"),
    ("recent_success_rate", "Recent success rate"),
    ("recent_mean_reward", "Recent mean reward"),
    ("recent_mean_episode_length", "Recent mean episode length"),
    ("recent_mean_repeat_visit_ratio", "Recent repeat visit ratio"),
    ("recent_timeout_rate", "Recent timeout rate"),
    ("loss", "Loss"),
    ("td_abs_mean", "TD abs mean"),
    ("q_mean", "Q mean"),
    ("grad_norm", "Grad norm"),
]

KNOWN_COLORS = {
    "A": "#1f77b4",
    "AB": "#2ca02c",
    "AN": "#9467bd",
    "C": "#7f7f7f",
    "D": "#2ca02c",
    "E": "#9467bd",
    "F1": "#ff7f0e",
    "F2": "#ffbb78",
    "F3": "#d62728",
    "F4": "#8c564b",
    "F5": "#e377c2",
    "F6": "#4c78a8",
    "F7": "#54a24b",
    "R1": "#17becf",
    "R2": "#bcbd22",
    "R3": "#aec7e8",
    "R4": "#98df8a",
    "R5": "#c5b0d5",
    "R6": "#8dd3c7",
}


@dataclass(frozen=True)
class MethodData:
    method_id: str
    run_dir: Path | None
    train_steps_path: Path | None
    rows: list[dict[str, Any]]


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


def parse_method_ids(raw: str) -> list[str]:
    return [item.strip() for item in re.split(r"[,，、\s]+", raw) if item.strip()]


def method_color(method_id: str) -> str:
    if method_id in KNOWN_COLORS:
        return KNOWN_COLORS[method_id]
    palette = [
        "#003f5c",
        "#58508d",
        "#bc5090",
        "#ff6361",
        "#ffa600",
        "#2f4b7c",
        "#665191",
        "#a05195",
        "#d45087",
        "#f95d6a",
    ]
    return palette[sum(ord(ch) for ch in method_id) % len(palette)]


def relpath(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _matches_prefix(path: Path, prefix: str) -> bool:
    name = path.name.lower()
    key = prefix.lower()
    return name == key or name.startswith(f"{key}_")


def train_steps_file(run_dir: Path) -> Path | None:
    for candidate in (run_dir / "logs" / "train_steps.csv", run_dir / "train_steps.csv"):
        if candidate.is_file():
            return candidate
    return None


def _has_train_steps(run_dir: Path) -> bool:
    return train_steps_file(run_dir) is not None


def resolve_run_dir(outputs_root: Path, method_id: str) -> Path | None:
    if not outputs_root.exists():
        raise FileNotFoundError(f"outputs root does not exist: {outputs_root}")

    exact = outputs_root / method_id
    if exact.is_dir() and _has_train_steps(exact):
        return exact

    candidates = [
        path
        for path in outputs_root.iterdir()
        if path.is_dir() and _matches_prefix(path, method_id) and _has_train_steps(path)
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def load_method_data(outputs_root: Path, method_id: str) -> MethodData:
    run_dir = resolve_run_dir(outputs_root, method_id)
    train_steps_path = train_steps_file(run_dir) if run_dir is not None else None
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
        run_dir=run_dir,
        train_steps_path=train_steps_path,
        rows=rows,
    )


def default_output_path(method_ids: list[str], output_dir: Path) -> Path:
    safe = "_".join(re.sub(r"[^A-Za-z0-9_.-]+", "_", method_id) for method_id in method_ids)
    return output_dir / f"selected_training_curves_{safe}.png"


def optimize_png(path: Path) -> None:
    image = Image.open(path).convert("RGB")
    quantized = image.convert("P", palette=Image.Palette.ADAPTIVE, colors=128)
    quantized.save(path, optimize=True, dpi=(300, 300))


def plot_selected(method_data: dict[str, MethodData], method_ids: list[str], output_path: Path) -> None:
    fig, axes = plt.subplots(5, 2, figsize=(11, 15), sharex=False)
    axes_flat = list(axes.flat)
    for axis, (metric, label) in zip(axes_flat, METRICS):
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
                linewidth=1.35 if method_id == "A" else 1.05,
                color=method_color(method_id),
                alpha=0.95,
            )
        axis.set_title(label, fontsize=10)
        axis.set_xlabel("env_steps")
        axis.grid(True, alpha=0.25, linewidth=0.5)
        axis.ticklabel_format(axis="x", style="sci", scilimits=(5, 5))

    axes_flat[0].legend(ncol=min(5, len(method_ids)), fontsize=8, loc="best")
    fig.suptitle(f"Training curves: {' / '.join(method_ids)}", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, facecolor="white", pil_kwargs={"optimize": True})
    plt.close(fig)
    optimize_png(output_path)


def write_manifest(
    output_path: Path,
    outputs_root: Path,
    method_data: dict[str, MethodData],
    method_ids: list[str],
) -> Path:
    manifest_path = output_path.with_suffix(".manifest.json")
    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "outputs_root": relpath(outputs_root),
        "methods": method_ids,
        "figure_path": relpath(output_path),
        "inputs": {
            method_id: {
                "run_dir": relpath(method_data[method_id].run_dir),
                "train_steps_path": relpath(method_data[method_id].train_steps_path),
                "row_count": len(method_data[method_id].rows),
            }
            for method_id in method_ids
        },
    }
    manifest_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot selected training curves by outputs directory prefix.")
    parser.add_argument(
        "--methods",
        required=True,
        help="Comma/Chinese-comma/whitespace separated output prefixes, for example: A,AB,AN,F1,F6,F7",
    )
    parser.add_argument(
        "--outputs-root",
        type=Path,
        default=DEFAULT_OUTPUTS_ROOT,
        help="Directory containing run folders. Defaults to this repo's outputs directory.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_FIGURE_DIR)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--allow-missing", action="store_true")
    args = parser.parse_args()

    method_ids = parse_method_ids(args.methods)
    if not method_ids:
        parser.error("--methods did not contain any method prefixes")

    outputs_root = args.outputs_root
    if not outputs_root.is_absolute():
        outputs_root = REPO_ROOT / outputs_root
    output_path = args.output if args.output is not None else default_output_path(method_ids, args.output_dir)
    if not output_path.is_absolute():
        output_path = REPO_ROOT / output_path

    method_data = {method_id: load_method_data(outputs_root, method_id) for method_id in method_ids}
    missing = [method_id for method_id, data in method_data.items() if not data.rows]
    if missing and not args.allow_missing:
        details = ", ".join(f"{method_id} (prefix under {outputs_root})" for method_id in missing)
        raise FileNotFoundError(f"Missing train_steps rows for: {details}")

    plot_ids = [method_id for method_id in method_ids if method_data[method_id].rows]
    if not plot_ids:
        raise FileNotFoundError("No selected prefix has train_steps.csv rows to plot.")

    plot_selected(method_data, plot_ids, output_path)
    manifest_path = write_manifest(output_path, outputs_root, method_data, method_ids)
    print(f"figure: {output_path}")
    print(f"manifest: {manifest_path}")
    for method_id in method_ids:
        data = method_data[method_id]
        print(f"{method_id}: rows={len(data.rows)} train_steps={data.train_steps_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
