from __future__ import annotations

import csv
from pathlib import Path
from typing import Mapping


class CSVMetricLogger:
    """Append-only CSV logger for formal-train metrics."""

    def __init__(self, run_dir: Path):
        self.run_dir = Path(run_dir)
        self.log_dir = self.run_dir / "logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.train_episode_csv = self.log_dir / "train_episodes.csv"
        self.final_probe_csv = self.log_dir / "final_probe.csv"
        self.train_step_csv = self.log_dir / "train_steps.csv"
        self.model_select_eval_csv = self.log_dir / "model_select_eval.csv"
        self.best_recheck_eval_csv = self.log_dir / "best_recheck_eval.csv"

    @staticmethod
    def _append_row(path: Path, row: Mapping[str, object]) -> None:
        write_header = (not path.exists()) or path.stat().st_size == 0
        with path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(dict(row))

    def log_train_episode(self, row: Mapping[str, object]) -> None:
        self._append_row(self.train_episode_csv, row)

    def log_final_probe(self, row: Mapping[str, object]) -> None:
        self._append_row(self.final_probe_csv, row)

    def log_train_step(self, row: Mapping[str, object]) -> None:
        self._append_row(self.train_step_csv, row)

    def log_model_select_eval(self, row: Mapping[str, object]) -> None:
        self._append_row(self.model_select_eval_csv, row)

    def log_best_recheck_eval(self, row: Mapping[str, object]) -> None:
        self._append_row(self.best_recheck_eval_csv, row)
