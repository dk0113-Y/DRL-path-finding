#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.formal_artifacts import (  # noqa: E402
    _read_csv_rows,
    build_config_snapshot,
    select_best_eval_row,
    write_formal_run_artifacts,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backfill formal_train JSON summaries for existing output runs.")
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT / "outputs")
    parser.add_argument("--run-name", action="append", default=[], help="Optional run directory name. Repeatable.")
    return parser.parse_args()


def iter_target_runs(output_root: Path, requested_runs: list[str]) -> list[Path]:
    if requested_runs:
        return [output_root / run_name for run_name in requested_runs]
    return [
        child
        for child in sorted(output_root.iterdir())
        if child.is_dir() and child.name != "scheduler_runs"
    ]


def main() -> int:
    args = parse_args()
    output_root = args.output_root.resolve()
    if not output_root.exists():
        print(f"error=output_root_missing:{output_root}", file=sys.stderr)
        return 1

    updated = 0
    skipped = 0
    for run_dir in iter_target_runs(output_root, args.run_name):
        if not run_dir.exists():
            print(f"skip=missing_run:{run_dir.name}")
            skipped += 1
            continue

        logs_dir = run_dir / "logs"
        train_steps_rows = _read_csv_rows(logs_dir / "train_steps.csv")
        eval_rows = _read_csv_rows(logs_dir / "eval_metrics.csv")
        final_probe_rows = _read_csv_rows(logs_dir / "final_probe.csv")
        if not train_steps_rows or not eval_rows or not final_probe_rows:
            print(f"skip=incomplete_core_artifacts:{run_dir.name}")
            skipped += 1
            continue

        recent_train_row = train_steps_rows[-1]
        last_eval_row = eval_rows[-1]
        best_eval_row = select_best_eval_row(eval_rows)
        final_probe_row = final_probe_rows[-1]

        flags = [
            "backfilled_from_historical_run",
            "train_config_unavailable_in_backfill_context",
        ]
        if not (run_dir / "checkpoints" / "best.pt").exists():
            flags.append("best_checkpoint_missing")

        write_formal_run_artifacts(
            run_dir=run_dir,
            cfg=None,
            run_mode="historical_backfill",
            recent_train_row=recent_train_row,
            last_eval_row=last_eval_row,
            best_eval_row=best_eval_row,
            final_probe_row=final_probe_row,
            best_checkpoint_source=(
                "checkpoints/best.pt::historical_backfill_success_rate_then_eval_mean_coverage"
                if (run_dir / "checkpoints" / "best.pt").exists()
                else "checkpoints/best.pt_missing"
            ),
            best_checkpoint_env_steps=int(best_eval_row["env_steps"]) if best_eval_row is not None else None,
            last_checkpoint_env_steps=int(recent_train_row["env_steps"]),
            final_probe_source="historical_backfill_unknown_best_or_online_last",
            total_runtime_sec=None,
            total_runtime_hms=None,
            source_of_truth_repo=str(REPO_ROOT),
            extra_insufficient_evidence_flags=flags,
        )

        config_snapshot_path = logs_dir / "config_snapshot.json"
        if config_snapshot_path.exists():
            config_snapshot = build_config_snapshot(
                cfg=None,
                run_dir=run_dir,
                run_mode="historical_backfill",
                source_of_truth_repo=str(REPO_ROOT),
                insufficient_evidence_flags=[
                    "backfilled_from_historical_run",
                    "complete_train_config_not_recoverable_without_checkpoint_loader",
                ],
            )
            observed_contract = {
                "final_env_steps": int(recent_train_row["env_steps"]),
                "train_steps_header": list(recent_train_row.keys()),
                "eval_metrics_header": list(last_eval_row.keys()),
                "final_probe_header": list(final_probe_row.keys()),
            }
            config_snapshot["observed_run_contract"] = observed_contract
            config_snapshot["comparability"]["bootstrap_signature"] = {
                "env_steps": int(recent_train_row["env_steps"]),
                "eval_columns": list(last_eval_row.keys()),
            }
            config_snapshot_path.write_text(
                __import__("json").dumps(config_snapshot, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

        print(f"updated={run_dir.name}")
        updated += 1

    print(f"summary=updated:{updated},skipped:{skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
