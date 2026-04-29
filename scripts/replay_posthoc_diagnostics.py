from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# Ensure repo root is importable when script is executed as `python scripts/...`.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.posthoc_selection import select_posthoc_candidates


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _run_replay(run_dir: Path, output_path: Path | None) -> Path:
    logs_dir = run_dir / "logs"
    summary_path = logs_dir / "posthoc_selection_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"missing summary: {summary_path}")
    base_summary = _load_json(summary_path)
    replay = select_posthoc_candidates(
        run_dir=run_dir,
        candidate_start_step=int(base_summary.get("candidate_start_step") or 0),
        candidate_end_step=int(base_summary.get("candidate_end_step") or 0),
        checkpoint_interval=int(base_summary.get("checkpoint_interval") or 1),
        window_env_steps=int(base_summary.get("selection_window_env_steps") or 1),
        topk=int(base_summary.get("selected_candidate_count") or 0),
        weights=base_summary.get("selection_weights") or None,
    )
    replay_summary = dict(replay.get("summary", {}))
    preview = {
        "replay_mode": "read_only_diagnostics_preview",
        "source_run_dir": str(run_dir),
        "source_summary_json": str(summary_path),
        "selected_steps_original": list(base_summary.get("selected_candidate_steps", [])),
        "selected_steps_replay": list(replay_summary.get("selected_candidate_steps", [])),
        "train_scores_original": {
            int(row.get("candidate_step")): row.get("selection_score")
            for row in base_summary.get("top_candidates", [])
            if row.get("candidate_step") is not None
        },
        "train_scores_replay": {
            int(row.get("candidate_step")): row.get("selection_score")
            for row in replay_summary.get("top_candidates", [])
            if row.get("candidate_step") is not None
        },
        "selected_steps_unchanged": (
            list(base_summary.get("selected_candidate_steps", []))
            == list(replay_summary.get("selected_candidate_steps", []))
        ),
        "train_score_semantics_unchanged": all(
            (
                (base_summary_score is None and replay_score is None)
                or (
                    base_summary_score is not None
                    and replay_score is not None
                    and abs(float(base_summary_score) - float(replay_score)) <= 1e-12
                )
            )
            for step, base_summary_score in {
                int(row.get("candidate_step")): row.get("selection_score")
                for row in base_summary.get("top_candidates", [])
                if row.get("candidate_step") is not None
            }.items()
            for replay_score in [
                {
                    int(row.get("candidate_step")): row.get("selection_score")
                    for row in replay_summary.get("top_candidates", [])
                    if row.get("candidate_step") is not None
                }.get(step)
            ]
        ),
        "diagnostics_preview": replay_summary.get("diagnostics"),
        "notes": [
            "This replay uses existing logs/checkpoint filenames only and does not start training.",
            "It does not load checkpoint binaries for inference/training.",
            "It writes preview JSON only; original source summary is untouched.",
        ],
    }
    if output_path is None:
        output_path = logs_dir / "enhanced_posthoc_selection_summary.diagnostics_preview.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(preview, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay posthoc diagnostics from an existing run directory.")
    parser.add_argument("--run-dir", required=True, help="Existing run output directory.")
    parser.add_argument(
        "--output",
        default="",
        help="Output JSON path for diagnostics preview. Defaults to logs/enhanced_posthoc_selection_summary.diagnostics_preview.json under run dir.",
    )
    args = parser.parse_args()
    run_dir = Path(args.run_dir).resolve()
    output_path = Path(args.output).resolve() if args.output else None
    out = _run_replay(run_dir, output_path)
    print(str(out))


if __name__ == "__main__":
    main()
