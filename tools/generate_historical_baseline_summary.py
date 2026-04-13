#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.formal_artifacts import write_historical_baseline_summary  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a historical_baseline_summary.json from local outputs.")
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT / "outputs")
    parser.add_argument(
        "--output-path",
        type=Path,
        default=REPO_ROOT / "formal_artifacts" / "historical_baseline_summary.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_path = write_historical_baseline_summary(
        output_root=args.output_root.resolve(),
        output_path=args.output_path.resolve(),
        source_of_truth_repo=str(REPO_ROOT),
    )
    print(f"historical_baseline_summary={output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
