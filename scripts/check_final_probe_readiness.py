from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.run_final_probe_matrix import DEFAULT_EPISODES, DEFAULT_OUTPUT_ROOT, DEFAULT_SEED_BASE, run_matrix


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Dry-run readiness audit for the final probe matrix.")
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES)
    parser.add_argument("--seed-base", type=int, default=DEFAULT_SEED_BASE)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--include-r6", action="store_true")
    parser.add_argument("--allow-missing", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.dry_run = True
    args.continue_on_error = False
    return run_matrix(args)


if __name__ == "__main__":
    raise SystemExit(main())
