from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from baselines.frontier_greedy_policy import (  # noqa: E402
    BASELINE_NAME,
    FrontierGreedyPolicy,
    FrontierGreedyPolicyConfig,
    baseline_policy_config_dict,
)
from training.baseline_evaluator import FrontierGreedyBaselineEvaluator  # noqa: E402
from training.collector import DERIVED_TRAIN_DIAGNOSTIC_FIELDS, SEMANTIC_EPISODE_FIELDS  # noqa: E402
from training.evaluator import EvaluatorConfig  # noqa: E402
from training.rewarding import REWARD_BREAKDOWN_FIELDS, REWARD_EVENT_SUMMARY_FIELDS  # noqa: E402


DEFAULT_RUN_PREFIX = "baseline_frontier_greedy_v1"


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _git_output(args: list[str]) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            check=True,
        )
    except Exception:
        return None
    text = result.stdout.strip()
    return text or None


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(dict(payload)), ensure_ascii=False, indent=2), encoding="utf-8")


def _run_dir_from_args(args: argparse.Namespace) -> Path:
    if args.output_dir:
        return Path(args.output_dir).resolve()
    run_name = str(args.run_name).strip() if args.run_name else ""
    if not run_name:
        run_name = f"{DEFAULT_RUN_PREFIX}_{args.run_stage}_{_timestamp()}"
    return (Path(args.output_root) / run_name).resolve()


def _episode_fieldnames(episodes: list[Mapping[str, Any]]) -> list[str]:
    preferred = [
        "episode_idx",
        "episode_seed",
        "map_fingerprint",
        "episode_reward",
        "episode_length",
        "final_coverage",
        "success",
        "repeat_visit_ratio",
        "done_reason",
        "timeout_flag",
        "weighted_info_gain_sum",
        "policy_frontier_target_step_count",
        "policy_fallback_step_count",
        *SEMANTIC_EPISODE_FIELDS,
        *REWARD_BREAKDOWN_FIELDS,
        *REWARD_EVENT_SUMMARY_FIELDS,
        *DERIVED_TRAIN_DIAGNOSTIC_FIELDS,
    ]
    seen = set()
    out = []
    for field in preferred:
        if field not in seen:
            seen.add(field)
            out.append(field)
    for row in episodes:
        for field in row.keys():
            if field not in seen:
                seen.add(field)
                out.append(str(field))
    return out


def _write_final_probe_csv(path: Path, episodes: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = _episode_fieldnames(episodes)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in episodes:
            writer.writerow({field: row.get(field) for field in fieldnames})


def _mean(values: list[float]) -> float | None:
    arr = np.asarray(values, dtype=np.float32)
    if arr.size <= 0 or not np.any(np.isfinite(arr)):
        return None
    return float(np.nanmean(arr))


def _summary_from_result(result: Mapping[str, Any]) -> dict[str, Any]:
    episodes = list(result.get("episodes", []))
    reward_breakdown = {
        field: _mean([float(ep.get(field, float("nan"))) for ep in episodes])
        for field in REWARD_BREAKDOWN_FIELDS
    }
    reward_events = {
        field: _mean([float(ep.get(field, float("nan"))) for ep in episodes])
        for field in REWARD_EVENT_SUMMARY_FIELDS
    }
    semantic = {
        field: _mean([float(ep.get(field, float("nan"))) for ep in episodes])
        for field in SEMANTIC_EPISODE_FIELDS
    }
    diagnostics = {
        field: _mean([float(ep.get(field, float("nan"))) for ep in episodes])
        for field in DERIVED_TRAIN_DIAGNOSTIC_FIELDS
    }
    return {
        "episodes": int(result.get("eval_episodes", len(episodes))),
        "metrics": {
            "reward": result.get("eval_mean_reward"),
            "coverage": result.get("eval_mean_coverage"),
            "success_rate": result.get("eval_success_rate"),
            "episode_length": result.get("eval_mean_episode_length"),
            "repeat_visit_ratio": result.get("eval_mean_repeat_visit_ratio"),
        },
        "reward_breakdown": reward_breakdown,
        "reward_events": reward_events,
        "semantic_monitoring": semantic,
        "derived_diagnostics": diagnostics,
        "policy_decision_counts": result.get("policy_decision_counts", {}),
    }


def _artifact_index(run_dir: Path) -> dict[str, Any]:
    required = [
        "logs/final_probe.csv",
        "logs/metric_snapshot.json",
        "logs/benchmark_summary.json",
        "logs/config_snapshot.json",
        "logs/reproducibility_contract.json",
        "logs/artifact_index.json",
        "logs/baseline_policy_summary.json",
        "logs/training_summary.txt",
    ]
    return {
        "artifact_type": "baseline_artifact_index",
        "run_type": "baseline",
        "baseline_name": BASELINE_NAME,
        "run_dir": str(run_dir.resolve()),
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "artifacts": [
            {
                "path": item,
                "exists": bool((run_dir / item).exists()),
                "required": True,
                "category": "baseline_logs",
            }
            for item in required
        ],
    }


def _training_summary_text(
    *,
    run_dir: Path,
    run_stage: str,
    final_probe_episodes: int,
    seed_base: int | None,
    summary: Mapping[str, Any],
) -> str:
    metrics = dict(summary.get("metrics", {}))
    return "\n".join(
        [
            "No-Training Baseline Evaluation Summary",
            "run_type: baseline",
            f"baseline_name: {BASELINE_NAME}",
            f"run_stage: {run_stage}",
            f"run_dir: {run_dir.resolve()}",
            "no_training: true",
            "no_q_network: true",
            "no_checkpoint: true",
            "no_ground_truth_map_for_decision: true",
            f"final_probe_episodes: {int(final_probe_episodes)}",
            f"fixed_final_probe_seed_base: {seed_base}",
            f"mean_reward: {metrics.get('reward')}",
            f"mean_coverage: {metrics.get('coverage')}",
            f"success_rate: {metrics.get('success_rate')}",
            f"mean_episode_length: {metrics.get('episode_length')}",
            f"mean_repeat_visit_ratio: {metrics.get('repeat_visit_ratio')}",
            "",
        ]
    )


def write_baseline_artifacts(
    *,
    run_dir: Path,
    cfg: EvaluatorConfig,
    policy: FrontierGreedyPolicy,
    result: Mapping[str, Any],
    args: argparse.Namespace,
    raw_argv: list[str],
    runtime_sec: float,
    seed_base: int | None,
) -> dict[str, Path]:
    logs_dir = run_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    episodes = list(result["episodes"])
    final_probe_csv = logs_dir / "final_probe.csv"
    _write_final_probe_csv(final_probe_csv, episodes)

    git_sha = _git_output(["rev-parse", "HEAD"])
    git_branch = _git_output(["rev-parse", "--abbrev-ref", "HEAD"])
    policy_summary = policy.policy_summary()
    metric_summary = _summary_from_result(result)

    config_snapshot = {
        "artifact_type": "baseline_config_snapshot",
        "run_type": "baseline",
        "baseline_name": BASELINE_NAME,
        "run_stage": str(args.run_stage),
        "final_probe_episodes": int(args.final_probe_episodes),
        "fixed_final_probe_seed_base": seed_base,
        "rows": int(cfg.rows),
        "cols": int(cfg.cols),
        "obs_size": int(cfg.obs_size),
        "scan_radius": int(cfg.scan_radius),
        "obstacle_ratio": float(cfg.obstacle_ratio),
        "max_episode_steps": int(cfg.max_episode_steps),
        "coverage_stop_threshold": float(cfg.coverage_stop_threshold),
        "trajectory_history_steps": int(cfg.trajectory_history_steps),
        "whether_use_shared_semantic_snapshot": bool(policy.cfg.use_shared_semantic_snapshot),
        "policy_tie_break_rule": str(policy.cfg.tie_break_rule),
        "fallback_rule": str(policy.cfg.fallback_rule),
        "git_sha": git_sha,
        "git_branch": git_branch,
        "eval_config": asdict(cfg),
        "policy_config": baseline_policy_config_dict(policy.cfg),
    }
    metric_snapshot = {
        "artifact_type": "baseline_metric_snapshot",
        "run_type": "baseline",
        "baseline_name": BASELINE_NAME,
        "run_stage": str(args.run_stage),
        "final_probe_source": "logs/final_probe.csv",
        "final_probe": metric_summary,
    }
    benchmark_summary = {
        "artifact_type": "baseline_benchmark_summary",
        "run_type": "baseline",
        "baseline_name": BASELINE_NAME,
        "run_stage": str(args.run_stage),
        "run_dir": str(run_dir.resolve()),
        "total_runtime_sec": float(runtime_sec),
        "final_probe_episodes": int(args.final_probe_episodes),
        "fixed_final_probe_seed_base": seed_base,
        "no_training": True,
        "no_q_network": True,
        "no_checkpoint": True,
    }
    reproducibility_contract = {
        "artifact_type": "baseline_reproducibility_contract",
        "run_type": "baseline",
        "baseline_name": BASELINE_NAME,
        "run_stage": str(args.run_stage),
        "source_of_truth_repo": str(REPO_ROOT.resolve()),
        "git_sha": git_sha,
        "git_branch": git_branch,
        "argv": raw_argv,
        "python": sys.version,
        "platform": platform.platform(),
        "numpy_version": np.__version__,
        "seed_policy": {
            "use_fixed_final_probe_seeds": bool(args.use_fixed_final_probe_seeds),
            "fixed_final_probe_seed_base": seed_base,
            "episode_seed_rule": "seed_base + zero_based_episode_index" if seed_base is not None else "unfixed",
        },
        "decision_input_contract": {
            "uses_cumulative_belief_map": True,
            "uses_frontier_cache": True,
            "uses_shared_semantic_snapshot": bool(policy.cfg.use_shared_semantic_snapshot),
            "uses_valid_action_set": True,
            "uses_recent_trajectory_positions": True,
            "uses_ground_truth_full_map_for_decision": False,
        },
        "no_training": True,
        "no_q_network": True,
        "no_checkpoint": True,
    }

    paths = {
        "final_probe_csv": final_probe_csv,
        "metric_snapshot": logs_dir / "metric_snapshot.json",
        "benchmark_summary": logs_dir / "benchmark_summary.json",
        "config_snapshot": logs_dir / "config_snapshot.json",
        "reproducibility_contract": logs_dir / "reproducibility_contract.json",
        "baseline_policy_summary": logs_dir / "baseline_policy_summary.json",
        "training_summary": logs_dir / "training_summary.txt",
        "artifact_index": logs_dir / "artifact_index.json",
    }
    _write_json(paths["metric_snapshot"], metric_snapshot)
    _write_json(paths["benchmark_summary"], benchmark_summary)
    _write_json(paths["config_snapshot"], config_snapshot)
    _write_json(paths["reproducibility_contract"], reproducibility_contract)
    _write_json(paths["baseline_policy_summary"], policy_summary)
    paths["training_summary"].write_text(
        _training_summary_text(
            run_dir=run_dir,
            run_stage=str(args.run_stage),
            final_probe_episodes=int(args.final_probe_episodes),
            seed_base=seed_base,
            summary=metric_summary,
        ),
        encoding="utf-8",
    )
    _write_json(paths["artifact_index"], _artifact_index(run_dir))
    return paths


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run the classical frontier greedy baseline final probe.")
    p.add_argument("--final-probe-episodes", "--episodes", dest="final_probe_episodes", type=int, default=100)
    p.add_argument("--fixed-final-probe-seed-base", type=int, default=20261323)
    p.add_argument(
        "--use-fixed-final-probe-seeds",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use seed_base + episode index for final probe map generation.",
    )
    p.add_argument("--run-stage", choices=("smoke", "formal", "diagnostic"), default="formal")
    p.add_argument("--smoke", action="store_true", help="Shortcut for run_stage=smoke; defaults to 3 episodes if unset.")
    p.add_argument("--output-root", type=Path, default=Path("outputs"))
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--run-name", type=str, default="")

    p.add_argument("--rows", type=int, default=40)
    p.add_argument("--cols", type=int, default=60)
    p.add_argument("--obs-size", type=int, default=6)
    p.add_argument("--scan-radius", type=int, default=10)
    p.add_argument("--obstacle-ratio", type=float, default=0.20)
    p.add_argument("--max-episode-steps", type=int, default=600)
    p.add_argument("--coverage-stop-threshold", type=float, default=0.95)
    p.add_argument("--trajectory-history-steps", type=int, default=10)

    p.add_argument("--reward-info-scale", type=float, default=3.0)
    p.add_argument("--reward-obstacle-weight", type=float, default=0.25)
    p.add_argument("--reward-step-penalty", type=float, default=0.02)
    p.add_argument("--reward-terminal-bonus", type=float, default=20.0)
    p.add_argument("--reward-revisit-penalty", type=float, default=0.10)
    p.add_argument("--reward-turn-penalty-scale", type=float, default=0.05)
    p.add_argument("--reward-turn-weight-45", type=float, default=0.0)
    p.add_argument("--reward-turn-weight-90", type=float, default=(1.0 / 3.0))
    p.add_argument("--reward-turn-weight-135", type=float, default=(2.0 / 3.0))
    p.add_argument("--reward-turn-weight-180", type=float, default=1.0)
    p.add_argument("--reward-timeout-penalty", type=float, default=8.0)
    p.add_argument(
        "--use-shared-semantic-snapshot",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use current belief-derived shared semantic snapshot as the preferred frontier source.",
    )
    p.add_argument(
        "--debug-check-incremental-frontier",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    return p


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    raw = sys.argv[1:] if argv is None else list(argv)
    args = build_parser().parse_args(raw)
    episode_arg_present = any(
        item == "--episodes"
        or item == "--final-probe-episodes"
        or item.startswith("--episodes=")
        or item.startswith("--final-probe-episodes=")
        for item in raw
    )
    if bool(args.smoke):
        args.run_stage = "smoke"
        if not episode_arg_present:
            args.final_probe_episodes = 3
    args.final_probe_episodes = int(max(1, args.final_probe_episodes))
    return args


def main(argv: list[str] | None = None) -> Path:
    args = parse_args(argv)
    run_dir = _run_dir_from_args(args)
    cfg = EvaluatorConfig(
        rows=int(args.rows),
        cols=int(args.cols),
        obs_size=int(args.obs_size),
        scan_radius=int(args.scan_radius),
        obstacle_ratio=float(args.obstacle_ratio),
        max_episode_steps=int(args.max_episode_steps),
        coverage_stop_threshold=float(args.coverage_stop_threshold),
        trajectory_history_steps=int(args.trajectory_history_steps),
        reward_info_scale=float(args.reward_info_scale),
        reward_obstacle_weight=float(args.reward_obstacle_weight),
        reward_step_penalty=float(args.reward_step_penalty),
        reward_terminal_bonus=float(args.reward_terminal_bonus),
        reward_revisit_penalty=float(args.reward_revisit_penalty),
        reward_turn_penalty_scale=float(args.reward_turn_penalty_scale),
        reward_turn_weight_45=float(args.reward_turn_weight_45),
        reward_turn_weight_90=float(args.reward_turn_weight_90),
        reward_turn_weight_135=float(args.reward_turn_weight_135),
        reward_turn_weight_180=float(args.reward_turn_weight_180),
        reward_timeout_penalty=float(args.reward_timeout_penalty),
        debug_check_incremental_frontier=bool(args.debug_check_incremental_frontier),
    )
    policy = FrontierGreedyPolicy(
        FrontierGreedyPolicyConfig(
            scan_radius=int(args.scan_radius),
            use_shared_semantic_snapshot=bool(args.use_shared_semantic_snapshot),
        )
    )
    evaluator = FrontierGreedyBaselineEvaluator(cfg, policy)
    seed_base = int(args.fixed_final_probe_seed_base) if bool(args.use_fixed_final_probe_seeds) else None

    start = time.perf_counter()
    result = evaluator.evaluate(num_episodes=int(args.final_probe_episodes), seed_base=seed_base)
    runtime_sec = time.perf_counter() - start
    write_baseline_artifacts(
        run_dir=run_dir,
        cfg=cfg,
        policy=policy,
        result=result,
        args=args,
        raw_argv=[str(Path(sys.argv[0]).name), *(sys.argv[1:] if argv is None else argv)],
        runtime_sec=runtime_sec,
        seed_base=seed_base,
    )
    print(
        "[frontier_greedy_baseline] "
        f"run_stage={args.run_stage} episodes={int(args.final_probe_episodes)} "
        f"seed_base={seed_base} run_dir={run_dir}"
    )
    print(
        "[frontier_greedy_baseline] "
        f"mean_reward={float(result['eval_mean_reward']):.4f} "
        f"mean_coverage={float(result['eval_mean_coverage']):.4f} "
        f"success_rate={float(result['eval_success_rate']):.4f} "
        f"mean_len={float(result['eval_mean_episode_length']):.2f}"
    )
    return run_dir


if __name__ == "__main__":
    main()
