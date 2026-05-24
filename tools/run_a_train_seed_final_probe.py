from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import tools.run_final_probe_matrix as final_probe_matrix
from agents.q_value_agent import ExplorationQConfig, ExplorationQNetwork, StateTensorAdapter
from train_q_agent import configure_torch_runtime, set_seed
from training.evaluator import GreedyEvaluator


DEFAULT_EPISODES = 100
DEFAULT_SEED_BASE = 20259323
DEFAULT_OUTPUT_ROOT = Path("experiment_records/final_probe_train_seed_diagnostic")
CHECKPOINT_PATH = REPO_ROOT / "checkpoint_store" / "full_method_main" / "A_full_method.pt"
CONFIG_SNAPSHOT_PATH = REPO_ROOT / "experiment_records" / "full_method_main" / "logs" / "config_snapshot.json"
OFFICIAL_FINAL_PROBE_ROOT = REPO_ROOT / "experiment_records" / "final_probe"
TRAINING_COMPARISON_CSV = OFFICIAL_FINAL_PROBE_ROOT / "analysis" / "training_vs_final_probe_comparison.csv"
OFFICIAL_SUMMARY_CSV = OFFICIAL_FINAL_PROBE_ROOT / "final_probe_summary.csv"

EXPECTED_ENV_PARAMS = {
    "rows": 40,
    "cols": 60,
    "obs_size": 6,
    "scan_radius": 10,
    "obstacle_ratio": 0.20,
    "max_episode_steps": 600,
    "coverage_stop_threshold": 0.95,
    "trajectory_history_steps": 10,
}

A_METHOD = {
    "method_id": "A",
    "group": "A",
    "display_name": "full_method_main",
    "checkpoint_required": True,
    "checkpoint_path": str(CHECKPOINT_PATH.resolve()),
    "config_snapshot_path": str(CONFIG_SNAPSHOT_PATH.resolve()),
    "model_factory": "ExplorationQNetwork",
    "state_adapter_factory": "StateTensorAdapter",
    "evaluation_order": 1,
}


class DiagnosticReadinessError(RuntimeError):
    pass


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _first_a_row(path: Path) -> dict[str, str]:
    for row in _read_csv_rows(path):
        if row.get("method_id") == "A" or row.get("group") == "A":
            return row
    raise DiagnosticReadinessError(f"No A row found in {path}")


def _float_or_none(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _metric(summary: Mapping[str, Any], name: str) -> Any:
    metrics = summary.get("summary", {}).get("metrics", {})
    if isinstance(metrics, Mapping):
        return metrics.get(name)
    return None


def _validate_expected_env_params(cfg: Any) -> dict[str, Any]:
    checked: dict[str, Any] = {}
    mismatches: list[str] = []
    for key, expected in EXPECTED_ENV_PARAMS.items():
        actual = getattr(cfg, key)
        checked[key] = actual
        if isinstance(expected, float):
            if abs(float(actual) - expected) > 1e-9:
                mismatches.append(f"{key}: expected {expected}, got {actual}")
        elif int(actual) != int(expected):
            mismatches.append(f"{key}: expected {expected}, got {actual}")
    if mismatches:
        raise DiagnosticReadinessError("Checkpoint train_config env parameter mismatch: " + "; ".join(mismatches))
    return checked


def readiness_check(device: torch.device) -> dict[str, Any]:
    missing: list[dict[str, str]] = []
    if not CHECKPOINT_PATH.exists():
        missing.append({"kind": "checkpoint", "path": str(CHECKPOINT_PATH)})
    if not CONFIG_SNAPSHOT_PATH.exists():
        missing.append({"kind": "config_snapshot", "path": str(CONFIG_SNAPSHOT_PATH)})
    if not TRAINING_COMPARISON_CSV.exists():
        missing.append({"kind": "training_comparison_csv", "path": str(TRAINING_COMPARISON_CSV)})
    if not OFFICIAL_SUMMARY_CSV.exists():
        missing.append({"kind": "official_summary_csv", "path": str(OFFICIAL_SUMMARY_CSV)})
    if missing:
        raise DiagnosticReadinessError("Missing required artifacts: " + json.dumps(missing, ensure_ascii=False))

    payload = final_probe_matrix._load_checkpoint_payload(CHECKPOINT_PATH)
    cfg = final_probe_matrix.train_config_from_payload(payload, device)
    env_params = _validate_expected_env_params(cfg)

    model = ExplorationQNetwork(ExplorationQConfig())
    load_result = model.load_state_dict(payload["online_state_dict"], strict=True)
    if load_result.missing_keys or load_result.unexpected_keys:
        raise DiagnosticReadinessError(
            "Strict load_state_dict was not clean: "
            + json.dumps(
                {
                    "missing_keys": list(load_result.missing_keys),
                    "unexpected_keys": list(load_result.unexpected_keys),
                },
                ensure_ascii=False,
            )
        )

    adapter_cfg = final_probe_matrix.state_adapter_config_from_train_config(cfg)
    _ = StateTensorAdapter(cfg=adapter_cfg, device="cpu")

    return {
        "checkpoint_path": str(CHECKPOINT_PATH.resolve()),
        "checkpoint_status": "ok",
        "config_snapshot_path": str(CONFIG_SNAPSHOT_PATH.resolve()),
        "config_snapshot_status": "ok",
        "model_factory": "ExplorationQNetwork",
        "model_factory_status": "ok",
        "state_adapter_factory": "StateTensorAdapter",
        "state_adapter_factory_status": "ok",
        "load_state_dict": {
            "missing_keys": list(load_result.missing_keys),
            "unexpected_keys": list(load_result.unexpected_keys),
        },
        "checkpoint_metadata": {
            "env_steps": payload.get("env_steps"),
            "learn_steps": payload.get("learn_steps"),
            "train_episode_idx": payload.get("train_episode_idx"),
        },
        "env_params_from_checkpoint_train_config": env_params,
    }


def write_protocol(output_root: Path, *, episodes: int, seed_base: int, device_text: str) -> None:
    payload = {
        "protocol_name": "a_train_seed_final_probe_diagnostic_v1",
        "created_at": final_probe_matrix._now_iso(),
        "diagnostic_scope": "A_only_train_seed_probe",
        "purpose": "Diagnose train-seed vs held-out final-seed differences for A_full_method_main.",
        "not_replacement_for": str(OFFICIAL_FINAL_PROBE_ROOT.resolve()),
        "method_id": "A",
        "episodes": int(episodes),
        "seed_base": int(seed_base),
        "episode_seed_rule": "seed_base + zero_based_episode_index",
        "policy": "greedy",
        "device": str(device_text),
        "checkpoint_path": str(CHECKPOINT_PATH.resolve()),
        "config_snapshot_path": str(CONFIG_SNAPSHOT_PATH.resolve()),
        "method_loading": {
            "model_factory": "ExplorationQNetwork",
            "state_adapter_factory": "StateTensorAdapter",
        },
        "env_params": dict(EXPECTED_ENV_PARAMS),
        "checkpoint_copy_policy": "checkpoint is referenced in place and is not copied into experiment_records",
        "official_final_probe_reference": str(OFFICIAL_FINAL_PROBE_ROOT.resolve()),
    }
    final_probe_matrix._write_json(output_root / "protocol.json", payload)


def evaluate_a(*, episodes: int, seed_base: int, device: torch.device, output_root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = final_probe_matrix._load_checkpoint_payload(CHECKPOINT_PATH)
    cfg = final_probe_matrix.train_config_from_payload(payload, device)
    _validate_expected_env_params(cfg)
    set_seed(int(cfg.seed))
    configure_torch_runtime(cfg)

    model = ExplorationQNetwork(ExplorationQConfig())
    model.load_state_dict(payload["online_state_dict"], strict=True)
    model.to(device)
    model.eval()

    state_adapter = StateTensorAdapter(
        cfg=final_probe_matrix.state_adapter_config_from_train_config(cfg),
        device="cpu",
    )
    evaluator = GreedyEvaluator.from_collector_config(
        final_probe_matrix.collector_config_from_train_config(cfg),
        state_adapter=state_adapter,
        device=str(device),
    )
    probe = evaluator.evaluate(model, num_episodes=int(episodes), seed_base=int(seed_base))

    episode_rows: list[dict[str, Any]] = []
    for idx, row in enumerate(probe.get("episodes", []), start=1):
        out = final_probe_matrix._episode_csv_row(row)
        out["method_id"] = "A"
        out["group"] = "A"
        out["display_name"] = "full_method_main"
        out["checkpoint_path"] = str(CHECKPOINT_PATH.resolve())
        out.setdefault("episode_idx", idx)
        out.setdefault("episode_seed", int(seed_base) + idx - 1)
        episode_rows.append(out)

    summary_payload = {
        "method_id": "A",
        "group": "A",
        "display_name": "full_method_main",
        "status": "ok",
        "checkpoint_path": str(CHECKPOINT_PATH.resolve()),
        "config_snapshot_path": str(CONFIG_SNAPSHOT_PATH.resolve()),
        "episodes": int(episodes),
        "seed_base": int(seed_base),
        "episode_seed_rule": "seed_base + zero_based_episode_index",
        "policy": "greedy",
        "checkpoint_metadata": {
            "env_steps": payload.get("env_steps"),
            "learn_steps": payload.get("learn_steps"),
            "train_episode_idx": payload.get("train_episode_idx"),
        },
        "method_loading": {
            "model_factory": "ExplorationQNetwork",
            "state_adapter_factory": "StateTensorAdapter",
        },
        "summary": final_probe_matrix._summary_from_probe(probe),
    }

    method_dir = output_root / "A"
    final_probe_matrix._write_csv(
        method_dir / "per_episode.csv",
        episode_rows,
        preferred=("method_id", "group", "display_name", "episode_idx", "episode_seed", "success", "final_coverage", "episode_reward"),
    )
    final_probe_matrix._write_json(method_dir / "summary.json", summary_payload)
    return summary_payload, episode_rows


def build_comparison_rows(summary: Mapping[str, Any], *, seed_base: int) -> list[dict[str, Any]]:
    train_row = _first_a_row(TRAINING_COMPARISON_CSV)
    official_row = _first_a_row(OFFICIAL_SUMMARY_CSV)
    return [
        {
            "source": "A_train_recent_from_metric_snapshot",
            "seed_base": "",
            "success_rate": _float_or_none(train_row.get("train_recent_success_rate")),
            "coverage": _float_or_none(train_row.get("train_recent_coverage")),
            "reward": _float_or_none(train_row.get("train_recent_reward")),
            "episode_length": _float_or_none(train_row.get("train_recent_episode_length")),
            "repeat_visit_ratio": _float_or_none(train_row.get("train_recent_repeat_visit_ratio")),
            "timeout_rate": _float_or_none(train_row.get("train_recent_timeout_rate")),
            "timeout_flag": "",
        },
        {
            "source": "A_official_final_probe_seed_20261323",
            "seed_base": 20261323,
            "success_rate": _float_or_none(official_row.get("success_rate")),
            "coverage": _float_or_none(official_row.get("coverage")),
            "reward": _float_or_none(official_row.get("reward")),
            "episode_length": _float_or_none(official_row.get("episode_length")),
            "repeat_visit_ratio": _float_or_none(official_row.get("repeat_visit_ratio")),
            "timeout_rate": "",
            "timeout_flag": _float_or_none(official_row.get("timeout_flag")),
        },
        {
            "source": "A_train_seed_greedy_probe_20259323",
            "seed_base": int(seed_base),
            "success_rate": _metric(summary, "success_rate"),
            "coverage": _metric(summary, "coverage"),
            "reward": _metric(summary, "reward"),
            "episode_length": _metric(summary, "episode_length"),
            "repeat_visit_ratio": _metric(summary, "repeat_visit_ratio"),
            "timeout_rate": "",
            "timeout_flag": _metric(summary, "timeout_flag"),
        },
    ]


def _delta(new_value: Any, old_value: Any) -> float | None:
    new_float = _float_or_none(new_value)
    old_float = _float_or_none(old_value)
    if new_float is None or old_float is None:
        return None
    return new_float - old_float


def write_interpretation(output_root: Path, comparison_rows: Sequence[Mapping[str, Any]]) -> None:
    by_source = {str(row["source"]): row for row in comparison_rows}
    train = by_source["A_train_recent_from_metric_snapshot"]
    official = by_source["A_official_final_probe_seed_20261323"]
    diagnostic = by_source["A_train_seed_greedy_probe_20259323"]
    metrics = ("success_rate", "coverage", "reward", "episode_length", "repeat_visit_ratio")
    lines = [
        "# A Train-Seed Final Probe Diagnostic",
        "",
        "- This diagnostic uses the training seed base and is not a replacement for the held-out final probe.",
        "- The official held-out final probe remains experiment_records/final_probe/.",
        "- If train-seed greedy probe is closer to train recent-window metrics than held-out final probe, the drop is likely related to seed-distribution shift or held-out generalization.",
        "- If train-seed greedy probe is still much lower than train recent-window metrics, the drop may be related to greedy determinization, endpoint checkpoint quality, or train-window vs checkpoint mismatch.",
        "",
        "## Metric Deltas",
        "",
        "| metric | train_seed_minus_train_recent | train_seed_minus_official_held_out |",
        "| --- | ---: | ---: |",
    ]
    for metric in metrics:
        lines.append(
            "| {metric} | {train_delta} | {official_delta} |".format(
                metric=metric,
                train_delta=_delta(diagnostic.get(metric), train.get(metric)),
                official_delta=_delta(diagnostic.get(metric), official.get(metric)),
            )
        )
    lines.append(
        "| timeout_flag_or_rate | {train_delta} | {official_delta} |".format(
            train_delta=_delta(diagnostic.get("timeout_flag"), train.get("timeout_rate")),
            official_delta=_delta(diagnostic.get("timeout_flag"), official.get("timeout_flag")),
        )
    )
    lines.append("")
    lines.append("Positive deltas mean the train-seed diagnostic value is higher than the reference row.")
    (output_root / "interpretation.md").write_text("\n".join(str(line) for line in lines) + "\n", encoding="utf-8")


def write_run_manifest(
    output_root: Path,
    *,
    status: str,
    args: argparse.Namespace,
    device_info: Mapping[str, Any],
    readiness: Mapping[str, Any] | None,
    runtime_sec: float,
    error: str | None = None,
    error_trace: str | None = None,
) -> None:
    manifest = {
        "schema_version": "a_train_seed_final_probe_run_manifest/v1",
        "created_at": final_probe_matrix._now_iso(),
        "status": status,
        "repo_root": str(REPO_ROOT.resolve()),
        "git_sha": final_probe_matrix._git_output(["rev-parse", "HEAD"]),
        "git_branch": final_probe_matrix._git_output(["rev-parse", "--abbrev-ref", "HEAD"]),
        "python": sys.version,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "episodes": int(args.episodes),
        "seed_base": int(args.seed_base),
        "episode_seed_rule": "seed_base + zero_based_episode_index",
        "checkpoint_path": str(CHECKPOINT_PATH.resolve()),
        "config_snapshot_path": str(CONFIG_SNAPSHOT_PATH.resolve()),
        "policy": "greedy",
        "official_final_probe_reference": str(OFFICIAL_FINAL_PROBE_ROOT.resolve()),
        "diagnostic_scope": "A_only_train_seed_probe",
        "arguments": vars(args),
        "device": dict(device_info),
        "readiness": readiness,
        "runtime_sec": runtime_sec,
        "output_files": {
            "protocol_json": str((output_root / "protocol.json").resolve()),
            "run_manifest_json": str((output_root / "run_manifest.json").resolve()),
            "per_episode_csv": str((output_root / "A" / "per_episode.csv").resolve()),
            "summary_json": str((output_root / "A" / "summary.json").resolve()),
            "comparison_with_existing_csv": str((output_root / "comparison_with_existing.csv").resolve()),
            "interpretation_md": str((output_root / "interpretation.md").resolve()),
        },
    }
    if error:
        manifest["error"] = error
    if error_trace:
        manifest["traceback"] = error_trace
    final_probe_matrix._write_json(output_root / "run_manifest.json", manifest)


def run(args: argparse.Namespace) -> int:
    output_root = Path(args.output_root).expanduser().resolve()
    device, device_info = final_probe_matrix.resolve_device(str(args.device), require_available=not bool(args.dry_run))
    readiness = readiness_check(device)

    if args.dry_run:
        plan = {
            "dry_run": True,
            "episodes": int(args.episodes),
            "seed_base": int(args.seed_base),
            "episode_seed_rule": "seed_base + zero_based_episode_index",
            "device": device_info,
            "output_root": str(output_root),
            "checkpoint_path": str(CHECKPOINT_PATH.resolve()),
            "config_snapshot_path": str(CONFIG_SNAPSHOT_PATH.resolve()),
            "official_final_probe_reference": str(OFFICIAL_FINAL_PROBE_ROOT.resolve()),
            "diagnostic_scope": "A_only_train_seed_probe",
            "readiness": readiness,
            "planned_outputs": {
                "protocol_json": str((output_root / "protocol.json").resolve()),
                "run_manifest_json": str((output_root / "run_manifest.json").resolve()),
                "per_episode_csv": str((output_root / "A" / "per_episode.csv").resolve()),
                "summary_json": str((output_root / "A" / "summary.json").resolve()),
                "comparison_with_existing_csv": str((output_root / "comparison_with_existing.csv").resolve()),
                "interpretation_md": str((output_root / "interpretation.md").resolve()),
            },
        }
        print("[a_train_seed_final_probe] dry_run=true")
        print(json.dumps(final_probe_matrix._json_safe(plan), ensure_ascii=False, indent=2))
        return 0

    output_root.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    try:
        write_protocol(output_root, episodes=int(args.episodes), seed_base=int(args.seed_base), device_text=str(args.device))
        summary, _ = evaluate_a(
            episodes=int(args.episodes),
            seed_base=int(args.seed_base),
            device=device,
            output_root=output_root,
        )
        comparison_rows = build_comparison_rows(summary, seed_base=int(args.seed_base))
        final_probe_matrix._write_csv(
            output_root / "comparison_with_existing.csv",
            comparison_rows,
            preferred=(
                "source",
                "seed_base",
                "success_rate",
                "coverage",
                "reward",
                "episode_length",
                "repeat_visit_ratio",
                "timeout_rate",
                "timeout_flag",
            ),
        )
        write_interpretation(output_root, comparison_rows)
        write_run_manifest(
            output_root,
            status="ok",
            args=args,
            device_info=device_info,
            readiness=readiness,
            runtime_sec=time.perf_counter() - start,
        )
        metrics = summary["summary"]["metrics"]
        print(
            "[a_train_seed_final_probe] ok "
            f"success_rate={metrics.get('success_rate')} "
            f"coverage={metrics.get('coverage')} "
            f"reward={metrics.get('reward')} "
            f"episode_length={metrics.get('episode_length')} "
            f"repeat_visit_ratio={metrics.get('repeat_visit_ratio')} "
            f"timeout_flag={metrics.get('timeout_flag')}"
        )
        return 0
    except Exception as exc:
        error_text = f"{type(exc).__name__}: {exc}"
        error_trace = traceback.format_exc()
        write_run_manifest(
            output_root,
            status="error",
            args=args,
            device_info=device_info,
            readiness=readiness,
            runtime_sec=time.perf_counter() - start,
            error=error_text,
            error_trace=error_trace,
        )
        print(f"[a_train_seed_final_probe] error: {error_text}", file=sys.stderr)
        return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the A-only train-seed final probe diagnostic.")
    parser.add_argument("--dry-run", action="store_true", help="Check artifacts and print the plan without running episodes.")
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES)
    parser.add_argument("--seed-base", type=int, default=DEFAULT_SEED_BASE)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if int(args.episodes) <= 0:
        raise SystemExit("--episodes must be > 0")
    try:
        return run(args)
    except DiagnosticReadinessError as exc:
        print(f"[a_train_seed_final_probe] readiness error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
