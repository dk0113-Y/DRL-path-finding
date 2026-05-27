from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import train_q_agent  # noqa: E402
from experiments.final_method.run_a_new_classical_frontier_baseline import (  # noqa: E402
    METHOD_ID as B_METHOD_ID,
    METHOD_NAME as B_METHOD_NAME,
    ClassicalFrontierBenchmark,
    build_config as build_b_config,
)
from experiments.final_method.run_a_new_local_state_ddqn_baseline import (  # noqa: E402
    apply_local_state_baseline_contract,
    local_state_model_factory,
    make_local_state_adapter_factory,
)
from experiments.final_method.run_a_new_no_dual_state_split_ablation import (  # noqa: E402
    apply_no_dual_state_split_contract,
    e_model_factory,
)
from training.collector import SEMANTIC_EPISODE_FIELDS  # noqa: E402
from training.rewarding import REWARD_BREAKDOWN_FIELDS, REWARD_EVENT_SUMMARY_FIELDS  # noqa: E402


RUNNER_ENTRYPOINT = "experiments/final_method/run_a_new_unified_final_probe.py"
DEFAULT_CHECKPOINT_STORE_ROOT = Path("checkpoint_store")
DEFAULT_OUTPUT_ROOT = Path("experiment_records/final_method/unified_final_probe")
DEFAULT_RUN_STAGE = "formal"


@dataclass(frozen=True)
class NeuralProbeSpec:
    label: str
    checkpoint_relpath: str
    model_kind: str


NEURAL_SPECS = (
    NeuralProbeSpec("A", "full_method_main/A_full_method_last.pt", "default"),
    NeuralProbeSpec("C", "final_method/A_new_minimum_closure/Anew_C_local_state_ddqn.pt", "local_state"),
    NeuralProbeSpec("D", "final_method/A_new_minimum_closure/Anew_D_no_value_tree.pt", "default"),
    NeuralProbeSpec("E", "final_method/A_new_minimum_closure/Anew_E_no_dual_state_split.pt", "no_dual_state_split"),
    NeuralProbeSpec("F_key", "final_method/A_new_minimum_closure/Anew_F3_no_behavior_memory.pt", "default"),
    NeuralProbeSpec("R_key", "final_method/A_new_reward_ablations/Anew_R5.pt", "default"),
)

SUMMARY_METRICS = (
    ("reward", "episode_reward"),
    ("coverage", "final_coverage"),
    ("success_rate", "success"),
    ("episode_length", "episode_length"),
    ("repeat_visit_ratio", "repeat_visit_ratio"),
    ("timeout_rate", "timeout_flag"),
    ("zero_info_step_count", "zero_info_step_count"),
    ("stall_trigger_count", "stall_trigger_count"),
    ("recent_revisit_trigger_count", "recent_revisit_trigger_count"),
)

PREFERRED_EPISODE_FIELDS = (
    "method_label",
    "method_id",
    "method_name",
    "run_stage",
    "checkpoint_used",
    "checkpoint_path",
    "checkpoint_env_steps",
    "checkpoint_train_episode_idx",
    "episode_index",
    "seed",
    "episode_seed",
    "map_fingerprint",
    "episode_reward",
    "episode_length",
    "final_coverage",
    "success",
    "repeat_visit_ratio",
    "done_reason",
    "timeout_flag",
    "zero_info_step_count",
    "stall_trigger_count",
    "recent_revisit_trigger_count",
    "turn_ge_90_count",
    "turn_135_count",
    "turn_180_count",
)

PREFERRED_SUMMARY_FIELDS = (
    "method_label",
    "method_id",
    "method_name",
    "run_stage",
    "checkpoint_used",
    "checkpoint_path",
    "checkpoint_env_steps",
    "checkpoint_train_episode_idx",
    "episodes",
    "seed_base",
    "seed_start",
    "seed_end",
    "reward",
    "coverage",
    "success_rate",
    "episode_length",
    "repeat_visit_ratio",
    "timeout_rate",
    "zero_info_step_count",
    "stall_trigger_count",
    "recent_revisit_trigger_count",
    "runtime_sec",
    "episode_csv",
    "summary_json",
)


def _now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        number = float(value)
        return None if not math.isfinite(number) else number
    if isinstance(value, float):
        return None if not math.isfinite(value) else value
    if isinstance(value, torch.Tensor):
        return None
    return value


def _is_scalar_for_csv(value: Any) -> bool:
    if value is None or isinstance(value, (str, int, float, bool, np.integer, np.floating)):
        return True
    return False


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(dict(payload)), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _fieldnames(rows: list[Mapping[str, Any]], preferred: tuple[str, ...]) -> list[str]:
    keys = {str(key) for row in rows for key in row.keys()}
    return [key for key in preferred if key in keys] + sorted(keys - set(preferred))


def _write_csv(path: Path, rows: list[Mapping[str, Any]], preferred: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = _fieldnames(rows, preferred)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _json_safe(row.get(field)) for field in fieldnames})


def _mean(rows: list[Mapping[str, Any]], field: str) -> float | None:
    values: list[float] = []
    for row in rows:
        try:
            value = float(row.get(field, float("nan")))
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            values.append(value)
    if not values:
        return None
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def _git_output(args: list[str]) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=str(Path(__file__).resolve().parents[2]),
            capture_output=True,
            text=True,
            check=True,
        )
    except Exception:
        return None
    text = result.stdout.strip()
    return text or None


def _command_text(command: list[str]) -> str:
    try:
        return subprocess.list2cmdline([str(part) for part in command])
    except Exception:
        return " ".join(str(part) for part in command)


def _default_seed_base() -> int:
    return int(train_q_agent.TrainConfig().fixed_final_probe_seed_base)


def _create_run_dir(output_root: Path, run_id: str | None) -> Path:
    resolved_id = str(run_id or f"unified_final_probe_{_now_stamp()}")
    run_dir = Path(output_root) / resolved_id
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _summary_row(
    *,
    method_label: str,
    method_id: str,
    method_name: str,
    run_stage: str,
    checkpoint_used: bool,
    checkpoint_path: str | None,
    checkpoint_env_steps: int | None,
    checkpoint_train_episode_idx: int | None,
    rows: list[Mapping[str, Any]],
    episodes: int,
    seed_base: int,
    runtime_sec: float,
    episode_csv: Path,
    summary_json: Path,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "method_label": method_label,
        "method_id": method_id,
        "method_name": method_name,
        "run_stage": run_stage,
        "checkpoint_used": bool(checkpoint_used),
        "checkpoint_path": checkpoint_path,
        "checkpoint_env_steps": checkpoint_env_steps,
        "checkpoint_train_episode_idx": checkpoint_train_episode_idx,
        "episodes": int(episodes),
        "seed_base": int(seed_base),
        "seed_start": int(seed_base),
        "seed_end": int(seed_base) + int(episodes) - 1,
        "runtime_sec": float(runtime_sec),
        "episode_csv": str(episode_csv),
        "summary_json": str(summary_json),
    }
    for output_name, episode_field in SUMMARY_METRICS:
        row[output_name] = _mean(rows, episode_field)
    return row


def _augment_episode_rows(
    *,
    rows: list[Mapping[str, Any]],
    method_label: str,
    method_id: str,
    method_name: str,
    run_stage: str,
    checkpoint_used: bool,
    checkpoint_path: str | None,
    checkpoint_env_steps: int | None,
    checkpoint_train_episode_idx: int | None,
    seed_base: int,
) -> list[dict[str, Any]]:
    augmented: list[dict[str, Any]] = []
    for index, source in enumerate(rows):
        row = {
            "method_label": method_label,
            "method_id": method_id,
            "method_name": method_name,
            "run_stage": run_stage,
            "checkpoint_used": bool(checkpoint_used),
            "checkpoint_path": checkpoint_path,
            "checkpoint_env_steps": checkpoint_env_steps,
            "checkpoint_train_episode_idx": checkpoint_train_episode_idx,
            "episode_index": int(source.get("episode_index", index)),
            "seed": int(source.get("seed", int(seed_base) + index)),
            "episode_seed": int(source.get("episode_seed", int(seed_base) + index)),
        }
        for key, value in source.items():
            if key in row or not _is_scalar_for_csv(value):
                continue
            row[str(key)] = _json_safe(value)
        augmented.append(row)
    return augmented


def _config_from_checkpoint(payload: Mapping[str, Any], device: str) -> train_q_agent.TrainConfig:
    raw_cfg = payload.get("train_config")
    if not isinstance(raw_cfg, Mapping):
        raise TypeError("checkpoint payload is missing a train_config mapping")
    cfg = train_q_agent.TrainConfig(**dict(raw_cfg))
    return replace(cfg, device=str(device))


def _build_eval_system(cfg: train_q_agent.TrainConfig, spec: NeuralProbeSpec):
    if spec.model_kind == "local_state":
        cfg = apply_local_state_baseline_contract(cfg)
        return (
            cfg,
            train_q_agent.build_system(
                cfg,
                state_adapter_factory=make_local_state_adapter_factory(cfg),
                model_factory=local_state_model_factory,
            ),
        )
    if spec.model_kind == "no_dual_state_split":
        cfg = apply_no_dual_state_split_contract(cfg)
        return cfg, train_q_agent.build_system(cfg, model_factory=e_model_factory)
    if spec.model_kind == "default":
        return cfg, train_q_agent.build_system(cfg)
    raise ValueError(f"Unsupported model_kind for {spec.label}: {spec.model_kind!r}")


def _run_b_probe(
    *,
    run_dir: Path,
    episodes: int,
    seed_base: int,
    run_stage: str,
    continue_on_failure: bool,
) -> dict[str, Any] | None:
    method_dir = run_dir / "B"
    method_dir.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    try:
        cfg = build_b_config(
            run_stage=run_stage,
            device="cpu",
            output_root=str(run_dir),
            run_name="B_unified_final_probe",
            episodes=int(episodes),
            seed_base=int(seed_base),
        )
        raw_rows = ClassicalFrontierBenchmark(cfg).run()
        runtime_sec = time.perf_counter() - start
        rows = _augment_episode_rows(
            rows=raw_rows,
            method_label="B",
            method_id=B_METHOD_ID,
            method_name=B_METHOD_NAME,
            run_stage=run_stage,
            checkpoint_used=False,
            checkpoint_path=None,
            checkpoint_env_steps=None,
            checkpoint_train_episode_idx=None,
            seed_base=seed_base,
        )
        episode_csv = method_dir / "final_probe.csv"
        summary_json = method_dir / "summary.json"
        _write_csv(episode_csv, rows, PREFERRED_EPISODE_FIELDS)
        summary = _summary_row(
            method_label="B",
            method_id=B_METHOD_ID,
            method_name=B_METHOD_NAME,
            run_stage=run_stage,
            checkpoint_used=False,
            checkpoint_path=None,
            checkpoint_env_steps=None,
            checkpoint_train_episode_idx=None,
            rows=rows,
            episodes=episodes,
            seed_base=seed_base,
            runtime_sec=runtime_sec,
            episode_csv=episode_csv,
            summary_json=summary_json,
        )
        _write_json(summary_json, {"summary": summary, "episodes": rows})
        print(
            "[unified_final_probe] B "
            f"episodes={episodes} success={summary.get('success_rate')} "
            f"coverage={summary.get('coverage')} reward={summary.get('reward')}"
        )
        return summary
    except Exception as exc:
        if not continue_on_failure:
            raise
        failure = {
            "method_label": "B",
            "method_id": B_METHOD_ID,
            "error": f"{type(exc).__name__}: {exc}",
        }
        _write_json(method_dir / "failure.json", failure)
        print(f"[unified_final_probe] B failed: {failure['error']}")
        return None


def _run_neural_probe(
    *,
    spec: NeuralProbeSpec,
    run_dir: Path,
    checkpoint_store_root: Path,
    episodes: int,
    seed_base: int,
    run_stage: str,
    device: str,
    continue_on_failure: bool,
) -> dict[str, Any] | None:
    method_dir = run_dir / spec.label
    method_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = (Path(checkpoint_store_root) / spec.checkpoint_relpath).resolve()
    start = time.perf_counter()
    try:
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
        payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
        cfg = _config_from_checkpoint(payload, device=device)
        cfg, system = _build_eval_system(cfg, spec)
        online_net = system[0]
        online_net.load_state_dict(payload["online_state_dict"])
        evaluator = system[-1]
        eval_result = evaluator.evaluate(online_net, num_episodes=int(episodes), seed_base=int(seed_base))
        runtime_sec = time.perf_counter() - start

        method_id = str(cfg.method_id)
        method_name = str(cfg.method_name)
        checkpoint_env_steps = int(payload.get("env_steps", 0))
        checkpoint_train_episode_idx = int(payload.get("train_episode_idx", 0))
        rows = _augment_episode_rows(
            rows=list(eval_result.get("episodes", [])),
            method_label=spec.label,
            method_id=method_id,
            method_name=method_name,
            run_stage=run_stage,
            checkpoint_used=True,
            checkpoint_path=str(checkpoint_path),
            checkpoint_env_steps=checkpoint_env_steps,
            checkpoint_train_episode_idx=checkpoint_train_episode_idx,
            seed_base=seed_base,
        )
        episode_csv = method_dir / "final_probe.csv"
        summary_json = method_dir / "summary.json"
        _write_csv(episode_csv, rows, PREFERRED_EPISODE_FIELDS)
        summary = _summary_row(
            method_label=spec.label,
            method_id=method_id,
            method_name=method_name,
            run_stage=run_stage,
            checkpoint_used=True,
            checkpoint_path=str(checkpoint_path),
            checkpoint_env_steps=checkpoint_env_steps,
            checkpoint_train_episode_idx=checkpoint_train_episode_idx,
            rows=rows,
            episodes=episodes,
            seed_base=seed_base,
            runtime_sec=runtime_sec,
            episode_csv=episode_csv,
            summary_json=summary_json,
        )
        _write_json(
            summary_json,
            {
                "summary": summary,
                "checkpoint_train_config": asdict(cfg),
                "episodes": rows,
            },
        )
        print(
            f"[unified_final_probe] {spec.label} "
            f"episodes={episodes} success={summary.get('success_rate')} "
            f"coverage={summary.get('coverage')} reward={summary.get('reward')}"
        )
        return summary
    except Exception as exc:
        if not continue_on_failure:
            raise
        failure = {
            "method_label": spec.label,
            "checkpoint_path": str(checkpoint_path),
            "error": f"{type(exc).__name__}: {exc}",
        }
        _write_json(method_dir / "failure.json", failure)
        print(f"[unified_final_probe] {spec.label} failed: {failure['error']}")
        return None


def _build_plan(args: argparse.Namespace) -> dict[str, Any]:
    seed_base = int(args.seed_base) if args.seed_base is not None else _default_seed_base()
    checkpoint_store_root = Path(args.checkpoint_store_root)
    return {
        "schema_version": "a_new_unified_final_probe_plan/v1",
        "run_stage": args.run_stage,
        "episodes": int(args.episodes),
        "seed_base": seed_base,
        "seed_start": seed_base,
        "seed_end": seed_base + int(args.episodes) - 1,
        "neural_device": args.device,
        "b_device": "cpu",
        "output_root": str(args.output_root),
        "checkpoint_store_root": str(checkpoint_store_root),
        "run_order": ["B", *[spec.label for spec in NEURAL_SPECS]],
        "methods": [
            {
                "label": "B",
                "method_id": B_METHOD_ID,
                "checkpoint_used": False,
                "policy": "ClassicalFrontierGreedyPolicy",
            },
            *[
                {
                    "label": spec.label,
                    "checkpoint_used": True,
                    "checkpoint_path": str(checkpoint_store_root / spec.checkpoint_relpath),
                    "model_kind": spec.model_kind,
                    "checkpoint_exists": bool((checkpoint_store_root / spec.checkpoint_relpath).exists()),
                }
                for spec in NEURAL_SPECS
            ],
        ],
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run unified 100-episode final probe for A_new matrix checkpoints.")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed-base", type=int, default=None)
    parser.add_argument("--run-stage", choices=("formal",), default=DEFAULT_RUN_STAGE)
    parser.add_argument("--checkpoint-store-root", type=Path, default=DEFAULT_CHECKPOINT_STORE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--continue-on-failure", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if int(args.episodes) <= 0:
        raise ValueError("episodes must be > 0")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("device=cuda was requested, but torch.cuda.is_available() is false")

    plan = _build_plan(args)
    if bool(args.dry_run):
        print(json.dumps({**plan, "dry_run": True}, indent=2, ensure_ascii=False))
        return 0

    seed_base = int(plan["seed_base"])
    run_dir = _create_run_dir(Path(args.output_root), args.run_id)
    print(f"[unified_final_probe] run_dir: {run_dir}")
    print(f"[unified_final_probe] episodes={int(args.episodes)} seed_base={seed_base} device={args.device}")
    _write_json(
        run_dir / "run_manifest.json",
        {
            **plan,
            "run_dir": str(run_dir.resolve()),
            "runner_entrypoint": RUNNER_ENTRYPOINT,
            "raw_argv": list(argv if argv is not None else sys.argv[1:]),
            "launch_command": _command_text([sys.executable, RUNNER_ENTRYPOINT, *(argv if argv is not None else sys.argv[1:])]),
            "source_commit": _git_output(["rev-parse", "HEAD"]),
            "source_branch": _git_output(["branch", "--show-current"]),
            "source_remote": _git_output(["remote", "get-url", "origin"]),
            "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        },
    )

    summaries: list[dict[str, Any]] = []
    b_summary = _run_b_probe(
        run_dir=run_dir,
        episodes=int(args.episodes),
        seed_base=seed_base,
        run_stage=args.run_stage,
        continue_on_failure=bool(args.continue_on_failure),
    )
    if b_summary is not None:
        summaries.append(b_summary)

    for spec in NEURAL_SPECS:
        summary = _run_neural_probe(
            spec=spec,
            run_dir=run_dir,
            checkpoint_store_root=Path(args.checkpoint_store_root),
            episodes=int(args.episodes),
            seed_base=seed_base,
            run_stage=args.run_stage,
            device=args.device,
            continue_on_failure=bool(args.continue_on_failure),
        )
        if summary is not None:
            summaries.append(summary)

    summary_csv = run_dir / "unified_final_probe_summary.csv"
    summary_json = run_dir / "unified_final_probe_summary.json"
    _write_csv(summary_csv, summaries, PREFERRED_SUMMARY_FIELDS)
    _write_json(
        summary_json,
        {
            "schema_version": "a_new_unified_final_probe_summary/v1",
            "run_dir": str(run_dir.resolve()),
            "episodes": int(args.episodes),
            "seed_base": seed_base,
            "seed_start": seed_base,
            "seed_end": seed_base + int(args.episodes) - 1,
            "summary_csv": str(summary_csv),
            "method_summaries": summaries,
        },
    )
    print(f"[unified_final_probe] summary_csv: {summary_csv}")
    print(f"[unified_final_probe] summary_json: {summary_json}")
    return 0 if len(summaries) == len(NEURAL_SPECS) + 1 else 1


if __name__ == "__main__":
    raise SystemExit(main())
