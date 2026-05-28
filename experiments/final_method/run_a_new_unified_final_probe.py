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
    paper_label: str
    internal_label: str
    checkpoint_relpath: str
    model_kind: str


@dataclass(frozen=True)
class ScenarioConfig:
    scenario_id: str
    rows: int
    cols: int
    obstacle_ratio: float
    max_episode_steps: int
    coverage_stop_threshold: float


NEURAL_SPECS = (
    NeuralProbeSpec("A", "A", "full_method_main/A_full_method_last.pt", "default"),
    NeuralProbeSpec("C", "C", "final_method/A_new_minimum_closure/Anew_C_local_state_ddqn.pt", "local_state"),
    NeuralProbeSpec("D", "D", "final_method/A_new_minimum_closure/Anew_D_no_value_tree.pt", "default"),
    NeuralProbeSpec("E", "E", "final_method/A_new_minimum_closure/Anew_E_no_dual_state_split.pt", "no_dual_state_split"),
    NeuralProbeSpec("F", "F_key", "final_method/A_new_minimum_closure/Anew_F3_no_behavior_memory.pt", "default"),
    NeuralProbeSpec("R", "R_key", "final_method/A_new_reward_ablations/Anew_R5.pt", "default"),
)
NEURAL_SPECS_BY_PAPER_LABEL = {spec.paper_label: spec for spec in NEURAL_SPECS}
SUPPORTED_GROUP_LABELS = ("A", "B", "C", "D", "E", "F", "R")

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
    "scenario_id",
    "rows",
    "cols",
    "obstacle_ratio",
    "max_episode_steps",
    "coverage_stop_threshold",
    "groups",
    "method_label",
    "internal_label",
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
    "scenario_id",
    "rows",
    "cols",
    "obstacle_ratio",
    "max_episode_steps",
    "coverage_stop_threshold",
    "groups",
    "method_label",
    "internal_label",
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
    "source_commit",
    "runner_entrypoint",
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


def _default_scenario_config() -> ScenarioConfig:
    reference = train_q_agent.TrainConfig()
    return ScenarioConfig(
        scenario_id="S0_default_training_matched",
        rows=int(reference.rows),
        cols=int(reference.cols),
        obstacle_ratio=float(reference.obstacle_ratio),
        max_episode_steps=int(reference.max_episode_steps),
        coverage_stop_threshold=float(reference.coverage_stop_threshold),
    )


def _scenario_from_args(args: argparse.Namespace) -> ScenarioConfig:
    default = _default_scenario_config()
    return ScenarioConfig(
        scenario_id=str(args.scenario_id or default.scenario_id),
        rows=int(args.rows) if args.rows is not None else int(default.rows),
        cols=int(args.cols) if args.cols is not None else int(default.cols),
        obstacle_ratio=(
            float(args.obstacle_ratio)
            if args.obstacle_ratio is not None
            else float(default.obstacle_ratio)
        ),
        max_episode_steps=(
            int(args.max_episode_steps)
            if args.max_episode_steps is not None
            else int(default.max_episode_steps)
        ),
        coverage_stop_threshold=(
            float(args.coverage_stop_threshold)
            if args.coverage_stop_threshold is not None
            else float(default.coverage_stop_threshold)
        ),
    )


def _scenario_config_dict(scenario: ScenarioConfig) -> dict[str, Any]:
    return {
        "scenario_id": str(scenario.scenario_id),
        "rows": int(scenario.rows),
        "cols": int(scenario.cols),
        "obstacle_ratio": float(scenario.obstacle_ratio),
        "max_episode_steps": int(scenario.max_episode_steps),
        "coverage_stop_threshold": float(scenario.coverage_stop_threshold),
    }


def _scenario_csv_fields(scenario: ScenarioConfig, groups: tuple[str, ...]) -> dict[str, Any]:
    return {
        **_scenario_config_dict(scenario),
        "groups": ",".join(groups),
    }


def _apply_scenario_overrides(
    cfg: train_q_agent.TrainConfig,
    scenario: ScenarioConfig,
) -> train_q_agent.TrainConfig:
    return replace(
        cfg,
        rows=int(scenario.rows),
        cols=int(scenario.cols),
        obstacle_ratio=float(scenario.obstacle_ratio),
        max_episode_steps=int(scenario.max_episode_steps),
        coverage_stop_threshold=float(scenario.coverage_stop_threshold),
    )


def _environment_config_from_train_config(cfg: train_q_agent.TrainConfig) -> dict[str, Any]:
    return {
        "rows": int(cfg.rows),
        "cols": int(cfg.cols),
        "obs_size": int(cfg.obs_size),
        "scan_radius": int(cfg.scan_radius),
        "obstacle_ratio": float(cfg.obstacle_ratio),
        "max_episode_steps": int(cfg.max_episode_steps),
        "coverage_stop_threshold": float(cfg.coverage_stop_threshold),
        "trajectory_history_steps": int(cfg.trajectory_history_steps),
    }


def _environment_config_from_baseline_config(cfg: Any) -> dict[str, Any]:
    return {
        "rows": int(cfg.rows),
        "cols": int(cfg.cols),
        "obs_size": int(cfg.obs_size),
        "scan_radius": int(cfg.scan_radius),
        "obstacle_ratio": float(cfg.obstacle_ratio),
        "max_episode_steps": int(cfg.max_episode_steps),
        "coverage_stop_threshold": float(cfg.coverage_stop_threshold),
        "trajectory_history_steps": int(cfg.trajectory_history_steps),
    }


def _parse_groups(value: str | None) -> tuple[str, ...]:
    raw = str(value or "B,A,C,D,E,F,R")
    labels = [item.strip().upper() for item in raw.replace(";", ",").split(",") if item.strip()]
    if not labels:
        raise ValueError("groups must contain at least one group label")
    seen: set[str] = set()
    groups: list[str] = []
    for label in labels:
        if label not in SUPPORTED_GROUP_LABELS:
            allowed = ",".join(SUPPORTED_GROUP_LABELS)
            raise ValueError(f"Unsupported group label {label!r}; expected one of {allowed}")
        if label in seen:
            raise ValueError(f"Duplicate group label {label!r}")
        seen.add(label)
        groups.append(label)
    return tuple(groups)


def _create_run_dir(output_root: Path, run_id: str | None) -> Path:
    resolved_id = str(run_id or f"unified_final_probe_{_now_stamp()}")
    run_dir = Path(output_root) / resolved_id
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _summary_row(
    *,
    scenario: ScenarioConfig,
    groups: tuple[str, ...],
    method_label: str,
    internal_label: str,
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
    source_commit: str | None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        **_scenario_csv_fields(scenario, groups),
        "method_label": method_label,
        "internal_label": internal_label,
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
        "source_commit": source_commit,
        "runner_entrypoint": RUNNER_ENTRYPOINT,
    }
    for output_name, episode_field in SUMMARY_METRICS:
        row[output_name] = _mean(rows, episode_field)
    return row


def _augment_episode_rows(
    *,
    rows: list[Mapping[str, Any]],
    scenario: ScenarioConfig,
    groups: tuple[str, ...],
    method_label: str,
    internal_label: str,
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
            **_scenario_csv_fields(scenario, groups),
            "method_label": method_label,
            "internal_label": internal_label,
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
    raise ValueError(f"Unsupported model_kind for {spec.paper_label}: {spec.model_kind!r}")


def _run_b_probe(
    *,
    run_dir: Path,
    scenario: ScenarioConfig,
    groups: tuple[str, ...],
    episodes: int,
    seed_base: int,
    run_stage: str,
    continue_on_failure: bool,
    source_commit: str | None,
) -> dict[str, Any] | None:
    method_dir = run_dir / "B"
    method_dir.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    try:
        cfg = build_b_config(
            scenario_id=scenario.scenario_id,
            run_stage=run_stage,
            device="cpu",
            output_root=str(run_dir),
            run_name="B_unified_final_probe",
            episodes=int(episodes),
            seed_base=int(seed_base),
            rows=int(scenario.rows),
            cols=int(scenario.cols),
            obstacle_ratio=float(scenario.obstacle_ratio),
            max_episode_steps=int(scenario.max_episode_steps),
            coverage_stop_threshold=float(scenario.coverage_stop_threshold),
        )
        raw_rows = ClassicalFrontierBenchmark(cfg).run()
        runtime_sec = time.perf_counter() - start
        rows = _augment_episode_rows(
            rows=raw_rows,
            scenario=scenario,
            groups=groups,
            method_label="B",
            internal_label="B",
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
            scenario=scenario,
            groups=groups,
            method_label="B",
            internal_label="B",
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
            source_commit=source_commit,
        )
        _write_json(
            summary_json,
            {
                "summary": summary,
                "scenario_config": _scenario_config_dict(scenario),
                "environment_config": _environment_config_from_baseline_config(cfg),
                "baseline_config": asdict(cfg),
                "episodes": rows,
            },
        )
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
            "internal_label": "B",
            "scenario_id": scenario.scenario_id,
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
    scenario: ScenarioConfig,
    groups: tuple[str, ...],
    episodes: int,
    seed_base: int,
    run_stage: str,
    device: str,
    continue_on_failure: bool,
    source_commit: str | None,
) -> dict[str, Any] | None:
    method_dir = run_dir / spec.paper_label
    method_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = (Path(checkpoint_store_root) / spec.checkpoint_relpath).resolve()
    start = time.perf_counter()
    try:
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
        payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
        checkpoint_cfg = _config_from_checkpoint(payload, device=device)
        cfg = _apply_scenario_overrides(checkpoint_cfg, scenario)
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
            scenario=scenario,
            groups=groups,
            method_label=spec.paper_label,
            internal_label=spec.internal_label,
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
            scenario=scenario,
            groups=groups,
            method_label=spec.paper_label,
            internal_label=spec.internal_label,
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
            source_commit=source_commit,
        )
        _write_json(
            summary_json,
            {
                "summary": summary,
                "scenario_config": _scenario_config_dict(scenario),
                "environment_config": _environment_config_from_train_config(cfg),
                "checkpoint_train_config": asdict(checkpoint_cfg),
                "evaluation_train_config": asdict(cfg),
                "paper_facing_label": spec.paper_label,
                "internal_label": spec.internal_label,
                "episodes": rows,
            },
        )
        print(
            f"[unified_final_probe] {spec.paper_label} "
            f"episodes={episodes} success={summary.get('success_rate')} "
            f"coverage={summary.get('coverage')} reward={summary.get('reward')}"
        )
        return summary
    except Exception as exc:
        if not continue_on_failure:
            raise
        failure = {
            "method_label": spec.paper_label,
            "internal_label": spec.internal_label,
            "scenario_id": scenario.scenario_id,
            "checkpoint_path": str(checkpoint_path),
            "error": f"{type(exc).__name__}: {exc}",
        }
        _write_json(method_dir / "failure.json", failure)
        print(f"[unified_final_probe] {spec.paper_label} failed: {failure['error']}")
        return None


def _build_plan(args: argparse.Namespace) -> dict[str, Any]:
    seed_base = int(args.seed_base) if args.seed_base is not None else _default_seed_base()
    checkpoint_store_root = Path(args.checkpoint_store_root)
    scenario = _scenario_from_args(args)
    groups = _parse_groups(args.groups)
    methods: list[dict[str, Any]] = []
    for group in groups:
        if group == "B":
            methods.append(
                {
                    "label": "B",
                    "paper_facing_label": "B",
                    "internal_label": "B",
                    "method_id": B_METHOD_ID,
                    "checkpoint_used": False,
                    "policy": "ClassicalFrontierGreedyPolicy",
                }
            )
            continue
        spec = NEURAL_SPECS_BY_PAPER_LABEL[group]
        methods.append(
            {
                "label": spec.paper_label,
                "paper_facing_label": spec.paper_label,
                "internal_label": spec.internal_label,
                "checkpoint_used": True,
                "checkpoint_path": str(checkpoint_store_root / spec.checkpoint_relpath),
                "model_kind": spec.model_kind,
                "checkpoint_exists": bool((checkpoint_store_root / spec.checkpoint_relpath).exists()),
            }
        )
    return {
        "schema_version": "a_new_unified_final_probe_plan/v1",
        "scenario_id": scenario.scenario_id,
        "scenario_config": _scenario_config_dict(scenario),
        "environment_config": _scenario_config_dict(scenario),
        "run_stage": args.run_stage,
        "episodes": int(args.episodes),
        "seed_base": seed_base,
        "seed_start": seed_base,
        "seed_end": seed_base + int(args.episodes) - 1,
        "neural_device": args.device,
        "b_device": "cpu",
        "output_root": str(args.output_root),
        "run_id": args.run_id,
        "checkpoint_store_root": str(checkpoint_store_root),
        "groups": list(groups),
        "run_order": list(groups),
        "methods": methods,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run unified 100-episode final probe for A_new matrix checkpoints.")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed-base", type=int, default=None)
    parser.add_argument("--run-stage", choices=("smoke", "formal"), default=DEFAULT_RUN_STAGE)
    parser.add_argument("--checkpoint-store-root", type=Path, default=DEFAULT_CHECKPOINT_STORE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--scenario-id", type=str, default="S0_default_training_matched")
    parser.add_argument("--rows", type=int, default=None)
    parser.add_argument("--cols", type=int, default=None)
    parser.add_argument("--obstacle-ratio", type=float, default=None)
    parser.add_argument("--max-episode-steps", type=int, default=None)
    parser.add_argument("--coverage-stop-threshold", type=float, default=None)
    parser.add_argument("--groups", type=str, default="B,A,C,D,E,F,R")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--continue-on-failure", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if int(args.episodes) <= 0:
        raise ValueError("episodes must be > 0")

    plan = _build_plan(args)
    if bool(args.dry_run):
        print(json.dumps({**plan, "dry_run": True}, indent=2, ensure_ascii=False))
        return 0

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("device=cuda was requested, but torch.cuda.is_available() is false")

    seed_base = int(plan["seed_base"])
    scenario = _scenario_from_args(args)
    groups = _parse_groups(args.groups)
    source_commit = _git_output(["rev-parse", "HEAD"])
    run_dir = _create_run_dir(Path(args.output_root), args.run_id)
    print(f"[unified_final_probe] run_dir: {run_dir}")
    print(
        "[unified_final_probe] "
        f"scenario_id={scenario.scenario_id} groups={','.join(groups)} "
        f"episodes={int(args.episodes)} seed_base={seed_base} device={args.device}"
    )
    _write_json(
        run_dir / "run_manifest.json",
        {
            **plan,
            "run_dir": str(run_dir.resolve()),
            "runner_entrypoint": RUNNER_ENTRYPOINT,
            "raw_argv": list(argv if argv is not None else sys.argv[1:]),
            "launch_command": _command_text([sys.executable, RUNNER_ENTRYPOINT, *(argv if argv is not None else sys.argv[1:])]),
            "source_commit": source_commit,
            "source_branch": _git_output(["branch", "--show-current"]),
            "source_remote": _git_output(["remote", "get-url", "origin"]),
            "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        },
    )
    _write_json(
        run_dir / "scenario_manifest.json",
        {
            "schema_version": "environment_shift_final_probe_scenario/v1",
            "scenario_id": scenario.scenario_id,
            "scenario_config": _scenario_config_dict(scenario),
            "groups": list(groups),
            "episodes": int(args.episodes),
            "seed_base": seed_base,
            "seed_start": seed_base,
            "seed_end": seed_base + int(args.episodes) - 1,
            "source_commit": source_commit,
            "runner_entrypoint": RUNNER_ENTRYPOINT,
        },
    )

    summaries: list[dict[str, Any]] = []
    for group in groups:
        if group == "B":
            summary = _run_b_probe(
                run_dir=run_dir,
                scenario=scenario,
                groups=groups,
                episodes=int(args.episodes),
                seed_base=seed_base,
                run_stage=args.run_stage,
                continue_on_failure=bool(args.continue_on_failure),
                source_commit=source_commit,
            )
        else:
            summary = _run_neural_probe(
                spec=NEURAL_SPECS_BY_PAPER_LABEL[group],
                run_dir=run_dir,
                checkpoint_store_root=Path(args.checkpoint_store_root),
                scenario=scenario,
                groups=groups,
                episodes=int(args.episodes),
                seed_base=seed_base,
                run_stage=args.run_stage,
                device=args.device,
                continue_on_failure=bool(args.continue_on_failure),
                source_commit=source_commit,
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
            "scenario_id": scenario.scenario_id,
            "scenario_config": _scenario_config_dict(scenario),
            "groups": list(groups),
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
    return 0 if len(summaries) == len(groups) else 1


if __name__ == "__main__":
    raise SystemExit(main())
