from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import subprocess
import sys
import time
import traceback
from dataclasses import fields, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agents.q_value_agent import ExplorationQConfig, ExplorationQNetwork, StateAdapterConfig, StateTensorAdapter
from baselines.local_state_ddqn import LocalStateQNetwork, LocalStateTensorAdapter
from experiments.ablations.ablation_specs import get_ablation_spec
from experiments.ablations.semantic_split_ablation import NoSemanticDualStateSplitQNetwork
from experiments.ablations.state_adapter_wrapper import AblationStateTensorAdapter
from experiments.ablations.value_tree_ablation import VALUE_REPLACEMENT_STRATEGY_ZERO
from train_q_agent import TrainConfig, configure_torch_runtime, set_seed
from training.collector import (
    DERIVED_TRAIN_DIAGNOSTIC_FIELDS,
    SEMANTIC_EPISODE_FIELDS,
    CollectorConfig,
)
from training.evaluator import GreedyEvaluator
from training.rewarding import REWARD_BREAKDOWN_FIELDS, REWARD_EVENT_SUMMARY_FIELDS


DEFAULT_OUTPUT_ROOT = Path("experiment_records/final_probe")
DEFAULT_EPISODES = 100
DEFAULT_SEED_BASE = 20261323


class ReadinessError(RuntimeError):
    pass


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


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
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return {"array_shape": list(value.shape), "array_dtype": str(value.dtype)}
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        if value.numel() > 32:
            return {"tensor_shape": list(value.shape), "tensor_dtype": str(value.dtype)}
        return value.detach().cpu().tolist()
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _csv_scalar(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, (str, int, float, bool)):
        return True
    if isinstance(value, np.generic):
        return True
    return False


def _episode_csv_row(row: Mapping[str, Any]) -> dict[str, Any]:
    skip = {"true_grid", "belief_map", "trajectory_positions", "semantic_viz"}
    return {str(key): _json_safe(value) for key, value in row.items() if key not in skip and _csv_scalar(value)}


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _fieldnames(rows: Sequence[Mapping[str, Any]], preferred: Sequence[str] = ()) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for field in preferred:
        if field not in seen:
            seen.add(field)
            out.append(str(field))
    for row in rows:
        for field in row.keys():
            if str(field) not in seen:
                seen.add(str(field))
                out.append(str(field))
    return out


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], preferred: Sequence[str] = ()) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields_out = _fieldnames(rows, preferred=preferred)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields_out, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _mean_from_episodes(episodes: Sequence[Mapping[str, Any]], field: str) -> float | None:
    values: list[float] = []
    for row in episodes:
        try:
            values.append(float(row[field]))
        except Exception:
            continue
    if not values:
        return None
    arr = np.asarray(values, dtype=np.float32)
    if not np.any(np.isfinite(arr)):
        return None
    return float(np.nanmean(arr))


def _summary_from_probe(probe: Mapping[str, Any]) -> dict[str, Any]:
    episodes = list(probe.get("episodes", []))
    summary = {
        "episodes": int(probe.get("eval_episodes", len(episodes))),
        "metrics": {
            "reward": probe.get("eval_mean_reward"),
            "coverage": probe.get("eval_mean_coverage"),
            "success_rate": probe.get("eval_success_rate"),
            "episode_length": probe.get("eval_mean_episode_length"),
            "repeat_visit_ratio": probe.get("eval_mean_repeat_visit_ratio"),
        },
    }
    for field in (
        *SEMANTIC_EPISODE_FIELDS,
        *REWARD_BREAKDOWN_FIELDS,
        *REWARD_EVENT_SUMMARY_FIELDS,
        *DERIVED_TRAIN_DIAGNOSTIC_FIELDS,
    ):
        key = f"eval_mean_{field}"
        if key in probe:
            summary["metrics"][field] = probe.get(key)
        else:
            summary["metrics"][field] = _mean_from_episodes(episodes, field)
    return summary


def _flatten_summary_row(method: Mapping[str, Any], summary: Mapping[str, Any], status: str, error: str = "") -> dict[str, Any]:
    metrics = dict(summary.get("metrics", {})) if isinstance(summary.get("metrics"), Mapping) else {}
    row = {
        "method_id": method["method_id"],
        "group": method["group"],
        "display_name": method["display_name"],
        "status": status,
        "error": error,
        "checkpoint_path": method.get("checkpoint_path") or "",
        "episodes": summary.get("episodes", ""),
    }
    for key, value in metrics.items():
        row[str(key)] = value
    return row


def train_config_from_payload(payload: Mapping[str, Any], device: torch.device) -> TrainConfig:
    cfg_payload = payload.get("train_config")
    if cfg_payload is None:
        cfg = TrainConfig()
    elif isinstance(cfg_payload, Mapping):
        valid = {field.name for field in fields(TrainConfig)}
        cfg = TrainConfig(**{key: value for key, value in cfg_payload.items() if key in valid})
    elif isinstance(cfg_payload, TrainConfig):
        cfg = cfg_payload
    else:
        raise TypeError(f"Unsupported checkpoint train_config type: {type(cfg_payload).__name__}")
    return replace(cfg, device=str(device))


def collector_config_from_train_config(cfg: TrainConfig) -> CollectorConfig:
    amp_dtype = str(cfg.amp_dtype).lower()
    if amp_dtype not in {"fp16", "bf16"}:
        raise ValueError(f"Unsupported amp_dtype: {cfg.amp_dtype!r}; expected 'fp16' or 'bf16'")
    return CollectorConfig(
        rows=int(cfg.rows),
        cols=int(cfg.cols),
        obs_size=int(cfg.obs_size),
        scan_radius=int(cfg.scan_radius),
        obstacle_ratio=float(cfg.obstacle_ratio),
        max_episode_steps=int(cfg.max_episode_steps),
        coverage_stop_threshold=float(cfg.coverage_stop_threshold),
        trajectory_history_steps=int(cfg.trajectory_history_steps),
        reward_info_scale=float(cfg.reward_info_scale),
        reward_obstacle_weight=float(cfg.reward_obstacle_weight),
        reward_step_penalty=float(cfg.reward_step_penalty),
        reward_terminal_bonus=float(cfg.reward_terminal_bonus),
        reward_revisit_penalty=float(cfg.reward_revisit_penalty),
        reward_turn_penalty_scale=float(cfg.reward_turn_penalty_scale),
        reward_turn_weight_45=float(cfg.reward_turn_weight_45),
        reward_turn_weight_90=float(cfg.reward_turn_weight_90),
        reward_turn_weight_135=float(cfg.reward_turn_weight_135),
        reward_turn_weight_180=float(cfg.reward_turn_weight_180),
        reward_timeout_penalty=float(cfg.reward_timeout_penalty),
        n_step=int(cfg.n_step),
        gamma=float(cfg.gamma),
        enable_timing=False,
        enable_cummap_timing=False,
        enable_inference_amp=bool(cfg.enable_inference_amp),
        inference_amp_dtype=amp_dtype,
        debug_check_incremental_frontier=bool(cfg.debug_check_incremental_frontier),
        prefer_batch_replay_add=bool(cfg.prefer_batch_replay_add),
        use_fixed_train_episode_seeds=bool(cfg.use_fixed_train_episode_seeds),
        fixed_train_episode_seed_base=int(cfg.fixed_train_episode_seed_base),
        record_episode_artifacts=False,
    )


def state_adapter_config_from_train_config(cfg: TrainConfig) -> StateAdapterConfig:
    from env.advantage_state_builder import AdvantageStateConfig
    from env.shared_semantic_layer import SharedSemanticConfig
    from env.value_state_builder import ValueStateConfig

    return StateAdapterConfig(
        shared_semantics=SharedSemanticConfig(enable_timing=False),
        advantage_state=AdvantageStateConfig(
            trajectory_history_steps=int(cfg.trajectory_history_steps),
            enable_timing=False,
        ),
        value_state=ValueStateConfig(
            max_accessible_blocks=int(cfg.max_accessible_blocks),
            max_entries_per_block=int(cfg.max_entries_per_block),
            enable_timing=False,
        ),
        pin_memory=True,
        non_blocking_transfer=True,
        channels_last_on_cuda=bool(cfg.enable_channels_last),
        enable_timing=False,
    )


def resolve_device(device_text: str, *, require_available: bool) -> tuple[torch.device, dict[str, Any]]:
    device = torch.device(str(device_text))
    info = {
        "requested": str(device_text),
        "resolved": str(device),
        "cuda_available": bool(torch.cuda.is_available()),
        "status": "ok",
    }
    if device.type == "cuda" and not torch.cuda.is_available():
        info["status"] = "unavailable"
        if require_available:
            raise ReadinessError("CUDA was requested with --device cuda, but torch.cuda.is_available() is false.")
    return device, info


def _full_method_checkpoint() -> Path:
    direct = REPO_ROOT / "checkpoint_store" / "full_method_main.pt"
    if direct.exists():
        return direct
    candidates = sorted((REPO_ROOT / "checkpoint_store" / "full_method_main").glob("*.pt"))
    if candidates:
        return candidates[0]
    return REPO_ROOT / "checkpoint_store" / "full_method_main" / "A_full_method.pt"


def _checkpoint_path(method_id: str) -> Path:
    if method_id == "A":
        return _full_method_checkpoint()
    if method_id == "C":
        return REPO_ROOT / "checkpoint_store" / "baselines" / "C_baseline_local_state_ddqn.pt"
    name_by_id = {
        "D": "D_ablation_no_value_tree.pt",
        "E": "E_ablation_no_semantic_dual_state_split.pt",
        "F1": "F1_ablation_no_frontier_channel.pt",
        "F2": "F2_ablation_no_visit_count_channel.pt",
        "F3": "F3_ablation_no_recent_trajectory_channel.pt",
        "F4": "F4_ablation_no_visit_traj_channels.pt",
        "F5": "F5_ablation_occupancy_only_canvas.pt",
        "R1": "R1_ablation_no_step_penalty.pt",
        "R2": "R2_ablation_no_revisit_penalty.pt",
        "R3": "R3_ablation_no_turn_penalty.pt",
        "R4": "R4_ablation_no_timeout_penalty.pt",
        "R5": "R5_ablation_no_efficiency_penalties.pt",
        "R6": "R6_ablation_sparse_reward_variant.pt",
    }
    return REPO_ROOT / "checkpoint_store" / "ablations" / name_by_id[method_id]


def _config_snapshot_path(method_id: str) -> Path:
    if method_id == "A":
        return REPO_ROOT / "experiment_records" / "full_method_main" / "logs" / "config_snapshot.json"
    if method_id == "C":
        return (
            REPO_ROOT
            / "experiment_records"
            / "baselines"
            / "C_baseline_local_state_ddqn"
            / "logs"
            / "config_snapshot.json"
        )
    dirname = {
        "D": "D_ablation_no_value_tree",
        "E": "E_ablation_no_semantic_dual_state_split",
        "F1": "F1_ablation_no_frontier_channel",
        "F2": "F2_ablation_no_visit_count_channel",
        "F3": "F3_ablation_no_recent_trajectory_channel",
        "F4": "F4_ablation_no_visit_traj_channels",
        "F5": "F5_ablation_occupancy_only_canvas",
        "R1": "R1_ablation_no_step_penalty",
        "R2": "R2_ablation_no_revisit_penalty",
        "R3": "R3_ablation_no_turn_penalty",
        "R4": "R4_ablation_no_timeout_penalty",
        "R5": "R5_ablation_no_efficiency_penalties",
        "R6": "R6_ablation_sparse_reward_variant",
    }[method_id]
    return REPO_ROOT / "experiment_records" / "ablations" / dirname / "logs" / "config_snapshot.json"


def build_method_specs(include_r6: bool = False) -> list[dict[str, Any]]:
    methods: list[dict[str, Any]] = [
        {
            "method_id": "B",
            "group": "B",
            "display_name": "classical_frontier_greedy",
            "runner": str((REPO_ROOT / "scripts" / "run_frontier_greedy_baseline.py").resolve()),
            "checkpoint_required": False,
            "checkpoint_path": None,
            "config_snapshot_path": None,
            "model_factory": "none",
            "state_adapter_factory": "FrontierGreedyBaselineEvaluator internal StateTensorAdapter",
            "evaluation_order": 0,
        },
        {
            "method_id": "A",
            "group": "A",
            "display_name": "full_method_main",
            "checkpoint_required": True,
            "checkpoint_path": str(_checkpoint_path("A").resolve()),
            "config_snapshot_path": str(_config_snapshot_path("A").resolve()),
            "model_factory": "ExplorationQNetwork",
            "state_adapter_factory": "StateTensorAdapter",
            "evaluation_order": 1,
        },
        {
            "method_id": "C",
            "group": "C",
            "display_name": "local_state_ddqn",
            "checkpoint_required": True,
            "checkpoint_path": str(_checkpoint_path("C").resolve()),
            "config_snapshot_path": str(_config_snapshot_path("C").resolve()),
            "model_factory": "LocalStateQNetwork",
            "state_adapter_factory": "LocalStateTensorAdapter",
            "evaluation_order": 2,
        },
        {
            "method_id": "D",
            "group": "D",
            "display_name": "no_value_tree",
            "checkpoint_required": True,
            "checkpoint_path": str(_checkpoint_path("D").resolve()),
            "config_snapshot_path": str(_config_snapshot_path("D").resolve()),
            "model_factory": "ExplorationQNetwork",
            "state_adapter_factory": "AblationStateTensorAdapter(value_replacement_strategy=zero_value_state)",
            "value_replacement_strategy": VALUE_REPLACEMENT_STRATEGY_ZERO,
            "evaluation_order": 3,
        },
        {
            "method_id": "E",
            "group": "E",
            "display_name": "no_semantic_dual_state_split",
            "checkpoint_required": True,
            "checkpoint_path": str(_checkpoint_path("E").resolve()),
            "config_snapshot_path": str(_config_snapshot_path("E").resolve()),
            "model_factory": "NoSemanticDualStateSplitQNetwork",
            "state_adapter_factory": "StateTensorAdapter",
            "evaluation_order": 4,
        },
    ]
    for method_id in ("F1", "F2", "F3", "F4", "F5", "R1", "R2", "R3", "R4", "R5"):
        spec = get_ablation_spec(method_id)
        is_f = method_id.startswith("F")
        methods.append(
            {
                "method_id": method_id,
                "group": "F" if is_f else "R",
                "display_name": spec.ablation_id,
                "checkpoint_required": True,
                "checkpoint_path": str(_checkpoint_path(method_id).resolve()),
                "config_snapshot_path": str(_config_snapshot_path(method_id).resolve()),
                "model_factory": "ExplorationQNetwork",
                "state_adapter_factory": (
                    f"AblationStateTensorAdapter(zeroed_channels={list(spec.zeroed_channels)})"
                    if is_f
                    else "StateTensorAdapter"
                ),
                "zeroed_advantage_channels": list(spec.zeroed_channels),
                "reward_override": dict(spec.reward_overrides),
                "evaluation_order": len(methods),
            }
        )
    if include_r6:
        spec = get_ablation_spec("R6")
        methods.append(
            {
                "method_id": "R6",
                "group": "R",
                "display_name": spec.ablation_id,
                "checkpoint_required": True,
                "checkpoint_path": str(_checkpoint_path("R6").resolve()),
                "config_snapshot_path": str(_config_snapshot_path("R6").resolve()),
                "model_factory": "ExplorationQNetwork",
                "state_adapter_factory": "StateTensorAdapter",
                "reward_override": dict(spec.reward_overrides),
                "evaluation_order": len(methods),
            }
        )
    return methods


def _model_for_method(method: Mapping[str, Any]) -> torch.nn.Module:
    method_id = str(method["method_id"])
    if method_id == "C":
        return LocalStateQNetwork()
    if method_id == "E":
        return NoSemanticDualStateSplitQNetwork()
    return ExplorationQNetwork(ExplorationQConfig())


def _adapter_for_method(method: Mapping[str, Any], cfg: TrainConfig) -> StateTensorAdapter:
    adapter_cfg = state_adapter_config_from_train_config(cfg)
    method_id = str(method["method_id"])
    if method_id == "C":
        return LocalStateTensorAdapter(cfg=adapter_cfg, device="cpu")
    if method_id == "D":
        return AblationStateTensorAdapter(
            cfg=adapter_cfg,
            device="cpu",
            value_replacement_strategy=VALUE_REPLACEMENT_STRATEGY_ZERO,
        )
    if method_id.startswith("F"):
        return AblationStateTensorAdapter(
            cfg=adapter_cfg,
            device="cpu",
            zeroed_channels=tuple(method.get("zeroed_advantage_channels", ())),
            value_replacement_strategy="none",
        )
    return StateTensorAdapter(cfg=adapter_cfg, device="cpu")


def _load_checkpoint_payload(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise TypeError(f"Checkpoint payload is {type(payload).__name__}, expected mapping")
    if "online_state_dict" not in payload:
        raise KeyError("Checkpoint is missing online_state_dict")
    return dict(payload)


def audit_methods(
    methods: Sequence[Mapping[str, Any]],
    *,
    output_root: Path,
    device: torch.device,
    validate_factories: bool,
) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    missing: list[dict[str, str]] = []
    for method in methods:
        method_id = str(method["method_id"])
        output_dir = output_root / "methods" / method_id
        entry = {
            "method_id": method_id,
            "display_name": method["display_name"],
            "group": method["group"],
            "output_dir": str(output_dir.resolve()),
            "per_episode_csv": str((output_dir / "per_episode.csv").resolve()),
            "summary_json": str((output_dir / "summary.json").resolve()),
            "checkpoint_required": bool(method.get("checkpoint_required")),
            "checkpoint_path": method.get("checkpoint_path"),
            "checkpoint_status": "not_required",
            "config_snapshot_path": method.get("config_snapshot_path"),
            "config_snapshot_status": "not_required",
            "model_factory": method["model_factory"],
            "model_factory_status": "not_checked",
            "state_adapter_factory": method["state_adapter_factory"],
            "state_adapter_factory_status": "not_checked",
        }
        if method_id == "B":
            runner = Path(str(method["runner"]))
            entry["runner"] = str(runner)
            entry["runner_status"] = "ok" if runner.exists() else "missing"
            if not runner.exists():
                missing.append({"method_id": method_id, "kind": "runner", "path": str(runner)})
            entries.append(entry)
            continue

        checkpoint_path = Path(str(method["checkpoint_path"]))
        config_path = Path(str(method["config_snapshot_path"]))
        entry["checkpoint_status"] = "ok" if checkpoint_path.exists() else "missing"
        entry["config_snapshot_status"] = "ok" if config_path.exists() else "missing"
        if not checkpoint_path.exists():
            missing.append({"method_id": method_id, "kind": "checkpoint", "path": str(checkpoint_path)})
        if not config_path.exists():
            missing.append({"method_id": method_id, "kind": "config_snapshot", "path": str(config_path)})
        if validate_factories and checkpoint_path.exists():
            try:
                payload = _load_checkpoint_payload(checkpoint_path)
                cfg = train_config_from_payload(payload, device)
                model = _model_for_method(method)
                load_result = model.load_state_dict(payload["online_state_dict"], strict=True)
                _ = _adapter_for_method(method, cfg)
                entry["model_factory_status"] = "ok"
                entry["state_adapter_factory_status"] = "ok"
                entry["checkpoint_train_config_status"] = "ok"
                entry["checkpoint_env_steps"] = payload.get("env_steps")
                entry["checkpoint_learn_steps"] = payload.get("learn_steps")
                entry["checkpoint_train_episode_idx"] = payload.get("train_episode_idx")
                entry["load_state_dict"] = {
                    "missing_keys": list(load_result.missing_keys),
                    "unexpected_keys": list(load_result.unexpected_keys),
                }
            except Exception as exc:
                text = f"{type(exc).__name__}: {exc}"
                entry["model_factory_status"] = "error"
                entry["state_adapter_factory_status"] = "error"
                entry["factory_error"] = text
                missing.append({"method_id": method_id, "kind": "factory", "path": text})
        entries.append(entry)
    output_files = {
        "final_probe_summary_csv": str((output_root / "final_probe_summary.csv").resolve()),
        "final_probe_per_episode_csv": str((output_root / "final_probe_per_episode.csv").resolve()),
        "final_probe_protocol_json": str((output_root / "final_probe_protocol.json").resolve()),
        "method_registry_json": str((output_root / "method_registry.json").resolve()),
        "run_manifest_json": str((output_root / "run_manifest.json").resolve()),
    }
    return {
        "created_at": _now_iso(),
        "repo_root": str(REPO_ROOT.resolve()),
        "method_count": len(entries),
        "missing_count": len(missing),
        "missing": missing,
        "methods": entries,
        "output_root": str(output_root.resolve()),
        "output_files": output_files,
    }


def run_baseline_method(method: Mapping[str, Any], *, episodes: int, seed_base: int, output_root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    import scripts.run_frontier_greedy_baseline as baseline_runner

    method_dir = output_root / "methods" / "B"
    runner_args = [
        "--episodes",
        str(int(episodes)),
        "--fixed-final-probe-seed-base",
        str(int(seed_base)),
        "--use-fixed-final-probe-seeds",
        "--run-stage",
        "formal",
        "--output-dir",
        str(method_dir),
    ]
    run_dir = baseline_runner.main(runner_args)
    episode_rows = _read_csv_rows(Path(run_dir) / "logs" / "final_probe.csv")
    for row in episode_rows:
        row["method_id"] = "B"
        row["group"] = method["group"]
        row["display_name"] = method["display_name"]
        row["checkpoint_path"] = ""
    metric_snapshot = _read_json(Path(run_dir) / "logs" / "metric_snapshot.json")
    final_probe = dict(metric_snapshot.get("final_probe", {}))
    summary = {
        "method_id": "B",
        "group": method["group"],
        "display_name": method["display_name"],
        "status": "ok",
        "checkpoint_path": None,
        "run_dir": str(Path(run_dir).resolve()),
        "episodes": int(episodes),
        "seed_base": int(seed_base),
        "summary": final_probe,
    }
    _write_csv(method_dir / "per_episode.csv", episode_rows, preferred=("method_id", "episode_idx", "episode_seed"))
    _write_json(method_dir / "summary.json", summary)
    return summary, episode_rows


def evaluate_checkpoint_method(
    method: Mapping[str, Any],
    *,
    episodes: int,
    seed_base: int,
    device: torch.device,
    output_root: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    checkpoint_path = Path(str(method["checkpoint_path"]))
    payload = _load_checkpoint_payload(checkpoint_path)
    cfg = train_config_from_payload(payload, device)
    set_seed(int(cfg.seed))
    configure_torch_runtime(cfg)

    model = _model_for_method(method)
    model.load_state_dict(payload["online_state_dict"], strict=True)
    model.to(device)
    model.eval()

    state_adapter = _adapter_for_method(method, cfg)
    evaluator = GreedyEvaluator.from_collector_config(
        collector_config_from_train_config(cfg),
        state_adapter=state_adapter,
        device=str(device),
    )
    probe = evaluator.evaluate(model, num_episodes=int(episodes), seed_base=int(seed_base))
    method_dir = output_root / "methods" / str(method["method_id"])
    episode_rows = []
    for idx, row in enumerate(probe.get("episodes", []), start=1):
        out = _episode_csv_row(row)
        out["method_id"] = method["method_id"]
        out["group"] = method["group"]
        out["display_name"] = method["display_name"]
        out["checkpoint_path"] = str(checkpoint_path.resolve())
        out.setdefault("episode_idx", idx)
        out.setdefault("episode_seed", int(seed_base) + idx - 1)
        episode_rows.append(out)

    summary_payload = {
        "method_id": method["method_id"],
        "group": method["group"],
        "display_name": method["display_name"],
        "status": "ok",
        "checkpoint_path": str(checkpoint_path.resolve()),
        "config_snapshot_path": method.get("config_snapshot_path"),
        "episodes": int(episodes),
        "seed_base": int(seed_base),
        "checkpoint_metadata": {
            "env_steps": payload.get("env_steps"),
            "learn_steps": payload.get("learn_steps"),
            "train_episode_idx": payload.get("train_episode_idx"),
        },
        "method_loading": {
            "model_factory": method["model_factory"],
            "state_adapter_factory": method["state_adapter_factory"],
        },
        "summary": _summary_from_probe(probe),
    }
    _write_csv(method_dir / "per_episode.csv", episode_rows, preferred=("method_id", "episode_idx", "episode_seed"))
    _write_json(method_dir / "summary.json", summary_payload)
    return summary_payload, episode_rows


def _baseline_command_for_dry_run(episodes: int, seed_base: int, output_root: Path) -> list[str]:
    return [
        sys.executable,
        str((REPO_ROOT / "scripts" / "run_frontier_greedy_baseline.py").resolve()),
        "--episodes",
        str(int(episodes)),
        "--fixed-final-probe-seed-base",
        str(int(seed_base)),
        "--use-fixed-final-probe-seeds",
        "--run-stage",
        "formal",
        "--output-dir",
        str((output_root / "methods" / "B").resolve()),
    ]


def write_protocol(
    *,
    output_root: Path,
    methods: Sequence[Mapping[str, Any]],
    episodes: int,
    seed_base: int,
    device_text: str,
    include_r6: bool,
) -> None:
    payload = {
        "protocol_name": "final_probe_matrix_v1",
        "created_at": _now_iso(),
        "episodes": int(episodes),
        "seed_base": int(seed_base),
        "episode_seed_rule": "seed_base + zero_based_episode_index",
        "device": str(device_text),
        "run_order": [method["method_id"] for method in methods],
        "b_group_rule": "B classical frontier greedy baseline runs first and requires no checkpoint.",
        "checkpoint_rule": "A/C/D/E/F/R are loaded from checkpoint_store with method-aware model and state adapter factories.",
        "include_r6": bool(include_r6),
        "r6_default": "excluded unless --include-r6 is specified",
        "checkpoint_copy_policy": "checkpoints are referenced in place and are not copied into experiment_records",
    }
    _write_json(output_root / "final_probe_protocol.json", payload)


def write_registry(output_root: Path, methods: Sequence[Mapping[str, Any]]) -> None:
    _write_json(
        output_root / "method_registry.json",
        {
            "schema_version": "final_probe_method_registry/v1",
            "created_at": _now_iso(),
            "methods": list(methods),
        },
    )


def run_matrix(args: argparse.Namespace) -> int:
    output_root = Path(args.output_root).expanduser().resolve()
    methods = build_method_specs(include_r6=bool(args.include_r6))
    device, device_info = resolve_device(str(args.device), require_available=not bool(args.dry_run))
    audit = audit_methods(
        methods,
        output_root=output_root,
        device=device,
        validate_factories=True,
    )
    audit["device"] = device_info
    audit["baseline_command"] = _baseline_command_for_dry_run(int(args.episodes), int(args.seed_base), output_root)

    if args.dry_run:
        print("[final_probe_matrix] dry_run=true")
        print("[final_probe_matrix] B_runner=" + str((REPO_ROOT / "scripts" / "run_frontier_greedy_baseline.py").resolve()))
        print("[final_probe_matrix] B_command=" + " ".join(str(part) for part in audit["baseline_command"]))
        for entry in audit["methods"]:
            print(
                "[final_probe_matrix] "
                f"{entry['method_id']}: checkpoint={entry['checkpoint_status']} "
                f"config={entry['config_snapshot_status']} "
                f"model={entry['model_factory']}:{entry['model_factory_status']} "
                f"adapter={entry['state_adapter_factory']}:{entry['state_adapter_factory_status']} "
                f"output={entry['output_dir']}"
            )
        if audit["missing"]:
            print("[final_probe_matrix] missing=" + json.dumps(audit["missing"], ensure_ascii=False))
        else:
            print("[final_probe_matrix] missing=[]")
        print(json.dumps(_json_safe(audit), ensure_ascii=False, indent=2))
        return 1 if audit["missing"] and not bool(args.allow_missing) else 0

    if audit["missing"] and not bool(args.allow_missing):
        _write_json(output_root / "run_manifest.json", {"status": "aborted", "audit": audit})
        raise ReadinessError("Readiness audit found missing or invalid required artifacts; use --allow-missing to skip missing methods.")

    output_root.mkdir(parents=True, exist_ok=True)
    write_protocol(
        output_root=output_root,
        methods=methods,
        episodes=int(args.episodes),
        seed_base=int(args.seed_base),
        device_text=str(args.device),
        include_r6=bool(args.include_r6),
    )
    write_registry(output_root, methods)

    summary_rows: list[dict[str, Any]] = []
    all_episode_rows: list[dict[str, Any]] = []
    run_entries: list[dict[str, Any]] = []
    start_all = time.perf_counter()

    for method in methods:
        method_id = str(method["method_id"])
        if method_id != "B" and not Path(str(method["checkpoint_path"])).exists():
            if not bool(args.allow_missing):
                raise ReadinessError(f"Missing checkpoint for {method_id}: {method['checkpoint_path']}")
            run_entries.append({"method_id": method_id, "status": "missing", "checkpoint_path": method.get("checkpoint_path")})
            continue
        print(f"[final_probe_matrix] evaluating {method_id} {method['display_name']}")
        start = time.perf_counter()
        try:
            if method_id == "B":
                summary, episodes = run_baseline_method(
                    method,
                    episodes=int(args.episodes),
                    seed_base=int(args.seed_base),
                    output_root=output_root,
                )
            else:
                summary, episodes = evaluate_checkpoint_method(
                    method,
                    episodes=int(args.episodes),
                    seed_base=int(args.seed_base),
                    device=device,
                    output_root=output_root,
                )
            summary_rows.append(_flatten_summary_row(method, summary.get("summary", {}), "ok"))
            all_episode_rows.extend(episodes)
            run_entries.append(
                {
                    "method_id": method_id,
                    "status": "ok",
                    "runtime_sec": time.perf_counter() - start,
                    "summary_json": str((output_root / "methods" / method_id / "summary.json").resolve()),
                    "per_episode_csv": str((output_root / "methods" / method_id / "per_episode.csv").resolve()),
                }
            )
        except Exception as exc:
            error_text = f"{type(exc).__name__}: {exc}"
            error_trace = traceback.format_exc()
            print(f"[final_probe_matrix] error {method_id}: {error_text}", file=sys.stderr)
            method_dir = output_root / "methods" / method_id
            failed_summary = {
                "method_id": method_id,
                "group": method["group"],
                "display_name": method["display_name"],
                "status": "error",
                "error": error_text,
                "traceback": error_trace,
                "checkpoint_path": method.get("checkpoint_path"),
            }
            _write_json(method_dir / "summary.json", failed_summary)
            summary_rows.append(_flatten_summary_row(method, {}, "error", error_text))
            run_entries.append(
                {
                    "method_id": method_id,
                    "status": "error",
                    "runtime_sec": time.perf_counter() - start,
                    "error": error_text,
                    "traceback": error_trace,
                }
            )
            if not bool(args.continue_on_error):
                break

    _write_csv(
        output_root / "final_probe_summary.csv",
        summary_rows,
        preferred=("method_id", "group", "display_name", "status", "episodes", "success_rate", "coverage", "reward"),
    )
    _write_csv(
        output_root / "final_probe_per_episode.csv",
        all_episode_rows,
        preferred=("method_id", "group", "display_name", "episode_idx", "episode_seed", "success", "final_coverage", "episode_reward"),
    )
    manifest = {
        "schema_version": "final_probe_run_manifest/v1",
        "created_at": _now_iso(),
        "status": "ok" if all(entry.get("status") in {"ok", "missing"} for entry in run_entries) else "error",
        "repo_root": str(REPO_ROOT.resolve()),
        "git_sha": _git_output(["rev-parse", "HEAD"]),
        "git_branch": _git_output(["rev-parse", "--abbrev-ref", "HEAD"]),
        "python": sys.version,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "arguments": vars(args),
        "audit": audit,
        "methods": run_entries,
        "runtime_sec": time.perf_counter() - start_all,
        "output_files": audit["output_files"],
    }
    _write_json(output_root / "run_manifest.json", manifest)
    return 0 if manifest["status"] == "ok" else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the unified 100-episode final probe matrix.")
    parser.add_argument("--dry-run", action="store_true", help="Audit artifacts and print planned work without evaluating episodes.")
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES)
    parser.add_argument("--seed-base", type=int, default=DEFAULT_SEED_BASE)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--include-r6", action="store_true", help="Include R6 sparse reward variant.")
    parser.add_argument("--allow-missing", action="store_true", help="Skip missing checkpoint methods instead of aborting formal runs.")
    parser.add_argument("--continue-on-error", action="store_true", help="Continue after a method evaluation failure.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if int(args.episodes) <= 0:
        raise SystemExit("--episodes must be > 0")
    try:
        return run_matrix(args)
    except ReadinessError as exc:
        print(f"[final_probe_matrix] readiness error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
