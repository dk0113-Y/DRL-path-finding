from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Iterable, Mapping

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import train_q_agent
from env.value_state_builder import VALUE_BLOCK_FEATURE_COUNT, VALUE_ENTRY_FEATURE_COUNT
from experiments.final_method.a_new_local_state_ddqn import (
    BASELINE_GROUP,
    BASELINE_NAME,
    BASELINE_TYPE,
    DUMMY_VALUE_MASK_RULE,
    EXPERIMENT_ID,
    LOCAL_STATE_CANVAS_ROLE,
    LOCAL_STATE_CANVAS_SCHEMA,
    LOCAL_STATE_CARRIER_KEY,
    LOCAL_STATE_CHANNELS,
    LOCAL_STATE_SOURCE,
    METHOD_ID,
    METHOD_NAME,
    VALUE_REPLACEMENT_STRATEGY,
    LocalStateTensorAdapter,
    build_local_state_model,
    local_state_patch_size_from_scan_radius,
)
from agents.local_state_q_network import local_state_model_parameter_count


RUNNER_ENTRYPOINT = "experiments/final_method/run_a_new_local_state_ddqn_baseline.py"
PLANNED_ARTIFACTS = (
    "logs/config_snapshot.json",
    "logs/baseline_manifest.json",
    "logs/training_summary.txt",
    "logs/train_episodes.csv",
    "logs/train_steps.csv",
    "logs/metric_snapshot.json",
    "logs/reproducibility_contract.json",
    "logs/artifact_index.json",
    "checkpoints/last.pt",
)


def _has_option(args: Iterable[str], option_name: str) -> bool:
    prefix = f"{option_name}="
    return any(arg == option_name or arg.startswith(prefix) for arg in args)


def _normalize_passthrough(args: list[str]) -> list[str]:
    return args[1:] if args and args[0] == "--" else args


def _command_text(command: list[str]) -> str:
    try:
        return subprocess.list2cmdline([str(item) for item in command])
    except Exception:
        return " ".join(str(item) for item in command)


def _parse_train_config(train_args: list[str]) -> train_q_agent.TrainConfig:
    original_argv = sys.argv
    sys.argv = ["train_q_agent.py", *train_args]
    try:
        return train_q_agent.parse_args()
    finally:
        sys.argv = original_argv


def build_train_args(
    *,
    run_stage: str,
    device: str,
    output_root: str,
    experiment_id: str,
    method_id: str,
    method_name: str,
    run_name: str,
    passthrough: list[str],
) -> list[str]:
    train_args = list(passthrough)
    if run_stage == "smoke" and not _has_option(train_args, "--smoke"):
        train_args.append("--smoke")
    train_args.extend(["--device", device])
    train_args.extend(["--output-root", output_root])
    train_args.extend(["--experiment-id", experiment_id])
    train_args.extend(["--method-id", method_id])
    train_args.extend(["--method-name", method_name])
    train_args.extend(["--run-stage", run_stage])
    train_args.extend(["--run-name", run_name])
    return train_args


def _environment_config(cfg: train_q_agent.TrainConfig) -> dict[str, Any]:
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


def _reward_config(cfg: train_q_agent.TrainConfig) -> dict[str, Any]:
    return {
        "reward_info_scale": float(cfg.reward_info_scale),
        "reward_obstacle_weight": float(cfg.reward_obstacle_weight),
        "reward_step_penalty": float(cfg.reward_step_penalty),
        "reward_terminal_bonus": float(cfg.reward_terminal_bonus),
        "reward_revisit_penalty": float(cfg.reward_revisit_penalty),
        "reward_turn_penalty_scale": float(cfg.reward_turn_penalty_scale),
        "reward_turn_weight_45": float(cfg.reward_turn_weight_45),
        "reward_turn_weight_90": float(cfg.reward_turn_weight_90),
        "reward_turn_weight_135": float(cfg.reward_turn_weight_135),
        "reward_turn_weight_180": float(cfg.reward_turn_weight_180),
        "reward_timeout_penalty": float(cfg.reward_timeout_penalty),
    }


def apply_local_state_baseline_contract(cfg: train_q_agent.TrainConfig) -> train_q_agent.TrainConfig:
    patch_size = local_state_patch_size_from_scan_radius(int(cfg.scan_radius))
    model_parameter_count = local_state_model_parameter_count(patch_size=patch_size)
    return replace(
        cfg,
        experiment_id=EXPERIMENT_ID,
        method_id=METHOD_ID,
        method_name=METHOD_NAME,
        baseline_id=METHOD_ID,
        baseline_group=BASELINE_GROUP,
        baseline_name=BASELINE_NAME,
        baseline_type=BASELINE_TYPE,
        is_learning_baseline=True,
        is_ablation=False,
        uses_structured_value_tree=False,
        behavior_memory_channels_used=False,
        checkpoint_source="trained_from_scratch",
        no_shared_semantic_dual_state=True,
        no_value_tree=False,
        no_frontier_cluster_input=True,
        no_accessible_unknown_block_input=True,
        no_ground_truth_map_for_decision=True,
        local_state_channels=tuple(LOCAL_STATE_CHANNELS),
        local_state_patch_size=int(patch_size),
        local_state_source=LOCAL_STATE_SOURCE,
        local_state_carrier_key=LOCAL_STATE_CARRIER_KEY,
        local_state_canvas_role=LOCAL_STATE_CANVAS_ROLE,
        model_class="LocalStateQNetwork",
        model_parameter_count=int(model_parameter_count),
        dummy_value_tensors_for_interface=True,
        value_tensors_used_by_model=False,
        dummy_value_block_shape=(int(cfg.max_accessible_blocks), int(VALUE_BLOCK_FEATURE_COUNT)),
        dummy_value_entry_shape=(
            int(cfg.max_accessible_blocks),
            int(cfg.max_entries_per_block),
            int(VALUE_ENTRY_FEATURE_COUNT),
        ),
        dummy_value_mask_rule=DUMMY_VALUE_MASK_RULE,
        value_tree_enabled=False,
        value_tree_unchanged=False,
        value_branch_source="not_applicable_to_baseline",
        value_branch_representation="not_applicable_to_baseline",
        frontier_raster_used=False,
        reward_override={},
    )


def _factory_patch_size(cfg: train_q_agent.TrainConfig) -> int:
    patch_size = int(cfg.local_state_patch_size)
    if patch_size <= 0:
        patch_size = local_state_patch_size_from_scan_radius(int(cfg.scan_radius))
    return patch_size


def local_state_model_factory(*, cfg: train_q_agent.TrainConfig):
    return build_local_state_model(patch_size=_factory_patch_size(cfg))


def make_local_state_adapter_factory(train_cfg: train_q_agent.TrainConfig):
    patch_size = _factory_patch_size(train_cfg)

    def _factory(*, cfg, device: str):
        return LocalStateTensorAdapter(cfg=cfg, device=device, patch_size=patch_size)

    return _factory


def _baseline_manifest(
    *,
    cfg: train_q_agent.TrainConfig,
    runner_entrypoint: str,
) -> dict[str, Any]:
    return {
        "schema_version": "a_new_local_state_ddqn_baseline_manifest/v1",
        "experiment_id": EXPERIMENT_ID,
        "method_id": METHOD_ID,
        "method_name": METHOD_NAME,
        "baseline_id": METHOD_ID,
        "baseline_group": BASELINE_GROUP,
        "baseline_name": BASELINE_NAME,
        "baseline_type": BASELINE_TYPE,
        "is_learning_baseline": True,
        "is_ablation": False,
        "model_class": "LocalStateQNetwork",
        "model_parameter_count": int(cfg.model_parameter_count),
        "local_state_channels": list(LOCAL_STATE_CHANNELS),
        "local_state_patch_size": int(cfg.local_state_patch_size),
        "local_state_source": LOCAL_STATE_SOURCE,
        "local_state_carrier_key": LOCAL_STATE_CARRIER_KEY,
        "local_state_canvas_role": LOCAL_STATE_CANVAS_ROLE,
        "no_ground_truth_map_for_decision": True,
        "uses_structured_value_tree": False,
        "value_tree_enabled": False,
        "value_tree_unchanged": False,
        "value_branch_source": "not_applicable_to_baseline",
        "value_branch_representation": "not_applicable_to_baseline",
        "value_tensors_used_by_model": False,
        "dummy_value_tensors_for_interface": True,
        "dummy_value_block_shape": list(cfg.dummy_value_block_shape),
        "dummy_value_entry_shape": list(cfg.dummy_value_entry_shape),
        "dummy_value_mask_rule": DUMMY_VALUE_MASK_RULE,
        "value_replacement_strategy": VALUE_REPLACEMENT_STRATEGY,
        "advantage_canvas_schema": LOCAL_STATE_CANVAS_SCHEMA,
        "frontier_raster_used": False,
        "behavior_memory_channels_used": False,
        "reward_override": {},
        "checkpoint_source": "trained_from_scratch",
        "run_stage": str(cfg.run_stage),
        "train_side_only_tuning": bool(cfg.train_side_only_tuning),
        "seed_policy": {
            "use_fixed_train_episode_seeds": bool(cfg.use_fixed_train_episode_seeds),
            "fixed_train_episode_seed_base": int(cfg.fixed_train_episode_seed_base),
            "use_fixed_eval_seeds": bool(cfg.use_fixed_eval_seeds),
            "fixed_final_probe_seed_base": int(cfg.fixed_final_probe_seed_base),
            "use_fixed_model_select_seeds": bool(cfg.use_fixed_model_select_seeds),
            "fixed_model_select_seed_base": int(cfg.fixed_model_select_seed_base),
        },
        "environment_config": _environment_config(cfg),
        "reward_config": _reward_config(cfg),
        "training_contract": {
            "learner_updates_per_iter": int(cfg.learner_updates_per_iter),
            "min_replay_size": int(cfg.min_replay_size),
            "epsilon_end": float(cfg.epsilon_end),
            "epsilon_decay_steps": int(cfg.epsilon_decay_steps),
            "train_side_only_tuning": bool(cfg.train_side_only_tuning),
        },
        "artifact_schema": list(PLANNED_ARTIFACTS),
        "source_entrypoint": "train_q_agent.py",
        "runner_entrypoint": runner_entrypoint,
        "execution_mode": "in_process_train_q_agent_with_local_state_factories",
        "legacy_inheritance": "none",
        "restores_legacy_baselines": False,
        "restores_experiments_ablations": False,
        "notes": [
            "Anew_C_local_state_ddqn is a learning baseline, not an A_new ablation.",
            "The policy/model input is a three-channel cumulative-belief patch.",
            "Structured value-tree tensors are replaced by all-zero interface tensors and ignored by LocalStateQNetwork.",
            "Behavior-memory channels, frontier rasters, and older baseline artifacts are not used.",
            "Smoke and pilot runs are local checks only, not paper Results evidence.",
        ],
    }


def dry_run_payload(
    *,
    cfg: train_q_agent.TrainConfig,
    command: list[str],
    train_args: list[str],
    runner_entrypoint: str,
) -> dict[str, Any]:
    manifest = _baseline_manifest(cfg=cfg, runner_entrypoint=runner_entrypoint)
    return {
        **manifest,
        "dry_run": True,
        "method_family": "A_new_learning_baseline",
        "baseline_method": "A_new_final_4ch_no_frontier_raster",
        "command": command,
        "command_text": _command_text(command),
        "train_args": list(train_args),
        "train_config": asdict(cfg),
        "reward_info_scale": float(cfg.reward_info_scale),
        "reward_obstacle_weight": float(cfg.reward_obstacle_weight),
        "learner_updates_per_iter": int(cfg.learner_updates_per_iter),
        "min_replay_size": int(cfg.min_replay_size),
        "epsilon_end": float(cfg.epsilon_end),
        "epsilon_decay_steps": int(cfg.epsilon_decay_steps),
    }


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _try_append_artifact_index(run_dir: Path, manifest_path: Path) -> None:
    artifact_index_path = run_dir / "logs" / "artifact_index.json"
    if not artifact_index_path.exists():
        print(f"[A_new_C] warning: artifact_index.json not found at {artifact_index_path}")
        return
    try:
        payload = json.loads(artifact_index_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise TypeError("artifact_index root is not a JSON object")
        rel_path = manifest_path.relative_to(run_dir).as_posix()
        record = {
            "path": rel_path,
            "required": True,
            "exists": True,
            "category": "a_new_local_state_ddqn_baseline_manifest",
        }
        payload.setdefault("baseline_artifacts", [])
        if isinstance(payload["baseline_artifacts"], list):
            payload["baseline_artifacts"].append(record)
        structured = payload.setdefault("structured_summaries", [])
        if isinstance(structured, list):
            structured.append(record)
        _write_json(artifact_index_path, payload)
    except Exception as exc:
        print(f"[A_new_C] warning: failed to append baseline manifest to artifact_index: {exc}")


def _runner_command(args: argparse.Namespace, passthrough: list[str]) -> list[str]:
    command = [
        sys.executable,
        RUNNER_ENTRYPOINT,
        "--run-stage",
        str(args.run_stage),
        "--device",
        str(args.device),
        "--output-root",
        str(args.output_root),
        "--experiment-id",
        str(args.experiment_id),
        "--method-id",
        str(args.method_id),
        "--method-name",
        str(args.method_name),
        "--run-name",
        str(args.run_name),
    ]
    if bool(args.dry_run):
        command.append("--dry-run")
    if passthrough:
        command.append("--")
        command.extend(passthrough)
    return command


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="A_new C local-state DDQN learning baseline launcher")
    parser.add_argument("--run-stage", choices=("smoke", "pilot", "formal"), default="smoke")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output-root", type=str, default="outputs")
    parser.add_argument("--experiment-id", type=str, default=EXPERIMENT_ID)
    parser.add_argument("--method-id", type=str, default=METHOD_ID)
    parser.add_argument("--method-name", type=str, default=METHOD_NAME)
    parser.add_argument("--run-name", type=str, default=None)
    args, passthrough = parser.parse_known_args(argv)
    passthrough = _normalize_passthrough(passthrough)

    if args.experiment_id != EXPERIMENT_ID:
        raise ValueError(f"Anew_C experiment_id must be {EXPERIMENT_ID!r}.")
    if args.method_id != METHOD_ID:
        raise ValueError(f"Anew_C method_id must be {METHOD_ID!r}.")
    if args.method_name != METHOD_NAME:
        raise ValueError(f"Anew_C method_name must be {METHOD_NAME!r}.")

    run_name = args.run_name or f"{METHOD_ID}_{args.run_stage}"
    train_args = build_train_args(
        run_stage=args.run_stage,
        device=args.device,
        output_root=args.output_root,
        experiment_id=args.experiment_id,
        method_id=args.method_id,
        method_name=args.method_name,
        run_name=run_name,
        passthrough=passthrough,
    )
    cfg = apply_local_state_baseline_contract(_parse_train_config(train_args))
    runner_args = argparse.Namespace(**vars(args))
    runner_args.run_name = run_name
    command = _runner_command(runner_args, passthrough)
    print(f"[A_new_C] command: {_command_text(command)}", flush=True)

    if args.dry_run:
        print(json.dumps(
            dry_run_payload(
                cfg=cfg,
                command=command,
                train_args=train_args,
                runner_entrypoint=RUNNER_ENTRYPOINT,
            ),
            indent=2,
            ensure_ascii=False,
        ))
        return 0

    print("[A_new_C] runtime_contract:")
    print(json.dumps(
        _baseline_manifest(cfg=cfg, runner_entrypoint=RUNNER_ENTRYPOINT),
        indent=2,
        ensure_ascii=False,
    ))
    run_dir = train_q_agent.run_training(
        cfg,
        run_mode=f"a_new_local_state_ddqn_baseline_{args.run_stage}",
        state_adapter_factory=make_local_state_adapter_factory(cfg),
        model_factory=local_state_model_factory,
    )
    manifest_path = run_dir / "logs" / "baseline_manifest.json"
    _write_json(manifest_path, _baseline_manifest(cfg=cfg, runner_entrypoint=RUNNER_ENTRYPOINT))
    _try_append_artifact_index(run_dir, manifest_path)
    print(f"baseline_manifest_json: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
