from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Iterable

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import train_q_agent
from agents.no_dual_state_split_q_network import (
    NoDualStateSplitQNetwork,
    no_dual_state_split_model_parameter_count,
)
from env.advantage_state_builder import (
    ADVANTAGE_CANVAS_SCHEMA_FINAL_4CH_NO_FRONTIER_RASTER,
    ADVANTAGE_CANVAS_SCHEMAS,
    FINAL_4CH_ADVANTAGE_CANVAS_CHANNELS,
    advantage_canvas_channel_count_for_schema,
    advantage_canvas_channels_for_schema,
    advantage_canvas_uses_frontier_raster,
)
from experiments.final_method.artifact_archiving import (
    DEFAULT_CHECKPOINT_STORE_ROOT,
    DEFAULT_RECORDS_ROOT,
    archive_dry_run_payload,
    archive_training_run,
)


DEFAULT_EXPERIMENT_ID = "Anew_E"
DEFAULT_METHOD_ID = "Anew_E_no_dual_state_split"
DEFAULT_METHOD_NAME = "no_dual_state_split_flat_value_injected_q"
DEFAULT_ABLATION_GROUP = "structural"
DEFAULT_ABLATION_ID = "E"
DEFAULT_ABLATION_NAME = "no_dual_state_split"
RUNNER_ENTRYPOINT = "experiments/final_method/run_a_new_no_dual_state_split_ablation.py"
MODEL_CLASS = "NoDualStateSplitQNetwork"


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
    advantage_canvas_schema: str,
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
    train_args.extend(["--advantage-canvas-schema", advantage_canvas_schema])
    train_args.extend(["--ablation-group", DEFAULT_ABLATION_GROUP])
    train_args.extend(["--ablation-id", DEFAULT_ABLATION_ID])
    train_args.extend(["--ablation-name", DEFAULT_ABLATION_NAME])
    return train_args


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
        "--advantage-canvas-schema",
        str(args.advantage_canvas_schema),
    ]
    if bool(args.dry_run):
        command.append("--dry-run")
    if passthrough:
        command.append("--")
        command.extend(passthrough)
    return command


def e_model_factory(*, cfg: train_q_agent.TrainConfig) -> NoDualStateSplitQNetwork:
    _ = cfg
    return NoDualStateSplitQNetwork()


def apply_no_dual_state_split_contract(cfg: train_q_agent.TrainConfig) -> train_q_agent.TrainConfig:
    if tuple(cfg.zeroed_advantage_channels):
        raise ValueError("Anew_E_no_dual_state_split must not zero behavior-memory channels.")
    if cfg.no_value_tree or not cfg.value_tree_enabled or str(cfg.value_replacement_strategy) != "none":
        raise ValueError("Anew_E_no_dual_state_split must keep real value-tree tensors enabled.")
    if cfg.reward_override:
        raise ValueError("Anew_E_no_dual_state_split must keep reward_override empty.")
    if str(cfg.advantage_canvas_schema) != ADVANTAGE_CANVAS_SCHEMA_FINAL_4CH_NO_FRONTIER_RASTER:
        raise ValueError(
            "Anew_E_no_dual_state_split must use "
            f"{ADVANTAGE_CANVAS_SCHEMA_FINAL_4CH_NO_FRONTIER_RASTER!r}."
        )

    return replace(
        cfg,
        experiment_id=DEFAULT_EXPERIMENT_ID,
        method_id=DEFAULT_METHOD_ID,
        method_name=DEFAULT_METHOD_NAME,
        ablation_group=DEFAULT_ABLATION_GROUP,
        ablation_id=DEFAULT_ABLATION_ID,
        ablation_name=DEFAULT_ABLATION_NAME,
        channel_ablation="none",
        zeroed_advantage_channels=(),
        reward_override={},
        value_replacement_strategy="none",
        value_tree_enabled=True,
        value_tree_unchanged=False,
        value_branch_source="SharedSemanticSnapshot",
        value_branch_representation="flattened_value_summary_injected_into_local_q",
        baseline_id="none",
        baseline_group="none",
        baseline_name="none",
        baseline_type="none",
        is_learning_baseline=False,
        is_ablation=True,
        uses_structured_value_tree=True,
        behavior_memory_channels_used=True,
        checkpoint_source="trained_from_scratch",
        no_shared_semantic_dual_state=True,
        dual_state_split_enabled=False,
        explicit_advantage_value_split=False,
        semantic_dueling_head_used=False,
        no_semantic_dueling_head=True,
        value_tree_information_used=True,
        value_tree_branch_separate=False,
        value_tree_summary_injected=True,
        no_value_tree=False,
        no_frontier_cluster_input=False,
        no_accessible_unknown_block_input=False,
        no_ground_truth_map_for_decision=True,
        local_state_channels=(),
        local_state_patch_size=0,
        local_state_source="none",
        local_state_carrier_key="none",
        local_state_canvas_role="full_method_advantage_canvas",
        model_class=MODEL_CLASS,
        model_parameter_count=no_dual_state_split_model_parameter_count(),
        dummy_value_tensors_for_interface=False,
        value_tensors_used_by_model=True,
        dummy_value_block_shape=(),
        dummy_value_entry_shape=(),
        dummy_value_mask_rule="none",
    )


def _identity_payload(cfg: train_q_agent.TrainConfig) -> dict[str, Any]:
    channels = advantage_canvas_channels_for_schema(cfg.advantage_canvas_schema)
    channel_count = advantage_canvas_channel_count_for_schema(cfg.advantage_canvas_schema)
    frontier_raster_used = advantage_canvas_uses_frontier_raster(cfg.advantage_canvas_schema)
    return {
        "experiment_id": cfg.experiment_id,
        "method_id": cfg.method_id,
        "method_name": cfg.method_name,
        "ablation_group": cfg.ablation_group,
        "ablation_id": cfg.ablation_id,
        "ablation_name": cfg.ablation_name,
        "dual_state_split_enabled": bool(cfg.dual_state_split_enabled),
        "explicit_advantage_value_split": bool(cfg.explicit_advantage_value_split),
        "semantic_dueling_head_used": bool(cfg.semantic_dueling_head_used),
        "no_semantic_dueling_head": bool(cfg.no_semantic_dueling_head),
        "value_tree_information_used": bool(cfg.value_tree_information_used),
        "value_tree_enabled": bool(cfg.value_tree_enabled),
        "value_tree_unchanged": bool(cfg.value_tree_unchanged),
        "value_tree_branch_separate": bool(cfg.value_tree_branch_separate),
        "value_tree_summary_injected": bool(cfg.value_tree_summary_injected),
        "value_replacement_strategy": cfg.value_replacement_strategy,
        "value_branch_source": cfg.value_branch_source,
        "value_branch_representation": cfg.value_branch_representation,
        "advantage_canvas_schema": cfg.advantage_canvas_schema,
        "advantage_canvas_channels": list(channels),
        "advantage_canvas_channel_count": int(channel_count),
        "frontier_raster_used": bool(frontier_raster_used),
        "zeroed_advantage_channels": list(cfg.zeroed_advantage_channels),
        "behavior_memory_channels_used": bool(cfg.behavior_memory_channels_used),
        "dummy_value_tensors_for_interface": bool(cfg.dummy_value_tensors_for_interface),
        "dummy_value_mask_rule": cfg.dummy_value_mask_rule,
        "value_tensors_used_by_model": bool(cfg.value_tensors_used_by_model),
        "model_class": cfg.model_class,
        "model_parameter_count": int(cfg.model_parameter_count),
        "reward_override": dict(cfg.reward_override),
        "reward_info_scale": float(cfg.reward_info_scale),
        "reward_obstacle_weight": float(cfg.reward_obstacle_weight),
        "reward_step_penalty": float(cfg.reward_step_penalty),
        "reward_terminal_bonus": float(cfg.reward_terminal_bonus),
        "reward_revisit_penalty": float(cfg.reward_revisit_penalty),
        "reward_turn_penalty_scale": float(cfg.reward_turn_penalty_scale),
        "reward_timeout_penalty": float(cfg.reward_timeout_penalty),
        "learner_updates_per_iter": int(cfg.learner_updates_per_iter),
        "min_replay_size": int(cfg.min_replay_size),
        "epsilon_end": float(cfg.epsilon_end),
        "epsilon_decay_steps": int(cfg.epsilon_decay_steps),
        "train_side_only_tuning": bool(cfg.train_side_only_tuning),
    }


def dry_run_payload(
    *,
    cfg: train_q_agent.TrainConfig,
    command: list[str],
    train_args: list[str],
    runner_entrypoint: str,
    records_root: Path,
    checkpoint_store_root: Path,
    copy_checkpoints: bool,
) -> dict[str, Any]:
    return {
        **archive_dry_run_payload(
            method_id=cfg.method_id,
            records_root=records_root,
            checkpoint_store_root=checkpoint_store_root,
            copy_checkpoints=copy_checkpoints,
        ),
        "dry_run": True,
        "method_family": "A_new_structural_ablation",
        "baseline_method": "A_new_final_4ch_no_frontier_raster",
        "run_name": cfg.run_name,
        "run_stage": cfg.run_stage,
        **_identity_payload(cfg),
        "source_entrypoint": "train_q_agent.py",
        "runner_entrypoint": runner_entrypoint,
        "execution_mode": "in_process_train_q_agent_with_no_dual_state_split_model_factory",
        "command": command,
        "command_text": _command_text(command),
        "train_args": train_args,
        "train_config": asdict(cfg),
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _try_append_artifact_index(run_dir: Path, manifest_path: Path) -> None:
    artifact_index_path = run_dir / "logs" / "artifact_index.json"
    if not artifact_index_path.exists():
        print(f"[A_new_E] warning: artifact_index.json not found at {artifact_index_path}")
        return
    try:
        payload = json.loads(artifact_index_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise TypeError("artifact_index root is not a JSON object")
        rel_path = manifest_path.relative_to(run_dir).as_posix()
        payload.setdefault("final_method_artifacts", [])
        if isinstance(payload["final_method_artifacts"], list):
            payload["final_method_artifacts"].append(
                {
                    "path": rel_path,
                    "required": True,
                    "category": "a_new_no_dual_state_split_ablation_manifest",
                }
            )
        else:
            raise TypeError("final_method_artifacts is present but is not a list")
        _write_json(artifact_index_path, payload)
    except Exception as exc:
        print(f"[A_new_E] warning: failed to append no-dual-state-split manifest to artifact_index: {exc}")


def manifest_payload(
    *,
    cfg: train_q_agent.TrainConfig,
    runner_entrypoint: str,
) -> dict[str, Any]:
    return {
        "schema_version": "a_new_no_dual_state_split_ablation_manifest/v1",
        "method_family": "A_new_structural_ablation",
        "baseline_method": "A_new_final_4ch_no_frontier_raster",
        "run_name": cfg.run_name,
        "run_stage": cfg.run_stage,
        **_identity_payload(cfg),
        "source_entrypoint": "train_q_agent.py",
        "runner_entrypoint": runner_entrypoint,
        "execution_mode": "in_process_train_q_agent_with_no_dual_state_split_model_factory",
        "notes": [
            "Anew_E_no_dual_state_split is aligned to the current A_new four-channel advantage canvas.",
            "The behavior-memory channels visit_count_log_norm and recent_trajectory_decay remain enabled.",
            "Structured value-tree tensors remain enabled and are summarized before local action conditioning.",
            "The model directly predicts Q values and does not use the A_new semantic dueling decision head.",
            "No legacy 5-channel frontier raster, frontier_block_area_map, or legacy E artifact is restored.",
            "Smoke and pilot runs are local checks only, not paper Results evidence.",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="A_new E no-dual-state-split structural ablation launcher")
    parser.add_argument("--run-stage", choices=("smoke", "pilot", "formal"), default="smoke")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output-root", type=str, default="outputs")
    parser.add_argument("--experiment-id", type=str, default=DEFAULT_EXPERIMENT_ID)
    parser.add_argument("--method-id", type=str, default=DEFAULT_METHOD_ID)
    parser.add_argument("--method-name", type=str, default=DEFAULT_METHOD_NAME)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--records-root", type=Path, default=DEFAULT_RECORDS_ROOT)
    parser.add_argument("--checkpoint-store-root", type=Path, default=DEFAULT_CHECKPOINT_STORE_ROOT)
    parser.add_argument("--copy-checkpoints", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--advantage-canvas-schema",
        choices=ADVANTAGE_CANVAS_SCHEMAS,
        default=ADVANTAGE_CANVAS_SCHEMA_FINAL_4CH_NO_FRONTIER_RASTER,
    )
    args, passthrough = parser.parse_known_args(argv)
    passthrough = _normalize_passthrough(passthrough)

    if args.experiment_id != DEFAULT_EXPERIMENT_ID:
        raise ValueError(f"Anew_E experiment_id must be {DEFAULT_EXPERIMENT_ID!r}.")
    if args.method_id != DEFAULT_METHOD_ID:
        raise ValueError(f"Anew_E method_id must be {DEFAULT_METHOD_ID!r}.")
    if args.method_name != DEFAULT_METHOD_NAME:
        raise ValueError(f"Anew_E method_name must be {DEFAULT_METHOD_NAME!r}.")
    if args.advantage_canvas_schema != ADVANTAGE_CANVAS_SCHEMA_FINAL_4CH_NO_FRONTIER_RASTER:
        raise ValueError(
            "Anew_E_no_dual_state_split must use "
            f"{ADVANTAGE_CANVAS_SCHEMA_FINAL_4CH_NO_FRONTIER_RASTER!r}."
        )

    run_name = args.run_name or f"{DEFAULT_METHOD_ID}_{args.run_stage}"
    runner_args = argparse.Namespace(**vars(args))
    runner_args.run_name = run_name
    command = _runner_command(runner_args, passthrough)
    train_args = build_train_args(
        run_stage=args.run_stage,
        device=args.device,
        output_root=args.output_root,
        experiment_id=args.experiment_id,
        method_id=args.method_id,
        method_name=args.method_name,
        run_name=run_name,
        advantage_canvas_schema=args.advantage_canvas_schema,
        passthrough=passthrough,
    )
    cfg = apply_no_dual_state_split_contract(_parse_train_config(train_args))
    print(f"[A_new_E] command: {_command_text(command)}", flush=True)

    if args.dry_run:
        print(json.dumps(
            dry_run_payload(
                cfg=cfg,
                command=command,
                train_args=train_args,
                runner_entrypoint=RUNNER_ENTRYPOINT,
                records_root=args.records_root,
                checkpoint_store_root=args.checkpoint_store_root,
                copy_checkpoints=bool(args.copy_checkpoints),
            ),
            indent=2,
            ensure_ascii=False,
        ))
        return 0

    print("[A_new_E] runtime_contract:")
    print(json.dumps(
        manifest_payload(
            cfg=cfg,
            runner_entrypoint=RUNNER_ENTRYPOINT,
        ),
        indent=2,
        ensure_ascii=False,
    ))
    run_dir = train_q_agent.run_training(
        cfg,
        run_mode=f"a_new_no_dual_state_split_ablation_{args.run_stage}",
        model_factory=e_model_factory,
    )
    manifest_path = run_dir / "logs" / "final_method_manifest.json"
    _write_json(
        manifest_path,
        manifest_payload(
            cfg=cfg,
            runner_entrypoint=RUNNER_ENTRYPOINT,
        ),
    )
    _try_append_artifact_index(run_dir, manifest_path)
    archive_record = archive_training_run(
        run_dir=run_dir,
        method_id=cfg.method_id,
        method_name=cfg.method_name,
        run_stage=args.run_stage,
        records_root=args.records_root,
        checkpoint_store_root=args.checkpoint_store_root,
        copy_checkpoints=bool(args.copy_checkpoints),
    )
    print(f"archive_record_json: {archive_record['run_record_path']}")
    print(f"checkpoint_store_path: {archive_record['checkpoint_store_path']}")
    print(f"records_logs_dir: {archive_record['records_logs_dir']}")
    print(f"final_method_manifest_json: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
