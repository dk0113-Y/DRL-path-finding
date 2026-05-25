from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import train_q_agent
from agents.q_value_agent import ExplorationQConfig, ExplorationQNetwork
from encoders.advantage_encoder import AdvantageEncoderConfig
from env.advantage_state_builder import (
    ADVANTAGE_CANVAS_SCHEMA_FINAL_4CH_NO_FRONTIER_RASTER,
    ADVANTAGE_CANVAS_SCHEMAS,
    advantage_canvas_channel_count_for_schema,
    advantage_canvas_channels_for_schema,
    advantage_canvas_uses_frontier_raster,
)


DEFAULT_EXPERIMENT_ID = "Anew_F"
DEFAULT_METHOD_ID = "Anew_F3_no_behavior_memory"
DEFAULT_METHOD_NAME = "no_behavior_memory"
DEFAULT_ABLATION_GROUP = "input_state"
DEFAULT_ABLATION_ID = "F_key"
DEFAULT_ABLATION_NAME = "no_behavior_memory"
DEFAULT_CHANNEL_ABLATION = "no_behavior_memory"
ZEROED_ADVANTAGE_CHANNELS = ("visit_count_log_norm", "recent_trajectory_decay")


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


def _model_parameter_count(cfg: train_q_agent.TrainConfig) -> int:
    channels = tuple(advantage_canvas_channels_for_schema(cfg.advantage_canvas_schema))
    model = ExplorationQNetwork(
        ExplorationQConfig(
            advantage_encoder=AdvantageEncoderConfig(
                canvas_in_channels=len(channels),
                canvas_channels=channels,
            )
        )
    )
    return int(sum(parameter.numel() for parameter in model.parameters()))


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
    train_args.extend(["--channel-ablation", DEFAULT_CHANNEL_ABLATION])
    train_args.extend(["--zeroed-advantage-channels", ",".join(ZEROED_ADVANTAGE_CHANNELS)])
    return train_args


def build_command(train_args: list[str]) -> list[str]:
    return [sys.executable, "train_q_agent.py", *train_args]


def dry_run_payload(
    *,
    cfg: train_q_agent.TrainConfig,
    command: list[str],
    train_args: list[str],
    runner_entrypoint: str,
) -> dict[str, object]:
    channels = advantage_canvas_channels_for_schema(cfg.advantage_canvas_schema)
    channel_count = advantage_canvas_channel_count_for_schema(cfg.advantage_canvas_schema)
    frontier_raster_used = advantage_canvas_uses_frontier_raster(cfg.advantage_canvas_schema)
    return {
        "dry_run": True,
        "method_family": "A_new_input_state_ablation",
        "baseline_method": "A_new_final_4ch_no_frontier_raster",
        "experiment_id": cfg.experiment_id,
        "method_id": cfg.method_id,
        "method_name": cfg.method_name,
        "ablation_group": cfg.ablation_group,
        "ablation_id": cfg.ablation_id,
        "ablation_name": cfg.ablation_name,
        "channel_ablation": cfg.channel_ablation,
        "zeroed_advantage_channels": list(cfg.zeroed_advantage_channels),
        "occupancy_only_alias": True,
        "separate_occupancy_only_formal_row": False,
        "a_new_f4_occupancy_only_note": (
            "Anew_F4_occupancy_only is not kept as a separate formal row; "
            "under the current A_new 4-channel schema it is equivalent to Anew_F3_no_behavior_memory."
        ),
        "run_name": cfg.run_name,
        "run_stage": cfg.run_stage,
        "advantage_canvas_schema": cfg.advantage_canvas_schema,
        "advantage_canvas_channels": list(channels),
        "advantage_canvas_channel_count": channel_count,
        "frontier_raster_used": bool(frontier_raster_used),
        "value_tree_enabled": bool(cfg.value_tree_enabled),
        "value_tree_unchanged": bool(cfg.value_tree_unchanged),
        "value_replacement_strategy": cfg.value_replacement_strategy,
        "value_branch_source": cfg.value_branch_source,
        "value_branch_representation": cfg.value_branch_representation,
        "dummy_value_tensors_for_interface": bool(cfg.dummy_value_tensors_for_interface),
        "dummy_value_mask_rule": cfg.dummy_value_mask_rule,
        "model_class": cfg.model_class,
        "model_parameter_count": _model_parameter_count(cfg),
        "advantage_encoder.canvas_in_channels": channel_count,
        "reward_override": dict(cfg.reward_override),
        "reward_info_scale": float(cfg.reward_info_scale),
        "reward_obstacle_weight": float(cfg.reward_obstacle_weight),
        "learner_updates_per_iter": int(cfg.learner_updates_per_iter),
        "min_replay_size": int(cfg.min_replay_size),
        "epsilon_end": float(cfg.epsilon_end),
        "epsilon_decay_steps": int(cfg.epsilon_decay_steps),
        "train_side_only_tuning": bool(cfg.train_side_only_tuning),
        "source_entrypoint": "train_q_agent.py",
        "runner_entrypoint": runner_entrypoint,
        "command": command,
        "command_text": _command_text(command),
        "train_args": train_args,
        "train_config": asdict(cfg),
    }


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _try_append_artifact_index(run_dir: Path, manifest_path: Path) -> None:
    artifact_index_path = run_dir / "logs" / "artifact_index.json"
    if not artifact_index_path.exists():
        print(f"[A_new_F] warning: artifact_index.json not found at {artifact_index_path}")
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
                    "category": "a_new_no_behavior_memory_ablation_manifest",
                }
            )
        else:
            raise TypeError("final_method_artifacts is present but is not a list")
        _write_json(artifact_index_path, payload)
    except Exception as exc:
        print(f"[A_new_F] warning: failed to append no-behavior-memory manifest to artifact_index: {exc}")


def manifest_payload(
    *,
    cfg: train_q_agent.TrainConfig,
    runner_entrypoint: str,
) -> dict[str, object]:
    channels = advantage_canvas_channels_for_schema(cfg.advantage_canvas_schema)
    channel_count = advantage_canvas_channel_count_for_schema(cfg.advantage_canvas_schema)
    return {
        "schema_version": "a_new_no_behavior_memory_ablation_manifest/v1",
        "method_family": "A_new_input_state_ablation",
        "baseline_method": "A_new_final_4ch_no_frontier_raster",
        "experiment_id": cfg.experiment_id,
        "method_id": cfg.method_id,
        "method_name": cfg.method_name,
        "ablation_group": cfg.ablation_group,
        "ablation_id": cfg.ablation_id,
        "ablation_name": cfg.ablation_name,
        "channel_ablation": cfg.channel_ablation,
        "zeroed_advantage_channels": list(cfg.zeroed_advantage_channels),
        "occupancy_only_alias": True,
        "separate_occupancy_only_formal_row": False,
        "run_name": cfg.run_name,
        "run_stage": cfg.run_stage,
        "advantage_canvas_schema": cfg.advantage_canvas_schema,
        "advantage_canvas_channels": list(channels),
        "advantage_canvas_channel_count": int(channel_count),
        "frontier_raster_used": bool(cfg.frontier_raster_used),
        "value_tree_enabled": bool(cfg.value_tree_enabled),
        "value_tree_unchanged": bool(cfg.value_tree_unchanged),
        "value_replacement_strategy": cfg.value_replacement_strategy,
        "value_branch_source": cfg.value_branch_source,
        "value_branch_representation": cfg.value_branch_representation,
        "dummy_value_tensors_for_interface": bool(cfg.dummy_value_tensors_for_interface),
        "dummy_value_mask_rule": cfg.dummy_value_mask_rule,
        "model_class": cfg.model_class,
        "model_parameter_count": _model_parameter_count(cfg),
        "advantage_encoder.canvas_in_channels": int(channel_count),
        "reward_override": dict(cfg.reward_override),
        "source_entrypoint": "train_q_agent.py",
        "runner_entrypoint": runner_entrypoint,
        "notes": [
            "Anew_F3_no_behavior_memory is aligned to the current A_new four-channel advantage canvas.",
            "The advantage branch keeps free and obstacle unchanged.",
            "The behavior-memory channels visit_count_log_norm and recent_trajectory_decay are zeroed after canvas construction.",
            "The structured frontier-block value tree remains enabled and unchanged.",
            "Under the current schema this tensor operation is equivalent to an occupancy-only advantage canvas.",
            "Anew_F4_occupancy_only is not kept as a separate formal row, run name, or artifact row.",
            "No legacy 5-channel frontier raster, frontier_block_area_map, or legacy F artifact is restored.",
            "Smoke and pilot runs are local checks only, not paper Results evidence.",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="A_new F_key no-behavior-memory input-state ablation launcher")
    parser.add_argument("--run-stage", choices=("smoke", "pilot", "formal"), default="smoke")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output-root", type=str, default="outputs")
    parser.add_argument("--experiment-id", type=str, default=DEFAULT_EXPERIMENT_ID)
    parser.add_argument("--method-id", type=str, default=DEFAULT_METHOD_ID)
    parser.add_argument("--method-name", type=str, default=DEFAULT_METHOD_NAME)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument(
        "--advantage-canvas-schema",
        choices=ADVANTAGE_CANVAS_SCHEMAS,
        default=ADVANTAGE_CANVAS_SCHEMA_FINAL_4CH_NO_FRONTIER_RASTER,
    )
    args, passthrough = parser.parse_known_args(argv)
    passthrough = _normalize_passthrough(passthrough)

    if args.advantage_canvas_schema != ADVANTAGE_CANVAS_SCHEMA_FINAL_4CH_NO_FRONTIER_RASTER:
        raise ValueError(
            "Anew_F3_no_behavior_memory must use "
            f"{ADVANTAGE_CANVAS_SCHEMA_FINAL_4CH_NO_FRONTIER_RASTER!r}, "
            f"got {args.advantage_canvas_schema!r}."
        )
    if args.method_id != DEFAULT_METHOD_ID:
        raise ValueError(f"Anew_F3_no_behavior_memory method_id must be {DEFAULT_METHOD_ID!r}.")
    if args.method_name != DEFAULT_METHOD_NAME:
        raise ValueError(f"Anew_F3_no_behavior_memory method_name must be {DEFAULT_METHOD_NAME!r}.")

    run_name = args.run_name or f"{DEFAULT_METHOD_ID}_{args.run_stage}"
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
    command = build_command(train_args)
    print(f"[A_new_F] command: {_command_text(command)}", flush=True)

    cfg = _parse_train_config(train_args)
    runner_entrypoint = "experiments/final_method/run_a_new_no_behavior_memory_ablation.py"
    if tuple(cfg.zeroed_advantage_channels) != ZEROED_ADVANTAGE_CHANNELS:
        raise ValueError(
            "Anew_F3_no_behavior_memory zeroed channels must be "
            f"{ZEROED_ADVANTAGE_CHANNELS!r}; got {cfg.zeroed_advantage_channels!r}."
        )
    if cfg.no_value_tree or not cfg.value_tree_enabled or not cfg.value_tree_unchanged:
        raise ValueError("Anew_F3_no_behavior_memory must keep the A_new value tree enabled and unchanged.")

    if args.dry_run:
        print(json.dumps(
            dry_run_payload(
                cfg=cfg,
                command=command,
                train_args=train_args,
                runner_entrypoint=runner_entrypoint,
            ),
            indent=2,
            ensure_ascii=False,
        ))
        return 0

    print("[A_new_F] runtime_contract:")
    print(json.dumps(
        manifest_payload(
            cfg=cfg,
            runner_entrypoint=runner_entrypoint,
        ),
        indent=2,
        ensure_ascii=False,
    ))
    run_dir = train_q_agent.run_training(cfg, run_mode=f"a_new_no_behavior_memory_ablation_{args.run_stage}")
    manifest_path = run_dir / "logs" / "final_method_manifest.json"
    _write_json(
        manifest_path,
        manifest_payload(
            cfg=cfg,
            runner_entrypoint=runner_entrypoint,
        ),
    )
    _try_append_artifact_index(run_dir, manifest_path)
    print(f"final_method_manifest_json: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
