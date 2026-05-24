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
from env.advantage_state_builder import (
    ADVANTAGE_CANVAS_SCHEMA_FINAL_4CH_NO_FRONTIER_RASTER,
    ADVANTAGE_CANVAS_SCHEMAS,
    advantage_canvas_channel_count_for_schema,
    advantage_canvas_channels_for_schema,
    advantage_canvas_uses_frontier_raster,
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
        "method_family": "A_new_final_method",
        "baseline_method": "A_new_final_4ch_no_frontier_raster",
        "method_id": cfg.method_id,
        "method_name": cfg.method_name,
        "experiment_id": cfg.experiment_id,
        "run_name": cfg.run_name,
        "run_stage": cfg.run_stage,
        "advantage_canvas_schema": cfg.advantage_canvas_schema,
        "advantage_canvas_channels": list(channels),
        "advantage_canvas_channel_count": channel_count,
        "frontier_raster_used": bool(frontier_raster_used),
        "value_tree_enabled": bool(cfg.value_tree_enabled),
        "value_tree_unchanged": bool(cfg.value_tree_unchanged),
        "value_branch_source": cfg.value_branch_source,
        "value_branch_representation": cfg.value_branch_representation,
        "model_class": cfg.model_class,
        "advantage_encoder.canvas_in_channels": channel_count,
        "reward_override": dict(cfg.reward_override),
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
        print(f"[A_new] warning: artifact_index.json not found at {artifact_index_path}")
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
                    "category": "final_method_manifest",
                }
            )
        else:
            raise TypeError("final_method_artifacts is present but is not a list")
        _write_json(artifact_index_path, payload)
    except Exception as exc:
        print(f"[A_new] warning: failed to append final_method_manifest to artifact_index: {exc}")


def manifest_payload(
    *,
    cfg: train_q_agent.TrainConfig,
    runner_entrypoint: str,
) -> dict[str, object]:
    channels = advantage_canvas_channels_for_schema(cfg.advantage_canvas_schema)
    channel_count = advantage_canvas_channel_count_for_schema(cfg.advantage_canvas_schema)
    return {
        "schema_version": "final_method_manifest/v1",
        "method_id": cfg.method_id,
        "method_name": cfg.method_name,
        "experiment_id": cfg.experiment_id,
        "run_name": cfg.run_name,
        "run_stage": cfg.run_stage,
        "advantage_canvas_schema": cfg.advantage_canvas_schema,
        "advantage_canvas_channels": list(channels),
        "advantage_canvas_channel_count": int(channel_count),
        "frontier_raster_used": bool(cfg.frontier_raster_used),
        "value_tree_enabled": bool(cfg.value_tree_enabled),
        "value_tree_unchanged": bool(cfg.value_tree_unchanged),
        "value_branch_source": cfg.value_branch_source,
        "value_branch_representation": cfg.value_branch_representation,
        "model_class": cfg.model_class,
        "advantage_encoder.canvas_in_channels": int(channel_count),
        "reward_override": dict(cfg.reward_override),
        "source_entrypoint": "train_q_agent.py",
        "runner_entrypoint": runner_entrypoint,
        "notes": [
            "A_new removes frontier_block_area_map from the advantage canvas.",
            "The structured frontier-block value tree remains enabled and unchanged.",
            "F1 remains a legacy 5-channel zero-frontier diagnostic, not the final network schema.",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="A_new final 4-channel no-frontier-raster launcher")
    parser.add_argument("--run-stage", choices=("smoke", "pilot", "formal"), default="smoke")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output-root", type=str, default="outputs")
    parser.add_argument("--experiment-id", type=str, default="A_new")
    parser.add_argument("--method-id", type=str, default=None)
    parser.add_argument("--method-name", type=str, default=None)
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
            "A_new final method must use "
            f"{ADVANTAGE_CANVAS_SCHEMA_FINAL_4CH_NO_FRONTIER_RASTER!r}, "
            f"got {args.advantage_canvas_schema!r}."
        )

    run_name = args.run_name or f"{args.experiment_id}_final_4ch_no_frontier_raster_{args.run_stage}"
    method_id = args.method_id or args.experiment_id
    method_name = args.method_name or "final_4ch_no_frontier_raster"
    train_args = build_train_args(
        run_stage=args.run_stage,
        device=args.device,
        output_root=args.output_root,
        experiment_id=args.experiment_id,
        method_id=method_id,
        method_name=method_name,
        run_name=run_name,
        advantage_canvas_schema=args.advantage_canvas_schema,
        passthrough=passthrough,
    )
    command = build_command(train_args)
    print(f"[A_new] command: {_command_text(command)}", flush=True)

    if args.dry_run:
        cfg = _parse_train_config(train_args)
        print(json.dumps(
            dry_run_payload(
                cfg=cfg,
                command=command,
                train_args=train_args,
                runner_entrypoint="experiments/final_method/run_a_new_final_method.py",
            ),
            indent=2,
            ensure_ascii=False,
        ))
        return 0

    cfg = _parse_train_config(train_args)
    run_dir = train_q_agent.run_training(cfg, run_mode=f"final_method_{args.run_stage}")
    manifest_path = run_dir / "logs" / "final_method_manifest.json"
    _write_json(
        manifest_path,
        manifest_payload(
            cfg=cfg,
            runner_entrypoint="experiments/final_method/run_a_new_final_method.py",
        ),
    )
    _try_append_artifact_index(run_dir, manifest_path)
    print(f"final_method_manifest_json: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
