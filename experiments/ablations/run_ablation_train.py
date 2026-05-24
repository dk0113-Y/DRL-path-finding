from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path
from typing import Iterable

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.ablations.ablation_specs import (
    AblationSpec,
    ablation_slug,
    get_ablation_spec,
    is_channel_ablation,
    is_frontier_channel_variant,
    is_reward_ablation,
    is_semantic_split_ablation,
    is_value_tree_ablation,
    list_ablation_specs,
)
from experiments.ablations.frontier_channel_variants import describe_frontier_channel_mode
from experiments.ablations.reward_overrides import apply_reward_overrides
from experiments.ablations.semantic_split_ablation import (
    NoSemanticDualStateSplitQNetwork,
    build_no_semantic_dual_state_split_manifest,
    count_model_parameters,
)
from experiments.ablations.state_adapter_wrapper import AblationStateTensorAdapter
from experiments.ablations.value_tree_ablation import VALUE_REPLACEMENT_STRATEGY_ZERO
import train_q_agent


def _has_option(args: Iterable[str], option_name: str) -> bool:
    prefix = f"{option_name}="
    return any(arg == option_name or arg.startswith(prefix) for arg in args)


def _normalize_passthrough(args: list[str]) -> list[str]:
    return args[1:] if args and args[0] == "--" else args


def _build_train_args(spec: AblationSpec, run_stage: str, passthrough: list[str]) -> list[str]:
    train_args = list(passthrough)
    if run_stage == "smoke" and not _has_option(train_args, "--smoke"):
        train_args.append("--smoke")
    if not _has_option(train_args, "--run-name"):
        train_args.extend(["--run-name", f"{ablation_slug(spec)}_{run_stage}"])
    if not _has_option(train_args, "--output-root"):
        train_args.extend(["--output-root", "outputs"])
    return train_args


def _validate_stage_train_args(run_stage: str, train_args: list[str]) -> None:
    if run_stage in {"pilot", "formal"} and _has_option(train_args, "--smoke"):
        raise ValueError(
            "--smoke is only allowed with --run-stage smoke. "
            f"Got run_stage={run_stage!r} with --smoke in passthrough args. "
            "Use --run-stage smoke for smoke tests, or remove --smoke for pilot/formal runs."
        )


def _parse_train_config(train_args: list[str]) -> train_q_agent.TrainConfig:
    original_argv = sys.argv
    sys.argv = ["train_q_agent.py", *train_args]
    try:
        return train_q_agent.parse_args()
    finally:
        sys.argv = original_argv


def _apply_ablation_config(
    cfg: train_q_agent.TrainConfig,
    spec: AblationSpec,
    run_stage: str,
) -> train_q_agent.TrainConfig:
    if is_reward_ablation(spec):
        cfg = apply_reward_overrides(cfg, spec)
        return replace(
            cfg,
            ablation_group=spec.group,
            ablation_id=spec.ablation_id,
            experiment_id=spec.experiment_id or spec.short_id,
            ablation_name=spec.ablation_name or spec.ablation_id,
            channel_ablation="none",
            zeroed_advantage_channels=(),
            reward_override=dict(spec.reward_overrides),
            value_replacement_strategy="none",
            value_tree_enabled=True,
            run_stage=run_stage,
        )
    if is_channel_ablation(spec):
        return replace(
            cfg,
            ablation_group=spec.group,
            ablation_id=spec.ablation_id,
            experiment_id=spec.experiment_id or spec.short_id,
            ablation_name=spec.ablation_name or spec.ablation_id,
            channel_ablation=spec.ablation_id,
            zeroed_advantage_channels=tuple(spec.zeroed_channels),
            reward_override={},
            value_replacement_strategy="none",
            value_tree_enabled=True,
            run_stage=run_stage,
        )
    if is_frontier_channel_variant(spec):
        if not spec.frontier_channel_mode:
            raise ValueError(f"Frontier channel variant {spec.short_id} is missing frontier_channel_mode.")
        return replace(
            cfg,
            ablation_group=spec.group,
            ablation_id=spec.ablation_id,
            experiment_id=spec.experiment_id or spec.short_id,
            ablation_name=spec.ablation_name or spec.ablation_id,
            channel_ablation="none",
            zeroed_advantage_channels=(),
            reward_override={},
            value_replacement_strategy="none",
            value_tree_enabled=True,
            advantage_frontier_channel_mode=str(spec.frontier_channel_mode),
            run_stage=run_stage,
        )
    if is_value_tree_ablation(spec):
        return replace(
            cfg,
            ablation_group=spec.group,
            ablation_id=spec.ablation_id,
            experiment_id=spec.experiment_id or spec.short_id,
            ablation_name=spec.ablation_name or spec.ablation_id,
            channel_ablation="none",
            zeroed_advantage_channels=(),
            reward_override={},
            value_replacement_strategy=VALUE_REPLACEMENT_STRATEGY_ZERO,
            value_tree_enabled=False,
            run_stage=run_stage,
        )
    if is_semantic_split_ablation(spec):
        model_parameter_count = count_model_parameters(NoSemanticDualStateSplitQNetwork())
        return replace(
            cfg,
            ablation_group=spec.group,
            ablation_id=spec.ablation_id,
            experiment_id=spec.experiment_id or spec.short_id,
            ablation_name=spec.ablation_name or spec.ablation_id,
            channel_ablation="none",
            zeroed_advantage_channels=(),
            reward_override={},
            value_replacement_strategy="none",
            value_tree_enabled=True,
            is_ablation=True,
            no_shared_semantic_dual_state=True,
            no_value_tree=False,
            model_class="NoSemanticDualStateSplitQNetwork",
            model_parameter_count=model_parameter_count,
            dummy_value_tensors_for_interface=False,
            value_tensors_used_by_model=True,
            run_stage=run_stage,
        )
    raise ValueError(f"Unsupported ablation group: {spec.group!r}")


def _state_adapter_factory_for(spec: AblationSpec):
    if not is_channel_ablation(spec) and not is_value_tree_ablation(spec):
        return None

    def factory(cfg=None, device="cpu"):
        return AblationStateTensorAdapter(
            cfg=cfg,
            device=device,
            zeroed_channels=spec.zeroed_channels,
            value_replacement_strategy=(
                spec.value_replacement_strategy if is_value_tree_ablation(spec) else "none"
            ),
        )

    return factory


def _model_factory_for(spec: AblationSpec):
    if not is_semantic_split_ablation(spec):
        return None

    def factory(cfg=None):
        _ = cfg
        return NoSemanticDualStateSplitQNetwork()

    return factory


def _dry_run_payload(
    *,
    spec: AblationSpec,
    cfg: train_q_agent.TrainConfig,
    run_stage: str,
    train_args: list[str],
) -> dict[str, object]:
    reward_fields = {
        "reward_info_scale": cfg.reward_info_scale,
        "reward_obstacle_weight": cfg.reward_obstacle_weight,
        "reward_step_penalty": cfg.reward_step_penalty,
        "reward_terminal_bonus": cfg.reward_terminal_bonus,
        "reward_revisit_penalty": cfg.reward_revisit_penalty,
        "reward_turn_penalty_scale": cfg.reward_turn_penalty_scale,
        "reward_timeout_penalty": cfg.reward_timeout_penalty,
    }
    frontier_descriptor = describe_frontier_channel_mode(cfg.advantage_frontier_channel_mode)
    return {
        "dry_run": True,
        "ablation_spec": asdict(spec),
        "run_name": cfg.run_name,
        "run_stage": run_stage,
        **frontier_descriptor,
        "zeroed_advantage_channels": list(cfg.zeroed_advantage_channels),
        "reward_override": dict(cfg.reward_override),
        "value_replacement_strategy": cfg.value_replacement_strategy,
        "value_tree_enabled": cfg.value_tree_enabled,
        "channel_ablation": cfg.channel_ablation,
        "no_semantic_dual_state_split": bool(is_semantic_split_ablation(spec)),
        "semantic_dual_state_split_used": not is_semantic_split_ablation(spec),
        "model_class": cfg.model_class,
        "model_parameter_count": cfg.model_parameter_count,
        "advantage_canvas_channels": list(cfg.advantage_canvas_channels),
        "train_args": train_args,
        "train_config": {
            "device": cfg.device,
            "output_root": cfg.output_root,
            "total_env_steps": cfg.total_env_steps,
            "final_greedy_episodes": cfg.final_greedy_episodes,
            "experiment_id": cfg.experiment_id,
            "ablation_group": cfg.ablation_group,
            "ablation_id": cfg.ablation_id,
            "ablation_name": cfg.ablation_name,
            "run_stage": cfg.run_stage,
            "value_replacement_strategy": cfg.value_replacement_strategy,
            "value_tree_enabled": cfg.value_tree_enabled,
            "channel_ablation": cfg.channel_ablation,
            "advantage_frontier_channel_mode": cfg.advantage_frontier_channel_mode,
            "no_semantic_dual_state_split": bool(is_semantic_split_ablation(spec)),
            "semantic_dual_state_split_used": not is_semantic_split_ablation(spec),
            "model_class": cfg.model_class,
            "model_parameter_count": cfg.model_parameter_count,
            **reward_fields,
        },
    }


def _manifest_payload(spec: AblationSpec, cfg: train_q_agent.TrainConfig) -> dict[str, object]:
    frontier_descriptor = describe_frontier_channel_mode(cfg.advantage_frontier_channel_mode)
    return {
        "schema_version": "ablation_manifest/v1",
        "run_stage": cfg.run_stage,
        "experiment_id": cfg.experiment_id,
        "ablation_group": cfg.ablation_group,
        "ablation_id": cfg.ablation_id,
        "ablation_name": cfg.ablation_name,
        "short_id": spec.short_id,
        "description": spec.description,
        **frontier_descriptor,
        "zeroed_advantage_channels": list(cfg.zeroed_advantage_channels),
        "reward_override": dict(cfg.reward_override),
        "reward_overrides": "none" if not cfg.reward_override else dict(cfg.reward_override),
        "channel_ablation": cfg.channel_ablation,
        "value_replacement_strategy": cfg.value_replacement_strategy,
        "value_tree_enabled": cfg.value_tree_enabled,
        "value_tree_unchanged": not bool(is_value_tree_ablation(spec)),
        "no_semantic_dual_state_split": bool(is_semantic_split_ablation(spec)),
        "semantic_dual_state_split_used": not is_semantic_split_ablation(spec),
        "value_tree_used_by_model": bool(cfg.value_tensors_used_by_model),
        "model_class": cfg.model_class,
        "model_parameter_count": cfg.model_parameter_count,
        "safe_zero_dummy_value_state": False,
        "advantage_canvas_channels": list(cfg.advantage_canvas_channels),
        "full_method_default_unchanged": True,
        "source_entrypoint": "experiments/ablations/run_ablation_train.py",
        "notes": list(spec.notes),
    }


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _try_append_artifact_index(run_dir: Path, manifest_path: Path) -> None:
    artifact_index_path = run_dir / "logs" / "artifact_index.json"
    if not artifact_index_path.exists():
        print(f"[ablation] warning: artifact_index.json not found at {artifact_index_path}")
        return
    try:
        payload = json.loads(artifact_index_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise TypeError("artifact_index root is not a JSON object")
        rel_path = manifest_path.relative_to(run_dir).as_posix()
        payload.setdefault("ablation_artifacts", [])
        if isinstance(payload["ablation_artifacts"], list):
            payload["ablation_artifacts"].append(
                {
                    "path": rel_path,
                    "required": True,
                    "category": "ablation_manifest",
                }
            )
        else:
            raise TypeError("ablation_artifacts is present but is not a list")
        _write_json(artifact_index_path, payload)
    except Exception as exc:
        print(f"[ablation] warning: failed to append ablation_manifest to artifact_index: {exc}")


def _print_specs() -> None:
    for spec in list_ablation_specs():
        zeroed = ", ".join(spec.zeroed_channels) if spec.zeroed_channels else "-"
        rewards = ", ".join(f"{k}={v}" for k, v in spec.reward_overrides.items()) or "-"
        value_strategy = spec.value_replacement_strategy or "none"
        recommended = "yes" if spec.recommended else "no"
        frontier_mode = spec.frontier_channel_mode or "-"
        print(
            f"{spec.short_id:>2}  {spec.ablation_id:<34} "
            f"group={spec.group:<17} recommended={recommended:<3} "
            f"zeroed=[{zeroed}] reward_overrides=[{rewards}] "
            f"value_strategy={value_strategy} frontier_mode={frontier_mode}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Structural, channel, and reward ablation training entrypoint")
    parser.add_argument("--ablation-id", type=str, default=None)
    parser.add_argument("--run-stage", choices=("smoke", "pilot", "formal"), default="smoke")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--list", action="store_true")
    args, passthrough = parser.parse_known_args(argv)
    passthrough = _normalize_passthrough(passthrough)

    if args.list:
        _print_specs()
        return 0
    if not args.ablation_id:
        parser.error("--ablation-id is required unless --list is used")

    spec = get_ablation_spec(args.ablation_id)
    train_args = _build_train_args(spec, args.run_stage, passthrough)
    _validate_stage_train_args(args.run_stage, train_args)
    cfg = _parse_train_config(train_args)
    cfg = _apply_ablation_config(cfg, spec, args.run_stage)
    state_adapter_factory = _state_adapter_factory_for(spec)
    model_factory = _model_factory_for(spec)

    if args.dry_run:
        print(json.dumps(
            _dry_run_payload(spec=spec, cfg=cfg, run_stage=args.run_stage, train_args=train_args),
            indent=2,
            ensure_ascii=False,
        ))
        return 0

    run_dir = train_q_agent.run_training(
        cfg,
        run_mode=f"ablation_{args.run_stage}",
        state_adapter_factory=state_adapter_factory,
        model_factory=model_factory,
    )
    manifest_path = run_dir / "logs" / "ablation_manifest.json"
    manifest = _manifest_payload(spec, cfg)
    if is_semantic_split_ablation(spec):
        manifest.update(
            build_no_semantic_dual_state_split_manifest(
                model=NoSemanticDualStateSplitQNetwork(),
            )
        )
        manifest["source_entrypoint"] = "experiments/ablations/run_ablation_train.py"
        manifest["notes"] = list(spec.notes)
    _write_json(manifest_path, manifest)
    _try_append_artifact_index(run_dir, manifest_path)
    print(f"ablation_manifest_json: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
