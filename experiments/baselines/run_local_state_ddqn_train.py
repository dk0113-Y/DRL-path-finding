from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import Iterable

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from baselines.local_state_ddqn import (
    DUMMY_VALUE_MASK_RULE,
    LOCAL_STATE_BASELINE_ID,
    LOCAL_STATE_CARRIER_KEY,
    LOCAL_STATE_CHANNELS,
    LocalStateQNetwork,
    LocalStateTensorAdapter,
    build_baseline_manifest,
    count_model_parameters,
)
import train_q_agent


def _has_option(args: Iterable[str], option_name: str) -> bool:
    prefix = f"{option_name}="
    return any(arg == option_name or arg.startswith(prefix) for arg in args)


def _normalize_passthrough(args: list[str]) -> list[str]:
    return args[1:] if args and args[0] == "--" else args


def _build_train_args(run_stage: str, passthrough: list[str]) -> list[str]:
    train_args = list(passthrough)
    if run_stage == "smoke" and not _has_option(train_args, "--smoke"):
        train_args.append("--smoke")
    if not _has_option(train_args, "--run-name"):
        train_args.extend(["--run-name", f"{LOCAL_STATE_BASELINE_ID}_{run_stage}"])
    if not _has_option(train_args, "--output-root"):
        train_args.extend(["--output-root", "outputs"])
    return train_args


def _validate_stage_train_args(run_stage: str, train_args: list[str]) -> None:
    if run_stage in {"pilot", "formal"} and _has_option(train_args, "--smoke"):
        raise ValueError(
            "--smoke is only allowed with --run-stage smoke. "
            f"Got run_stage={run_stage!r} with --smoke in passthrough args."
        )


def _parse_train_config(train_args: list[str]) -> train_q_agent.TrainConfig:
    original_argv = sys.argv
    sys.argv = ["train_q_agent.py", *train_args]
    try:
        return train_q_agent.parse_args()
    finally:
        sys.argv = original_argv


def _apply_baseline_config(
    cfg: train_q_agent.TrainConfig,
    *,
    run_stage: str,
    model_parameter_count: int,
) -> train_q_agent.TrainConfig:
    patch_size = int(2 * int(cfg.scan_radius) + 1)
    return replace(
        cfg,
        experiment_id="C",
        ablation_group="not_applicable",
        ablation_id="not_applicable",
        ablation_name="not_applicable",
        channel_ablation="none",
        zeroed_advantage_channels=(),
        reward_override={},
        value_replacement_strategy="not_applicable",
        value_tree_enabled=False,
        advantage_canvas_channels=tuple(LOCAL_STATE_CHANNELS),
        baseline_id=LOCAL_STATE_BASELINE_ID,
        baseline_group="learning_baseline",
        baseline_name="local_state_ddqn",
        baseline_type="simpler_drl_baseline",
        is_ablation=False,
        no_shared_semantic_dual_state=True,
        no_value_tree=True,
        no_frontier_cluster_input=True,
        no_accessible_unknown_block_input=True,
        no_ground_truth_map_for_decision=True,
        local_state_channels=tuple(LOCAL_STATE_CHANNELS),
        local_state_patch_size=patch_size,
        local_state_carrier_key=LOCAL_STATE_CARRIER_KEY,
        local_state_canvas_role="local_belief_patch_not_full_method_semantic_advantage_canvas",
        model_class="LocalStateQNetwork",
        model_parameter_count=int(model_parameter_count),
        dummy_value_tensors_for_interface=True,
        value_tensors_used_by_model=False,
        dummy_value_block_shape=(int(cfg.max_accessible_blocks), 2),
        dummy_value_entry_shape=(int(cfg.max_accessible_blocks), int(cfg.max_entries_per_block), 4),
        dummy_value_mask_rule=DUMMY_VALUE_MASK_RULE,
        run_stage=run_stage,
    )


def _state_adapter_factory(cfg=None, device="cpu"):
    return LocalStateTensorAdapter(cfg=cfg, device=device)


def _model_factory(cfg=None):
    _ = cfg
    return LocalStateQNetwork()


def _git_sha(repo_dir: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_dir),
            check=True,
            text=True,
            capture_output=True,
        )
    except Exception:
        return None
    return result.stdout.strip() or None


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _try_append_artifact_index(run_dir: Path, manifest_path: Path) -> None:
    artifact_index_path = run_dir / "logs" / "artifact_index.json"
    if not artifact_index_path.exists():
        print(f"[baseline] warning: artifact_index.json not found at {artifact_index_path}")
        return
    try:
        payload = json.loads(artifact_index_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise TypeError("artifact_index root is not a JSON object")
        rel_path = manifest_path.relative_to(run_dir).as_posix()
        payload.setdefault("baseline_artifacts", [])
        if isinstance(payload["baseline_artifacts"], list):
            payload["baseline_artifacts"].append(
                {
                    "path": rel_path,
                    "required": True,
                    "category": "baseline_manifest",
                }
            )
        else:
            raise TypeError("baseline_artifacts is present but is not a list")
        _write_json(artifact_index_path, payload)
    except Exception as exc:
        print(f"[baseline] warning: failed to append baseline_manifest to artifact_index: {exc}")


def _dry_run_payload(
    *,
    cfg: train_q_agent.TrainConfig,
    run_stage: str,
    train_args: list[str],
) -> dict[str, object]:
    return {
        "dry_run": True,
        "experiment_id": cfg.experiment_id,
        "baseline_id": cfg.baseline_id,
        "baseline_group": cfg.baseline_group,
        "baseline_name": cfg.baseline_name,
        "baseline_type": cfg.baseline_type,
        "is_ablation": cfg.is_ablation,
        "run_stage": run_stage,
        "local_state_channels": list(cfg.local_state_channels),
        "local_state_patch_size": cfg.local_state_patch_size,
        "local_state_carrier_key": cfg.local_state_carrier_key,
        "model_class": cfg.model_class,
        "model_parameter_count": cfg.model_parameter_count,
        "value_tensors_used_by_model": cfg.value_tensors_used_by_model,
        "dummy_value_tensors_for_interface": cfg.dummy_value_tensors_for_interface,
        "train_args": train_args,
        "reward_overrides": "none",
        "channel_ablation": "none",
        "value_replacement_strategy": "not_applicable",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="C local-state DDQN learning baseline training entrypoint")
    parser.add_argument("--baseline-id", type=str, default=LOCAL_STATE_BASELINE_ID)
    parser.add_argument("--run-stage", choices=("smoke", "pilot", "formal"), default="smoke")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--list", action="store_true")
    args, passthrough = parser.parse_known_args(argv)
    passthrough = _normalize_passthrough(passthrough)

    if args.list:
        print(f"{LOCAL_STATE_BASELINE_ID}  group=learning_baseline  name=local_state_ddqn")
        return 0
    if args.baseline_id != LOCAL_STATE_BASELINE_ID:
        parser.error(f"--baseline-id must be {LOCAL_STATE_BASELINE_ID!r}; got {args.baseline_id!r}")

    train_args = _build_train_args(args.run_stage, passthrough)
    _validate_stage_train_args(args.run_stage, train_args)
    model_parameter_count = count_model_parameters(LocalStateQNetwork())
    cfg = _parse_train_config(train_args)
    cfg = _apply_baseline_config(
        cfg,
        run_stage=args.run_stage,
        model_parameter_count=model_parameter_count,
    )

    if args.dry_run:
        print(json.dumps(
            _dry_run_payload(cfg=cfg, run_stage=args.run_stage, train_args=train_args),
            indent=2,
            ensure_ascii=False,
        ))
        return 0

    run_dir = train_q_agent.run_training(
        cfg,
        run_mode=f"baseline_{args.run_stage}",
        state_adapter_factory=_state_adapter_factory,
        model_factory=_model_factory,
    )
    manifest_path = run_dir / "logs" / "baseline_manifest.json"
    manifest = build_baseline_manifest(
        cfg=cfg,
        model=LocalStateQNetwork(),
        git_sha=_git_sha(Path(__file__).resolve().parents[2]),
    )
    manifest["source_entrypoint"] = "experiments/baselines/run_local_state_ddqn_train.py"
    _write_json(manifest_path, manifest)
    _try_append_artifact_index(run_dir, manifest_path)
    print(f"baseline_manifest_json: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
