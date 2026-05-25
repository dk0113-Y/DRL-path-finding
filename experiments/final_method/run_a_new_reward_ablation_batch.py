from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from env.advantage_state_builder import (
    ADVANTAGE_CANVAS_SCHEMA_FINAL_4CH_NO_FRONTIER_RASTER,
    advantage_canvas_channel_count_for_schema,
    advantage_canvas_channels_for_schema,
)


DEFAULT_RECORDS_ROOT = Path("experiment_records/final_method/A_new_reward_ablations")
DEFAULT_CHECKPOINT_STORE_ROOT = Path("checkpoint_store/final_method/A_new_reward_ablations")

COPIED_LOG_FILES = (
    "final_probe.csv",
    "final_probe_summary.json",
    "metric_snapshot.json",
    "config_snapshot.json",
    "reproducibility_contract.json",
    "posthoc_selection_summary.json",
    "formal_selection_manifest.json",
    "artifact_index.json",
    "training_summary.txt",
)


@dataclass(frozen=True)
class ANewRewardAblationSpec:
    selector: str
    method_id: str
    name: str
    reward_override: dict[str, float]


SPECS: tuple[ANewRewardAblationSpec, ...] = (
    ANewRewardAblationSpec(
        selector="R1",
        method_id="Anew_R1",
        name="no_step_penalty",
        reward_override={"reward_step_penalty": 0.0},
    ),
    ANewRewardAblationSpec(
        selector="R2",
        method_id="Anew_R2",
        name="no_revisit_penalty",
        reward_override={"reward_revisit_penalty": 0.0},
    ),
    ANewRewardAblationSpec(
        selector="R3",
        method_id="Anew_R3",
        name="no_turn_penalty",
        reward_override={"reward_turn_penalty_scale": 0.0},
    ),
    ANewRewardAblationSpec(
        selector="R4",
        method_id="Anew_R4",
        name="no_timeout_penalty",
        reward_override={"reward_timeout_penalty": 0.0},
    ),
    ANewRewardAblationSpec(
        selector="R5",
        method_id="Anew_R5",
        name="no_efficiency_penalties",
        reward_override={
            "reward_step_penalty": 0.0,
            "reward_revisit_penalty": 0.0,
            "reward_turn_penalty_scale": 0.0,
            "reward_timeout_penalty": 0.0,
        },
    ),
)


def _alias_map() -> dict[str, ANewRewardAblationSpec]:
    aliases: dict[str, ANewRewardAblationSpec] = {}
    for spec in SPECS:
        aliases[spec.selector.lower()] = spec
        aliases[spec.method_id.lower()] = spec
        aliases[f"{spec.method_id}_{spec.name}".lower()] = spec
    return aliases


def _normalize_specs(raw: str | None) -> list[ANewRewardAblationSpec]:
    raw_ids = raw or "R1,R2,R3,R4,R5"
    aliases = _alias_map()
    selected: list[ANewRewardAblationSpec] = []
    for item in [part.strip() for part in raw_ids.split(",") if part.strip()]:
        key = item.lower()
        if key not in aliases:
            available = ", ".join(spec.selector for spec in SPECS)
            raise ValueError(f"Unknown A_new reward ablation id {item!r}. Available: {available}")
        spec = aliases[key]
        if spec not in selected:
            selected.append(spec)
    if not selected:
        raise ValueError("No A_new reward ablations were selected.")
    return selected


def _command_text(command: list[str]) -> str:
    try:
        return subprocess.list2cmdline([str(item) for item in command])
    except Exception:
        return " ".join(str(item) for item in command)


def _reward_override_args(reward_override: Mapping[str, float]) -> list[str]:
    option_by_field = {
        "reward_step_penalty": "--reward-step-penalty",
        "reward_revisit_penalty": "--reward-revisit-penalty",
        "reward_turn_penalty_scale": "--reward-turn-penalty-scale",
        "reward_timeout_penalty": "--reward-timeout-penalty",
    }
    args: list[str] = []
    for field_name, value in reward_override.items():
        if field_name not in option_by_field:
            raise ValueError(f"Unsupported A_new reward override field: {field_name}")
        args.extend([option_by_field[field_name], str(float(value))])
    return args


def _run_name(spec: ANewRewardAblationSpec, run_stage: str) -> str:
    return f"{spec.method_id}_{spec.name}_{run_stage}"


def _build_command(
    *,
    spec: ANewRewardAblationSpec,
    run_stage: str,
    device: str,
    output_root: str,
) -> list[str]:
    return [
        sys.executable,
        "experiments/final_method/run_a_new_final_method.py",
        "--run-stage",
        run_stage,
        "--device",
        device,
        "--output-root",
        output_root,
        "--experiment-id",
        spec.method_id,
        "--method-id",
        spec.method_id,
        "--method-name",
        spec.name,
        "--run-name",
        _run_name(spec, run_stage),
        "--advantage-canvas-schema",
        ADVANTAGE_CANVAS_SCHEMA_FINAL_4CH_NO_FRONTIER_RASTER,
        "--",
        *_reward_override_args(spec.reward_override),
    ]


def _records_logs_dir(records_root: Path, spec: ANewRewardAblationSpec) -> Path:
    return records_root / spec.method_id / "logs"


def _checkpoint_target_path(checkpoint_store_root: Path, spec: ANewRewardAblationSpec) -> Path:
    return checkpoint_store_root / f"{spec.method_id}.pt"


def _dry_run_method_payload(
    *,
    spec: ANewRewardAblationSpec,
    run_stage: str,
    command: list[str],
    records_root: Path,
    checkpoint_store_root: Path,
    copy_checkpoints: bool,
) -> dict[str, object]:
    schema = ADVANTAGE_CANVAS_SCHEMA_FINAL_4CH_NO_FRONTIER_RASTER
    channels = advantage_canvas_channels_for_schema(schema)
    channel_count = advantage_canvas_channel_count_for_schema(schema)
    return {
        "method_family": "A_new_reward_ablation",
        "baseline_method": "A_new_final_4ch_no_frontier_raster",
        "selector": spec.selector,
        "method_id": spec.method_id,
        "experiment_id": spec.method_id,
        "run_name": _run_name(spec, run_stage),
        "advantage_canvas_schema": schema,
        "advantage_canvas_channels": list(channels),
        "advantage_canvas_channel_count": channel_count,
        "frontier_raster_used": False,
        "value_tree_enabled": True,
        "value_tree_unchanged": True,
        "model_class": "ExplorationQNetwork",
        "advantage_encoder.canvas_in_channels": channel_count,
        "reward_override": dict(spec.reward_override),
        "run_stage": run_stage,
        "source_entrypoint": "experiments/final_method/run_a_new_final_method.py",
        "train_entrypoint": "train_q_agent.py",
        "records_logs_dir": str(_records_logs_dir(records_root, spec)),
        "checkpoint_store_path": str(_checkpoint_target_path(checkpoint_store_root, spec)),
        "checkpoint_copying": "enabled" if copy_checkpoints else "disabled",
        "command": command,
        "command_text": _command_text(command),
    }


def _print_dry_run(
    *,
    specs: list[ANewRewardAblationSpec],
    args: argparse.Namespace,
) -> None:
    methods: list[dict[str, object]] = []
    for spec in specs:
        command = _build_command(
            spec=spec,
            run_stage=args.run_stage,
            device=args.device,
            output_root=args.output_root,
        )
        print(f"[A_new_R:dry-run] {spec.method_id}/{spec.name}")
        print(f"  command: {_command_text(command)}")
        methods.append(
            _dry_run_method_payload(
                spec=spec,
                run_stage=args.run_stage,
                command=command,
                records_root=Path(args.records_root),
                checkpoint_store_root=Path(args.checkpoint_store_root),
                copy_checkpoints=bool(args.copy_checkpoints),
            )
        )
    payload = {
        "dry_run": True,
        "method_family": "A_new_reward_ablation",
        "baseline_method": "A_new_final_4ch_no_frontier_raster",
        "run_stage": args.run_stage,
        "device": args.device,
        "selected_methods": [spec.method_id for spec in specs],
        "methods": methods,
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))


def _resolve_run_dir_from_output(output_lines: list[str], cwd: Path) -> Path:
    run_dir: Path | None = None
    for line in output_lines:
        stripped = line.strip()
        if stripped.startswith("run_dir:"):
            run_dir = Path(stripped.split(":", 1)[1].strip())
    if run_dir is None:
        raise RuntimeError("Could not determine run_dir from A_new reward ablation output.")
    if not run_dir.is_absolute():
        run_dir = cwd / run_dir
    return run_dir.resolve()


def _run_child_command(command: list[str], cwd: Path) -> Path:
    print(f"[A_new_R] running: {_command_text(command)}")
    process = subprocess.Popen(
        command,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    output_lines: list[str] = []
    assert process.stdout is not None
    for line in process.stdout:
        output_lines.append(line)
        print(line, end="")
    return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"A_new reward ablation command failed with exit code {return_code}: {_command_text(command)}")
    return _resolve_run_dir_from_output(output_lines, cwd)


def _copy_curated_logs(run_dir: Path, records_root: Path, spec: ANewRewardAblationSpec) -> tuple[Path, list[str], list[str]]:
    source_logs = run_dir / "logs"
    target_logs = _records_logs_dir(records_root, spec)
    target_logs.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    missing: list[str] = []
    for file_name in COPIED_LOG_FILES:
        source_path = source_logs / file_name
        if source_path.exists() and source_path.is_file():
            shutil.copy2(source_path, target_logs / file_name)
            copied.append(file_name)
        else:
            missing.append(file_name)
    return target_logs, copied, missing


def _copy_last_checkpoint(
    run_dir: Path,
    checkpoint_store_root: Path,
    spec: ANewRewardAblationSpec,
    *,
    copy_checkpoints: bool,
) -> tuple[Path | None, Path | None, bool, str | None]:
    source_path = run_dir / "checkpoints" / "last.pt"
    target_path = _checkpoint_target_path(checkpoint_store_root, spec)
    if not copy_checkpoints:
        return source_path, target_path, False, "disabled_by_user"
    if not source_path.exists():
        return source_path, target_path, False, "missing_last_checkpoint"
    checkpoint_store_root.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, target_path)
    return source_path, target_path, True, None


def _write_run_record(
    *,
    target_logs: Path,
    spec: ANewRewardAblationSpec,
    run_stage: str,
    run_dir: Path,
    copied: list[str],
    missing: list[str],
    checkpoint_source: Path | None,
    checkpoint_store_path: Path | None,
    checkpoint_copied: bool,
    checkpoint_copy_reason: str | None,
) -> Path:
    record = [
        "# A_new Reward Ablation Run Record",
        "",
        f"- method_id: {spec.method_id}",
        f"- selector: {spec.selector}",
        f"- name: {spec.name}",
        f"- baseline_method: A_new_final_4ch_no_frontier_raster",
        f"- run_stage: {run_stage}",
        f"- source run_dir: {run_dir}",
        f"- reward_override: {json.dumps(spec.reward_override, sort_keys=True)}",
        f"- copied artifact list: {', '.join(copied) if copied else 'none'}",
        f"- missing artifact list: {', '.join(missing) if missing else 'none'}",
        f"- checkpoint_source: {checkpoint_source if checkpoint_source is not None else 'none'}",
        f"- checkpoint_store_path: {checkpoint_store_path if checkpoint_store_path is not None else 'none'}",
        f"- checkpoint_copied: {str(checkpoint_copied).lower()}",
        f"- checkpoint_copy_reason: {checkpoint_copy_reason if checkpoint_copy_reason is not None else 'none'}",
        "",
        "## Method Contract",
        "",
        "- advantage_canvas_schema: final_4ch_no_frontier_raster",
        "- frontier_raster_used: false",
        "- value_tree_enabled: true",
        "- model_class: ExplorationQNetwork",
        "- advantage_encoder.canvas_in_channels: 4",
    ]
    record_path = target_logs.parent / "run_record.md"
    record_path.write_text("\n".join(record) + "\n", encoding="utf-8")
    return record_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Batch runner for A_new R1-R5 reward ablations")
    parser.add_argument("--reward-ablation-ids", type=str, default="R1,R2,R3,R4,R5")
    parser.add_argument("--run-stage", choices=("smoke", "pilot", "formal"), default="smoke")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output-root", type=str, default="outputs")
    parser.add_argument("--records-root", type=Path, default=DEFAULT_RECORDS_ROOT)
    parser.add_argument("--checkpoint-store-root", type=Path, default=DEFAULT_CHECKPOINT_STORE_ROOT)
    parser.add_argument("--copy-checkpoints", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--stop-on-failure", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args(argv)

    specs = _normalize_specs(args.reward_ablation_ids)

    if args.dry_run:
        _print_dry_run(specs=specs, args=args)
        return 0

    repo_root = Path.cwd()
    failures: list[str] = []
    for spec in specs:
        command = _build_command(
            spec=spec,
            run_stage=args.run_stage,
            device=args.device,
            output_root=args.output_root,
        )
        print(f"[A_new_R] command: {_command_text(command)}")
        try:
            run_dir = _run_child_command(command, repo_root)
            target_logs, copied, missing = _copy_curated_logs(run_dir, args.records_root, spec)
            checkpoint_source, checkpoint_store_path, checkpoint_copied, checkpoint_copy_reason = _copy_last_checkpoint(
                run_dir,
                args.checkpoint_store_root,
                spec,
                copy_checkpoints=bool(args.copy_checkpoints),
            )
            record_path = _write_run_record(
                target_logs=target_logs,
                spec=spec,
                run_stage=args.run_stage,
                run_dir=run_dir,
                copied=copied,
                missing=missing,
                checkpoint_source=checkpoint_source,
                checkpoint_store_path=checkpoint_store_path,
                checkpoint_copied=checkpoint_copied,
                checkpoint_copy_reason=checkpoint_copy_reason,
            )
            print(f"[A_new_R] archived {spec.method_id} to {target_logs}")
            print(f"[A_new_R] run_record: {record_path}")
        except Exception as exc:
            message = f"{spec.method_id}: {exc}"
            failures.append(message)
            print(f"[A_new_R] failure: {message}")
            if bool(args.stop_on_failure):
                raise

    if failures:
        print("[A_new_R] completed with failures:")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
