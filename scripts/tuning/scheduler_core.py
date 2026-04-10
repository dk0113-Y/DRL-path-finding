from __future__ import annotations

import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from .recipes import TrialSpec
from .result_reader import RunResult, find_latest_run_dir, read_run_result


@dataclass
class TrialExecution:
    trial_id: str
    trial_index: int
    trial_spec: TrialSpec
    command: list[str]
    started_at: str
    ended_at: str | None
    process_log_path: Path
    return_code: int | None
    run_dir: Path | None
    status: str
    status_reason: str
    result: RunResult | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "trial_id": self.trial_id,
            "trial_index": self.trial_index,
            "trial_spec": self.trial_spec.to_dict(),
            "command": self.command,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "process_log_path": str(self.process_log_path),
            "return_code": self.return_code,
            "run_dir": str(self.run_dir) if self.run_dir else None,
            "status": self.status,
            "status_reason": self.status_reason,
            "result": self.result.to_dict() if self.result else None,
        }


class SessionLogger:
    def __init__(self, log_path: Path) -> None:
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, message: str, *, echo: bool = True) -> None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{timestamp}] {message}"
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
        if echo:
            print(line, flush=True)


def _format_cli_float(value: float) -> str:
    return f"{value:.4f}".rstrip("0").rstrip(".")


def build_train_command(
    *,
    python_executable: str,
    train_script: str,
    output_root: Path,
    device: str,
    seed: int,
    total_env_steps: int,
    entry_cap: int,
    generate_plots_on_finish: bool,
    trial_spec: TrialSpec,
    run_name: str,
) -> list[str]:
    command = [
        python_executable,
        train_script,
        "--device",
        device,
        "--seed",
        str(seed),
        "--total-env-steps",
        str(total_env_steps),
        "--reward-turn-penalty-scale",
        _format_cli_float(trial_spec.turn_penalty_scale),
        "--reward-revisit-penalty",
        _format_cli_float(trial_spec.revisit_penalty),
        "--max-entries-per-block",
        str(entry_cap),
        "--run-name",
        run_name,
        "--output-root",
        str(output_root),
    ]
    command.append(
        "--generate-plots-on-finish" if generate_plots_on_finish else "--no-generate-plots-on-finish"
    )
    return command


def _stream_process_output(
    process: subprocess.Popen[str],
    session_logger: SessionLogger,
    process_log_path: Path,
    trial_id: str,
) -> None:
    process_log_path.parent.mkdir(parents=True, exist_ok=True)
    with process_log_path.open("w", encoding="utf-8") as process_log:
        if process.stdout is None:
            return
        for line in process.stdout:
            process_log.write(line)
            process_log.flush()
            print(line, end="", flush=True)
            session_logger.write(f"[{trial_id}] {line.rstrip()}", echo=False)


def execute_trial(
    *,
    repo_root: Path,
    session_logger: SessionLogger,
    session_dir: Path,
    python_executable: str,
    train_script: str,
    output_root: Path,
    device: str,
    seed: int,
    total_env_steps: int,
    entry_cap: int,
    generate_plots_on_finish: bool,
    trial_spec: TrialSpec,
    trial_index: int,
    dry_run: bool,
) -> TrialExecution:
    trial_id = f"trial_{trial_index:02d}"
    run_name = trial_spec.run_name(entry_cap)
    command = build_train_command(
        python_executable=python_executable,
        train_script=train_script,
        output_root=output_root,
        device=device,
        seed=seed,
        total_env_steps=total_env_steps,
        entry_cap=entry_cap,
        generate_plots_on_finish=generate_plots_on_finish,
        trial_spec=trial_spec,
        run_name=run_name,
    )
    process_log_path = session_dir / f"{trial_id}_process.log"
    started_at = datetime.now()
    command_text = subprocess.list2cmdline(command)

    session_logger.write(f"Starting {trial_id}: {trial_spec.note}")
    session_logger.write(f"{trial_id} command: {command_text}")

    if dry_run:
        return TrialExecution(
            trial_id=trial_id,
            trial_index=trial_index,
            trial_spec=trial_spec,
            command=command,
            started_at=started_at.isoformat(timespec="seconds"),
            ended_at=started_at.isoformat(timespec="seconds"),
            process_log_path=process_log_path,
            return_code=None,
            run_dir=None,
            status="dry_run",
            status_reason="dry-run mode: command not executed",
            result=None,
        )

    started_after_epoch = time.time()
    process = subprocess.Popen(
        command,
        cwd=repo_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    _stream_process_output(process, session_logger, process_log_path, trial_id)
    return_code = process.wait()
    ended_at = datetime.now()

    run_dir: Path | None = None
    result: RunResult | None = None
    status = "failed"
    status_reason = "run directory not found"
    try:
        run_dir = find_latest_run_dir(output_root, run_name, started_after_epoch)
        result = read_run_result(run_dir, return_code)
        status = result.status
        status_reason = result.status_reason
    except FileNotFoundError as exc:
        status_reason = str(exc)

    session_logger.write(
        f"Finished {trial_id}: return_code={return_code}, status={status}, reason={status_reason}"
    )

    return TrialExecution(
        trial_id=trial_id,
        trial_index=trial_index,
        trial_spec=trial_spec,
        command=command,
        started_at=started_at.isoformat(timespec="seconds"),
        ended_at=ended_at.isoformat(timespec="seconds"),
        process_log_path=process_log_path,
        return_code=return_code,
        run_dir=run_dir,
        status=status,
        status_reason=status_reason,
        result=result,
    )
