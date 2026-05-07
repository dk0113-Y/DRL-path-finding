from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping


EXPECTED_ENV = {
    "PYTHONHASHSEED": "0",
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
}

EXPECTED_BACKEND_FLAGS = {
    "enable_tf32": False,
    "enable_cudnn_benchmark": False,
    "enable_amp": False,
    "enable_inference_amp": False,
    "enable_torch_compile": False,
    "enable_channels_last": False,
}

REQUIRED_SEED_POLICY_FIELDS = (
    "python_random_seed_policy",
    "numpy_seed_policy",
    "torch_seed_policy",
    "torch_cuda_seed_policy",
    "model_initialization_seed_policy",
    "replay_sampling_rng_policy",
    "epsilon_action_rng_policy",
    "train_episode_seed_policy",
    "final_probe_seed_policy",
    "posthoc_candidate_final_probe_seed_policy",
)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"failed to parse JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("top-level JSON payload must be an object")
    return payload


def _bool_check(errors: list[str], payload: Mapping[str, Any], key: str, expected: bool) -> None:
    value = payload.get(key)
    if value is not expected:
        errors.append(f"{key} must be {expected}, got {value!r}")


def _env_check(errors: list[str], env_block: Mapping[str, Any], key: str, expected: str) -> None:
    item = env_block.get(key)
    if not isinstance(item, Mapping):
        errors.append(f"determinism_environment_variables.{key} is missing")
        return
    if item.get("present") is not True:
        errors.append(f"determinism_environment_variables.{key}.present must be true")
    if str(item.get("value")) != expected:
        errors.append(f"determinism_environment_variables.{key}.value must be {expected!r}, got {item.get('value')!r}")


def _optional_readback_check(
    errors: list[str],
    readback: Mapping[str, Any],
    key: str,
    expected: bool,
) -> None:
    if key not in readback or readback.get(key) is None:
        return
    value = readback.get(key)
    if isinstance(value, Mapping) and "unavailable" in value:
        return
    if value is not expected:
        errors.append(f"backend_runtime_readback.{key} must be {expected}, got {value!r}")


def _mapping(payload: Mapping[str, Any], key: str, errors: list[str]) -> Mapping[str, Any]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        errors.append(f"{key} must be an object")
        return {}
    return value


def validate_contract(payload: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []

    _bool_check(errors, payload, "strict_reproducibility", True)
    _bool_check(errors, payload, "deterministic_algorithms_enabled", True)
    _bool_check(errors, payload, "deterministic_algorithms_warn_only", False)

    env_block = _mapping(payload, "determinism_environment_variables", errors)
    for key, expected in EXPECTED_ENV.items():
        _env_check(errors, env_block, key, expected)

    readback = _mapping(payload, "backend_runtime_readback", errors)
    _optional_readback_check(errors, readback, "torch.backends.cudnn.deterministic", True)
    _optional_readback_check(errors, readback, "torch.backends.cudnn.benchmark", False)
    _optional_readback_check(errors, readback, "torch.backends.cuda.matmul.allow_tf32", False)
    _optional_readback_check(errors, readback, "torch.backends.cudnn.allow_tf32", False)

    requested_flags = _mapping(payload, "backend_requested_flags", errors)
    for key, expected in EXPECTED_BACKEND_FLAGS.items():
        if requested_flags.get(key) is not expected:
            errors.append(f"backend_requested_flags.{key} must be {expected}, got {requested_flags.get(key)!r}")

    fixed_seed_fields = _mapping(payload, "fixed_seed_fields", errors)
    if fixed_seed_fields.get("use_fixed_train_episode_seeds") is not True:
        errors.append("fixed_seed_fields.use_fixed_train_episode_seeds must be true")
    if fixed_seed_fields.get("use_fixed_eval_seeds") is not True:
        errors.append("fixed_seed_fields.use_fixed_eval_seeds must be true")

    seed_policy = _mapping(payload, "seed_policy", errors)
    for field in REQUIRED_SEED_POLICY_FIELDS:
        if not seed_policy.get(field):
            errors.append(f"seed_policy.{field} is missing or empty")

    if not payload.get("raw_argv") and not payload.get("raw_argv_sanitized"):
        errors.append("raw_argv or raw_argv_sanitized must be present")

    verdict = payload.get("contract_verdict")
    if verdict != "strict_contract_ready":
        notes = payload.get("contract_notes")
        errors.append(f"contract_verdict must be 'strict_contract_ready', got {verdict!r}; notes={notes!r}")

    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate a formal_train reproducibility_contract.json for stable same-seed runs."
    )
    parser.add_argument("contract_path", type=Path, help="Path to logs/reproducibility_contract.json")
    args = parser.parse_args(argv)

    try:
        payload = _load_json(args.contract_path)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    errors = validate_contract(payload)
    if errors:
        print("Reproducibility contract check failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        notes = payload.get("contract_notes")
        if notes:
            print("contract_notes:", file=sys.stderr)
            for note in notes if isinstance(notes, list) else [notes]:
                print(f"- {note}", file=sys.stderr)
        return 1

    print(f"Reproducibility contract check passed: {args.contract_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
