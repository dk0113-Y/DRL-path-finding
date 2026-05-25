from __future__ import annotations

import argparse
import copy
import os
import random
import sys
import time
from collections import deque
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from agents.q_value_agent import ExplorationQConfig, ExplorationQNetwork, StateAdapterConfig, StateTensorAdapter
from encoders.advantage_encoder import AdvantageEncoderConfig
from env.advantage_state_builder import (
    ADVANTAGE_CANVAS_SCHEMA_FINAL_4CH_NO_FRONTIER_RASTER,
    ADVANTAGE_CANVAS_SCHEMAS,
    FINAL_4CH_ADVANTAGE_CANVAS_CHANNELS,
    AdvantageStateConfig,
    advantage_canvas_channels_for_schema,
    advantage_canvas_uses_frontier_raster,
    normalize_advantage_canvas_schema,
    normalize_zeroed_advantage_channels,
)
from env.shared_semantic_layer import SharedSemanticConfig
from env.value_state_builder import (
    VALUE_BLOCK_FEATURE_COUNT,
    VALUE_ENTRY_FEATURE_COUNT,
    VALUE_REPLACEMENT_STRATEGIES,
    VALUE_REPLACEMENT_STRATEGY_NONE,
    VALUE_REPLACEMENT_STRATEGY_ZERO_VALUE_STATE,
    ValueStateConfig,
    normalize_value_replacement_strategy,
)
from training.checkpointing import CheckpointManager
from training.collector import (
    CollectorConfig,
    DERIVED_TRAIN_DIAGNOSTIC_FIELDS,
    SEMANTIC_EPISODE_FIELDS,
    TransitionCollector,
)
from training.evaluator import GreedyEvaluator
from training.formal_artifacts import write_formal_run_artifacts
from training.learner import DDQNLearner, DDQNLearnerConfig
from training.logger import CSVMetricLogger
from training.plotting import generate_all_plots
from training.posthoc_selection import (
    PROTOCOL_NAME as POSTHOC_PROTOCOL_NAME,
    final_probe_rank_key,
    select_posthoc_candidates,
    write_posthoc_final_artifacts,
)
from training.replay_buffer import ReplayBuffer, ReplayBufferConfig
from training.rewarding import REWARD_BREAKDOWN_FIELDS, REWARD_EVENT_SUMMARY_FIELDS
from training.trajectory_plotting import save_episode_trajectory_plots, save_train_special_trajectory_plots

EVAL_SEMANTIC_METRIC_NAMES = SEMANTIC_EPISODE_FIELDS


@dataclass(frozen=True)
class TrainConfig:
    seed: int = 0
    device: str = "cuda"
    # Performance-related side-overhead toggles only; they do not change the algorithm or metric definitions.
    enable_amp: bool = False  # AMP/autocast is a throughput toggle only; training semantics and metrics stay the same.
    enable_inference_amp: bool = False  # Greedy policy AMP toggle only; default stays off so baseline semantics remain unchanged.
    amp_dtype: str = "fp16"  # AMP dtype choice is performance-only; it does not redefine the algorithm or metrics.
    enable_torch_compile: bool = False  # torch.compile is a runtime wrapper only; algorithm behavior/metrics are unchanged.
    compile_mode: str = "default"  # torch.compile mode tunes runtime behavior only; it does not change the RL objective.
    enable_cudnn_benchmark: bool = True  # cuDNN autotune affects kernel selection only; it does not change metrics.
    enable_tf32: bool = True  # TF32 backend toggle is a runtime perf setting only; it is not an algorithm switch.
    strict_reproducibility: bool = False  # Optional runtime-determinism mode; kept non-default because it can reduce throughput.
    deterministic_warn_only: bool = True  # Strict-mode deterministic guard warns by default unless explicitly disabled.
    enable_channels_last: bool = False  # Tensor-layout toggle only; model math and metrics stay unchanged.
    episode_print_interval: int = 10  # Stdout throttling only; CSV episode logging remains unchanged.
    train_print_interval: int = 2_000  # Stdout throttling only; separated from CSV logging and algorithm behavior.
    save_train_representative_trajectories: bool = False  # Train-failure trajectory dumping is optional wall-clock overhead.
    save_train_special_trajectories: bool = False  # Optional train-side special-case trajectory export for post-run analysis.
    save_final_probe_trajectories: bool = False  # Final probe plotting is optional wall-clock overhead only.
    generate_plots_on_finish: bool = False  # End-of-run plotting is optional wall-clock overhead only.
    enable_collector_timing: bool = False  # Profiling only; collector timing does not change rollout/reward semantics.
    enable_learner_timing: bool = False  # Profiling only; learner timing does not change DDQN updates or metrics.
    enable_replay_timing: bool = False  # Profiling only; replay timing does not change storage or sampling semantics.
    enable_state_adapter_timing: bool = False  # Profiling only; adapter timing does not change state tensor definitions.
    enable_cummap_timing: bool = False  # Profiling only; cumulative-map timing does not change map/frontier math.
    enable_shared_semantic_timing: bool = False  # Profiling only; semantic parsing timing does not change state semantics.
    enable_advantage_state_timing: bool = False  # Profiling only; local decision canvas timing does not change channels.
    enable_value_state_timing: bool = False  # Profiling only; block-tree tensor timing does not change value semantics.
    timing_log_interval: int = 2000  # Stdout profiling cadence only; it does not affect training behavior.
    debug_check_incremental_frontier: bool = False  # Debug-only full-recompute compare; default stays off in normal rollout.
    prefer_batch_replay_add: bool = True  # Replay write-path optimization only; transition semantics stay unchanged.
    learner_debug_stats_every: int = 8  # Metric-sync throttling only; learner updates stay unchanged.
    rows: int = 40
    cols: int = 60
    obs_size: int = 6
    scan_radius: int = 10  # radar sensor radius only
    trajectory_history_steps: int = 10  # shared short horizon for the trajectory branch and recent revisit penalty
    obstacle_ratio: float = 0.20

    max_accessible_blocks: int = 16
    max_entries_per_block: int = 8

    budget_mode: str = "env_steps"
    total_env_steps: int = 500_000
    total_train_episodes: int = 600
    warmup_steps: int = 4_000
    warmup_episodes: int = 0
    collect_steps_per_iter: int = 16
    learner_updates_per_iter: int = 1
    train_every_env_steps: int = 16
    log_interval: int = 500
    log_interval_episodes: int = 10

    recent_episode_window: int = 100
    formal_protocol: str = POSTHOC_PROTOCOL_NAME
    train_side_only_tuning: bool = True
    final_greedy_episodes: int = 100
    train_print_interval_episodes: int = 20
    use_fixed_train_episode_seeds: bool = True
    fixed_train_episode_seed_base: int = 20259323
    use_fixed_eval_seeds: bool = True  # Legacy name retained; now this toggle only controls fixed final_probe seeds.
    fixed_final_probe_seed_base: int = 20261323
    periodic_checkpoint_interval_env_steps: int = 20_000
    posthoc_candidate_start_env_steps: int = 200_000
    posthoc_candidate_end_env_steps: int = 0  # 0 means "use total_env_steps".
    posthoc_selection_window_env_steps: int = 40_000
    posthoc_final_probe_topk: int = 3
    enable_best_checkpoint_selection: bool = False
    best_checkpoint_selection_start_env_steps: int = 300_000
    best_checkpoint_selection_interval_env_steps: int = 20_000
    best_checkpoint_validation_episodes: int = 24
    best_checkpoint_topk_recheck: int = 3
    best_checkpoint_recheck_episodes: int = 50
    use_fixed_model_select_seeds: bool = True
    fixed_model_select_seed_base: int = 20262323

    replay_capacity: int = 100_000
    batch_size: int = 128
    min_replay_size: int = 8_000

    gamma: float = 0.99
    n_step: int = 3

    learning_rate: float = 1.0e-4
    weight_decay: float = 0.0
    grad_clip_norm: float = 10.0
    target_update_interval: int = 1_000

    epsilon_start: float = 1.0
    epsilon_end: float = 0.04
    epsilon_decay_steps: int = 240_000

    max_episode_steps: int = 600  # tune with map scale as needed
    coverage_stop_threshold: float = 0.95

    reward_info_scale: float = 3.1
    reward_obstacle_weight: float = 0.20
    reward_step_penalty: float = 0.02
    reward_terminal_bonus: float = 20.0
    # Candidate formal defaults align A_new training/reward settings to the matched legacy A/F1 contract.
    reward_revisit_penalty: float = 0.10
    reward_turn_penalty_scale: float = 0.05  # total turn penalty scale; angle-specific weights are configured below.
    reward_turn_weight_45: float = 0.0
    reward_turn_weight_90: float = 1.0 / 3.0
    reward_turn_weight_135: float = 2.0 / 3.0
    reward_turn_weight_180: float = 1.0
    reward_timeout_penalty: float = 8.0

    ablation_group: str = "none"
    ablation_id: str = "none"
    experiment_id: str = "A_new"
    method_id: str = "A_new"
    method_name: str = "final_4ch_no_frontier_raster"
    ablation_name: str = "none"
    channel_ablation: str = "none"
    zeroed_advantage_channels: tuple[str, ...] = ()
    reward_override: dict[str, float] = field(default_factory=dict)
    value_replacement_strategy: str = "none"
    value_tree_enabled: bool = True
    advantage_canvas_schema: str = ADVANTAGE_CANVAS_SCHEMA_FINAL_4CH_NO_FRONTIER_RASTER
    advantage_canvas_channels: tuple[str, ...] = FINAL_4CH_ADVANTAGE_CANVAS_CHANNELS
    advantage_canvas_channel_count: int = 4
    frontier_raster_used: bool = False
    value_tree_unchanged: bool = True
    value_branch_source: str = "SharedSemanticSnapshot"
    value_branch_representation: str = "structured_frontier_block_value_tree"
    baseline_id: str = "none"
    baseline_group: str = "none"
    baseline_name: str = "none"
    baseline_type: str = "none"
    is_ablation: bool = False
    no_shared_semantic_dual_state: bool = False
    no_value_tree: bool = False
    no_frontier_cluster_input: bool = False
    no_accessible_unknown_block_input: bool = False
    no_ground_truth_map_for_decision: bool = False
    local_state_channels: tuple[str, ...] = ()
    local_state_patch_size: int = 0
    local_state_carrier_key: str = "none"
    local_state_canvas_role: str = "full_method_advantage_canvas"
    model_class: str = "ExplorationQNetwork"
    model_parameter_count: int = 0
    dummy_value_tensors_for_interface: bool = False
    value_tensors_used_by_model: bool = True
    dummy_value_block_shape: tuple[int, ...] = ()
    dummy_value_entry_shape: tuple[int, ...] = ()
    dummy_value_mask_rule: str = "none"
    run_stage: str = "formal"

    special_highcov_timeout_min_coverage: float = 0.85
    special_highcov_timeout_max_plots: int = 5
    special_long_success_gate_coverage: float = 0.80
    special_long_success_gate_window: int = 100
    special_long_success_min_length: int = 350
    special_long_success_percentile: float = 85.0
    special_long_success_max_plots: int = 5
    special_lowcov_gate_coverage: float = 0.80
    special_lowcov_gate_window: int = 100
    special_lowcov_absolute_threshold: float = 0.75
    special_lowcov_local_drop_margin: float = 0.12
    special_lowcov_max_plots: int = 5

    output_root: str = "outputs"
    run_name: str = "ddqn_explore_vscode_stage5"

    def __post_init__(self) -> None:
        schema = normalize_advantage_canvas_schema(self.advantage_canvas_schema)
        schema_channels = tuple(advantage_canvas_channels_for_schema(schema))
        configured_channels = tuple(str(channel) for channel in (self.advantage_canvas_channels or ()))
        if configured_channels and configured_channels != schema_channels:
            raise ValueError(
                "A_new main only supports the final 4-channel advantage canvas: "
                f"{schema_channels}; got {configured_channels}"
            )
        channels = schema_channels
        zeroed_advantage_channels = normalize_zeroed_advantage_channels(
            self.zeroed_advantage_channels,
            schema=schema,
        )

        object.__setattr__(self, "advantage_canvas_schema", schema)
        object.__setattr__(self, "advantage_canvas_channels", channels)
        object.__setattr__(self, "advantage_canvas_channel_count", int(len(channels)))
        object.__setattr__(self, "channel_ablation", str(self.channel_ablation or "none"))
        object.__setattr__(self, "zeroed_advantage_channels", zeroed_advantage_channels)
        if zeroed_advantage_channels:
            object.__setattr__(self, "is_ablation", True)
        object.__setattr__(
            self,
            "frontier_raster_used",
            bool(advantage_canvas_uses_frontier_raster(schema)),
        )

        value_replacement_strategy = normalize_value_replacement_strategy(self.value_replacement_strategy)
        no_value_tree = bool(self.no_value_tree) or (
            value_replacement_strategy == VALUE_REPLACEMENT_STRATEGY_ZERO_VALUE_STATE
        )
        object.__setattr__(self, "value_replacement_strategy", value_replacement_strategy)
        if no_value_tree:
            object.__setattr__(self, "no_value_tree", True)
            object.__setattr__(self, "value_tree_enabled", False)
            object.__setattr__(self, "value_tree_unchanged", False)
            object.__setattr__(self, "is_ablation", True)
            object.__setattr__(self, "dummy_value_tensors_for_interface", True)
            object.__setattr__(self, "value_tensors_used_by_model", True)
            object.__setattr__(self, "dummy_value_mask_rule", "all_false")
            object.__setattr__(
                self,
                "dummy_value_block_shape",
                (int(self.max_accessible_blocks), int(VALUE_BLOCK_FEATURE_COUNT)),
            )
            object.__setattr__(
                self,
                "dummy_value_entry_shape",
                (
                    int(self.max_accessible_blocks),
                    int(self.max_entries_per_block),
                    int(VALUE_ENTRY_FEATURE_COUNT),
                ),
            )
            object.__setattr__(self, "value_branch_source", "zero_value_state")
            object.__setattr__(self, "value_branch_representation", "zero_value_state")

        experiment_id = str(self.experiment_id or "A_new")
        method_id = str(self.method_id or experiment_id)
        method_name = str(self.method_name or schema)
        object.__setattr__(self, "experiment_id", experiment_id)
        object.__setattr__(self, "method_id", method_id)
        object.__setattr__(self, "method_name", method_name)


def linear_epsilon(step: int, cfg: TrainConfig) -> float:
    s = max(0, int(step))
    if s >= int(cfg.epsilon_decay_steps):
        return float(cfg.epsilon_end)
    ratio = float(s) / float(max(1, int(cfg.epsilon_decay_steps)))
    return float(cfg.epsilon_start + ratio * (cfg.epsilon_end - cfg.epsilon_start))


def resolve_budget_mode(cfg: TrainConfig) -> str:
    mode = str(cfg.budget_mode).strip().lower()
    if mode not in {"env_steps", "episodes"}:
        raise ValueError(f"Unsupported budget_mode: {cfg.budget_mode!r}; expected 'env_steps' or 'episodes'")
    return mode


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _ensure_strict_cublas_workspace_config(cfg: TrainConfig) -> dict[str, object]:
    existing = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if not bool(cfg.strict_reproducibility):
        return {
            "name": "CUBLAS_WORKSPACE_CONFIG",
            "value": existing,
            "preexisting": existing is not None,
            "set_by_script": False,
            "status": "not_required_when_strict_reproducibility_false",
        }
    if existing:
        return {
            "name": "CUBLAS_WORKSPACE_CONFIG",
            "value": existing,
            "preexisting": True,
            "set_by_script": False,
            "status": "preexisting",
        }
    value = ":4096:8"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = value
    return {
        "name": "CUBLAS_WORKSPACE_CONFIG",
        "value": value,
        "preexisting": False,
        "set_by_script": True,
        "status": "set_by_script_before_configure_torch_runtime_cuda_checks",
        "limitation": "torch is already imported; exact timing relative to lower-level CUDA/CUBLAS initialization is not guaranteed.",
    }


def _safe_backend_value(getter) -> object:
    try:
        return getter()
    except Exception as exc:
        return {"unavailable": type(exc).__name__}


def _collect_backend_readback(cfg: TrainConfig, cublas_workspace_action: Mapping[str, object]) -> dict[str, object]:
    cuda_backend = getattr(torch.backends, "cuda", None)
    cuda_matmul = getattr(cuda_backend, "matmul", None)
    cudnn_backend = getattr(torch.backends, "cudnn", None)
    return {
        "requested_device": str(cfg.device),
        "strict_reproducibility": bool(cfg.strict_reproducibility),
        "deterministic_algorithms_warn_only": bool(cfg.deterministic_warn_only),
        "cublas_workspace_config_action": dict(cublas_workspace_action),
        "torch.backends.cudnn.deterministic": _safe_backend_value(
            lambda: bool(torch.backends.cudnn.deterministic)
        ) if cudnn_backend is not None and hasattr(cudnn_backend, "deterministic") else None,
        "torch.backends.cudnn.benchmark": _safe_backend_value(
            lambda: bool(torch.backends.cudnn.benchmark)
        ) if cudnn_backend is not None and hasattr(cudnn_backend, "benchmark") else None,
        "torch.backends.cuda.matmul.allow_tf32": _safe_backend_value(
            lambda: bool(torch.backends.cuda.matmul.allow_tf32)
        ) if cuda_matmul is not None and hasattr(cuda_matmul, "allow_tf32") else None,
        "torch.backends.cudnn.allow_tf32": _safe_backend_value(
            lambda: bool(torch.backends.cudnn.allow_tf32)
        ) if cudnn_backend is not None and hasattr(cudnn_backend, "allow_tf32") else None,
        "torch.are_deterministic_algorithms_enabled()": _safe_backend_value(
            lambda: bool(torch.are_deterministic_algorithms_enabled())
        ) if hasattr(torch, "are_deterministic_algorithms_enabled") else None,
    }


def configure_torch_runtime(cfg: TrainConfig) -> dict[str, object]:
    """Apply performance-only backend toggles without changing training/eval semantics."""
    cublas_workspace_action = _ensure_strict_cublas_workspace_config(cfg)
    strict_mode = bool(cfg.strict_reproducibility)
    if hasattr(torch, "use_deterministic_algorithms"):
        torch.use_deterministic_algorithms(strict_mode, warn_only=bool(cfg.deterministic_warn_only))
    if hasattr(torch.backends, "cudnn") and hasattr(torch.backends.cudnn, "deterministic"):
        torch.backends.cudnn.deterministic = strict_mode

    if not str(cfg.device).lower().startswith("cuda") or not torch.cuda.is_available():
        return _collect_backend_readback(cfg, cublas_workspace_action)

    torch.backends.cudnn.benchmark = False if strict_mode else bool(cfg.enable_cudnn_benchmark)

    if (
        hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul")
        and hasattr(torch.backends.cuda.matmul, "allow_tf32")
    ):
        torch.backends.cuda.matmul.allow_tf32 = False if strict_mode else bool(cfg.enable_tf32)

    if hasattr(torch.backends.cudnn, "allow_tf32"):
        torch.backends.cudnn.allow_tf32 = False if strict_mode else bool(cfg.enable_tf32)

    return _collect_backend_readback(cfg, cublas_workspace_action)


def collect_reproducibility_runtime_info(
    cfg: TrainConfig,
    *,
    backend_readback: Mapping[str, object],
    online_net: torch.nn.Module | None = None,
) -> dict[str, object]:
    actual_device = None
    if online_net is not None:
        param = next(online_net.parameters(), None)
        if param is not None:
            actual_device = str(param.device)
    cudnn_version = None
    if hasattr(torch.backends, "cudnn") and hasattr(torch.backends.cudnn, "version"):
        cudnn_version = _safe_backend_value(lambda: torch.backends.cudnn.version())
    return {
        "requested_device": str(cfg.device),
        "actual_device": actual_device,
        "python_executable_basename": Path(sys.executable).name,
        "torch_version": str(torch.__version__),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": getattr(torch.version, "cuda", None),
        "cudnn_version": cudnn_version,
        "backend_readback": dict(backend_readback),
    }


class _CompileFallbackWrapper(torch.nn.Module):
    def __init__(self, raw_module: torch.nn.Module, compiled_module: torch.nn.Module):
        super().__init__()
        self.raw_module = raw_module
        self._compiled_module: torch.nn.Module | None = compiled_module
        self._reported_fallback = False

    def forward(self, *args, **kwargs):
        compiled_module = self._compiled_module
        if compiled_module is not None:
            try:
                return compiled_module(*args, **kwargs)
            except Exception as exc:
                self._compiled_module = None
                if not self._reported_fallback:
                    print(
                        "[startup] "
                        f"torch.compile forward failed ({type(exc).__name__}: {exc}); "
                        "falling back to eager."
                    )
                    self._reported_fallback = True
        return self.raw_module(*args, **kwargs)

    def train(self, mode: bool = True):
        super().train(mode)
        compiled_module = self._compiled_module
        if compiled_module is not None and compiled_module is not self.raw_module:
            compiled_module.train(mode)
        return self

    def state_dict(self, *args, **kwargs):
        return self.raw_module.state_dict(*args, **kwargs)

    def load_state_dict(self, state_dict, *args, **kwargs):
        return self.raw_module.load_state_dict(state_dict, *args, **kwargs)


def _compile_online_net(raw_online_net: torch.nn.Module, cfg: TrainConfig) -> torch.nn.Module:
    """Compile the online net while keeping raw-module state IO for sync/checkpoint compatibility."""
    if not bool(cfg.enable_torch_compile):
        return raw_online_net
    if not hasattr(torch, "compile"):
        print("[startup] torch.compile is unavailable in this torch build; using eager.")
        return raw_online_net

    compile_kwargs = {}
    compile_mode = str(cfg.compile_mode).strip()
    if compile_mode != "":
        compile_kwargs["mode"] = compile_mode
    try:
        compiled_net = torch.compile(raw_online_net, **compile_kwargs)
    except Exception as exc:
        print(
            "[startup] "
            f"torch.compile setup failed ({type(exc).__name__}: {exc}); using eager."
        )
        return raw_online_net

    # Keep target sync/checkpoint state_dict handling on the raw module; compile is a perf wrapper only.
    return _CompileFallbackWrapper(raw_online_net, compiled_net)


def _maybe_to_channels_last(module: torch.nn.Module, cfg: TrainConfig) -> torch.nn.Module:
    if not bool(cfg.enable_channels_last):
        return module
    device_text = str(cfg.device).lower()
    if not device_text.startswith("cuda"):
        return module
    return module.to(memory_format=torch.channels_last)


def create_run_dir(cfg: TrainConfig) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(cfg.output_root) / f"{cfg.run_name}_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def summarize_recent_episodes(recent: deque[dict]) -> dict:
    if len(recent) <= 0:
        out = {
            "mean_reward": float("nan"),
            "mean_coverage": float("nan"),
            "success_rate": float("nan"),
            "mean_length": float("nan"),
            "mean_repeat_visit_ratio": float("nan"),
        }
        for field in SEMANTIC_EPISODE_FIELDS:
            out[field] = float("nan")
        for field in REWARD_BREAKDOWN_FIELDS:
            out[field] = float("nan")
        for field in REWARD_EVENT_SUMMARY_FIELDS:
            out[field] = float("nan")
        for field in DERIVED_TRAIN_DIAGNOSTIC_FIELDS:
            out[field] = float("nan")
        return out

    rewards = np.asarray([float(ep["episode_reward"]) for ep in recent], dtype=np.float32)
    coverages = np.asarray([float(ep["final_coverage"]) for ep in recent], dtype=np.float32)
    successes = np.asarray([float(ep["success"]) for ep in recent], dtype=np.float32)
    lengths = np.asarray([float(ep["episode_length"]) for ep in recent], dtype=np.float32)
    repeats = np.asarray([float(ep["repeat_visit_ratio"]) for ep in recent], dtype=np.float32)

    out = {
        "mean_reward": float(np.mean(rewards)),
        "mean_coverage": float(np.mean(coverages)),
        "success_rate": float(np.mean(successes)),
        "mean_length": float(np.mean(lengths)),
        "mean_repeat_visit_ratio": float(np.mean(repeats)),
    }
    for field in SEMANTIC_EPISODE_FIELDS:
        values = np.asarray([float(ep.get(field, float("nan"))) for ep in recent], dtype=np.float32)
        out[field] = float(np.nanmean(values)) if np.any(np.isfinite(values)) else float("nan")
    for field in REWARD_BREAKDOWN_FIELDS:
        values = np.asarray([float(ep.get(field, float("nan"))) for ep in recent], dtype=np.float32)
        out[field] = float(np.nanmean(values)) if np.any(np.isfinite(values)) else float("nan")
    for field in REWARD_EVENT_SUMMARY_FIELDS:
        values = np.asarray([float(ep.get(field, float("nan"))) for ep in recent], dtype=np.float32)
        out[field] = float(np.nanmean(values)) if np.any(np.isfinite(values)) else float("nan")
    for field in DERIVED_TRAIN_DIAGNOSTIC_FIELDS:
        values = np.asarray([float(ep.get(field, float("nan"))) for ep in recent], dtype=np.float32)
        out[field] = float(np.nanmean(values)) if np.any(np.isfinite(values)) else float("nan")
    return out


def _timing_summary_enabled(cfg: TrainConfig) -> bool:
    return any(
        (
            bool(cfg.enable_collector_timing),
            bool(cfg.enable_learner_timing),
            bool(cfg.enable_replay_timing),
            bool(cfg.enable_state_adapter_timing),
            bool(cfg.enable_cummap_timing),
            bool(cfg.enable_shared_semantic_timing),
            bool(cfg.enable_advantage_state_timing),
            bool(cfg.enable_value_state_timing),
        )
    )


def _timing_flag_dict(cfg: TrainConfig) -> dict[str, bool]:
    return {
        "enable_collector_timing": bool(cfg.enable_collector_timing),
        "enable_learner_timing": bool(cfg.enable_learner_timing),
        "enable_replay_timing": bool(cfg.enable_replay_timing),
        "enable_state_adapter_timing": bool(cfg.enable_state_adapter_timing),
        "enable_cummap_timing": bool(cfg.enable_cummap_timing),
        "enable_shared_semantic_timing": bool(cfg.enable_shared_semantic_timing),
        "enable_advantage_state_timing": bool(cfg.enable_advantage_state_timing),
        "enable_value_state_timing": bool(cfg.enable_value_state_timing),
    }


def _describe_profiling_mode(cfg: TrainConfig, run_mode: str) -> str:
    profiling_enabled = _timing_summary_enabled(cfg)
    mode = str(run_mode).strip().lower()
    if mode == "vscode":
        return (
            "regular run, timing disabled by preset"
            if not profiling_enabled else "regular run, timing enabled by config"
        )
    if mode == "profile":
        return (
            "profiling run, timing enabled by preset"
            if profiling_enabled else "profiling run selected, but timing flags are disabled"
        )
    if mode == "profile_compile":
        return (
            "profiling+compile run, timing enabled by preset"
            if profiling_enabled else "profiling+compile run selected, but timing flags are disabled"
        )
    if mode == "fast_cuda":
        return (
            "fast cuda run with amp/compile enabled"
            if not profiling_enabled else "fast cuda run with amp/compile and timing enabled"
        )
    if mode == "fast_cuda_profile":
        return (
            "fast cuda profiling run with amp/compile enabled"
            if profiling_enabled else "fast cuda profiling preset selected, but timing flags are disabled"
        )
    if mode == "cli":
        return (
            "cli run, timing enabled via flags"
            if profiling_enabled else "cli run, timing disabled unless --profile or timing flags are set"
        )
    if mode == "smoke":
        return ("smoke run with timing enabled" if profiling_enabled else "smoke run with timing disabled")
    return ("timing enabled for this run" if profiling_enabled else "timing disabled for this run")


def _print_startup_summary(cfg: TrainConfig, run_mode: str) -> None:
    budget_mode = resolve_budget_mode(cfg)
    timing_flags = _timing_flag_dict(cfg)
    profiling_enabled = _timing_summary_enabled(cfg)
    timing_flag_text = " ".join(f"{key}={value}" for key, value in timing_flags.items())
    print(
        "[startup] "
        f"file={Path(__file__).resolve()} "
        f"run_mode={run_mode} "
        f"device={cfg.device} "
        f"profiling_enabled={profiling_enabled} "
        f"budget_mode={budget_mode} "
        f"experiment_id={cfg.experiment_id} "
        f"method_id={cfg.method_id} "
        f"method_name={cfg.method_name} "
        f"advantage_canvas_schema={cfg.advantage_canvas_schema} "
        f"advantage_canvas_channel_count={int(cfg.advantage_canvas_channel_count)} "
        f"advantage_canvas_channels={list(cfg.advantage_canvas_channels)} "
        f"frontier_raster_used={bool(cfg.frontier_raster_used)} "
        f"channel_ablation={cfg.channel_ablation} "
        f"zeroed_advantage_channels={list(cfg.zeroed_advantage_channels)} "
        f"value_tree_enabled={bool(cfg.value_tree_enabled)} "
        f"value_replacement_strategy={cfg.value_replacement_strategy} "
        f"model_class={cfg.model_class} "
        f"advantage_encoder.canvas_in_channels={int(cfg.advantage_canvas_channel_count)} "
        f"total_env_steps={int(cfg.total_env_steps)} "
        f"epsilon_decay_steps={int(cfg.epsilon_decay_steps)} "
        f"epsilon_end={float(cfg.epsilon_end):.4f} "
        f"train_amp={bool(cfg.enable_amp)} "
        f"torch_compile={bool(cfg.enable_torch_compile)} "
        f"strict_reproducibility={bool(cfg.strict_reproducibility)} "
        f"fixed_train_episode_seeds={bool(cfg.use_fixed_train_episode_seeds)} "
        f"inference_amp={bool(cfg.enable_inference_amp)} "
        f"amp_dtype={cfg.amp_dtype} "
        f"channels_last={bool(cfg.enable_channels_last)} "
        f"timing_log_interval={int(cfg.timing_log_interval)} "
        f"train_print_interval={int(cfg.train_print_interval)} "
        f"log_interval={int(cfg.log_interval)} "
        f"formal_protocol={_formal_protocol(cfg)} "
        f"train_side_only_tuning={bool(cfg.train_side_only_tuning)} "
        f"formal_final_probe_episodes={int(cfg.final_greedy_episodes)} "
        f"best_checkpoint_selection={bool(cfg.enable_best_checkpoint_selection)} "
        f"posthoc_checkpoint_interval={int(cfg.periodic_checkpoint_interval_env_steps)} "
        f"posthoc_candidate_start={int(cfg.posthoc_candidate_start_env_steps)} "
        f"posthoc_candidate_end={_resolve_posthoc_candidate_end(cfg)} "
        f"note=\"{_describe_profiling_mode(cfg, run_mode)}\""
    )
    print(f"[startup] timing_flags {timing_flag_text}")
    if not profiling_enabled:
        print(
            "[startup] timing summaries are disabled in this run, so no [timing] lines will be printed; "
            "use RUN_MODE='profile' or RUN_MODE='profile_compile' in VSCode direct-run, "
            "or pass --profile / explicit CLI timing flags to enable them."
        )


def _run_with_startup_summary(cfg: TrainConfig, run_mode: str) -> Path:
    _print_startup_summary(cfg, run_mode=run_mode)
    return run_training(cfg, run_mode=run_mode)


def _format_timing_line(
    name: str,
    stats: dict[str, float] | None,
    *,
    aliases: dict[str, str] | None = None,
    total_key: str | None = None,
) -> str | None:
    if not stats:
        return None

    aliases = aliases or {}
    total = float(stats.get(total_key, 0.0)) if total_key is not None else 0.0
    if total <= 0.0:
        total = sum(max(0.0, float(value)) for key, value in stats.items() if key != total_key)
    if total <= 0.0:
        return None

    parts = [f"{name} total={total:.2f}s"]
    for key, value in stats.items():
        if key == total_key:
            continue
        value_f = float(value)
        if value_f <= 0.0:
            continue
        label = aliases.get(key, key)
        parts.append(f"{label}={value_f:.2f}s({(100.0 * value_f / total):.0f}%)")
    return " ".join(parts)


def _print_timing_summary(env_steps: int, collector, learner, replay, state_adapter) -> None:
    lines: list[str] = []

    collector_line = _format_timing_line(
        "collector",
        collector.get_timing_stats() if hasattr(collector, "get_timing_stats") else None,
        aliases={
            "state_build_time": "state",
            "policy_forward_time": "policy",
            "env_step_time": "env",
            "action_apply_time": "action",
            "observe_time": "observe",
            "valid_action_refresh_time": "valid",
            "cummap_update_time": "cummap_update",
            "frontier_fetch_time": "frontier_fetch",
            "shared_artifact_rebuild_time": "shared_rebuild",
            "reward_bookkeeping_time": "reward_book",
        },
        total_key="total_time_sec",
    )
    if collector_line is not None:
        lines.append(collector_line)

    learner_line = _format_timing_line(
        "learner",
        learner.get_timing_stats() if hasattr(learner, "get_timing_stats") else None,
        aliases={
            "sample_time": "sample",
            "target_compute_time": "target",
            "forward_backward_time": "fwdbwd",
            "optimizer_time": "optim",
        },
    )
    if learner_line is not None:
        lines.append(learner_line)

    replay_line = _format_timing_line(
        "replay",
        replay.get_timing_stats() if hasattr(replay, "get_timing_stats") else None,
        aliases={
            "add_time": "add",
            "sample_time": "sample",
            "h2d_time": "h2d",
        },
    )
    if replay_line is not None:
        lines.append(replay_line)

    adapter_line = _format_timing_line(
        "adapter",
        state_adapter.get_timing_stats() if hasattr(state_adapter, "get_timing_stats") else None,
        aliases={
            "shared_artifact_time": "shared",
            "advantage_build_time": "adv",
            "value_build_time": "value",
            "tensor_transfer_time": "xfer",
        },
    )
    if adapter_line is not None:
        lines.append(adapter_line)

    semantic_builder = getattr(state_adapter, "shared_semantic_layer", None)
    semantic_line = _format_timing_line(
        "semantic",
        semantic_builder.get_timing_stats()
        if semantic_builder is not None and hasattr(semantic_builder, "get_timing_stats") else None,
    )
    if semantic_line is not None:
        lines.append(semantic_line)

    advantage_builder = getattr(state_adapter, "advantage_builder", None)
    advantage_line = _format_timing_line(
        "advantage",
        advantage_builder.get_timing_stats()
        if advantage_builder is not None and hasattr(advantage_builder, "get_timing_stats") else None,
    )
    if advantage_line is not None:
        lines.append(advantage_line)

    value_builder = getattr(state_adapter, "value_builder", None)
    value_line = _format_timing_line(
        "value",
        value_builder.get_timing_stats()
        if value_builder is not None and hasattr(value_builder, "get_timing_stats") else None,
    )
    if value_line is not None:
        lines.append(value_line)

    cum_map = getattr(collector, "cum_map", None)
    cummap_line = _format_timing_line(
        "cummap",
        cum_map.get_timing_stats()
        if cum_map is not None and hasattr(cum_map, "get_timing_stats") else None,
        aliases={
            "update_time": "update",
            "local_projection_time": "project",
            "local_observation_merge_time": "local_merge",
            "bounds_expand_time": "bounds",
            "visit_update_time": "visit",
            "map_merge_time": "map_merge",
            "frontier_dirty_update_time": "frontier_dirty",
            "frontier_full_rebuild_time": "frontier_full",
            "frontier_fetch_time": "frontier_fetch",
            "frontier_cache_invalidation_time": "frontier_cache",
            "coverage_update_time": "coverage",
            "analysis_box_time": "analysis_box",
            "frontier_stats_time": "frontier",
            "domain_extract_time": "domain",
            "aggregate_time": "agg",
        },
        total_key="total_time_sec",
    )
    if cummap_line is not None:
        lines.append(cummap_line)

    if len(lines) <= 0:
        return

    print(f"[timing] env_steps={int(env_steps)}")
    for line in lines:
        print(f"  {line}")


def build_system(cfg: TrainConfig, state_adapter_factory=None, model_factory=None):
    if model_factory is None:
        advantage_canvas_channels = tuple(
            cfg.advantage_canvas_channels
            or advantage_canvas_channels_for_schema(cfg.advantage_canvas_schema)
        )
        q_cfg = ExplorationQConfig(
            advantage_encoder=AdvantageEncoderConfig(
                canvas_in_channels=len(advantage_canvas_channels),
                canvas_channels=advantage_canvas_channels,
            )
        )
        raw_online_net = ExplorationQNetwork(q_cfg)
    else:
        raw_online_net = model_factory(cfg=cfg)
    raw_online_net = raw_online_net.to(cfg.device)
    raw_online_net = _maybe_to_channels_last(raw_online_net, cfg)
    target_net = copy.deepcopy(raw_online_net).to(cfg.device)
    target_net = _maybe_to_channels_last(target_net, cfg)
    online_net = _compile_online_net(raw_online_net, cfg)

    state_cfg = StateAdapterConfig(
        shared_semantics=SharedSemanticConfig(
            enable_timing=bool(cfg.enable_shared_semantic_timing),
        ),
        advantage_state=AdvantageStateConfig(
            advantage_canvas_schema=str(cfg.advantage_canvas_schema),
            trajectory_history_steps=int(cfg.trajectory_history_steps),
            zeroed_advantage_channels=tuple(cfg.zeroed_advantage_channels),
            enable_timing=bool(cfg.enable_advantage_state_timing),
        ),
        value_state=ValueStateConfig(
            max_accessible_blocks=int(cfg.max_accessible_blocks),
            max_entries_per_block=int(cfg.max_entries_per_block),
            enable_timing=bool(cfg.enable_value_state_timing),
            value_replacement_strategy=str(cfg.value_replacement_strategy),
        ),
        pin_memory=True,
        non_blocking_transfer=True,
        channels_last_on_cuda=bool(cfg.enable_channels_last),
        enable_timing=bool(cfg.enable_state_adapter_timing),
    )
    if state_adapter_factory is None:
        state_adapter = StateTensorAdapter(cfg=state_cfg, device="cpu")
    else:
        state_adapter = state_adapter_factory(cfg=state_cfg, device="cpu")

    replay = ReplayBuffer(
        ReplayBufferConfig(
            capacity=int(cfg.replay_capacity),
            prioritized=False,
            pin_memory=True,
            non_blocking_transfer=True,
            channels_last_on_cuda=bool(cfg.enable_channels_last),
            enable_timing=bool(cfg.enable_replay_timing),
        )
    )

    amp_dtype = str(cfg.amp_dtype).lower()
    if amp_dtype not in {"fp16", "bf16"}:
        raise ValueError(f"Unsupported amp_dtype: {cfg.amp_dtype!r}; expected 'fp16' or 'bf16'")

    collector_cfg = CollectorConfig(
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
        enable_timing=bool(cfg.enable_collector_timing),
        enable_cummap_timing=bool(cfg.enable_cummap_timing),
        enable_inference_amp=bool(cfg.enable_inference_amp),
        inference_amp_dtype=amp_dtype,
        debug_check_incremental_frontier=bool(cfg.debug_check_incremental_frontier),
        prefer_batch_replay_add=bool(cfg.prefer_batch_replay_add),
        use_fixed_train_episode_seeds=bool(cfg.use_fixed_train_episode_seeds),
        fixed_train_episode_seed_base=int(cfg.fixed_train_episode_seed_base),
        record_episode_artifacts=bool(
            cfg.save_train_representative_trajectories or cfg.save_train_special_trajectories
        ),
    )
    collector = TransitionCollector(collector_cfg, online_net, state_adapter, replay)

    learner_cfg_kwargs = {
        "batch_size": int(cfg.batch_size),
        "min_replay_size": int(cfg.min_replay_size),
        "learning_rate": float(cfg.learning_rate),
        "weight_decay": float(cfg.weight_decay),
        "grad_clip_norm": float(cfg.grad_clip_norm),
        "target_update_interval": int(cfg.target_update_interval),
        "enable_amp": bool(cfg.enable_amp),
        "amp_dtype": amp_dtype,
        "enable_timing": bool(cfg.enable_learner_timing),
        "return_debug_stats_every": int(max(1, cfg.learner_debug_stats_every)),
    }

    learner_cfg = DDQNLearnerConfig(**learner_cfg_kwargs)
    learner = DDQNLearner(online_net, target_net, learner_cfg, device=cfg.device)

    evaluator = GreedyEvaluator.from_collector_config(
        collector_cfg, state_adapter=state_adapter, device=cfg.device
    )

    return online_net, target_net, replay, collector, learner, evaluator


def _best_model_select_score(row: Mapping[str, Any]) -> tuple[float, float, float]:
    """
    Fixed checkpoint-selection rule.

    Validation and recheck candidates are ranked by success_rate first, then
    coverage, then reward. The rule intentionally ignores final-test seeds;
    model selection uses its own dedicated seed set.
    """

    def metric(name: str) -> float:
        value = row.get(name)
        if value is None:
            value = row.get(f"eval_{name}")
        try:
            return float(value)
        except Exception:
            return float("-inf")

    return (
        metric("success_rate"),
        metric("mean_coverage"),
        metric("mean_reward"),
    )


def _formal_protocol(cfg: TrainConfig) -> str:
    protocol = str(getattr(cfg, "formal_protocol", POSTHOC_PROTOCOL_NAME)).strip()
    return protocol or POSTHOC_PROTOCOL_NAME


def _use_posthoc_protocol(cfg: TrainConfig) -> bool:
    return _formal_protocol(cfg) == POSTHOC_PROTOCOL_NAME


def _resolve_posthoc_candidate_end(cfg: TrainConfig) -> int:
    configured = int(getattr(cfg, "posthoc_candidate_end_env_steps", 0))
    return int(cfg.total_env_steps) if configured <= 0 else configured


def _relative_checkpoint_path(run_dir: Path, checkpoint_path: Path) -> str:
    try:
        return Path(checkpoint_path).resolve().relative_to(run_dir.resolve()).as_posix()
    except Exception:
        return Path(checkpoint_path).as_posix()


def _load_checkpoint_into_model(model: torch.nn.Module, checkpoint_path: Path, device: str) -> dict[str, Any]:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(payload["online_state_dict"])
    return payload


def _build_eval_row(
    *,
    tag: str,
    source: str,
    budget_mode: str,
    env_steps: int,
    episode_idx: int,
    train_episode_idx: int,
    completed_train_episodes: int,
    learner_steps: int,
    eval_result: Mapping[str, Any],
    checkpoint_path: str,
    seed_base: int | None,
    selection_rank: int | None = None,
) -> dict[str, object]:
    row = {
        "tag": tag,
        "source": source,
        "budget_mode": budget_mode,
        "env_steps": int(env_steps),
        "episode_idx": int(episode_idx),
        "train_episode_idx": int(train_episode_idx),
        "completed_train_episodes": int(completed_train_episodes),
        "learner_steps": int(learner_steps),
        "checkpoint_path": str(checkpoint_path),
        "seed_base": None if seed_base is None else int(seed_base),
        "selection_rank": None if selection_rank is None else int(selection_rank),
        "eval_episodes": int(eval_result["eval_episodes"]),
        "eval_mean_reward": float(eval_result["eval_mean_reward"]),
        "eval_mean_coverage": float(eval_result["eval_mean_coverage"]),
        "eval_success_rate": float(eval_result["eval_success_rate"]),
        "eval_mean_episode_length": float(eval_result["eval_mean_episode_length"]),
        "eval_mean_repeat_visit_ratio": float(eval_result["eval_mean_repeat_visit_ratio"]),
        **{
            f"eval_mean_{metric_name}": float(eval_result[f"eval_mean_{metric_name}"])
            for metric_name in EVAL_SEMANTIC_METRIC_NAMES
        },
        **{
            f"eval_mean_{field}": float(eval_result[f"eval_mean_{field}"])
            for field in REWARD_BREAKDOWN_FIELDS
        },
        **{
            f"eval_mean_{field}": float(eval_result[f"eval_mean_{field}"])
            for field in REWARD_EVENT_SUMMARY_FIELDS
        },
    }
    return row


def run_training(cfg: TrainConfig, *, run_mode: str = "cli", state_adapter_factory=None, model_factory=None) -> Path:
    run_start_time = time.perf_counter()
    set_seed(int(cfg.seed))
    backend_readback = configure_torch_runtime(cfg)
    run_dir = create_run_dir(cfg)
    logger = CSVMetricLogger(run_dir)
    ckpt = CheckpointManager(run_dir)
    budget_mode = resolve_budget_mode(cfg)

    online_net, _, replay, collector, learner, evaluator = build_system(
        cfg,
        state_adapter_factory=state_adapter_factory,
        model_factory=model_factory,
    )
    reproducibility_runtime_info = collect_reproducibility_runtime_info(
        cfg,
        backend_readback=backend_readback,
        online_net=online_net,
    )

    recent_eps: deque[dict] = deque(maxlen=int(max(1, cfg.recent_episode_window)))
    warmup_phase_episodes = 0
    train_phase_episodes = 0
    last_train_metrics = {
        "loss": float("nan"),
        "q_mean": float("nan"),
        "target_q_mean": float("nan"),
        "td_abs_mean": float("nan"),
        "grad_norm": float("nan"),
    }
    trajectory_plot_paths: list[Path] = []
    train_trace_episodes: list[dict] = []
    episode_print_interval = int(max(0, cfg.episode_print_interval))
    model_select_rows: list[dict[str, object]] = []
    best_recheck_rows: list[dict[str, object]] = []
    best_validation_row: dict[str, object] | None = None
    final_best_selection_row: dict[str, object] | None = None
    last_checkpoint_diagnostic_row: dict[str, object] | None = None
    best_checkpoint_path: Path | None = None
    formal_protocol = _formal_protocol(cfg)
    train_side_only_tuning = bool(cfg.train_side_only_tuning)
    use_posthoc_protocol = (not train_side_only_tuning) and _use_posthoc_protocol(cfg)
    enable_best_selection = (
        (not train_side_only_tuning)
        and (not use_posthoc_protocol)
        and bool(cfg.enable_best_checkpoint_selection)
    )
    posthoc_candidate_rows: list[dict[str, Any]] = []
    posthoc_final_probe_rows: list[dict[str, object]] = []
    posthoc_selection_result: dict[str, Any] | None = None
    posthoc_candidate_start_step = int(max(0, cfg.posthoc_candidate_start_env_steps))
    posthoc_candidate_end_step = _resolve_posthoc_candidate_end(cfg)
    posthoc_checkpoint_interval = int(max(1, cfg.periodic_checkpoint_interval_env_steps))
    posthoc_selection_window = int(max(1, cfg.posthoc_selection_window_env_steps))
    posthoc_final_probe_topk = int(max(1, cfg.posthoc_final_probe_topk))
    next_posthoc_checkpoint_step = int(posthoc_candidate_start_step)
    saved_posthoc_checkpoint_steps: set[int] = set()
    model_select_seed_base = (
        int(cfg.fixed_model_select_seed_base) if bool(cfg.use_fixed_model_select_seeds) else None
    )
    final_probe_seed_base = int(cfg.fixed_final_probe_seed_base) if bool(cfg.use_fixed_eval_seeds) else None

    def completed_train_episodes() -> int:
        return int(train_phase_episodes)

    def handle_episodes(episodes: list[dict], phase: str, epsilon: float) -> None:
        nonlocal warmup_phase_episodes, train_phase_episodes
        for ep in episodes:
            if phase == "warmup":
                warmup_phase_episodes += 1
                phase_episode_idx = int(warmup_phase_episodes)
            else:
                train_phase_episodes += 1
                phase_episode_idx = int(train_phase_episodes)
            row = {
                "phase": phase,
                "budget_mode": budget_mode,
                "env_steps": int(ep["env_steps"]),
                "episode_idx": int(ep["episode_idx"]),
                "train_episode_idx": int(ep.get("train_episode_idx", ep["episode_idx"])),
                "phase_episode_idx": phase_episode_idx,
                "completed_train_episodes": int(train_phase_episodes),
                "episode_seed": (
                    None if ep.get("episode_seed") is None else int(ep["episode_seed"])
                ),
                "map_fingerprint": str(ep.get("map_fingerprint") or ""),
                "epsilon": float(epsilon),
                "episode_reward": float(ep["episode_reward"]),
                "episode_length": int(ep["episode_length"]),
                "final_coverage": float(ep["final_coverage"]),
                "success": int(ep["success"]),
                "repeat_visit_ratio": float(ep["repeat_visit_ratio"]),
                "done_reason": str(ep["done_reason"]),
                **{
                    field: float(ep[field])
                    for field in SEMANTIC_EPISODE_FIELDS
                    if field in ep
                },
                **{
                    field: float(ep[field])
                    for field in REWARD_BREAKDOWN_FIELDS
                },
                **{
                    field: float(ep[field])
                    for field in REWARD_EVENT_SUMMARY_FIELDS
                    if field in ep
                },
                **{
                    field: float(ep[field])
                    for field in DERIVED_TRAIN_DIAGNOSTIC_FIELDS
                    if field in ep
                },
            }
            logger.log_train_episode(row)
            recent_eps.append(row)
            if (
                phase == "train"
                and bool(cfg.save_train_representative_trajectories or cfg.save_train_special_trajectories)
                and (ep.get("trajectory_positions") is not None)
                and (ep.get("true_grid") is not None)
            ):
                trace_ep = dict(ep)
                trace_ep["phase"] = phase
                trace_ep["epsilon"] = float(epsilon)
                trace_ep["phase_episode_idx"] = phase_episode_idx
                trace_ep["completed_train_episodes"] = int(train_phase_episodes)
                train_trace_episodes.append(trace_ep)
            if episode_print_interval > 0 and int(row["episode_idx"]) % episode_print_interval == 0:
                print(
                    "[episode] "
                    f"phase={phase} idx={row['episode_idx']} phase_idx={row['phase_episode_idx']} "
                    f"env={row['env_steps']} seed={row['episode_seed']} "
                    f"reward={row['episode_reward']:.4f} len={row['episode_length']} "
                    f"cov={row['final_coverage']:.4f} succ={row['success']} "
                    f"repeat={row['repeat_visit_ratio']:.4f} reason={row['done_reason']}"
                )

    timing_log_interval = int(max(0, cfg.timing_log_interval))
    timing_summary_enabled = _timing_summary_enabled(cfg)
    if budget_mode == "episodes":
        warmup_episodes = int(max(0, cfg.warmup_episodes))
        if warmup_episodes > 0:
            warm_stats = collector.collect_steps(
                max(1, warmup_episodes * int(cfg.max_episode_steps)),
                epsilon=1.0,
                random_only=True,
                stop_after_episodes=warmup_episodes,
            )
            handle_episodes(warm_stats.get("episodes", []), phase="warmup", epsilon=1.0)
        env_steps = int(collector.total_env_steps)
        total_train_episodes = int(max(1, cfg.total_train_episodes))
        log_interval_episodes = int(max(0, cfg.log_interval_episodes))
        next_log_episode = (
            (((completed_train_episodes() // log_interval_episodes) + 1) * log_interval_episodes)
            if log_interval_episodes > 0 else total_train_episodes + 1
        )
        train_print_interval_episodes = int(max(0, cfg.train_print_interval_episodes))
        next_train_print_episode = (
            (((completed_train_episodes() // train_print_interval_episodes) + 1) * train_print_interval_episodes)
            if train_print_interval_episodes > 0 else total_train_episodes + 1
        )
        log_interval = int(cfg.log_interval)
        next_log_step = int(cfg.total_env_steps) + 1
        train_print_interval = int(max(0, cfg.train_print_interval))
        next_train_print_step = int(cfg.total_env_steps) + 1
    else:
        warmup = min(int(cfg.warmup_steps), int(cfg.total_env_steps))
        if warmup > 0:
            warm_stats = collector.collect_steps(warmup, epsilon=1.0, random_only=True)
            handle_episodes(warm_stats.get("episodes", []), phase="warmup", epsilon=1.0)
        env_steps = int(collector.total_env_steps)
        total_train_episodes = int(max(1, cfg.total_train_episodes))
        log_interval = int(cfg.log_interval)
        next_log_step = (
            (((env_steps // log_interval) + 1) * log_interval)
            if log_interval > 0 else int(cfg.total_env_steps) + 1
        )
        train_print_interval = int(max(0, cfg.train_print_interval))
        next_train_print_step = (
            (((env_steps // train_print_interval) + 1) * train_print_interval)
            if train_print_interval > 0 else int(cfg.total_env_steps) + 1
        )
        log_interval_episodes = 0
        next_log_episode = total_train_episodes + 1
        train_print_interval_episodes = 0
        next_train_print_episode = total_train_episodes + 1

    last_train_env_step = int(env_steps)
    next_timing_log_step = (
        timing_log_interval if timing_log_interval > 0 else int(cfg.total_env_steps) + 1
    )
    model_select_interval = int(max(1, cfg.best_checkpoint_selection_interval_env_steps))
    next_model_select_step = int(max(0, cfg.best_checkpoint_selection_start_env_steps))
    if not enable_best_selection:
        next_model_select_step = int(cfg.total_env_steps) + model_select_interval + 1

    def emit_train_snapshot(eps: float, *, should_log: bool, should_print_train: bool) -> None:
        rec = summarize_recent_episodes(recent_eps)
        step_row = {
            "budget_mode": budget_mode,
            "env_steps": int(env_steps),
            "episode_idx": int(collector.total_episodes),
            "train_episode_idx": int(collector.total_episodes),
            "completed_train_episodes": int(train_phase_episodes),
            "replay_size": int(len(replay)),
            "epsilon": float(eps),
            "loss": float(last_train_metrics["loss"]),
            "q_mean": float(last_train_metrics["q_mean"]),
            "target_q_mean": float(last_train_metrics["target_q_mean"]),
            "td_abs_mean": float(last_train_metrics["td_abs_mean"]),
            "grad_norm": float(last_train_metrics["grad_norm"]),
            "learner_steps": int(learner.learn_steps),
            "recent_mean_reward": float(rec["mean_reward"]),
            "recent_mean_coverage": float(rec["mean_coverage"]),
            "recent_success_rate": float(rec["success_rate"]),
            "recent_mean_episode_length": float(rec["mean_length"]),
            "recent_mean_repeat_visit_ratio": float(rec["mean_repeat_visit_ratio"]),
            **{
                f"recent_{field}": float(rec[field])
                for field in SEMANTIC_EPISODE_FIELDS
            },
            **{
                f"recent_{field}": float(rec[field])
                for field in REWARD_BREAKDOWN_FIELDS
            },
            **{
                f"recent_{field}": float(rec[field])
                for field in REWARD_EVENT_SUMMARY_FIELDS
            },
            **{
                f"recent_{field}": float(rec[field])
                for field in DERIVED_TRAIN_DIAGNOSTIC_FIELDS
            },
        }
        if should_log:
            logger.log_train_step(step_row)
        if should_print_train:
            print(
                "[train] "
                f"budget={budget_mode} env_steps={step_row['env_steps']} "
                f"train_eps={step_row['completed_train_episodes']} replay={step_row['replay_size']} "
                f"eps={step_row['epsilon']:.4f} loss={step_row['loss']:.5f} "
                f"q_mean={step_row['q_mean']:.5f} target_q_mean={step_row['target_q_mean']:.5f} "
                f"td_abs_mean={step_row['td_abs_mean']:.5f} grad_norm={step_row['grad_norm']:.5f} "
                f"learner_steps={step_row['learner_steps']} "
                f"recent_reward={step_row['recent_mean_reward']:.4f} "
                f"recent_cov={step_row['recent_mean_coverage']:.4f} "
                f"recent_succ={step_row['recent_success_rate']:.4f} "
                f"recent_blocks={step_row['recent_accessible_block_count']:.2f}"
            )

    def run_model_select_validation(trigger_env_steps: int) -> dict[str, object]:
        nonlocal best_validation_row, best_checkpoint_path

        candidate_path = ckpt.save_model_select_candidate(
            online_net,
            learner,
            env_steps=int(env_steps),
            train_episode_idx=int(train_phase_episodes),
            train_config=cfg,
            selection_metadata={
                "selection_phase": "periodic_validation",
                "trigger_env_steps": int(trigger_env_steps),
                "seed_base": model_select_seed_base,
                "episodes": int(max(1, cfg.best_checkpoint_validation_episodes)),
            },
        )
        validation = evaluator.evaluate(
            online_net,
            num_episodes=int(max(1, cfg.best_checkpoint_validation_episodes)),
            seed_base=model_select_seed_base,
        )
        row = _build_eval_row(
            tag="model_select_eval",
            source="periodic_validation",
            budget_mode=budget_mode,
            env_steps=int(env_steps),
            episode_idx=int(collector.total_episodes),
            train_episode_idx=int(collector.total_episodes),
            completed_train_episodes=int(train_phase_episodes),
            learner_steps=int(learner.learn_steps),
            eval_result=validation,
            checkpoint_path=_relative_checkpoint_path(run_dir, candidate_path),
            seed_base=model_select_seed_base,
        )
        row["trigger_env_steps"] = int(trigger_env_steps)
        logger.log_model_select_eval(row)
        model_select_rows.append(row)

        if best_validation_row is None or _best_model_select_score(row) > _best_model_select_score(best_validation_row):
            best_validation_row = row
            best_checkpoint_path = ckpt.save_best_from_checkpoint(
                candidate_path,
                selection_metadata={
                    "selection_phase": "periodic_validation_preliminary_best",
                    "selection_rule": "success_rate_then_coverage_then_reward",
                    "selection_row": row,
                },
            )
            best_marker = "new_best"
        else:
            best_marker = "kept_best"
        print(
            "[model_select] "
            f"{best_marker} trigger_env={int(trigger_env_steps)} env_steps={int(env_steps)} "
            f"episodes={row['eval_episodes']} success={row['eval_success_rate']:.4f} "
            f"coverage={row['eval_mean_coverage']:.4f} reward={row['eval_mean_reward']:.4f}"
        )
        return row

    def maybe_run_model_select_validation() -> None:
        nonlocal next_model_select_step
        if not enable_best_selection:
            return
        if int(env_steps) < int(next_model_select_step):
            return
        trigger_env_steps = int(next_model_select_step)
        # If a rollout crosses more than one validation boundary, evaluate the
        # current checkpoint once and advance past all missed boundaries. Default
        # formal settings collect in 16-step chunks, so planned 20k boundaries are
        # reached exactly.
        run_model_select_validation(trigger_env_steps)
        while int(next_model_select_step) <= int(env_steps):
            next_model_select_step += model_select_interval

    def maybe_save_posthoc_checkpoint(*, force: bool = False) -> None:
        nonlocal next_posthoc_checkpoint_step
        if not use_posthoc_protocol:
            return
        if int(env_steps) < int(posthoc_candidate_start_step):
            return
        if int(env_steps) > int(posthoc_candidate_end_step) and not force:
            return
        should_save = force or int(env_steps) >= int(next_posthoc_checkpoint_step)
        if not should_save:
            return
        checkpoint_step = int(env_steps)
        if checkpoint_step in saved_posthoc_checkpoint_steps:
            while int(next_posthoc_checkpoint_step) <= checkpoint_step:
                next_posthoc_checkpoint_step += posthoc_checkpoint_interval
            return
        checkpoint_path = ckpt.save_periodic_checkpoint(
            online_net,
            learner,
            env_steps=checkpoint_step,
            train_episode_idx=int(train_phase_episodes),
            train_config=cfg,
            selection_metadata={
                "protocol_name": formal_protocol,
                "checkpoint_role": "posthoc_train_side_candidate",
                "training_only_checkpoint": True,
                "note": "Saved during training without validation, recheck, or held-out probe.",
            },
        )
        saved_posthoc_checkpoint_steps.add(checkpoint_step)
        while int(next_posthoc_checkpoint_step) <= checkpoint_step:
            next_posthoc_checkpoint_step += posthoc_checkpoint_interval
        print(
            "[checkpoint] "
            f"posthoc_candidate env_steps={checkpoint_step} path={_relative_checkpoint_path(run_dir, checkpoint_path)}"
        )

    if timing_summary_enabled and timing_log_interval > 0:
        while env_steps >= next_timing_log_step:
            _print_timing_summary(env_steps, collector, learner, replay, collector.state_adapter)
            next_timing_log_step += timing_log_interval

    if budget_mode == "episodes":
        while completed_train_episodes() < total_train_episodes:
            eps = linear_epsilon(env_steps, cfg)
            remaining_episodes = max(1, total_train_episodes - completed_train_episodes())
            cstats = collector.collect_steps(
                int(max(1, cfg.collect_steps_per_iter)),
                epsilon=eps,
                stop_after_episodes=remaining_episodes,
            )
            env_steps += int(cstats["env_steps"])
            handle_episodes(cstats.get("episodes", []), phase="train", epsilon=eps)

            train_interval = int(max(1, cfg.train_every_env_steps))
            if (
                (env_steps - last_train_env_step) >= train_interval
                and len(replay) >= int(cfg.min_replay_size)
            ):
                for _ in range(int(cfg.learner_updates_per_iter)):
                    if len(replay) < int(cfg.min_replay_size):
                        break
                    lstats = learner.train_step(replay)
                    if lstats is not None:
                        for metric_name in last_train_metrics:
                            if metric_name in lstats:
                                last_train_metrics[metric_name] = float(lstats[metric_name])
                last_train_env_step = int(env_steps)

            if timing_summary_enabled and timing_log_interval > 0:
                while env_steps >= next_timing_log_step:
                    _print_timing_summary(env_steps, collector, learner, replay, collector.state_adapter)
                    next_timing_log_step += timing_log_interval

            should_log = (completed_train_episodes() == total_train_episodes)
            if log_interval_episodes > 0 and completed_train_episodes() >= next_log_episode:
                should_log = True
                while completed_train_episodes() >= next_log_episode:
                    next_log_episode += log_interval_episodes

            should_print_train = False
            if (
                train_print_interval_episodes > 0
                and completed_train_episodes() >= next_train_print_episode
            ):
                should_print_train = True
                while completed_train_episodes() >= next_train_print_episode:
                    next_train_print_episode += train_print_interval_episodes

            if should_log or should_print_train:
                emit_train_snapshot(eps, should_log=should_log, should_print_train=should_print_train)
            maybe_save_posthoc_checkpoint()
            maybe_run_model_select_validation()
    else:
        while env_steps < int(cfg.total_env_steps):
            eps = linear_epsilon(env_steps, cfg)
            collect_n = min(int(cfg.collect_steps_per_iter), int(cfg.total_env_steps) - env_steps)
            cstats = collector.collect_steps(collect_n, epsilon=eps)
            env_steps += int(cstats["env_steps"])
            handle_episodes(cstats.get("episodes", []), phase="train", epsilon=eps)

            train_interval = int(max(1, cfg.train_every_env_steps))
            if (
                (env_steps - last_train_env_step) >= train_interval
                and len(replay) >= int(cfg.min_replay_size)
            ):
                for _ in range(int(cfg.learner_updates_per_iter)):
                    if len(replay) < int(cfg.min_replay_size):
                        break
                    lstats = learner.train_step(replay)
                    if lstats is not None:
                        for metric_name in last_train_metrics:
                            if metric_name in lstats:
                                last_train_metrics[metric_name] = float(lstats[metric_name])
                last_train_env_step = int(env_steps)

            if timing_summary_enabled and timing_log_interval > 0:
                while env_steps >= next_timing_log_step:
                    _print_timing_summary(env_steps, collector, learner, replay, collector.state_adapter)
                    next_timing_log_step += timing_log_interval

            should_log = (env_steps == int(cfg.total_env_steps))
            if log_interval > 0 and env_steps >= next_log_step:
                should_log = True
                while env_steps >= next_log_step:
                    next_log_step += log_interval

            should_print_train = False
            if train_print_interval > 0 and env_steps >= next_train_print_step:
                should_print_train = True
                while env_steps >= next_train_print_step:
                    next_train_print_step += train_print_interval

            if should_log or should_print_train:
                emit_train_snapshot(eps, should_log=should_log, should_print_train=should_print_train)
            maybe_save_posthoc_checkpoint()
            maybe_run_model_select_validation()

    last_ckpt_path = ckpt.save_last(
        online_net,
        learner,
        env_steps=int(env_steps),
        train_episode_idx=int(train_phase_episodes),
        train_config=cfg,
    )

    best_env_steps: int | None = None
    best_train_episode_idx: int | None = None
    probe_source = "skipped_train_side_only_tuning"
    probe: dict[str, Any] = {}
    probe_row: dict[str, object] | None = None

    if train_side_only_tuning:
        print(
            "[train_side_only_tuning] "
            "skipping posthoc checkpoint selection, candidate scoring, final_probe evaluation, "
            "best-vs-last comparison, and automatic winner checkpoint selection."
        )
    elif use_posthoc_protocol:
        if int(env_steps) >= posthoc_candidate_start_step and int(env_steps) <= posthoc_candidate_end_step:
            # If the configured final step is itself a scheduled boundary, this
            # is a no-op because the loop already saved it. It never evaluates.
            if (int(env_steps) - posthoc_candidate_start_step) % posthoc_checkpoint_interval == 0:
                maybe_save_posthoc_checkpoint(force=True)

        posthoc_selection_result = select_posthoc_candidates(
            run_dir=run_dir,
            candidate_start_step=posthoc_candidate_start_step,
            candidate_end_step=posthoc_candidate_end_step,
            checkpoint_interval=posthoc_checkpoint_interval,
            window_env_steps=posthoc_selection_window,
            topk=posthoc_final_probe_topk,
        )
        posthoc_candidate_rows = [
            dict(row) for row in posthoc_selection_result.get("selected_candidates", [])
        ]
        if not posthoc_candidate_rows:
            fallback_path = ckpt.save_periodic_checkpoint(
                online_net,
                learner,
                env_steps=int(env_steps),
                train_episode_idx=int(train_phase_episodes),
                train_config=cfg,
                selection_metadata={
                    "protocol_name": formal_protocol,
                    "checkpoint_role": "posthoc_fallback_last_candidate",
                    "training_only_checkpoint": True,
                    "reason": "no_valid_periodic_posthoc_candidates",
                },
            )
            print(
                "[warning] no valid posthoc candidates were available; "
                f"saved fallback training-side checkpoint {_relative_checkpoint_path(run_dir, fallback_path)}"
            )
            posthoc_selection_result = select_posthoc_candidates(
                run_dir=run_dir,
                candidate_start_step=int(env_steps),
                candidate_end_step=int(env_steps),
                checkpoint_interval=posthoc_checkpoint_interval,
                window_env_steps=posthoc_selection_window,
                topk=1,
            )
            posthoc_candidate_rows = [
                dict(row) for row in posthoc_selection_result.get("selected_candidates", [])
            ]
        if not posthoc_candidate_rows:
            raise RuntimeError("post-hoc candidate selection produced no valid checkpoints")

        candidate_probe_payloads: dict[int, Mapping[str, Any]] = {}
        for candidate_row in posthoc_candidate_rows:
            candidate_step = int(candidate_row["candidate_step"])
            candidate_path = run_dir / str(candidate_row["checkpoint_path"])
            payload = _load_checkpoint_into_model(online_net, candidate_path, cfg.device)
            probe_result = evaluator.evaluate(
                online_net,
                num_episodes=int(max(1, cfg.final_greedy_episodes)),
                seed_base=final_probe_seed_base,
            )
            candidate_probe_payloads[candidate_step] = probe_result
            row = _build_eval_row(
                tag="final_probe",
                source="posthoc_candidate_final_probe",
                budget_mode=budget_mode,
                env_steps=int(payload.get("env_steps", candidate_step)),
                episode_idx=int(collector.total_episodes),
                train_episode_idx=int(payload.get("train_episode_idx", train_phase_episodes)),
                completed_train_episodes=int(train_phase_episodes),
                learner_steps=int(payload.get("learn_steps", learner.learn_steps)),
                eval_result=probe_result,
                checkpoint_path=str(candidate_row["checkpoint_path"]),
                seed_base=final_probe_seed_base,
                selection_rank=int(candidate_row.get("selection_rank") or 0),
            )
            row["posthoc_selection_rank"] = int(candidate_row.get("selection_rank") or 0)
            row["posthoc_selection_score"] = float(candidate_row.get("selection_score") or 0.0)
            row["posthoc_window_start_env_steps"] = int(candidate_row.get("window_start_env_steps") or 0)
            row["posthoc_window_end_env_steps"] = int(candidate_row.get("window_end_env_steps") or candidate_step)
            row["posthoc_window_row_count"] = int(candidate_row.get("window_row_count") or 0)
            row["formal_winner"] = False
            row["final_probe_rank"] = None
            posthoc_final_probe_rows.append(row)

        ranked_probe_rows = sorted(
            posthoc_final_probe_rows,
            key=lambda row: (
                *final_probe_rank_key(row),
                -int(row.get("posthoc_selection_rank") or 0),
            ),
            reverse=True,
        )
        winner_row = ranked_probe_rows[0]
        for rank, row in enumerate(ranked_probe_rows, start=1):
            row["final_probe_rank"] = rank
            is_winner = row is winner_row
            row["formal_winner"] = is_winner
            row["source"] = "posthoc_final_winner" if is_winner else "posthoc_candidate_final_probe"
            logger.log_final_probe(row)
            print(
                "[final_probe] "
                f"rank={rank} source={row['source']} checkpoint={row['checkpoint_path']} "
                f"env_steps={row['env_steps']} episodes={row['eval_episodes']} "
                f"mean_reward={row['eval_mean_reward']:.4f} mean_cov={row['eval_mean_coverage']:.4f} "
                f"success_rate={row['eval_success_rate']:.4f} mean_len={row['eval_mean_episode_length']:.2f}"
            )

        best_candidate_path = run_dir / str(winner_row["checkpoint_path"])
        best_checkpoint_path = ckpt.save_best_from_checkpoint(
            best_candidate_path,
            selection_metadata={
                "protocol_name": formal_protocol,
                "selection_phase": "posthoc_trainselect_final_probe_winner",
                "train_side_candidate_selection": posthoc_selection_result.get("summary", {}),
                "final_probe_selection_rule": "success_rate_then_coverage_then_reward",
                "winner_final_probe_row": winner_row,
            },
        )
        best_payload = _load_checkpoint_into_model(online_net, best_checkpoint_path, cfg.device)
        best_env_steps = int(best_payload.get("env_steps", winner_row["env_steps"]))
        best_train_episode_idx = int(best_payload.get("train_episode_idx", winner_row["train_episode_idx"]))
        probe_source = "posthoc_final_winner"
        probe_row = dict(winner_row)
        probe = dict(candidate_probe_payloads.get(best_env_steps, {}))
        if not probe:
            probe = dict(candidate_probe_payloads.get(int(winner_row["env_steps"]), {}))
        final_best_selection_row = probe_row
        last_probe_matches = [
            row for row in posthoc_final_probe_rows
            if int(row.get("env_steps", -1)) == int(env_steps)
        ]
        last_checkpoint_diagnostic_row = last_probe_matches[-1] if last_probe_matches else None

        recent_for_posthoc = summarize_recent_episodes(recent_eps)
        recent_for_posthoc_row = {
            "env_steps": int(env_steps),
            "recent_mean_reward": float(recent_for_posthoc["mean_reward"]),
            "recent_mean_coverage": float(recent_for_posthoc["mean_coverage"]),
            "recent_success_rate": float(recent_for_posthoc["success_rate"]),
            "recent_mean_episode_length": float(recent_for_posthoc["mean_length"]),
            "recent_mean_repeat_visit_ratio": float(recent_for_posthoc["mean_repeat_visit_ratio"]),
            **{
                f"recent_{field}": float(recent_for_posthoc[field])
                for field in DERIVED_TRAIN_DIAGNOSTIC_FIELDS
            },
        }
        write_posthoc_final_artifacts(
            run_dir=run_dir,
            total_env_steps=int(env_steps),
            candidate_start_step=posthoc_candidate_start_step,
            candidate_end_step=posthoc_candidate_end_step,
            checkpoint_interval=posthoc_checkpoint_interval,
            selected_candidates=posthoc_candidate_rows,
            final_probe_rows=ranked_probe_rows,
            winner_probe_row=probe_row,
            recent_train_row=recent_for_posthoc_row,
            last_checkpoint_path=_relative_checkpoint_path(run_dir, last_ckpt_path),
            best_pt_path=_relative_checkpoint_path(run_dir, best_checkpoint_path),
            final_probe_episode_count=int(max(1, cfg.final_greedy_episodes)),
            seed_base=final_probe_seed_base,
        )
    else:
        if enable_best_selection and not any(int(row.get("env_steps", -1)) == int(env_steps) for row in model_select_rows):
            if int(env_steps) >= int(cfg.best_checkpoint_selection_start_env_steps):
                run_model_select_validation(int(env_steps))

        if not model_select_rows:
            # Short smoke runs or disabled model selection still publish best.pt
            # by promoting last.pt without extra selection episodes.
            best_checkpoint_path = ckpt.save_best_from_checkpoint(
                last_ckpt_path,
                selection_metadata={
                    "selection_phase": "fallback_last_checkpoint_no_model_select_eval",
                    "selection_rule": "no_validation_candidates_available",
                },
            )
            final_best_selection_row = {
                "tag": "best_recheck_eval",
                "source": "fallback_last_checkpoint_no_model_select_eval",
                "env_steps": int(env_steps),
                "checkpoint_path": _relative_checkpoint_path(run_dir, last_ckpt_path),
                "selection_rank": None,
            }
            last_checkpoint_diagnostic_row = final_best_selection_row
        else:
            topk = int(max(1, cfg.best_checkpoint_topk_recheck))
            ranked_candidates = sorted(model_select_rows, key=_best_model_select_score, reverse=True)
            top_candidates = ranked_candidates[: min(topk, len(ranked_candidates))]
            for rank, candidate_row in enumerate(top_candidates, start=1):
                candidate_path = run_dir / str(candidate_row["checkpoint_path"])
                _load_checkpoint_into_model(online_net, candidate_path, cfg.device)
                recheck = evaluator.evaluate(
                    online_net,
                    num_episodes=int(max(1, cfg.best_checkpoint_recheck_episodes)),
                    seed_base=model_select_seed_base,
                )
                recheck_row = _build_eval_row(
                    tag="best_recheck_eval",
                    source="topk_model_select_recheck",
                    budget_mode=budget_mode,
                    env_steps=int(candidate_row["env_steps"]),
                    episode_idx=int(candidate_row["episode_idx"]),
                    train_episode_idx=int(candidate_row["train_episode_idx"]),
                    completed_train_episodes=int(candidate_row["completed_train_episodes"]),
                    learner_steps=int(candidate_row["learner_steps"]),
                    eval_result=recheck,
                    checkpoint_path=str(candidate_row["checkpoint_path"]),
                    seed_base=model_select_seed_base,
                    selection_rank=rank,
                )
                recheck_row["model_select_eval_success_rate"] = float(candidate_row["eval_success_rate"])
                recheck_row["model_select_eval_mean_coverage"] = float(candidate_row["eval_mean_coverage"])
                recheck_row["model_select_eval_mean_reward"] = float(candidate_row["eval_mean_reward"])
                logger.log_best_recheck_eval(recheck_row)
                best_recheck_rows.append(recheck_row)
                print(
                    "[best_recheck] "
                    f"rank={rank} env_steps={recheck_row['env_steps']} episodes={recheck_row['eval_episodes']} "
                    f"success={recheck_row['eval_success_rate']:.4f} "
                    f"coverage={recheck_row['eval_mean_coverage']:.4f} reward={recheck_row['eval_mean_reward']:.4f}"
                )

            final_best_selection_row = max(best_recheck_rows, key=_best_model_select_score)
            best_candidate_path = run_dir / str(final_best_selection_row["checkpoint_path"])
            best_checkpoint_path = ckpt.save_best_from_checkpoint(
                best_candidate_path,
                selection_metadata={
                    "selection_phase": "topk_recheck_final_best",
                    "selection_rule": "success_rate_then_coverage_then_reward",
                    "selection_row": final_best_selection_row,
                    "topk_recheck": int(topk),
                    "validation_candidate_count": int(len(model_select_rows)),
                },
            )

            last_checkpoint_matches = [
                row for row in best_recheck_rows
                if int(row.get("env_steps", -1)) == int(env_steps)
            ]
            if last_checkpoint_matches:
                last_checkpoint_diagnostic_row = last_checkpoint_matches[-1]
            else:
                last_validation_matches = [
                    row for row in model_select_rows
                    if int(row.get("env_steps", -1)) == int(env_steps)
                ]
                last_checkpoint_diagnostic_row = last_validation_matches[-1] if last_validation_matches else None

        if best_checkpoint_path is None:
            raise RuntimeError("best checkpoint path was not resolved before final formal test")

        best_payload = _load_checkpoint_into_model(online_net, best_checkpoint_path, cfg.device)
        best_env_steps = int(best_payload.get("env_steps", env_steps))
        best_train_episode_idx = int(best_payload.get("train_episode_idx", train_phase_episodes))
        probe_source = "best_checkpoint"
        probe = evaluator.evaluate(
            online_net,
            num_episodes=int(max(1, cfg.final_greedy_episodes)),
            seed_base=final_probe_seed_base,
        )
        probe_row = _build_eval_row(
            tag="final_probe",
            source=probe_source,
            budget_mode=budget_mode,
            env_steps=best_env_steps,
            episode_idx=int(collector.total_episodes),
            train_episode_idx=best_train_episode_idx,
            completed_train_episodes=int(train_phase_episodes),
            learner_steps=int(learner.learn_steps),
            eval_result=probe,
            checkpoint_path=_relative_checkpoint_path(run_dir, best_checkpoint_path),
            seed_base=final_probe_seed_base,
        )
        logger.log_final_probe(probe_row)
        print(
            "[final_probe] "
            f"source={probe_source} checkpoint={probe_row['checkpoint_path']} "
            f"env_steps={probe_row['env_steps']} episodes={probe_row['eval_episodes']} "
            f"train_eps={int(train_phase_episodes)} "
            f"mean_reward={probe_row['eval_mean_reward']:.4f} mean_cov={probe_row['eval_mean_coverage']:.4f} "
            f"success_rate={probe_row['eval_success_rate']:.4f} mean_len={probe_row['eval_mean_episode_length']:.2f} "
            f"blocks={probe_row['eval_mean_accessible_block_count']:.2f}"
        )
    if bool(cfg.save_train_representative_trajectories):
        try:
            trajectory_plot_paths.extend(
                save_episode_trajectory_plots(
                    run_dir,
                    train_trace_episodes,
                    prefix="train_postgate_fail",
                    max_episodes=max(1, len(train_trace_episodes)),
                    selection_mode="train_postgate_failures",
                    gate_window=int(cfg.recent_episode_window),
                    coverage_target=float(cfg.coverage_stop_threshold),
                )
            )
        except Exception as exc:
            print(
                "[warning] optional artifact export failed: "
                f"train representative trajectories "
                f"({type(exc).__name__}: {exc})"
            )
    if bool(cfg.save_train_special_trajectories):
        try:
            trajectory_plot_paths.extend(
                save_train_special_trajectory_plots(
                    run_dir,
                    train_trace_episodes,
                    highcov_timeout_min_coverage=float(cfg.special_highcov_timeout_min_coverage),
                    highcov_timeout_max_plots=int(cfg.special_highcov_timeout_max_plots),
                    long_success_gate_coverage=float(cfg.special_long_success_gate_coverage),
                    long_success_gate_window=int(cfg.special_long_success_gate_window),
                    long_success_min_length=int(cfg.special_long_success_min_length),
                    long_success_percentile=float(cfg.special_long_success_percentile),
                    long_success_max_plots=int(cfg.special_long_success_max_plots),
                    lowcov_gate_coverage=float(cfg.special_lowcov_gate_coverage),
                    lowcov_gate_window=int(cfg.special_lowcov_gate_window),
                    lowcov_absolute_threshold=float(cfg.special_lowcov_absolute_threshold),
                    lowcov_local_drop_margin=float(cfg.special_lowcov_local_drop_margin),
                    lowcov_max_plots=int(cfg.special_lowcov_max_plots),
                )
            )
        except Exception as exc:
            print(
                "[warning] optional artifact export failed: "
                f"train special trajectories "
                f"({type(exc).__name__}: {exc})"
            )
    if bool(cfg.save_final_probe_trajectories) and not probe:
        print("[warning] final probe trajectories requested but skipped by train_side_only_tuning.")
    if bool(cfg.save_final_probe_trajectories) and probe:
        try:
            trajectory_plot_paths.extend(
                save_episode_trajectory_plots(
                    run_dir,
                    probe.get("episodes", []),
                    prefix="final_probe",
                    max_episodes=1,
                    selection_mode="lowest_coverage",
                    gate_window=int(cfg.recent_episode_window),
                    coverage_target=float(cfg.coverage_stop_threshold),
                )
            )
        except Exception as exc:
            print(
                "[warning] optional artifact export failed: "
                f"final probe trajectories "
                f"({type(exc).__name__}: {exc})"
            )
    if bool(cfg.generate_plots_on_finish):
        generated_plots = generate_all_plots(run_dir)
    else:
        generated_plots = []

    recent_summary = summarize_recent_episodes(recent_eps)
    recent_train_row = {
        "budget_mode": budget_mode,
        "env_steps": int(env_steps),
        "episode_idx": int(collector.total_episodes),
        "train_episode_idx": int(collector.total_episodes),
        "completed_train_episodes": int(train_phase_episodes),
        "replay_size": int(len(replay)),
        "epsilon": float(linear_epsilon(env_steps, cfg)),
        "loss": float(last_train_metrics["loss"]),
        "q_mean": float(last_train_metrics["q_mean"]),
        "target_q_mean": float(last_train_metrics["target_q_mean"]),
        "td_abs_mean": float(last_train_metrics["td_abs_mean"]),
        "grad_norm": float(last_train_metrics["grad_norm"]),
        "learner_steps": int(learner.learn_steps),
        "recent_mean_reward": float(recent_summary["mean_reward"]),
        "recent_mean_coverage": float(recent_summary["mean_coverage"]),
        "recent_success_rate": float(recent_summary["success_rate"]),
        "recent_mean_episode_length": float(recent_summary["mean_length"]),
        "recent_mean_repeat_visit_ratio": float(recent_summary["mean_repeat_visit_ratio"]),
        **{
            f"recent_{field}": float(recent_summary[field])
            for field in SEMANTIC_EPISODE_FIELDS
        },
        **{
            f"recent_{field}": float(recent_summary[field])
            for field in REWARD_BREAKDOWN_FIELDS
        },
        **{
            f"recent_{field}": float(recent_summary[field])
            for field in REWARD_EVENT_SUMMARY_FIELDS
        },
        **{
            f"recent_{field}": float(recent_summary[field])
            for field in DERIVED_TRAIN_DIAGNOSTIC_FIELDS
        },
    }
    total_runtime_sec = time.perf_counter() - run_start_time
    total_runtime_sec_int = int(round(total_runtime_sec))
    hours, rem = divmod(total_runtime_sec_int, 3600)
    minutes, seconds = divmod(rem, 60)
    total_runtime_hms = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    print("=" * 72)
    print("Training Summary")
    print(f"run_dir: {run_dir}")
    print(f"budget_mode: {budget_mode}")
    print(f"final_env_steps: {env_steps}")
    print(f"completed_train_episodes: {int(train_phase_episodes)}")
    print(f"total_runtime_sec: {total_runtime_sec:.2f}")
    print(f"total_runtime_hms: {total_runtime_hms}")
    print(
        "recent_train_episodes: "
        f"reward={recent_summary['mean_reward']:.4f}, "
        f"coverage={recent_summary['mean_coverage']:.4f}, "
        f"success={recent_summary['success_rate']:.4f}, "
        f"length={recent_summary['mean_length']:.2f}, "
        f"repeat={recent_summary['mean_repeat_visit_ratio']:.4f}"
    )

    if probe_row is None:
        print("final_probe: skipped intentionally by train_side_only_tuning")
    else:
        print(
            "final_probe: "
            f"source={probe_source}, checkpoint={probe_row['checkpoint_path']}, reward={probe_row['eval_mean_reward']:.4f}, "
            f"coverage={probe_row['eval_mean_coverage']:.4f}, "
            f"success={probe_row['eval_success_rate']:.4f}, "
            f"length={probe_row['eval_mean_episode_length']:.2f}, "
            f"blocks={probe_row['eval_mean_accessible_block_count']:.2f}"
        )

    print(f"checkpoint_last: {last_ckpt_path}")
    print(f"checkpoint_best: {best_checkpoint_path}")
    print(f"train_episode_csv: {logger.train_episode_csv}")
    print(f"final_probe_csv: {logger.final_probe_csv}")
    print(f"model_select_eval_csv: {logger.model_select_eval_csv}")
    print(f"best_recheck_eval_csv: {logger.best_recheck_eval_csv}")
    print(f"posthoc_candidate_scores_csv: {run_dir / 'logs' / 'posthoc_candidate_scores.csv'}")
    print(f"posthoc_selection_summary_json: {run_dir / 'logs' / 'posthoc_selection_summary.json'}")
    print(f"final_probe_summary_json: {run_dir / 'logs' / 'final_probe_summary.json'}")
    print(f"formal_selection_manifest_json: {run_dir / 'logs' / 'formal_selection_manifest.json'}")
    print(f"train_step_csv: {logger.train_step_csv}")
    if len(generated_plots) > 0:
        print(f"plots_dir: {run_dir / 'plots'}")
    if len(trajectory_plot_paths) > 0:
        print(f"trajectories_dir: {run_dir / 'trajectories'}")
    structured_artifacts = write_formal_run_artifacts(
        run_dir=run_dir,
        cfg=cfg,
        run_mode=run_mode,
        recent_train_row=recent_train_row,
        final_probe_row=probe_row,
        last_checkpoint_env_steps=int(env_steps),
        last_checkpoint_train_episode_idx=int(train_phase_episodes),
        best_checkpoint_env_steps=best_env_steps,
        best_checkpoint_train_episode_idx=best_train_episode_idx,
        final_probe_source=probe_source,
        total_runtime_sec=float(total_runtime_sec),
        total_runtime_hms=total_runtime_hms,
        collector=collector,
        learner=learner,
        replay=replay,
        state_adapter=collector.state_adapter,
        source_of_truth_repo=str(Path(__file__).resolve().parent),
        raw_argv=list(sys.argv),
        runtime_info=reproducibility_runtime_info,
        model_select_rows=model_select_rows,
        best_recheck_rows=best_recheck_rows,
        best_checkpoint_selection_row=final_best_selection_row,
        last_checkpoint_diagnostic_row=last_checkpoint_diagnostic_row,
    )
    print(f"metric_snapshot_json: {structured_artifacts['metric_snapshot']}")
    print(f"benchmark_summary_json: {structured_artifacts['benchmark_summary']}")
    print(f"config_snapshot_json: {structured_artifacts['config_snapshot']}")
    print(f"reproducibility_contract_json: {structured_artifacts['reproducibility_contract']}")
    print(f"artifact_index_json: {structured_artifacts['artifact_index']}")
    print("=" * 72)
    return run_dir


def parse_args() -> TrainConfig:
    p = argparse.ArgumentParser(description="Double DQN training with post-hoc formal checkpoint selection")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--experiment-id", type=str, default="A_new")
    p.add_argument("--method-id", type=str, default=None)
    p.add_argument("--method-name", type=str, default=None)
    p.add_argument("--run-stage", type=str, choices=("smoke", "pilot", "formal"), default=None)
    p.add_argument("--ablation-group", type=str, default="none")
    p.add_argument("--ablation-id", type=str, default="none")
    p.add_argument("--ablation-name", type=str, default="none")
    p.add_argument("--channel-ablation", type=str, default="none")
    p.add_argument(
        "--zeroed-advantage-channels",
        type=str,
        default="",
        help="Comma-separated A_new advantage canvas channels to zero after state construction.",
    )
    p.add_argument(
        "--value-replacement-strategy",
        type=str,
        choices=VALUE_REPLACEMENT_STRATEGIES,
        default=VALUE_REPLACEMENT_STRATEGY_NONE,
        help="A_new structural ablation input replacement for the value branch.",
    )
    p.add_argument(
        "--no-value-tree",
        action="store_true",
        help="Use a zero-value-state tensor adapter while preserving the ExplorationQNetwork interface.",
    )
    p.add_argument(
        "--enable-amp",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=
        "Performance-side overhead toggle only; enables AMP in the learner hot path without changing algorithm or metric definitions.",
    )
    p.add_argument(
        "--amp-dtype",
        type=str,
        default="fp16",
        choices=("fp16", "bf16"),
        help="AMP dtype for performance tuning only; it does not redefine the algorithm or metrics.",
    )
    p.add_argument(
        "--enable-inference-amp",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Greedy-policy inference AMP toggle only; default stays off so baseline training/eval semantics "
            "remain unchanged unless you opt in."
        ),
    )
    p.add_argument(
        "--enable-torch-compile",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=
        "Performance-side overhead toggle only; wraps the online net with torch.compile without changing metrics.",
    )
    p.add_argument(
        "--compile-mode",
        type=str,
        default="default",
        help=
        "torch.compile mode for performance tuning only; algorithm behavior and metric definitions stay the same.",
    )
    p.add_argument(
        "--enable-cudnn-benchmark",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=
        "Performance-side overhead toggle only; controls cuDNN benchmark autotuning without changing training logic.",
    )
    p.add_argument(
        "--enable-tf32",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=
        "Performance-side overhead toggle only; controls TF32 backends without changing algorithm logic.",
    )
    p.add_argument(
        "--strict-reproducibility",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Optional runtime determinism mode: disables cuDNN benchmark / TF32 on CUDA and enables "
            "deterministic algorithm guards where supported. This does not replace fixed episode seeds."
        ),
    )
    p.add_argument(
        "--deterministic-warn-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Controls warn_only for torch.use_deterministic_algorithms in strict reproducibility mode; "
            "default warns instead of hard failing."
        ),
    )
    p.add_argument(
        "--enable-channels-last",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Tensor-layout toggle only; uses channels-last tensors on CUDA conv paths without changing "
            "training logic or metric definitions."
        ),
    )
    p.add_argument(
        "--episode-print-interval",
        type=int,
        default=10,
        help="Stdout throttling only; set 1 to print every episode. CSV episode metrics are unaffected.",
    )
    p.add_argument(
        "--train-print-interval",
        type=int,
        default=2000,
        help="Stdout throttling only; separate from --log-interval and does not affect CSV metrics.",
    )
    p.add_argument(
        "--save-train-representative-trajectories",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Save all post-gate failed train episodes with trajectory/belief overlays.",
    )
    p.add_argument(
        "--save-train-special-trajectories",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Save highcov-timeout, long-success, and low-coverage special train episodes into separate folders.",
    )
    p.add_argument(
        "--save-final-probe-trajectories",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Plot-saving toggle only; final probe metrics/logic are unchanged.",
    )
    p.add_argument(
        "--generate-plots-on-finish",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="End-of-run plotting toggle only; training, evaluation, and checkpoint logic are unchanged.",
    )
    p.add_argument(
        "--enable-collector-timing",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Profiling only; collector timing does not change rollout or reward semantics.",
    )
    p.add_argument(
        "--enable-learner-timing",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Profiling only; learner timing does not change DDQN updates or metric definitions.",
    )
    p.add_argument(
        "--enable-replay-timing",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Profiling only; replay timing does not change storage or sampling semantics.",
    )
    p.add_argument(
        "--enable-state-adapter-timing",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Profiling only; adapter timing does not change state tensor semantics.",
    )
    p.add_argument(
        "--enable-cummap-timing",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Profiling only; cumulative-map timing does not change map/frontier definitions.",
    )
    p.add_argument(
        "--enable-shared-semantic-timing",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Profiling only; shared-semantic timing does not change block/entry definitions.",
    )
    p.add_argument(
        "--enable-advantage-state-timing",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Profiling only; advantage-state timing does not change canvas semantics.",
    )
    p.add_argument(
        "--enable-value-state-timing",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Profiling only; value-state timing does not change block-tree semantics.",
    )
    p.add_argument(
        "--timing-log-interval",
        type=int,
        default=2000,
        help="Stdout profiling cadence only; 0 disables periodic timing summaries.",
    )
    p.add_argument(
        "--debug-check-incremental-frontier",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Debug-only rollout check: compare incrementally maintained frontier against a full "
            "recompute after reset/update and raise on mismatch."
        ),
    )
    p.add_argument(
        "--learner-debug-stats-every",
        type=int,
        default=8,
        help=(
            "Metric-sync throttling only; returns full learner debug scalars every N learner steps "
            "without changing optimization behavior."
        ),
    )
    p.add_argument(
        "--prefer-batch-replay-add",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Replay write-path optimization only; transition semantics stay unchanged.",
    )
    p.add_argument(
        "--profile",
        action="store_true",
        help="Enable all timing/profiling flags together without changing any training hyperparameters.",
    )
    p.add_argument(
        "--fast-cuda",
        action="store_true",
        help=(
            "Enable the experimental fast CUDA runtime path: AMP, inference AMP, torch.compile with safe fallback, "
            "channels-last, cuDNN benchmark, and TF32. This is kept for optional A/B testing and is not the "
            "default recommended training path on the current machine/model."
        ),
    )

    p.add_argument("--budget-mode", type=str, choices=("env_steps", "episodes"), default="env_steps")
    p.add_argument("--total-env-steps", type=int, default=500_000)
    p.add_argument("--total-train-episodes", type=int, default=600)
    p.add_argument("--warmup-steps", type=int, default=4_000)
    p.add_argument("--warmup-episodes", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--min-replay-size", type=int, default=8_000)
    p.add_argument("--replay-capacity", type=int, default=100_000)
    p.add_argument("--collect-steps-per-iter", type=int, default=16)
    p.add_argument("--learner-updates-per-iter", type=int, default=1)
    p.add_argument("--train-every-env-steps", type=int, default=16)
    p.add_argument("--n-step", type=int, default=3)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--target-update-interval", type=int, default=1_000)
    p.add_argument("--learning-rate", type=float, default=1.0e-4)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--grad-clip-norm", type=float, default=10.0)

    p.add_argument("--epsilon-start", type=float, default=1.0)
    p.add_argument("--epsilon-end", type=float, default=0.04)
    p.add_argument("--epsilon-decay-steps", type=int, default=240_000)

    p.add_argument("--recent-episode-window", type=int, default=100)
    p.add_argument(
        "--formal-protocol",
        type=str,
        default=POSTHOC_PROTOCOL_NAME,
        choices=(POSTHOC_PROTOCOL_NAME, "formal_best_checkpoint_v3"),
        help="Formal protocol lane. Default uses post-hoc train-side candidate selection.",
    )
    p.add_argument(
        "--train-side-only-tuning",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Formal tuning mode that preserves train-side logging and endpoint checkpoints while skipping "
            "posthoc selection, candidate scoring, final_probe, best-vs-last comparison, and automatic winner selection. "
            "Enabled by default to match the legacy A/F1 formal training contract."
        ),
    )
    p.add_argument(
        "--final-greedy-episodes",
        type=int,
        default=100,
        help=(
            "Held-out greedy episodes for the formal final_probe on best.pt. "
            "Default 100 under the best-checkpoint protocol."
        ),
    )
    p.add_argument(
        "--use-fixed-train-episode-seeds",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Bind each train episode index to an explicit seed so repeated runs see the same map stream.",
    )
    p.add_argument(
        "--fixed-train-episode-seed-base",
        type=int,
        default=20259323,
        help="Base seed for the train-episode map sequence when fixed train episode seeds are enabled.",
    )
    p.add_argument(
        "--use-fixed-eval-seeds",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Legacy-named toggle that now only controls fixed held-out map seeds for final_probe.",
    )
    p.add_argument(
        "--fixed-final-probe-seed-base",
        type=int,
        default=20261323,
        help="Base seed for the final-probe evaluation map set.",
    )
    p.add_argument("--periodic-checkpoint-interval-env-steps", type=int, default=20_000)
    p.add_argument("--posthoc-candidate-start-env-steps", type=int, default=200_000)
    p.add_argument(
        "--posthoc-candidate-end-env-steps",
        type=int,
        default=0,
        help="End of post-hoc candidate range; 0 means --total-env-steps.",
    )
    p.add_argument("--posthoc-selection-window-env-steps", type=int, default=40_000)
    p.add_argument("--posthoc-final-probe-topk", type=int, default=3)
    p.add_argument(
        "--enable-best-checkpoint-selection",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Legacy formal_best_checkpoint_v3 only: enable periodic validation, top-k recheck, and best.pt selection.",
    )
    p.add_argument("--best-checkpoint-selection-start-env-steps", type=int, default=300_000)
    p.add_argument("--best-checkpoint-selection-interval-env-steps", type=int, default=20_000)
    p.add_argument("--best-checkpoint-validation-episodes", type=int, default=24)
    p.add_argument("--best-checkpoint-topk-recheck", type=int, default=3)
    p.add_argument("--best-checkpoint-recheck-episodes", type=int, default=50)
    p.add_argument(
        "--use-fixed-model-select-seeds",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use the dedicated model-selection seed set for validation and recheck.",
    )
    p.add_argument(
        "--fixed-model-select-seed-base",
        type=int,
        default=20262323,
        help="Base seed for checkpoint validation; intentionally separate from final_probe seeds.",
    )

    p.add_argument("--log-interval", type=int, default=500)
    p.add_argument("--log-interval-episodes", type=int, default=10)
    p.add_argument("--train-print-interval-episodes", type=int, default=20)
    p.add_argument("--rows", type=int, default=40)
    p.add_argument("--cols", type=int, default=60)
    p.add_argument("--obs-size", type=int, default=6)
    p.add_argument("--scan-radius", type=int, default=10)
    p.add_argument("--max-accessible-blocks", type=int, default=16)
    p.add_argument("--max-entries-per-block", type=int, default=8)
    p.add_argument("--max-episode-steps", type=int, default=600)
    p.add_argument("--coverage-stop-threshold", type=float, default=0.95)
    p.add_argument("--obstacle-ratio", type=float, default=0.20)
    p.add_argument(
        "--reward-info-scale",
        type=float,
        default=3.1,
        help="weighted information gain scale under the fixed half-perimeter normalization rule",
    )
    p.add_argument(
        "--reward-obstacle-weight",
        type=float,
        default=0.20,
        help="obstacle reveal weight inside the fixed half-perimeter information gain",
    )
    p.add_argument("--reward-step-penalty", type=float, default=0.02, help="step penalty")
    p.add_argument("--reward-terminal-bonus", type=float, default=20.0, help="terminal success bonus")
    p.add_argument(
        "--reward-revisit-penalty",
        type=float,
        default=0.10,
        help="recent revisit penalty; its horizon is fixed to trajectory_history_steps",
    )
    p.add_argument(
        "--reward-turn-penalty-scale",
        type=float,
        default=0.05,
        help="overall turn penalty scale; the final per-step penalty is this scale times the explicit angle weight",
    )
    p.add_argument("--reward-turn-weight-45", type=float, default=0.0, help="turn-penalty weight for 45-degree turns")
    p.add_argument("--reward-turn-weight-90", type=float, default=(1.0 / 3.0), help="turn-penalty weight for 90-degree turns")
    p.add_argument("--reward-turn-weight-135", type=float, default=(2.0 / 3.0), help="turn-penalty weight for 135-degree turns")
    p.add_argument("--reward-turn-weight-180", type=float, default=1.0, help="turn-penalty weight for 180-degree turns")
    p.add_argument("--reward-timeout-penalty", type=float, default=8.0, help="timeout penalty")
    p.add_argument("--special-highcov-timeout-min-coverage", type=float, default=0.85)
    p.add_argument("--special-highcov-timeout-max-plots", type=int, default=5)
    p.add_argument("--special-long-success-gate-coverage", type=float, default=0.80)
    p.add_argument("--special-long-success-gate-window", type=int, default=100)
    p.add_argument("--special-long-success-min-length", type=int, default=350)
    p.add_argument("--special-long-success-percentile", type=float, default=85.0)
    p.add_argument("--special-long-success-max-plots", type=int, default=5)
    p.add_argument("--special-lowcov-gate-coverage", type=float, default=0.80)
    p.add_argument("--special-lowcov-gate-window", type=int, default=100)
    p.add_argument("--special-lowcov-absolute-threshold", type=float, default=0.75)
    p.add_argument("--special-lowcov-local-drop-margin", type=float, default=0.12)
    p.add_argument("--special-lowcov-max-plots", type=int, default=5)
    p.add_argument(
        "--advantage-canvas-schema",
        type=str,
        choices=ADVANTAGE_CANVAS_SCHEMAS,
        default=ADVANTAGE_CANVAS_SCHEMA_FINAL_4CH_NO_FRONTIER_RASTER,
        help="Advantage canvas schema. Main only supports final_4ch_no_frontier_raster.",
    )

    p.add_argument("--output-root", type=str, default="outputs")
    p.add_argument("--run-name", type=str, default="ddqn_explore_vscode_stage5")

    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()

    enable_collector_timing = bool(args.enable_collector_timing)
    enable_learner_timing = bool(args.enable_learner_timing)
    enable_replay_timing = bool(args.enable_replay_timing)
    enable_state_adapter_timing = bool(args.enable_state_adapter_timing)
    enable_cummap_timing = bool(args.enable_cummap_timing)
    enable_shared_semantic_timing = bool(args.enable_shared_semantic_timing)
    enable_advantage_state_timing = bool(args.enable_advantage_state_timing)
    enable_value_state_timing = bool(args.enable_value_state_timing)

    if bool(args.profile):
        enable_collector_timing = True
        enable_learner_timing = True
        enable_replay_timing = True
        enable_state_adapter_timing = True
        enable_cummap_timing = True
        enable_shared_semantic_timing = True
        enable_advantage_state_timing = True
        enable_value_state_timing = True

    enable_amp = bool(args.enable_amp)
    enable_inference_amp = bool(args.enable_inference_amp)
    enable_torch_compile = bool(args.enable_torch_compile)
    enable_cudnn_benchmark = bool(args.enable_cudnn_benchmark)
    enable_tf32 = bool(args.enable_tf32)
    strict_reproducibility = bool(args.strict_reproducibility)
    deterministic_warn_only = bool(args.deterministic_warn_only)
    enable_channels_last = bool(args.enable_channels_last)
    if bool(args.fast_cuda):
        enable_amp = True
        enable_inference_amp = True
        enable_torch_compile = True
        enable_cudnn_benchmark = True
        enable_tf32 = True
        enable_channels_last = True

    enable_best_checkpoint_selection = (
        bool(args.enable_best_checkpoint_selection)
        if args.enable_best_checkpoint_selection is not None
        else str(args.formal_protocol) == "formal_best_checkpoint_v3"
    )
    advantage_canvas_channels = advantage_canvas_channels_for_schema(args.advantage_canvas_schema)
    frontier_raster_used = advantage_canvas_uses_frontier_raster(args.advantage_canvas_schema)
    resolved_run_stage = str(args.run_stage or ("smoke" if args.smoke else "formal"))
    resolved_method_id = str(args.method_id or args.experiment_id)
    resolved_method_name = str(args.method_name or args.advantage_canvas_schema)
    zeroed_advantage_channels = normalize_zeroed_advantage_channels(
        args.zeroed_advantage_channels,
        schema=args.advantage_canvas_schema,
    )
    channel_ablation = str(args.channel_ablation or "none")
    value_replacement_strategy = normalize_value_replacement_strategy(args.value_replacement_strategy)
    no_value_tree = bool(args.no_value_tree) or (
        value_replacement_strategy == VALUE_REPLACEMENT_STRATEGY_ZERO_VALUE_STATE
    )
    ablation_group = str(args.ablation_group or "none")
    ablation_id = str(args.ablation_id or "none")
    ablation_name = str(args.ablation_name or "none")
    is_ablation = bool(
        no_value_tree
        or ablation_group != "none"
        or ablation_id != "none"
        or ablation_name != "none"
        or channel_ablation != "none"
        or zeroed_advantage_channels
    )
    dummy_value_block_shape = (
        (max(1, int(args.max_accessible_blocks)), VALUE_BLOCK_FEATURE_COUNT)
        if no_value_tree else ()
    )
    dummy_value_entry_shape = (
        (max(1, int(args.max_accessible_blocks)), max(1, int(args.max_entries_per_block)), VALUE_ENTRY_FEATURE_COUNT)
        if no_value_tree else ()
    )
    dummy_value_mask_rule = "all_false" if no_value_tree else "none"

    if args.smoke:
        return TrainConfig(
            device=args.device,
            seed=args.seed,
            experiment_id=args.experiment_id,
            method_id=resolved_method_id,
            method_name=resolved_method_name,
            ablation_group=ablation_group,
            ablation_id=ablation_id,
            ablation_name=ablation_name,
            channel_ablation=channel_ablation,
            zeroed_advantage_channels=zeroed_advantage_channels,
            is_ablation=is_ablation,
            value_replacement_strategy=value_replacement_strategy,
            value_tree_enabled=not no_value_tree,
            value_tree_unchanged=not no_value_tree,
            no_value_tree=no_value_tree,
            dummy_value_tensors_for_interface=no_value_tree,
            value_tensors_used_by_model=True,
            dummy_value_block_shape=dummy_value_block_shape,
            dummy_value_entry_shape=dummy_value_entry_shape,
            dummy_value_mask_rule=dummy_value_mask_rule,
            run_stage=resolved_run_stage,
            enable_amp=enable_amp,
            enable_inference_amp=enable_inference_amp,
            amp_dtype=args.amp_dtype,
            enable_torch_compile=enable_torch_compile,
            compile_mode=args.compile_mode,
            enable_cudnn_benchmark=enable_cudnn_benchmark,
            enable_tf32=enable_tf32,
            strict_reproducibility=strict_reproducibility,
            deterministic_warn_only=deterministic_warn_only,
            enable_channels_last=enable_channels_last,
            episode_print_interval=max(0, args.episode_print_interval),
            train_print_interval=max(0, args.train_print_interval),
            save_train_representative_trajectories=args.save_train_representative_trajectories,
            save_train_special_trajectories=args.save_train_special_trajectories,
            save_final_probe_trajectories=args.save_final_probe_trajectories,
            generate_plots_on_finish=args.generate_plots_on_finish,
            enable_collector_timing=enable_collector_timing,
            enable_learner_timing=enable_learner_timing,
            enable_replay_timing=enable_replay_timing,
            enable_state_adapter_timing=enable_state_adapter_timing,
            enable_cummap_timing=enable_cummap_timing,
            enable_shared_semantic_timing=enable_shared_semantic_timing,
            enable_advantage_state_timing=enable_advantage_state_timing,
            enable_value_state_timing=enable_value_state_timing,
            timing_log_interval=max(0, args.timing_log_interval),
            debug_check_incremental_frontier=bool(args.debug_check_incremental_frontier),
            prefer_batch_replay_add=args.prefer_batch_replay_add,
            learner_debug_stats_every=max(1, args.learner_debug_stats_every),
            rows=30,
            cols=40,
            scan_radius=args.scan_radius,
            max_accessible_blocks=max(1, args.max_accessible_blocks),
            max_entries_per_block=max(1, args.max_entries_per_block),
            budget_mode=str(args.budget_mode),
            total_env_steps=180,
            total_train_episodes=max(1, args.total_train_episodes),
            warmup_steps=40,
            warmup_episodes=max(0, args.warmup_episodes),
            collect_steps_per_iter=max(1, args.collect_steps_per_iter),
            learner_updates_per_iter=max(1, args.learner_updates_per_iter),
            train_every_env_steps=max(1, args.train_every_env_steps),
            batch_size=16,
            min_replay_size=32,
            replay_capacity=2_000,
            n_step=max(1, args.n_step),
            gamma=args.gamma,
            target_update_interval=20,
            learning_rate=args.learning_rate,
            grad_clip_norm=args.grad_clip_norm,
            epsilon_start=args.epsilon_start,
            epsilon_end=args.epsilon_end,
            epsilon_decay_steps=max(1, args.epsilon_decay_steps),
            recent_episode_window=10,
            formal_protocol=str(args.formal_protocol),
            train_side_only_tuning=bool(args.train_side_only_tuning),
            final_greedy_episodes=1,
            train_print_interval_episodes=max(0, args.train_print_interval_episodes),
            use_fixed_train_episode_seeds=bool(args.use_fixed_train_episode_seeds),
            fixed_train_episode_seed_base=int(args.fixed_train_episode_seed_base),
            use_fixed_eval_seeds=True,
            fixed_final_probe_seed_base=int(args.fixed_final_probe_seed_base),
            periodic_checkpoint_interval_env_steps=60,
            posthoc_candidate_start_env_steps=60,
            posthoc_candidate_end_env_steps=0,
            posthoc_selection_window_env_steps=60,
            posthoc_final_probe_topk=max(1, min(3, args.posthoc_final_probe_topk)),
            enable_best_checkpoint_selection=enable_best_checkpoint_selection,
            best_checkpoint_selection_start_env_steps=60,
            best_checkpoint_selection_interval_env_steps=60,
            best_checkpoint_validation_episodes=1,
            best_checkpoint_topk_recheck=1,
            best_checkpoint_recheck_episodes=1,
            use_fixed_model_select_seeds=bool(args.use_fixed_model_select_seeds),
            fixed_model_select_seed_base=int(args.fixed_model_select_seed_base),
            log_interval=20,
            log_interval_episodes=max(0, args.log_interval_episodes),
            max_episode_steps=80,
            reward_info_scale=args.reward_info_scale,
            reward_obstacle_weight=args.reward_obstacle_weight,
            reward_step_penalty=args.reward_step_penalty,
            reward_terminal_bonus=args.reward_terminal_bonus,
            reward_revisit_penalty=args.reward_revisit_penalty,
            reward_turn_penalty_scale=args.reward_turn_penalty_scale,
            reward_turn_weight_45=args.reward_turn_weight_45,
            reward_turn_weight_90=args.reward_turn_weight_90,
            reward_turn_weight_135=args.reward_turn_weight_135,
            reward_turn_weight_180=args.reward_turn_weight_180,
            reward_timeout_penalty=args.reward_timeout_penalty,
            special_highcov_timeout_min_coverage=args.special_highcov_timeout_min_coverage,
            special_highcov_timeout_max_plots=max(0, args.special_highcov_timeout_max_plots),
            special_long_success_gate_coverage=args.special_long_success_gate_coverage,
            special_long_success_gate_window=max(1, args.special_long_success_gate_window),
            special_long_success_min_length=max(1, args.special_long_success_min_length),
            special_long_success_percentile=args.special_long_success_percentile,
            special_long_success_max_plots=max(0, args.special_long_success_max_plots),
            special_lowcov_gate_coverage=args.special_lowcov_gate_coverage,
            special_lowcov_gate_window=max(1, args.special_lowcov_gate_window),
            special_lowcov_absolute_threshold=args.special_lowcov_absolute_threshold,
            special_lowcov_local_drop_margin=args.special_lowcov_local_drop_margin,
            special_lowcov_max_plots=max(0, args.special_lowcov_max_plots),
            advantage_canvas_schema=str(args.advantage_canvas_schema),
            advantage_canvas_channels=advantage_canvas_channels,
            advantage_canvas_channel_count=len(advantage_canvas_channels),
            frontier_raster_used=bool(frontier_raster_used),
            output_root=args.output_root,
            run_name=args.run_name,
        )

    return TrainConfig(
        device=args.device,
        seed=args.seed,
        experiment_id=args.experiment_id,
        method_id=resolved_method_id,
        method_name=resolved_method_name,
        ablation_group=ablation_group,
        ablation_id=ablation_id,
        ablation_name=ablation_name,
        channel_ablation=channel_ablation,
        zeroed_advantage_channels=zeroed_advantage_channels,
        is_ablation=is_ablation,
        value_replacement_strategy=value_replacement_strategy,
        value_tree_enabled=not no_value_tree,
        value_tree_unchanged=not no_value_tree,
        no_value_tree=no_value_tree,
        dummy_value_tensors_for_interface=no_value_tree,
        value_tensors_used_by_model=True,
        dummy_value_block_shape=dummy_value_block_shape,
        dummy_value_entry_shape=dummy_value_entry_shape,
        dummy_value_mask_rule=dummy_value_mask_rule,
        run_stage=resolved_run_stage,
        enable_amp=enable_amp,
        enable_inference_amp=enable_inference_amp,
        amp_dtype=args.amp_dtype,
        enable_torch_compile=enable_torch_compile,
        compile_mode=args.compile_mode,
        enable_cudnn_benchmark=enable_cudnn_benchmark,
        enable_tf32=enable_tf32,
        strict_reproducibility=strict_reproducibility,
        deterministic_warn_only=deterministic_warn_only,
        enable_channels_last=enable_channels_last,
        episode_print_interval=max(0, args.episode_print_interval),
        train_print_interval=max(0, args.train_print_interval),
        save_train_representative_trajectories=args.save_train_representative_trajectories,
        save_train_special_trajectories=args.save_train_special_trajectories,
        save_final_probe_trajectories=args.save_final_probe_trajectories,
        generate_plots_on_finish=args.generate_plots_on_finish,
        enable_collector_timing=enable_collector_timing,
        enable_learner_timing=enable_learner_timing,
        enable_replay_timing=enable_replay_timing,
        enable_state_adapter_timing=enable_state_adapter_timing,
        enable_cummap_timing=enable_cummap_timing,
        enable_shared_semantic_timing=enable_shared_semantic_timing,
        enable_advantage_state_timing=enable_advantage_state_timing,
        enable_value_state_timing=enable_value_state_timing,
        timing_log_interval=max(0, args.timing_log_interval),
        debug_check_incremental_frontier=bool(args.debug_check_incremental_frontier),
        prefer_batch_replay_add=args.prefer_batch_replay_add,
        learner_debug_stats_every=max(1, args.learner_debug_stats_every),
        budget_mode=str(args.budget_mode),
        total_env_steps=args.total_env_steps,
        total_train_episodes=max(1, args.total_train_episodes),
        warmup_steps=args.warmup_steps,
        warmup_episodes=max(0, args.warmup_episodes),
        collect_steps_per_iter=max(1, args.collect_steps_per_iter),
        learner_updates_per_iter=max(1, args.learner_updates_per_iter),
        train_every_env_steps=max(1, args.train_every_env_steps),
        batch_size=args.batch_size,
        min_replay_size=args.min_replay_size,
        replay_capacity=args.replay_capacity,
        n_step=max(1, args.n_step),
        gamma=args.gamma,
        target_update_interval=args.target_update_interval,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        grad_clip_norm=args.grad_clip_norm,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay_steps=max(1, args.epsilon_decay_steps),
        recent_episode_window=max(1, args.recent_episode_window),
        formal_protocol=str(args.formal_protocol),
        train_side_only_tuning=bool(args.train_side_only_tuning),
        final_greedy_episodes=max(1, args.final_greedy_episodes),
        train_print_interval_episodes=max(0, args.train_print_interval_episodes),
        use_fixed_train_episode_seeds=bool(args.use_fixed_train_episode_seeds),
        fixed_train_episode_seed_base=int(args.fixed_train_episode_seed_base),
        use_fixed_eval_seeds=bool(args.use_fixed_eval_seeds),
        fixed_final_probe_seed_base=int(args.fixed_final_probe_seed_base),
        periodic_checkpoint_interval_env_steps=max(1, args.periodic_checkpoint_interval_env_steps),
        posthoc_candidate_start_env_steps=max(0, args.posthoc_candidate_start_env_steps),
        posthoc_candidate_end_env_steps=max(0, args.posthoc_candidate_end_env_steps),
        posthoc_selection_window_env_steps=max(1, args.posthoc_selection_window_env_steps),
        posthoc_final_probe_topk=max(1, args.posthoc_final_probe_topk),
        enable_best_checkpoint_selection=enable_best_checkpoint_selection,
        best_checkpoint_selection_start_env_steps=max(0, args.best_checkpoint_selection_start_env_steps),
        best_checkpoint_selection_interval_env_steps=max(1, args.best_checkpoint_selection_interval_env_steps),
        best_checkpoint_validation_episodes=max(1, args.best_checkpoint_validation_episodes),
        best_checkpoint_topk_recheck=max(1, args.best_checkpoint_topk_recheck),
        best_checkpoint_recheck_episodes=max(1, args.best_checkpoint_recheck_episodes),
        use_fixed_model_select_seeds=bool(args.use_fixed_model_select_seeds),
        fixed_model_select_seed_base=int(args.fixed_model_select_seed_base),
        log_interval=args.log_interval,
        log_interval_episodes=max(0, args.log_interval_episodes),
        rows=args.rows,
        cols=args.cols,
        obs_size=args.obs_size,
        scan_radius=args.scan_radius,
        max_accessible_blocks=max(1, args.max_accessible_blocks),
        max_entries_per_block=max(1, args.max_entries_per_block),
        max_episode_steps=args.max_episode_steps,
        coverage_stop_threshold=args.coverage_stop_threshold,
        obstacle_ratio=args.obstacle_ratio,
        reward_info_scale=args.reward_info_scale,
        reward_obstacle_weight=args.reward_obstacle_weight,
        reward_step_penalty=args.reward_step_penalty,
        reward_terminal_bonus=args.reward_terminal_bonus,
        reward_revisit_penalty=args.reward_revisit_penalty,
        reward_turn_penalty_scale=args.reward_turn_penalty_scale,
        reward_turn_weight_45=args.reward_turn_weight_45,
        reward_turn_weight_90=args.reward_turn_weight_90,
        reward_turn_weight_135=args.reward_turn_weight_135,
        reward_turn_weight_180=args.reward_turn_weight_180,
        reward_timeout_penalty=args.reward_timeout_penalty,
        special_highcov_timeout_min_coverage=args.special_highcov_timeout_min_coverage,
        special_highcov_timeout_max_plots=max(0, args.special_highcov_timeout_max_plots),
        special_long_success_gate_coverage=args.special_long_success_gate_coverage,
        special_long_success_gate_window=max(1, args.special_long_success_gate_window),
        special_long_success_min_length=max(1, args.special_long_success_min_length),
        special_long_success_percentile=args.special_long_success_percentile,
        special_long_success_max_plots=max(0, args.special_long_success_max_plots),
        special_lowcov_gate_coverage=args.special_lowcov_gate_coverage,
        special_lowcov_gate_window=max(1, args.special_lowcov_gate_window),
        special_lowcov_absolute_threshold=args.special_lowcov_absolute_threshold,
        special_lowcov_local_drop_margin=args.special_lowcov_local_drop_margin,
        special_lowcov_max_plots=max(0, args.special_lowcov_max_plots),
        advantage_canvas_schema=str(args.advantage_canvas_schema),
        advantage_canvas_channels=advantage_canvas_channels,
        advantage_canvas_channel_count=len(advantage_canvas_channels),
        frontier_raster_used=bool(frontier_raster_used),
        output_root=args.output_root,
        run_name=args.run_name,
    )


def build_vscode_config() -> TrainConfig:
    """
    VSCode direct-run config entry for regular training runs.

    Edit values here for local experiments, then click Run in VSCode.
    Use build_profile_config() when you want the same experiment preset with profiling enabled.
    """
    return _build_vscode_preset(enable_profiling=False)


def build_profile_config() -> TrainConfig:
    """VSCode direct-run config entry for profiling bottlenecks with timing enabled."""
    return _build_vscode_preset(enable_profiling=True)


def _fast_cuda_overrides() -> dict[str, object]:
    # Experimental acceleration path only. Keep it available for controlled A/B testing,
    # but do not treat it as the default recommendation on this machine/model.
    return {
        "enable_amp": True,
        "enable_inference_amp": True,
        "amp_dtype": "fp16",
        "enable_torch_compile": True,
        "compile_mode": "default",
        "enable_cudnn_benchmark": True,
        "enable_tf32": True,
        "enable_channels_last": True,
    }


def build_fast_cuda_config() -> TrainConfig:
    """Experimental fast CUDA preset; kept for optional A/B testing, not the default recommended path."""
    return replace(build_vscode_config(), **_fast_cuda_overrides())


def build_fast_cuda_profile_config() -> TrainConfig:
    """Experimental fast CUDA profiling preset; optional only, not the default recommended path."""
    return replace(build_profile_config(), **_fast_cuda_overrides())


def build_profile_compile_config() -> TrainConfig:
    """Experimental profiling preset with fast CUDA toggles enabled for A/B testing."""
    return build_fast_cuda_profile_config()


def _build_vscode_preset(*, enable_profiling: bool) -> TrainConfig:
    return TrainConfig(
        device="cuda",
        seed=0,
        # Performance-side overhead toggles only; they do not change the algorithm or metric definitions.
        enable_amp=False,
        enable_inference_amp=False,
        amp_dtype="fp16",
        enable_torch_compile=False,
        compile_mode="default",
        enable_cudnn_benchmark=True,
        enable_tf32=True,
        strict_reproducibility=False,
        deterministic_warn_only=True,
        enable_channels_last=False,
        episode_print_interval=10,
        train_print_interval=2000,
        save_train_representative_trajectories=False,
        save_train_special_trajectories=False,
        save_final_probe_trajectories=False,
        generate_plots_on_finish=False,
        enable_collector_timing=enable_profiling,
        enable_learner_timing=enable_profiling,
        enable_replay_timing=enable_profiling,
        enable_state_adapter_timing=enable_profiling,
        enable_cummap_timing=enable_profiling,
        enable_shared_semantic_timing=enable_profiling,
        enable_advantage_state_timing=enable_profiling,
        enable_value_state_timing=enable_profiling,
        timing_log_interval=2000,
        debug_check_incremental_frontier=False,
        prefer_batch_replay_add=True,
        learner_debug_stats_every=8,
        rows=40,
        cols=60,
        scan_radius=10,
        obstacle_ratio=0.20,
        max_accessible_blocks=16,
        max_entries_per_block=8,
        budget_mode="env_steps",
        total_env_steps=500_000,
        total_train_episodes=600,
        warmup_steps=4_000,
        warmup_episodes=0,
        collect_steps_per_iter=16,
        learner_updates_per_iter=1,
        train_every_env_steps=16,
        batch_size=128,
        min_replay_size=8_000,
        replay_capacity=100_000,
        gamma=0.99,
        n_step=3,
        learning_rate=1.0e-4,
        target_update_interval=1_000,
        grad_clip_norm=10.0,
        epsilon_start=1.0,
        epsilon_end=0.04,
        epsilon_decay_steps=240_000,
        recent_episode_window=100,
        formal_protocol=POSTHOC_PROTOCOL_NAME,
        train_side_only_tuning=True,
        final_greedy_episodes=100,
        train_print_interval_episodes=20,
        use_fixed_train_episode_seeds=True,
        fixed_train_episode_seed_base=20259323,
        use_fixed_eval_seeds=True,
        fixed_final_probe_seed_base=20261323,
        periodic_checkpoint_interval_env_steps=20_000,
        posthoc_candidate_start_env_steps=200_000,
        posthoc_candidate_end_env_steps=0,
        posthoc_selection_window_env_steps=40_000,
        posthoc_final_probe_topk=3,
        enable_best_checkpoint_selection=False,
        best_checkpoint_selection_start_env_steps=300_000,
        best_checkpoint_selection_interval_env_steps=20_000,
        best_checkpoint_validation_episodes=24,
        best_checkpoint_topk_recheck=3,
        best_checkpoint_recheck_episodes=50,
        use_fixed_model_select_seeds=True,
        fixed_model_select_seed_base=20262323,
        log_interval=500,
        log_interval_episodes=10,
        max_episode_steps=600,
        coverage_stop_threshold=0.95,
        reward_info_scale=3.1,
        reward_obstacle_weight=0.20,
        reward_step_penalty=0.02,
        reward_terminal_bonus=20.0,
        reward_revisit_penalty=0.10,
        reward_turn_penalty_scale=0.05,
        reward_turn_weight_45=0.0,
        reward_turn_weight_90=(1.0 / 3.0),
        reward_turn_weight_135=(2.0 / 3.0),
        reward_turn_weight_180=1.0,
        reward_timeout_penalty=8.0,
        special_highcov_timeout_min_coverage=0.85,
        special_highcov_timeout_max_plots=5,
        special_long_success_gate_coverage=0.80,
        special_long_success_gate_window=100,
        special_long_success_min_length=350,
        special_long_success_percentile=85.0,
        special_long_success_max_plots=5,
        special_lowcov_gate_coverage=0.80,
        special_lowcov_gate_window=100,
        special_lowcov_absolute_threshold=0.75,
        special_lowcov_local_drop_margin=0.12,
        special_lowcov_max_plots=5,
    )


def _smoke_test() -> None:
    cfg = TrainConfig(
        total_env_steps=120,
        warmup_steps=30,
        rows=28,
        cols=36,
        replay_capacity=1024,
        batch_size=8,
        min_replay_size=16,
        target_update_interval=10,
        final_greedy_episodes=1,
        best_checkpoint_selection_start_env_steps=60,
        best_checkpoint_selection_interval_env_steps=60,
        best_checkpoint_validation_episodes=1,
        best_checkpoint_topk_recheck=1,
        best_checkpoint_recheck_episodes=1,
        log_interval=20,
        max_episode_steps=60,
    )
    _run_with_startup_summary(cfg, run_mode="smoke")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        cfg = parse_args()
        _run_with_startup_summary(cfg, run_mode="cli")
        raise SystemExit(0)

    # Switch run mode here:
    # - "vscode":            use build_vscode_config() for one-click regular runs in VSCode
    # - "profile":           use build_profile_config() for one-click profiling runs in VSCode
    # - "profile_compile":   experimental profiling preset with fast CUDA toggles; not default-recommended
    # - "fast_cuda":         experimental fast CUDA run entry; not default-recommended on current machine/model
    # - "fast_cuda_profile": experimental fast CUDA profiling entry; not default-recommended
    # - "cli":               use parse_args() for command-line parameter control
    # - "smoke":             run built-in smoke test quickly
    RUN_MODE = "vscode"

    if RUN_MODE == "vscode":
        cfg = build_vscode_config()
        _run_with_startup_summary(cfg, run_mode="vscode")
    elif RUN_MODE == "profile":
        cfg = build_profile_config()
        _run_with_startup_summary(cfg, run_mode="profile")
    elif RUN_MODE == "profile_compile":
        cfg = build_profile_compile_config()
        _run_with_startup_summary(cfg, run_mode="profile_compile")
    elif RUN_MODE == "fast_cuda":
        cfg = build_fast_cuda_config()
        _run_with_startup_summary(cfg, run_mode="fast_cuda")
    elif RUN_MODE == "fast_cuda_profile":
        cfg = build_fast_cuda_profile_config()
        _run_with_startup_summary(cfg, run_mode="fast_cuda_profile")
    elif RUN_MODE == "cli":
        cfg = parse_args()
        _run_with_startup_summary(cfg, run_mode="cli")
    elif RUN_MODE == "smoke":
        _smoke_test()
    else:
        raise ValueError(f"Unsupported RUN_MODE: {RUN_MODE}")
