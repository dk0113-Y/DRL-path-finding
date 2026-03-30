from __future__ import annotations

import argparse
import copy
import random
import sys
import time
from collections import deque
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from agents.q_value_agent import ExplorationQConfig, ExplorationQNetwork, StateAdapterConfig, StateTensorAdapter
from env.advantage_state_builder import AdvantageStateConfig
from env.shared_semantic_layer import SharedSemanticConfig
from env.value_state_builder import ValueStateConfig
from training.checkpointing import CheckpointManager
from training.collector import CollectorConfig, SEMANTIC_EPISODE_FIELDS, TransitionCollector
from training.evaluator import GreedyEvaluator
from training.learner import DDQNLearner, DDQNLearnerConfig
from training.logger import CSVMetricLogger
from training.plotting import generate_all_plots
from training.replay_buffer import ReplayBuffer, ReplayBufferConfig
from training.rewarding import REWARD_BREAKDOWN_FIELDS
from training.trajectory_plotting import save_episode_trajectory_plots

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
    enable_channels_last: bool = False  # Tensor-layout toggle only; model math and metrics stay unchanged.
    episode_print_interval: int = 1  # Stdout throttling only; 1 prints every episode without affecting logging/metrics.
    train_print_interval: int = 2_000  # Stdout throttling only; separated from CSV logging and algorithm behavior.
    save_eval_trajectories: bool = False  # Plot-saving side overhead only; evaluation logic and metrics are unchanged.
    save_train_representative_trajectories: bool = False  # Train-failure trajectory dumping is optional wall-clock overhead.
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
    learner_debug_stats_every: int = 1  # Metric-sync throttling only; learner updates stay unchanged.
    rows: int = 40
    cols: int = 60
    obs_size: int = 6
    scan_radius: int = 10  # radar sensor radius only
    obstacle_ratio: float = 0.20

    max_accessible_blocks: int = 16
    max_entries_per_block: int = 6

    total_env_steps: int = 500_000
    warmup_steps: int = 4_000
    collect_steps_per_iter: int = 16
    learner_updates_per_iter: int = 2
    train_every_env_steps: int = 16
    log_interval: int = 500

    eval_interval_env_steps: int = 24_000
    eval_episodes: int = 12
    recent_episode_window: int = 100
    final_greedy_episodes: int = 16
    use_fixed_eval_seeds: bool = True
    fixed_eval_seed_base: int = 20260323
    fixed_final_probe_seed_base: int = 20261323

    replay_capacity: int = 100_000
    batch_size: int = 128
    min_replay_size: int = 4_000

    gamma: float = 0.99
    n_step: int = 3

    learning_rate: float = 1.0e-4
    weight_decay: float = 0.0
    grad_clip_norm: float = 10.0
    target_update_interval: int = 1_000

    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 240_000

    max_episode_steps: int = 600  # tune with map scale as needed
    coverage_stop_threshold: float = 0.95

    reward_info_scale: float = 3.0
    reward_obstacle_weight: float = 0.25
    reward_info_norm: float | str | None = "half_perimeter"
    reward_recent_revisit_window: int = 15
    reward_stall_window: int = 8
    reward_step_penalty: float = 0.02
    reward_terminal_bonus: float = 20.0
    reward_revisit_penalty: float = 0.12
    reward_stall_penalty: float = 0.12
    reward_timeout_penalty: float = 8.0

    output_root: str = "outputs"
    run_name: str = "ddqn_explore_vscode_stage5"


def linear_epsilon(step: int, cfg: TrainConfig) -> float:
    s = max(0, int(step))
    if s >= int(cfg.epsilon_decay_steps):
        return float(cfg.epsilon_end)
    ratio = float(s) / float(max(1, int(cfg.epsilon_decay_steps)))
    return float(cfg.epsilon_start + ratio * (cfg.epsilon_end - cfg.epsilon_start))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_torch_runtime(cfg: TrainConfig) -> None:
    """Apply performance-only backend toggles without changing training/eval semantics."""
    if not str(cfg.device).lower().startswith("cuda"):
        return
    if not torch.cuda.is_available():
        return

    torch.backends.cudnn.benchmark = bool(cfg.enable_cudnn_benchmark)

    if (
        hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul")
        and hasattr(torch.backends.cuda.matmul, "allow_tf32")
    ):
        torch.backends.cuda.matmul.allow_tf32 = bool(cfg.enable_tf32)

    if hasattr(torch.backends.cudnn, "allow_tf32"):
        torch.backends.cudnn.allow_tf32 = bool(cfg.enable_tf32)


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


def _parse_reward_info_norm_arg(value: str) -> float | str:
    text = str(value).strip()
    if text == "":
        return None
    try:
        return float(text)
    except ValueError:
        return text


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
    timing_flags = _timing_flag_dict(cfg)
    profiling_enabled = _timing_summary_enabled(cfg)
    timing_flag_text = " ".join(f"{key}={value}" for key, value in timing_flags.items())
    print(
        "[startup] "
        f"file={Path(__file__).resolve()} "
        f"run_mode={run_mode} "
        f"device={cfg.device} "
        f"profiling_enabled={profiling_enabled} "
        f"train_amp={bool(cfg.enable_amp)} "
        f"torch_compile={bool(cfg.enable_torch_compile)} "
        f"inference_amp={bool(cfg.enable_inference_amp)} "
        f"amp_dtype={cfg.amp_dtype} "
        f"channels_last={bool(cfg.enable_channels_last)} "
        f"timing_log_interval={int(cfg.timing_log_interval)} "
        f"train_print_interval={int(cfg.train_print_interval)} "
        f"log_interval={int(cfg.log_interval)} "
        f"note=\"{_describe_profiling_mode(cfg, run_mode)}\""
    )
    print(f"[startup] timing_flags {timing_flag_text}")
    if not profiling_enabled:
        print(
            "[startup] timing summaries are disabled in this run, so no [timing] lines will be printed; "
            "use RUN_MODE='profile' or RUN_MODE='profile_compile' in VSCode direct-run, "
            "or pass --profile / explicit CLI timing flags to enable them."
        )


def _run_with_startup_summary(cfg: TrainConfig, run_mode: str) -> None:
    _print_startup_summary(cfg, run_mode=run_mode)
    run_training(cfg)


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
        },
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
            "frontier_stats_time": "frontier",
            "domain_extract_time": "domain",
            "aggregate_time": "agg",
        },
    )
    if cummap_line is not None:
        lines.append(cummap_line)

    if len(lines) <= 0:
        return

    print(f"[timing] env_steps={int(env_steps)}")
    for line in lines:
        print(f"  {line}")


def build_system(cfg: TrainConfig):
    q_cfg = ExplorationQConfig()
    raw_online_net = ExplorationQNetwork(q_cfg).to(cfg.device)
    raw_online_net = _maybe_to_channels_last(raw_online_net, cfg)
    target_net = copy.deepcopy(raw_online_net).to(cfg.device)
    target_net = _maybe_to_channels_last(target_net, cfg)
    online_net = _compile_online_net(raw_online_net, cfg)

    state_cfg = StateAdapterConfig(
        shared_semantics=SharedSemanticConfig(
            enable_timing=bool(cfg.enable_shared_semantic_timing),
        ),
        advantage_state=AdvantageStateConfig(
            enable_timing=bool(cfg.enable_advantage_state_timing),
        ),
        value_state=ValueStateConfig(
            max_accessible_blocks=int(cfg.max_accessible_blocks),
            max_entries_per_block=int(cfg.max_entries_per_block),
            enable_timing=bool(cfg.enable_value_state_timing),
        ),
        pin_memory=True,
        non_blocking_transfer=True,
        channels_last_on_cuda=bool(cfg.enable_channels_last),
        enable_timing=bool(cfg.enable_state_adapter_timing),
    )
    state_adapter = StateTensorAdapter(cfg=state_cfg, device="cpu")

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
        reward_info_scale=float(cfg.reward_info_scale),
        reward_obstacle_weight=float(cfg.reward_obstacle_weight),
        reward_info_norm=cfg.reward_info_norm,
        reward_recent_revisit_window=int(cfg.reward_recent_revisit_window),
        reward_stall_window=int(cfg.reward_stall_window),
        reward_step_penalty=float(cfg.reward_step_penalty),
        reward_terminal_bonus=float(cfg.reward_terminal_bonus),
        reward_revisit_penalty=float(cfg.reward_revisit_penalty),
        reward_stall_penalty=float(cfg.reward_stall_penalty),
        reward_timeout_penalty=float(cfg.reward_timeout_penalty),
        n_step=int(cfg.n_step),
        gamma=float(cfg.gamma),
        enable_timing=bool(cfg.enable_collector_timing),
        enable_cummap_timing=bool(cfg.enable_cummap_timing),
        enable_inference_amp=bool(cfg.enable_inference_amp),
        inference_amp_dtype=amp_dtype,
        debug_check_incremental_frontier=bool(cfg.debug_check_incremental_frontier),
        prefer_batch_replay_add=bool(cfg.prefer_batch_replay_add),
        record_episode_artifacts=bool(cfg.save_train_representative_trajectories),
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


def run_training(cfg: TrainConfig) -> None:
    run_start_time = time.perf_counter()
    set_seed(int(cfg.seed))
    configure_torch_runtime(cfg)
    run_dir = create_run_dir(cfg)
    logger = CSVMetricLogger(run_dir)
    ckpt = CheckpointManager(run_dir)

    online_net, _, replay, collector, learner, evaluator = build_system(cfg)

    recent_eps: deque[dict] = deque(maxlen=int(max(1, cfg.recent_episode_window)))
    last_eval: dict | None = None
    best_eval: dict | None = None
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

    def handle_episodes(episodes: list[dict], phase: str, epsilon: float) -> None:
        for ep in episodes:
            row = {
                "phase": phase,
                "env_steps": int(ep["env_steps"]),
                "episode_idx": int(ep["episode_idx"]),
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
            }
            logger.log_train_episode(row)
            recent_eps.append(row)
            if (
                phase == "train"
                and bool(cfg.save_train_representative_trajectories)
                and (ep.get("trajectory_positions") is not None)
                and (ep.get("true_grid") is not None)
            ):
                trace_ep = dict(ep)
                trace_ep["phase"] = phase
                trace_ep["epsilon"] = float(epsilon)
                train_trace_episodes.append(trace_ep)
            if episode_print_interval > 0 and int(row["episode_idx"]) % episode_print_interval == 0:
                print(
                    "[episode] "
                    f"phase={phase} idx={row['episode_idx']} env={row['env_steps']} "
                    f"reward={row['episode_reward']:.4f} len={row['episode_length']} "
                    f"cov={row['final_coverage']:.4f} succ={row['success']} "
                    f"repeat={row['repeat_visit_ratio']:.4f} reason={row['done_reason']}"
                )

    def run_eval(env_steps: int, tag: str = "periodic") -> tuple[dict, bool]:
        nonlocal last_eval, best_eval

        eval_seed_base = int(cfg.fixed_eval_seed_base) if bool(cfg.use_fixed_eval_seeds) else None
        em = evaluator.evaluate(
            online_net,
            num_episodes=int(cfg.eval_episodes),
            seed_base=eval_seed_base,
        )
        row = {
            "tag": tag,
            "env_steps": int(env_steps),
            "learner_steps": int(learner.learn_steps),
            "eval_episodes": int(em["eval_episodes"]),
            "eval_mean_reward": float(em["eval_mean_reward"]),
            "eval_mean_coverage": float(em["eval_mean_coverage"]),
            "eval_success_rate": float(em["eval_success_rate"]),
            "eval_mean_episode_length": float(em["eval_mean_episode_length"]),
            "eval_mean_repeat_visit_ratio": float(em["eval_mean_repeat_visit_ratio"]),
            **{
                f"eval_mean_{metric_name}": float(em[f"eval_mean_{metric_name}"])
                for metric_name in EVAL_SEMANTIC_METRIC_NAMES
            },
            **{
                f"eval_mean_{field}": float(em[f"eval_mean_{field}"])
                for field in REWARD_BREAKDOWN_FIELDS
            },
        }
        logger.log_eval(row)

        if bool(cfg.save_eval_trajectories):
            traj_prefix = f"eval_{int(env_steps):07d}" if tag == "periodic" else f"{tag}_{int(env_steps):07d}"
            trajectory_plot_paths.extend(
                save_episode_trajectory_plots(
                    run_dir,
                    em.get("episodes", []),
                    prefix=traj_prefix,
                    max_episodes=1,
                    selection_mode="lowest_coverage",
                    gate_window=int(cfg.recent_episode_window),
                    coverage_target=float(cfg.coverage_stop_threshold),
                )
            )

        is_best = ckpt.maybe_save_best(
            online_net,
            learner,
            env_steps=int(env_steps),
            eval_metrics=row,
            train_config=cfg,
        )

        last_eval = row
        if is_best:
            best_eval = dict(row)

        print(
            "[eval] "
            f"tag={tag} env_steps={row['env_steps']} episodes={row['eval_episodes']} "
            f"mean_reward={row['eval_mean_reward']:.4f} mean_cov={row['eval_mean_coverage']:.4f} "
            f"success_rate={row['eval_success_rate']:.4f} mean_len={row['eval_mean_episode_length']:.2f} "
            f"blocks={row['eval_mean_accessible_block_count']:.2f} "
            f"main_block_area={row['eval_mean_main_block_area']:.2f} "
            f"best_saved={int(is_best)}"
        )
        return row, is_best

    warmup = min(int(cfg.warmup_steps), int(cfg.total_env_steps))
    timing_log_interval = int(max(0, cfg.timing_log_interval))
    timing_summary_enabled = _timing_summary_enabled(cfg)
    if warmup > 0:
        warm_stats = collector.collect_steps(warmup, epsilon=1.0, random_only=True)
        handle_episodes(warm_stats.get("episodes", []), phase="warmup", epsilon=1.0)

    env_steps = warmup
    last_train_env_step = int(env_steps)

    eval_interval = int(max(0, cfg.eval_interval_env_steps))
    next_eval_step = eval_interval if eval_interval > 0 else int(cfg.total_env_steps) + 1
    log_interval = int(cfg.log_interval)
    next_log_step = (
        (((env_steps//log_interval) + 1) * log_interval) if log_interval > 0 else int(cfg.total_env_steps) +
        1
    )
    next_timing_log_step = (
        timing_log_interval if timing_log_interval > 0 else int(cfg.total_env_steps) + 1
    )
    train_print_interval = int(max(0, cfg.train_print_interval))
    next_train_print_step = (
        (((env_steps//train_print_interval) + 1) *
         train_print_interval) if train_print_interval > 0 else int(cfg.total_env_steps) + 1
    )

    if timing_summary_enabled and timing_log_interval > 0:
        while env_steps >= next_timing_log_step:
            _print_timing_summary(env_steps, collector, learner, replay, collector.state_adapter)
            next_timing_log_step += timing_log_interval

    while eval_interval > 0 and env_steps >= next_eval_step:
        run_eval(env_steps=env_steps, tag="periodic")
        next_eval_step += eval_interval

    while env_steps < int(cfg.total_env_steps):
        eps = linear_epsilon(env_steps, cfg)
        collect_n = min(int(cfg.collect_steps_per_iter), int(cfg.total_env_steps) - env_steps)
        cstats = collector.collect_steps(collect_n, epsilon=eps)
        env_steps += int(cstats["env_steps"])
        handle_episodes(cstats.get("episodes", []), phase="train", epsilon=eps)

        train_interval = int(max(1, cfg.train_every_env_steps))
        if (
            (env_steps - last_train_env_step) >= train_interval and len(replay) >= int(cfg.min_replay_size)
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

        if eval_interval > 0:
            while env_steps >= next_eval_step:
                run_eval(env_steps=env_steps, tag="periodic")
                next_eval_step += eval_interval

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
            rec = summarize_recent_episodes(recent_eps)
            step_row = {
                "env_steps": int(env_steps),
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
            }
            if should_log:
                logger.log_train_step(step_row)
            if should_print_train:
                print(
                    "[train] "
                    f"env_steps={step_row['env_steps']} replay={step_row['replay_size']} "
                    f"eps={step_row['epsilon']:.4f} loss={step_row['loss']:.5f} "
                    f"q_mean={step_row['q_mean']:.5f} target_q_mean={step_row['target_q_mean']:.5f} "
                    f"td_abs_mean={step_row['td_abs_mean']:.5f} grad_norm={step_row['grad_norm']:.5f} "
                    f"learner_steps={step_row['learner_steps']} "
                    f"recent_reward={step_row['recent_mean_reward']:.4f} "
                    f"recent_cov={step_row['recent_mean_coverage']:.4f} "
                    f"recent_succ={step_row['recent_success_rate']:.4f} "
                    f"recent_blocks={step_row['recent_accessible_block_count']:.2f}"
                )

    if last_eval is None or int(last_eval["env_steps"]) != int(env_steps):
        run_eval(env_steps=env_steps, tag="final")

    last_ckpt_path = ckpt.save_last(
        online_net,
        learner,
        env_steps=int(env_steps),
        eval_metrics=last_eval,
        train_config=cfg,
    )

    probe_source = "online_last"
    if ckpt.best_path.exists():
        payload = torch.load(ckpt.best_path, map_location=cfg.device, weights_only=False)
        online_net.load_state_dict(payload["online_state_dict"])
        probe_source = "best_checkpoint"

    final_probe_seed_base = int(cfg.fixed_final_probe_seed_base) if bool(cfg.use_fixed_eval_seeds) else None
    probe = evaluator.evaluate(
        online_net,
        num_episodes=int(max(1, cfg.final_greedy_episodes)),
        seed_base=final_probe_seed_base,
    )
    probe_row = {
        "tag": "final_probe",
        "env_steps": int(env_steps),
        "learner_steps": int(learner.learn_steps),
        "eval_episodes": int(probe["eval_episodes"]),
        "eval_mean_reward": float(probe["eval_mean_reward"]),
        "eval_mean_coverage": float(probe["eval_mean_coverage"]),
        "eval_success_rate": float(probe["eval_success_rate"]),
        "eval_mean_episode_length": float(probe["eval_mean_episode_length"]),
        "eval_mean_repeat_visit_ratio": float(probe["eval_mean_repeat_visit_ratio"]),
        **{
            f"eval_mean_{metric_name}": float(probe[f"eval_mean_{metric_name}"])
            for metric_name in EVAL_SEMANTIC_METRIC_NAMES
        },
        **{
            f"eval_mean_{field}": float(probe[f"eval_mean_{field}"])
            for field in REWARD_BREAKDOWN_FIELDS
        },
    }
    logger.log_final_probe(probe_row)
    print(
        "[final_probe] "
        f"source={probe_source} env_steps={probe_row['env_steps']} episodes={probe_row['eval_episodes']} "
        f"mean_reward={probe_row['eval_mean_reward']:.4f} mean_cov={probe_row['eval_mean_coverage']:.4f} "
        f"success_rate={probe_row['eval_success_rate']:.4f} mean_len={probe_row['eval_mean_episode_length']:.2f} "
        f"blocks={probe_row['eval_mean_accessible_block_count']:.2f}"
    )
    if bool(cfg.save_train_representative_trajectories):
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
    if bool(cfg.save_final_probe_trajectories):
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
    if bool(cfg.generate_plots_on_finish):
        generated_plots = generate_all_plots(run_dir)
    else:
        generated_plots = []

    recent_summary = summarize_recent_episodes(recent_eps)
    total_runtime_sec = time.perf_counter() - run_start_time
    total_runtime_sec_int = int(round(total_runtime_sec))
    hours, rem = divmod(total_runtime_sec_int, 3600)
    minutes, seconds = divmod(rem, 60)
    total_runtime_hms = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    print("=" * 72)
    print("Training Summary")
    print(f"run_dir: {run_dir}")
    print(f"final_env_steps: {env_steps}")
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

    if last_eval is not None:
        print(
            "last_eval: "
            f"reward={last_eval['eval_mean_reward']:.4f}, "
            f"coverage={last_eval['eval_mean_coverage']:.4f}, "
            f"success={last_eval['eval_success_rate']:.4f}, "
            f"length={last_eval['eval_mean_episode_length']:.2f}, "
            f"blocks={last_eval['eval_mean_accessible_block_count']:.2f}"
        )

    if best_eval is None and ckpt.best_path.exists():
        payload = torch.load(ckpt.best_path, map_location="cpu", weights_only=False)
        best_eval = payload.get("eval_metrics", None)

    if best_eval is not None:
        print(
            "best_eval: "
            f"reward={float(best_eval['eval_mean_reward']):.4f}, "
            f"coverage={float(best_eval['eval_mean_coverage']):.4f}, "
            f"success={float(best_eval['eval_success_rate']):.4f}, "
            f"length={float(best_eval['eval_mean_episode_length']):.2f}, "
            f"blocks={float(best_eval.get('eval_mean_accessible_block_count', float('nan'))):.2f}"
        )

    print(
        "final_probe: "
        f"source={probe_source}, reward={probe_row['eval_mean_reward']:.4f}, "
        f"coverage={probe_row['eval_mean_coverage']:.4f}, "
        f"success={probe_row['eval_success_rate']:.4f}, "
        f"length={probe_row['eval_mean_episode_length']:.2f}, "
        f"blocks={probe_row['eval_mean_accessible_block_count']:.2f}"
    )

    print(f"checkpoint_last: {last_ckpt_path}")
    print(f"checkpoint_best: {ckpt.best_path if ckpt.best_path.exists() else 'N/A'}")
    print(f"train_episode_csv: {logger.train_episode_csv}")
    print(f"eval_csv: {logger.eval_csv}")
    print(f"final_probe_csv: {logger.final_probe_csv}")
    print(f"train_step_csv: {logger.train_step_csv}")
    if len(generated_plots) > 0:
        print(f"plots_dir: {run_dir / 'plots'}")
    if len(trajectory_plot_paths) > 0:
        print(f"trajectories_dir: {run_dir / 'trajectories'}")
    print("=" * 72)


def parse_args() -> TrainConfig:
    p = argparse.ArgumentParser(description="Double DQN training with monitoring/eval/checkpoint loop")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=0)
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
        default=1,
        help="Stdout throttling only; 1 prints every episode and does not affect CSV metrics.",
    )
    p.add_argument(
        "--train-print-interval",
        type=int,
        default=2000,
        help="Stdout throttling only; separate from --log-interval and does not affect CSV metrics.",
    )
    p.add_argument(
        "--save-eval-trajectories",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Plot-saving toggle only; periodic/final eval logic and metrics are unchanged.",
    )
    p.add_argument(
        "--save-train-representative-trajectories",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Save all post-gate failed train episodes with trajectory/belief overlays.",
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
        default=1,
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

    p.add_argument("--total-env-steps", type=int, default=500_000)
    p.add_argument("--warmup-steps", type=int, default=4_000)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--min-replay-size", type=int, default=4_000)
    p.add_argument("--replay-capacity", type=int, default=100_000)
    p.add_argument("--collect-steps-per-iter", type=int, default=16)
    p.add_argument("--learner-updates-per-iter", type=int, default=2)
    p.add_argument("--train-every-env-steps", type=int, default=16)
    p.add_argument("--n-step", type=int, default=3)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--target-update-interval", type=int, default=1_000)
    p.add_argument("--learning-rate", type=float, default=1.0e-4)

    p.add_argument("--epsilon-start", type=float, default=1.0)
    p.add_argument("--epsilon-end", type=float, default=0.05)
    p.add_argument("--epsilon-decay-steps", type=int, default=240_000)

    p.add_argument("--eval-interval-env-steps", type=int, default=24_000)
    p.add_argument("--eval-episodes", type=int, default=12)
    p.add_argument("--recent-episode-window", type=int, default=100)
    p.add_argument("--final-greedy-episodes", type=int, default=16)
    p.add_argument(
        "--use-fixed-eval-seeds",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use a fixed held-out map seed set for periodic eval/final probe to reduce metric noise.",
    )
    p.add_argument(
        "--fixed-eval-seed-base",
        type=int,
        default=20260323,
        help="Base seed for the periodic evaluation map set.",
    )
    p.add_argument(
        "--fixed-final-probe-seed-base",
        type=int,
        default=20261323,
        help="Base seed for the final-probe evaluation map set.",
    )

    p.add_argument("--log-interval", type=int, default=500)
    p.add_argument("--rows", type=int, default=40)
    p.add_argument("--cols", type=int, default=60)
    p.add_argument("--obs-size", type=int, default=6)
    p.add_argument("--scan-radius", type=int, default=10)
    p.add_argument("--max-accessible-blocks", type=int, default=16)
    p.add_argument("--max-entries-per-block", type=int, default=6)
    p.add_argument("--obstacle-ratio", type=float, default=0.20)
    p.add_argument("--reward-info-scale", type=float, default=3.0, help="weighted information gain scale")
    p.add_argument(
        "--reward-obstacle-weight", type=float, default=0.25, help="obstacle reveal weight in info gain"
    )
    p.add_argument(
        "--reward-info-norm",
        type=_parse_reward_info_norm_arg,
        default="half_perimeter",
        help=
        "optional info gain normalization override; accepts a positive number or 'half_perimeter'; default uses radar theoretical visible cells / 2",
    )
    p.add_argument(
        "--reward-recent-revisit-window",
        type=int,
        default=15,
        help="recent position window size for revisit penalty",
    )
    p.add_argument(
        "--reward-stall-window",
        type=int,
        default=8,
        help="consecutive zero-info steps before stall penalty starts; larger values allow short backtracks",
    )
    p.add_argument("--reward-step-penalty", type=float, default=0.02, help="step penalty")
    p.add_argument("--reward-terminal-bonus", type=float, default=20.0, help="terminal success bonus")
    p.add_argument("--reward-revisit-penalty", type=float, default=0.12, help="revisit penalty")
    p.add_argument("--reward-stall-penalty", type=float, default=0.12, help="stall penalty")
    p.add_argument("--reward-timeout-penalty", type=float, default=8.0, help="timeout penalty")

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
    enable_channels_last = bool(args.enable_channels_last)
    if bool(args.fast_cuda):
        enable_amp = True
        enable_inference_amp = True
        enable_torch_compile = True
        enable_cudnn_benchmark = True
        enable_tf32 = True
        enable_channels_last = True

    if args.smoke:
        return TrainConfig(
            device=args.device,
            seed=args.seed,
            enable_amp=enable_amp,
            enable_inference_amp=enable_inference_amp,
            amp_dtype=args.amp_dtype,
            enable_torch_compile=enable_torch_compile,
            compile_mode=args.compile_mode,
            enable_cudnn_benchmark=enable_cudnn_benchmark,
            enable_tf32=enable_tf32,
            enable_channels_last=enable_channels_last,
            episode_print_interval=max(0, args.episode_print_interval),
            train_print_interval=max(0, args.train_print_interval),
            save_eval_trajectories=args.save_eval_trajectories,
            save_train_representative_trajectories=args.save_train_representative_trajectories,
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
            total_env_steps=180,
            warmup_steps=40,
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
            epsilon_start=args.epsilon_start,
            epsilon_end=args.epsilon_end,
            epsilon_decay_steps=max(1, args.epsilon_decay_steps),
            eval_interval_env_steps=60,
            eval_episodes=3,
            recent_episode_window=10,
            final_greedy_episodes=1,
            use_fixed_eval_seeds=True,
            fixed_eval_seed_base=int(args.fixed_eval_seed_base),
            fixed_final_probe_seed_base=int(args.fixed_final_probe_seed_base),
            log_interval=20,
            max_episode_steps=80,
            reward_info_scale=args.reward_info_scale,
            reward_obstacle_weight=args.reward_obstacle_weight,
            reward_info_norm=args.reward_info_norm,
            reward_recent_revisit_window=max(1, args.reward_recent_revisit_window),
            reward_stall_window=max(1, args.reward_stall_window),
            reward_step_penalty=args.reward_step_penalty,
            reward_terminal_bonus=args.reward_terminal_bonus,
            reward_revisit_penalty=args.reward_revisit_penalty,
            reward_stall_penalty=args.reward_stall_penalty,
            reward_timeout_penalty=args.reward_timeout_penalty,
            output_root=args.output_root,
            run_name=args.run_name,
        )

    return TrainConfig(
        device=args.device,
        seed=args.seed,
        enable_amp=enable_amp,
        enable_inference_amp=enable_inference_amp,
        amp_dtype=args.amp_dtype,
        enable_torch_compile=enable_torch_compile,
        compile_mode=args.compile_mode,
        enable_cudnn_benchmark=enable_cudnn_benchmark,
        enable_tf32=enable_tf32,
        enable_channels_last=enable_channels_last,
        episode_print_interval=max(0, args.episode_print_interval),
        train_print_interval=max(0, args.train_print_interval),
        save_eval_trajectories=args.save_eval_trajectories,
        save_train_representative_trajectories=args.save_train_representative_trajectories,
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
        total_env_steps=args.total_env_steps,
        warmup_steps=args.warmup_steps,
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
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay_steps=max(1, args.epsilon_decay_steps),
        eval_interval_env_steps=max(0, args.eval_interval_env_steps),
        eval_episodes=max(1, args.eval_episodes),
        recent_episode_window=max(1, args.recent_episode_window),
        final_greedy_episodes=max(1, args.final_greedy_episodes),
        use_fixed_eval_seeds=bool(args.use_fixed_eval_seeds),
        fixed_eval_seed_base=int(args.fixed_eval_seed_base),
        fixed_final_probe_seed_base=int(args.fixed_final_probe_seed_base),
        log_interval=args.log_interval,
        rows=args.rows,
        cols=args.cols,
        obs_size=args.obs_size,
        scan_radius=args.scan_radius,
        max_accessible_blocks=max(1, args.max_accessible_blocks),
        max_entries_per_block=max(1, args.max_entries_per_block),
        obstacle_ratio=args.obstacle_ratio,
        reward_info_scale=args.reward_info_scale,
        reward_obstacle_weight=args.reward_obstacle_weight,
        reward_info_norm=args.reward_info_norm,
        reward_recent_revisit_window=max(1, args.reward_recent_revisit_window),
        reward_stall_window=max(1, args.reward_stall_window),
        reward_step_penalty=args.reward_step_penalty,
        reward_terminal_bonus=args.reward_terminal_bonus,
        reward_revisit_penalty=args.reward_revisit_penalty,
        reward_stall_penalty=args.reward_stall_penalty,
        reward_timeout_penalty=args.reward_timeout_penalty,
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
        enable_channels_last=False,
        episode_print_interval=1,
        train_print_interval=2000,
        save_eval_trajectories=False,
        save_train_representative_trajectories=False,
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
        learner_debug_stats_every=1,
        rows=40,
        cols=60,
        scan_radius=10,
        obstacle_ratio=0.20,
        max_accessible_blocks=16,
        max_entries_per_block=6,
        total_env_steps=500_000,
        warmup_steps=4_000,
        collect_steps_per_iter=16,
        learner_updates_per_iter=2,
        train_every_env_steps=16,
        batch_size=128,
        min_replay_size=4_000,
        replay_capacity=100_000,
        gamma=0.99,
        n_step=3,
        learning_rate=1.0e-4,
        target_update_interval=1_000,
        grad_clip_norm=10.0,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay_steps=240_000,
        eval_interval_env_steps=24_000,
        eval_episodes=12,
        recent_episode_window=100,
        final_greedy_episodes=16,
        use_fixed_eval_seeds=True,
        fixed_eval_seed_base=20260323,
        fixed_final_probe_seed_base=20261323,
        log_interval=500,
        max_episode_steps=600,
        coverage_stop_threshold=0.95,
        reward_info_scale=3.0,
        reward_obstacle_weight=0.25,
        reward_info_norm="half_perimeter",
        reward_recent_revisit_window=15,
        reward_stall_window=8,
        reward_step_penalty=0.02,
        reward_terminal_bonus=20.0,
        reward_revisit_penalty=0.12,
        reward_stall_penalty=0.12,
        reward_timeout_penalty=8.0,
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
        eval_interval_env_steps=40,
        eval_episodes=2,
        final_greedy_episodes=1,
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
