from __future__ import annotations

import argparse
import csv
import json
import sys
import traceback
from dataclasses import fields, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import agents.q_value_agent as q_value_agent_module
from agents.q_value_agent import ExplorationQConfig, ExplorationQNetwork, StateAdapterConfig, StateTensorAdapter
from encoders.advantage_encoder import AdvantageEncoderConfig
from env.advantage_state_builder import ADVANTAGE_CANVAS_CHANNEL_COUNT, AdvantageStateBuilder, AdvantageStateConfig
from env.grid_topology import EMPTY, INVISIBLE, OBSTACLE
from env.shared_semantic_layer import SharedSemanticConfig, SharedSemanticSnapshot
from env.value_state_builder import ValueStateConfig
from train_q_agent import TrainConfig, build_system, configure_torch_runtime, set_seed
from training.collector import CollectorConfig
from training.evaluator import GreedyEvaluator


SCRIPT_ROLE = "supplementary_confidence_check"
SUPPLEMENTARY_NOTE = (
    "This is a supplementary held-out diagnostic for multi-checkpoint, non-default episode-count, "
    "recovery, or extra-confidence analysis. It does not overwrite the current formal final_probe; "
    "100-episode held-out evaluation is now the default formal lane rather than supplementary-only evidence."
)

CHANNEL_LAYOUTS: dict[int, tuple[str, ...]] = {
    7: (
        "unknown",
        "free",
        "obstacle",
        "frontier_mask",
        "frontier_block_area_map",
        "visit_count_log_norm",
        "recent_trajectory_decay",
    ),
    6: (
        "unknown",
        "free",
        "obstacle",
        "frontier_block_area_map",
        "visit_count_log_norm",
        "recent_trajectory_decay",
    ),
    5: (
        "free",
        "obstacle",
        "frontier_block_area_map",
        "visit_count_log_norm",
        "recent_trajectory_decay",
    ),
}

SUMMARY_FIELDS = (
    "checkpoint_path",
    "checkpoint_label",
    "run_dir",
    "run_name",
    "git_commit_sha",
    "status",
    "error",
    "inferred_canvas_channels",
    "channel_layout",
    "checkpoint_env_steps",
    "checkpoint_learn_steps",
    "checkpoint_train_episode_idx",
    "episodes",
    "seed_base",
    "success_rate",
    "coverage",
    "reward",
    "episode_length",
    "repeat_visit_ratio",
    "timeout_flag",
    "stall_trigger_count",
    "zero_info_step_count",
    "accessible_block_count",
    "total_accessible_unknown_area",
    "total_frontier_cluster_count",
    "local_frontier_coverage",
    "local_frontier_block_area_mean",
    "value_truncated_entry_count",
    "value_entry_cap_hit_flag",
)

EPISODE_FIELDS = (
    "checkpoint_path",
    "checkpoint_label",
    "episode_index",
    "episode_seed",
    "success",
    "final_coverage",
    "episode_reward",
    "episode_length",
    "repeat_visit_ratio",
    "timeout_flag",
    "stall_trigger_count",
    "zero_info_step_count",
    "accessible_block_count",
    "total_accessible_unknown_area",
    "total_frontier_cluster_count",
    "local_frontier_coverage",
    "local_frontier_block_area_mean",
    "value_truncated_entry_count",
    "value_entry_cap_hit_flag",
)


class SupplementaryAdvantageStateBuilder:
    """Eval-only canvas builder for historical 7/6/5 advantage checkpoint layouts."""

    def __init__(self, channel_layout: Sequence[str], config: Optional[AdvantageStateConfig] = None):
        self.channel_layout = tuple(str(name) for name in channel_layout)
        self.channel_to_index = {name: idx for idx, name in enumerate(self.channel_layout)}
        self.config = config if config is not None else AdvantageStateConfig()
        self._base = AdvantageStateBuilder(self.config)
        self._canvas_cache: dict[tuple[int, int], np.ndarray] = {}

    def _canvas_buffer(self, shape: tuple[int, int]) -> np.ndarray:
        cached = self._canvas_cache.get(shape)
        if cached is None:
            cached = np.zeros((len(self.channel_layout), int(shape[0]), int(shape[1])), dtype=np.float32)
            self._canvas_cache[shape] = cached
        cached.fill(0.0)
        return cached

    def build(
        self,
        cum_map,
        agent_state: tuple[int, int],
        semantic_snapshot: SharedSemanticSnapshot,
        recent_trajectory_positions: Optional[Sequence[tuple[int, int]]] = None,
    ) -> tuple[np.ndarray, dict[str, float]]:
        local_shape = (int(cum_map.local_shape[0]), int(cum_map.local_shape[1]))
        canvas = self._canvas_buffer(local_shape)
        arr_rows, arr_cols, inside = AdvantageStateBuilder._local_index_arrays(cum_map, agent_state)

        sampled_map = np.full(local_shape, INVISIBLE, dtype=np.int8)
        sampled_visit = np.zeros(local_shape, dtype=np.float32)
        if np.any(inside):
            sampled_map[inside] = cum_map.map[arr_rows[inside], arr_cols[inside]]
            sampled_visit[inside] = cum_map.visit_count[arr_rows[inside], arr_cols[inside]].astype(np.float32)

        if "unknown" in self.channel_to_index:
            canvas[self.channel_to_index["unknown"]] = sampled_map == INVISIBLE
        if "free" in self.channel_to_index:
            canvas[self.channel_to_index["free"]] = sampled_map == EMPTY
        if "obstacle" in self.channel_to_index:
            canvas[self.channel_to_index["obstacle"]] = sampled_map == OBSTACLE

        agent_arr = cum_map.world_to_array(agent_state)
        total_unknown_area = float(max(1, semantic_snapshot.total_accessible_unknown_area))
        frontier_mask_idx = self.channel_to_index.get("frontier_mask")
        frontier_block_idx = self.channel_to_index.get("frontier_block_area_map")
        for block in semantic_snapshot.accessible_blocks:
            block_area_ratio = float(block.block_area) / total_unknown_area
            for frontier_cluster in block.frontier_clusters:
                if frontier_mask_idx is not None:
                    frontier_cluster.paint_to_local_canvas(
                        canvas[frontier_mask_idx],
                        agent_arr=agent_arr,
                        local_shape=local_shape,
                    )
                if frontier_block_idx is not None:
                    AdvantageStateBuilder._paint_geometry_value_to_local_canvas(
                        frontier_cluster.frontier_geometry,
                        canvas[frontier_block_idx],
                        value=block_area_ratio,
                        agent_arr=agent_arr,
                        local_shape=local_shape,
                    )

        visit_idx = self.channel_to_index.get("visit_count_log_norm")
        if visit_idx is not None:
            revisit_count = np.maximum(sampled_visit - 1.0, 0.0).astype(np.float32, copy=False)
            visit_log_denominator = float(np.log1p(max(1e-6, float(self.config.visit_count_log_saturation))))
            visit_count_log_norm = (
                np.log1p(revisit_count).astype(np.float32, copy=False)
                / max(1e-6, visit_log_denominator)
            )
            canvas[visit_idx] = np.clip(visit_count_log_norm, 0.0, 1.0).astype(np.float32, copy=False)

        trajectory_idx = self.channel_to_index.get("recent_trajectory_decay")
        if trajectory_idx is not None:
            history_limit = max(1, int(self.config.trajectory_history_steps))
            raw_history = list(recent_trajectory_positions or ())
            decayed_history = raw_history[:-1] if len(raw_history) > 1 else []
            if len(decayed_history) > history_limit:
                decayed_history = decayed_history[-history_limit:]
            decayed_history_arr = [cum_map.world_to_array(world_rc) for world_rc in decayed_history]
            AdvantageStateBuilder._paint_recent_trajectory_to_local_canvas(
                decayed_history_arr,
                canvas[trajectory_idx],
                current_agent_arr=agent_arr,
                local_shape=local_shape,
            )

        window_area = float(max(1, local_shape[0] * local_shape[1]))
        if frontier_mask_idx is not None:
            frontier_visible_mask = canvas[frontier_mask_idx] > 0.0
        elif frontier_block_idx is not None:
            frontier_visible_mask = canvas[frontier_block_idx] > 0.0
        else:
            frontier_visible_mask = np.zeros(local_shape, dtype=bool)
        frontier_visible = int(np.count_nonzero(frontier_visible_mask))
        frontier_values = canvas[frontier_block_idx][frontier_visible_mask] if frontier_block_idx is not None else ()
        meta = {
            "local_frontier_coverage": float(frontier_visible) / window_area,
            "local_frontier_block_area_mean": float(np.mean(frontier_values)) if frontier_visible > 0 else 0.0,
        }
        return canvas.copy(), meta

    def get_timing_stats(self) -> dict[str, float]:
        return {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run supplementary held-out greedy evaluation for multiple last.pt checkpoints. "
            "This does not overwrite formal run artifacts; the current formal final_probe default is 100 episodes."
        )
    )
    parser.add_argument(
        "--checkpoint",
        action="append",
        default=[],
        help="Path to a checkpoint last.pt. Can be provided multiple times.",
    )
    parser.add_argument(
        "--checkpoint-list-file",
        type=str,
        default=None,
        help="Text file containing one checkpoint path per line. Blank lines and # comments are ignored.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=100,
        help=(
            "Greedy evaluation episodes per checkpoint. Default matches the current formal final_probe, "
            "but this tool remains supplementary because it can compare extra checkpoints or non-default counts."
        ),
    )
    parser.add_argument("--seed-base", type=int, default=20261323, help="First held-out episode seed.")
    parser.add_argument("--device", type=str, default="cuda", help="Evaluation device, e.g. cuda or cpu.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory. Defaults to outputs/supplementary_probes/<timestamped_dir>/.",
    )
    return parser.parse_args()


def read_checkpoint_paths(args: argparse.Namespace) -> list[Path]:
    raw_paths = list(args.checkpoint or [])
    if args.checkpoint_list_file:
        list_path = Path(args.checkpoint_list_file)
        with list_path.open("r", encoding="utf-8-sig") as handle:
            for line in handle:
                text = line.strip()
                if not text or text.startswith("#"):
                    continue
                raw_paths.append(text)
    if not raw_paths:
        raise SystemExit("Provide at least one --checkpoint or --checkpoint-list-file entry.")
    return [Path(path).expanduser().resolve() for path in raw_paths]


def default_output_dir() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return (REPO_ROOT / "outputs" / "supplementary_probes" / f"multi_checkpoint_probe_{timestamp}").resolve()


def resolve_device(device_text: str) -> torch.device:
    device = torch.device(str(device_text))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested with --device cuda, but torch.cuda.is_available() is false.")
    return device


def train_config_from_payload(payload: Mapping[str, Any], device: torch.device) -> TrainConfig:
    cfg_payload = payload.get("train_config")
    if cfg_payload is None:
        cfg = TrainConfig()
    elif isinstance(cfg_payload, Mapping):
        valid_fields = {field.name for field in fields(TrainConfig)}
        cfg = TrainConfig(**{key: value for key, value in cfg_payload.items() if key in valid_fields})
    elif isinstance(cfg_payload, TrainConfig):
        cfg = cfg_payload
    else:
        raise TypeError(f"Unsupported checkpoint train_config type: {type(cfg_payload).__name__}")
    return replace(cfg, device=str(device))


def infer_run_dir(checkpoint_path: Path) -> Path:
    if checkpoint_path.parent.name == "checkpoints":
        return checkpoint_path.parent.parent
    return checkpoint_path.parent


def load_config_snapshot(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "logs" / "config_snapshot.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def infer_canvas_channels(state_dict: Mapping[str, torch.Tensor]) -> int:
    preferred = state_dict.get("advantage_encoder.backbone.0.weight")
    if isinstance(preferred, torch.Tensor) and preferred.dim() == 4:
        return int(preferred.shape[1])
    for key, value in state_dict.items():
        if key.endswith("advantage_encoder.backbone.0.weight") and isinstance(value, torch.Tensor) and value.dim() == 4:
            return int(value.shape[1])
    raise KeyError("Could not infer advantage canvas channel count from online_state_dict.")


def collector_config_from_train_config(cfg: TrainConfig) -> CollectorConfig:
    amp_dtype = str(cfg.amp_dtype).lower()
    if amp_dtype not in {"fp16", "bf16"}:
        raise ValueError(f"Unsupported amp_dtype: {cfg.amp_dtype!r}; expected 'fp16' or 'bf16'")
    return CollectorConfig(
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
        enable_timing=False,
        enable_cummap_timing=False,
        enable_inference_amp=bool(cfg.enable_inference_amp),
        inference_amp_dtype=amp_dtype,
        debug_check_incremental_frontier=bool(cfg.debug_check_incremental_frontier),
        prefer_batch_replay_add=bool(cfg.prefer_batch_replay_add),
        use_fixed_train_episode_seeds=bool(cfg.use_fixed_train_episode_seeds),
        fixed_train_episode_seed_base=int(cfg.fixed_train_episode_seed_base),
        record_episode_artifacts=False,
    )


def build_legacy_eval_system(cfg: TrainConfig, canvas_channels: int, device: torch.device):
    layout = CHANNEL_LAYOUTS.get(int(canvas_channels))
    if layout is None:
        raise ValueError(
            f"Unsupported inferred canvas channel count {canvas_channels}. "
            f"Supported layouts: {sorted(CHANNEL_LAYOUTS)}"
        )

    q_cfg = ExplorationQConfig(
        advantage_encoder=AdvantageEncoderConfig(canvas_in_channels=int(canvas_channels))
    )
    original_canvas_channel_count = q_value_agent_module.ADVANTAGE_CANVAS_CHANNEL_COUNT
    try:
        q_value_agent_module.ADVANTAGE_CANVAS_CHANNEL_COUNT = int(canvas_channels)
        model = ExplorationQNetwork(q_cfg).to(device)
    finally:
        q_value_agent_module.ADVANTAGE_CANVAS_CHANNEL_COUNT = original_canvas_channel_count

    state_cfg = StateAdapterConfig(
        shared_semantics=SharedSemanticConfig(enable_timing=False),
        advantage_state=AdvantageStateConfig(
            trajectory_history_steps=int(cfg.trajectory_history_steps),
            enable_timing=False,
        ),
        value_state=ValueStateConfig(
            max_accessible_blocks=int(cfg.max_accessible_blocks),
            max_entries_per_block=int(cfg.max_entries_per_block),
            enable_timing=False,
        ),
        pin_memory=True,
        non_blocking_transfer=True,
        channels_last_on_cuda=bool(cfg.enable_channels_last),
        enable_timing=False,
    )
    state_adapter = StateTensorAdapter(cfg=state_cfg, device="cpu")
    state_adapter.advantage_builder = SupplementaryAdvantageStateBuilder(
        layout,
        AdvantageStateConfig(
            trajectory_history_steps=int(cfg.trajectory_history_steps),
            enable_timing=False,
        ),
    )
    evaluator = GreedyEvaluator.from_collector_config(
        collector_config_from_train_config(cfg),
        state_adapter=state_adapter,
        device=str(device),
    )
    return model, evaluator, layout


def build_eval_system(cfg: TrainConfig, canvas_channels: int, device: torch.device):
    if int(canvas_channels) == int(ADVANTAGE_CANVAS_CHANNEL_COUNT):
        online_net, _, _, _, _, evaluator = build_system(cfg)
        return online_net, evaluator, CHANNEL_LAYOUTS.get(int(canvas_channels), ())
    return build_legacy_eval_system(cfg, int(canvas_channels), device)


def metric_from_probe(probe: Mapping[str, Any], name: str) -> float:
    if name == "success_rate":
        return float(probe["eval_success_rate"])
    if name == "coverage":
        return float(probe["eval_mean_coverage"])
    if name == "reward":
        return float(probe["eval_mean_reward"])
    if name == "episode_length":
        return float(probe["eval_mean_episode_length"])
    if name == "repeat_visit_ratio":
        return float(probe["eval_mean_repeat_visit_ratio"])
    return float(probe[f"eval_mean_{name}"])


def build_summary_row(
    *,
    checkpoint_path: Path,
    checkpoint_label: str,
    run_dir: Path,
    run_name: str,
    git_commit_sha: str,
    status: str,
    error: str,
    canvas_channels: Optional[int],
    channel_layout: Sequence[str],
    checkpoint_env_steps: Any,
    checkpoint_learn_steps: Any,
    checkpoint_train_episode_idx: Any,
    episodes: int,
    seed_base: int,
    probe: Optional[Mapping[str, Any]],
) -> dict[str, Any]:
    row = {field: "" for field in SUMMARY_FIELDS}
    row.update(
        {
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_label": checkpoint_label,
            "run_dir": str(run_dir),
            "run_name": run_name,
            "git_commit_sha": git_commit_sha,
            "status": status,
            "error": error,
            "inferred_canvas_channels": "" if canvas_channels is None else int(canvas_channels),
            "channel_layout": "|".join(channel_layout),
            "checkpoint_env_steps": "" if checkpoint_env_steps is None else checkpoint_env_steps,
            "checkpoint_learn_steps": "" if checkpoint_learn_steps is None else checkpoint_learn_steps,
            "checkpoint_train_episode_idx": "" if checkpoint_train_episode_idx is None else checkpoint_train_episode_idx,
            "episodes": int(episodes),
            "seed_base": int(seed_base),
        }
    )
    if probe is None:
        return row

    for metric_name in (
        "success_rate",
        "coverage",
        "reward",
        "episode_length",
        "repeat_visit_ratio",
        "timeout_flag",
        "stall_trigger_count",
        "zero_info_step_count",
        "accessible_block_count",
        "total_accessible_unknown_area",
        "total_frontier_cluster_count",
        "local_frontier_coverage",
        "local_frontier_block_area_mean",
        "value_truncated_entry_count",
        "value_entry_cap_hit_flag",
    ):
        row[metric_name] = metric_from_probe(probe, metric_name)
    return row


def build_episode_rows(
    *,
    checkpoint_path: Path,
    checkpoint_label: str,
    episodes: Sequence[Mapping[str, Any]],
    seed_base: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx, episode in enumerate(episodes):
        row = {field: "" for field in EPISODE_FIELDS}
        row.update(
            {
                "checkpoint_path": str(checkpoint_path),
                "checkpoint_label": checkpoint_label,
                "episode_index": int(idx),
                "episode_seed": int(seed_base) + int(idx),
                "success": int(episode["success"]),
                "final_coverage": float(episode["final_coverage"]),
                "episode_reward": float(episode["episode_reward"]),
                "episode_length": int(episode["episode_length"]),
                "repeat_visit_ratio": float(episode["repeat_visit_ratio"]),
                "timeout_flag": float(episode["timeout_flag"]),
                "stall_trigger_count": float(episode["stall_trigger_count"]),
                "zero_info_step_count": float(episode["zero_info_step_count"]),
                "accessible_block_count": float(episode["accessible_block_count"]),
                "total_accessible_unknown_area": float(episode["total_accessible_unknown_area"]),
                "total_frontier_cluster_count": float(episode["total_frontier_cluster_count"]),
                "local_frontier_coverage": float(episode["local_frontier_coverage"]),
                "local_frontier_block_area_mean": float(episode["local_frontier_block_area_mean"]),
                "value_truncated_entry_count": float(episode["value_truncated_entry_count"]),
                "value_entry_cap_hit_flag": float(episode["value_entry_cap_hit_flag"]),
            }
        )
        rows.append(row)
    return rows


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def json_default(value: Any):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def evaluate_checkpoint(
    checkpoint_path: Path,
    *,
    episodes: int,
    seed_base: int,
    device: torch.device,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    run_dir = infer_run_dir(checkpoint_path)
    checkpoint_label = run_dir.name
    config_snapshot = load_config_snapshot(run_dir)
    git_commit_sha = str(config_snapshot.get("git_commit_sha") or "")

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = payload["online_state_dict"]
    canvas_channels = infer_canvas_channels(state_dict)
    channel_layout = CHANNEL_LAYOUTS.get(int(canvas_channels), ())

    cfg = train_config_from_payload(payload, device)
    run_name = str(getattr(cfg, "run_name", "") or checkpoint_label)
    if not git_commit_sha:
        git_commit_sha = str(config_snapshot.get("git_sha") or "")

    set_seed(int(cfg.seed))
    configure_torch_runtime(cfg)
    model, evaluator, channel_layout = build_eval_system(cfg, int(canvas_channels), device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    probe = evaluator.evaluate(model, num_episodes=int(episodes), seed_base=int(seed_base))
    summary_row = build_summary_row(
        checkpoint_path=checkpoint_path,
        checkpoint_label=checkpoint_label,
        run_dir=run_dir,
        run_name=run_name,
        git_commit_sha=git_commit_sha,
        status="ok",
        error="",
        canvas_channels=int(canvas_channels),
        channel_layout=channel_layout,
        checkpoint_env_steps=payload.get("env_steps"),
        checkpoint_learn_steps=payload.get("learn_steps"),
        checkpoint_train_episode_idx=payload.get("train_episode_idx"),
        episodes=int(episodes),
        seed_base=int(seed_base),
        probe=probe,
    )
    episode_rows = build_episode_rows(
        checkpoint_path=checkpoint_path,
        checkpoint_label=checkpoint_label,
        episodes=probe.get("episodes", []),
        seed_base=int(seed_base),
    )
    detail = {
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_label": checkpoint_label,
        "status": "ok",
        "inferred_canvas_channels": int(canvas_channels),
        "channel_layout": list(channel_layout),
        "checkpoint_env_steps": payload.get("env_steps"),
        "checkpoint_learn_steps": payload.get("learn_steps"),
        "checkpoint_train_episode_idx": payload.get("train_episode_idx"),
    }
    return summary_row, episode_rows, detail


def build_failed_summary_row(
    checkpoint_path: Path,
    *,
    episodes: int,
    seed_base: int,
    error: str,
) -> dict[str, Any]:
    run_dir = infer_run_dir(checkpoint_path)
    config_snapshot = load_config_snapshot(run_dir)
    return build_summary_row(
        checkpoint_path=checkpoint_path,
        checkpoint_label=run_dir.name,
        run_dir=run_dir,
        run_name=str(run_dir.name),
        git_commit_sha=str(config_snapshot.get("git_commit_sha") or ""),
        status="error",
        error=error,
        canvas_channels=None,
        channel_layout=(),
        checkpoint_env_steps=None,
        checkpoint_learn_steps=None,
        checkpoint_train_episode_idx=None,
        episodes=int(episodes),
        seed_base=int(seed_base),
        probe=None,
    )


def main() -> int:
    args = parse_args()
    checkpoint_paths = read_checkpoint_paths(args)
    episodes = int(args.episodes)
    if episodes <= 0:
        raise SystemExit("--episodes must be > 0")
    seed_base = int(args.seed_base)
    device = resolve_device(args.device)
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else default_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, Any]] = []
    episode_rows: list[dict[str, Any]] = []
    checkpoint_details: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    for checkpoint_path in checkpoint_paths:
        try:
            print(f"[probe] evaluating {checkpoint_path}")
            summary_row, rows, detail = evaluate_checkpoint(
                checkpoint_path,
                episodes=episodes,
                seed_base=seed_base,
                device=device,
            )
            summary_rows.append(summary_row)
            episode_rows.extend(rows)
            checkpoint_details.append(detail)
            print(
                "[probe] done "
                f"label={summary_row['checkpoint_label']} "
                f"success_rate={float(summary_row['success_rate']):.4f} "
                f"coverage={float(summary_row['coverage']):.4f} "
                f"reward={float(summary_row['reward']):.4f}"
            )
        except Exception as exc:
            error_text = f"{type(exc).__name__}: {exc}"
            print(f"[probe] error {checkpoint_path}: {error_text}", file=sys.stderr)
            traceback.print_exc()
            summary_rows.append(
                build_failed_summary_row(
                    checkpoint_path,
                    episodes=episodes,
                    seed_base=seed_base,
                    error=error_text,
                )
            )
            error_detail = {
                "checkpoint_path": str(checkpoint_path),
                "checkpoint_label": infer_run_dir(checkpoint_path).name,
                "status": "error",
                "error": error_text,
                "traceback": traceback.format_exc(),
            }
            checkpoint_details.append(error_detail)
            errors.append(error_detail)
            continue

    checkpoint_summary_path = output_dir / "checkpoint_summary.csv"
    checkpoint_episode_records_path = output_dir / "checkpoint_episode_records.csv"
    comparison_summary_path = output_dir / "comparison_summary.json"
    run_manifest_path = output_dir / "run_manifest.json"

    write_csv(checkpoint_summary_path, summary_rows, SUMMARY_FIELDS)
    write_csv(checkpoint_episode_records_path, episode_rows, EPISODE_FIELDS)

    comparison_summary = {
        "script_role": SCRIPT_ROLE,
        "episodes": int(episodes),
        "seed_base": int(seed_base),
        "checkpoint_count": int(len(checkpoint_paths)),
        "successful_checkpoint_count": int(sum(1 for row in summary_rows if row.get("status") == "ok")),
        "failed_checkpoint_count": int(sum(1 for row in summary_rows if row.get("status") != "ok")),
        "checkpoint_labels": [str(row.get("checkpoint_label", "")) for row in summary_rows],
        "metric_table": summary_rows,
        "errors": errors,
        "note": SUPPLEMENTARY_NOTE,
    }
    comparison_summary_path.write_text(
        json.dumps(comparison_summary, ensure_ascii=False, indent=2, default=json_default),
        encoding="utf-8",
    )

    manifest = {
        "script_role": SCRIPT_ROLE,
        "created_at": datetime.now().astimezone().isoformat(),
        "argv": sys.argv,
        "arguments": {
            "checkpoint": [str(path) for path in checkpoint_paths],
            "checkpoint_list_file": args.checkpoint_list_file,
            "episodes": int(episodes),
            "seed_base": int(seed_base),
            "device": str(device),
            "output_dir": str(output_dir),
        },
        "checkpoints": checkpoint_details,
        "output_files": {
            "checkpoint_summary_csv": str(checkpoint_summary_path),
            "checkpoint_episode_records_csv": str(checkpoint_episode_records_path),
            "comparison_summary_json": str(comparison_summary_path),
            "run_manifest_json": str(run_manifest_path),
        },
        "note": SUPPLEMENTARY_NOTE,
    }
    run_manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=json_default),
        encoding="utf-8",
    )

    print(f"[probe] wrote {checkpoint_summary_path}")
    print(f"[probe] wrote {checkpoint_episode_records_path}")
    print(f"[probe] wrote {comparison_summary_path}")
    print(f"[probe] wrote {run_manifest_path}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
