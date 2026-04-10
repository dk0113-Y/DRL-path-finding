from __future__ import annotations

from pathlib import Path

import numpy as np

from env.grid_topology import EMPTY, INVISIBLE, OBSTACLE


def _warn(message: str) -> None:
    print(f"[trajectory] warning: {message}")


def _format_background(true_grid: np.ndarray) -> np.ndarray:
    grid = np.asarray(true_grid, dtype=np.int8)
    return np.where(grid == OBSTACLE, 0.15, 1.0).astype(np.float32)


def _format_belief_background(belief_map: np.ndarray) -> np.ndarray:
    belief = np.asarray(belief_map, dtype=np.int8)
    bg = np.full(belief.shape, 0.38, dtype=np.float32)
    bg[belief == EMPTY] = 0.98
    bg[belief == OBSTACLE] = 0.14
    return bg


def _mask_boundary(mask: np.ndarray) -> np.ndarray:
    mask_bool = np.asarray(mask, dtype=bool)
    if mask_bool.ndim != 2 or not np.any(mask_bool):
        return np.zeros_like(mask_bool, dtype=bool)
    p = np.pad(mask_bool, 1, mode="constant", constant_values=False)
    interior = (
        mask_bool &
        p[:-2, 1:-1] &
        p[2:, 1:-1] &
        p[1:-1, :-2] &
        p[1:-1, 2:]
    )
    return mask_bool & (~interior)


def _coords_to_true_grid_mask(
    rows: np.ndarray,
    cols: np.ndarray,
    origin_world_rc: tuple[int, int],
    true_shape: tuple[int, int],
) -> np.ndarray:
    out = np.zeros(true_shape, dtype=bool)
    if rows.size <= 0 or cols.size <= 0:
        return out
    world_rows = np.asarray(rows, dtype=np.int32) + np.int32(origin_world_rc[0])
    world_cols = np.asarray(cols, dtype=np.int32) + np.int32(origin_world_rc[1])
    valid = (
        (world_rows >= 0) &
        (world_rows < int(true_shape[0])) &
        (world_cols >= 0) &
        (world_cols < int(true_shape[1]))
    )
    if not np.any(valid):
        return out
    out[world_rows[valid], world_cols[valid]] = True
    return out
def _episode_semantic_visualization_meta(ep: dict) -> dict[str, object] | None:
    raw = ep.get("semantic_viz")
    return raw if isinstance(raw, dict) else None


def _align_dynamic_map_to_true_grid(
    arr: np.ndarray | None,
    origin_world_rc,
    true_shape: tuple[int, int],
    *,
    fill_value: int,
    dtype,
) -> np.ndarray | None:
    if arr is None or origin_world_rc is None:
        return None

    src = np.asarray(arr, dtype=dtype)
    if src.ndim != 2:
        return None

    out = np.full(true_shape, fill_value, dtype=dtype)
    src_min_r = int(origin_world_rc[0])
    src_min_c = int(origin_world_rc[1])
    src_max_r = src_min_r + int(src.shape[0])
    src_max_c = src_min_c + int(src.shape[1])
    dst_h, dst_w = int(true_shape[0]), int(true_shape[1])

    ov_min_r = max(0, src_min_r)
    ov_min_c = max(0, src_min_c)
    ov_max_r = min(dst_h, src_max_r)
    ov_max_c = min(dst_w, src_max_c)
    if ov_min_r >= ov_max_r or ov_min_c >= ov_max_c:
        return out

    src_r0 = ov_min_r - src_min_r
    src_c0 = ov_min_c - src_min_c
    src_r1 = src_r0 + (ov_max_r - ov_min_r)
    src_c1 = src_c0 + (ov_max_c - ov_min_c)
    out[ov_min_r:ov_max_r, ov_min_c:ov_max_c] = src[src_r0:src_r1, src_c0:src_c1]
    return out


def _trailing_mean(values: list[float], window: int) -> list[float]:
    if window <= 1 or len(values) <= 0:
        return list(values)

    out: list[float] = []
    running = 0.0
    for idx, value in enumerate(values):
        running += float(value)
        if idx >= window:
            running -= float(values[idx - window])
        count = min(idx + 1, window)
        out.append(running / float(count))
    return out


def _centered_rolling_mean(values: list[float], window: int) -> list[float]:
    if len(values) <= 0:
        return []
    window_use = max(1, int(window))
    if (window_use % 2) == 0:
        window_use += 1
    window_use = min(window_use, len(values) if (len(values) % 2 == 1) else max(1, len(values) - 1))
    if window_use <= 1:
        return list(values)

    half = window_use // 2
    prefix = [0.0]
    for value in values:
        prefix.append(prefix[-1] + float(value))

    out: list[float] = []
    for idx in range(len(values)):
        start = max(0, idx - half)
        end = min(len(values), idx + half + 1)
        out.append((prefix[end] - prefix[start]) / float(max(1, end - start)))
    return out


def _episode_numeric_idx(ep: dict, fallback_idx: int) -> int:
    value = ep.get("episode_idx")
    if value is None:
        return int(fallback_idx)
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(fallback_idx)


def _episode_has_numeric_idx(ep: dict) -> bool:
    value = ep.get("episode_idx")
    if value is None:
        return False
    try:
        int(value)
        return True
    except (TypeError, ValueError):
        return False


def _episode_file_label(ep: dict, fallback_idx: int) -> str:
    if _episode_has_numeric_idx(ep):
        return f"ep{_episode_numeric_idx(ep, fallback_idx):04d}"
    return f"slot{int(fallback_idx):02d}"


def _episode_plot_title(prefix: str, ep: dict, fallback_idx: int) -> str:
    ep_label = _episode_file_label(ep, fallback_idx)
    reward = float(ep.get("episode_reward", 0.0))
    coverage = float(ep.get("final_coverage", 0.0))
    repeat = float(ep.get("repeat_visit_ratio", 0.0))
    reason = str(ep.get("done_reason", ""))
    return (
        f"{prefix} {ep_label} reward={reward:.3f} cov={coverage:.3f} "
        f"repeat={repeat:.3f} reason={reason}"
    )


def _selection_summary_lines(meta: dict[str, object]) -> list[str]:
    lines: list[str] = []
    for key in (
        "selection_mode",
        "episodes_total",
        "train_episode_count",
        "candidate_count",
        "success_episode_count",
        "timeout_episode_count",
        "gate_start_episode_idx",
        "gate_start_position",
        "gate_coverage",
        "gate_window",
        "min_coverage",
        "min_length",
        "length_percentile",
        "length_threshold",
        "absolute_low_threshold",
        "local_drop_margin",
        "selection_fallback",
    ):
        if key in meta:
            lines.append(f"{key}: {meta[key]}")

    selected = meta.get("selected")
    if isinstance(selected, list) and len(selected) > 0:
        lines.append("selected:")
        for item in selected:
            if not isinstance(item, dict):
                continue
            line = (
                f"  ep={item.get('episode_idx')} cov={item.get('final_coverage')} "
                f"baseline={item.get('baseline_coverage')} score={item.get('score')} "
                f"reason={item.get('done_reason')}"
            )
            lines.append(line)
    return lines


def _write_selection_summary(trajectories_dir: Path, prefix: str, meta: dict[str, object]) -> None:
    lines = _selection_summary_lines(meta)
    if len(lines) <= 0:
        return
    summary_path = trajectories_dir / f"{prefix}_selection.txt"
    try:
        summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    except Exception as exc:
        _warn(f"failed to write selection summary {summary_path}: {exc}")


def _select_first_episodes(episodes: list[dict], max_episodes: int) -> tuple[list[dict], dict[str, object]]:
    limit = min(int(max_episodes), len(episodes))
    selected = [dict(episodes[idx], _selection_rank=idx + 1) for idx in range(limit)]
    meta = {
        "selection_mode": "first",
        "episodes_total": int(len(episodes)),
        "selected": [
            {
                "episode_idx": _episode_numeric_idx(ep, idx + 1),
                "final_coverage": float(ep.get("final_coverage", 0.0)),
                "baseline_coverage": "",
                "score": "",
                "done_reason": str(ep.get("done_reason", "")),
            }
            for idx, ep in enumerate(selected, start=1)
        ],
    }
    return selected, meta


def _select_lowest_coverage_episodes(
    episodes: list[dict],
    max_episodes: int,
) -> tuple[list[dict], dict[str, object]]:
    ordered = sorted(
        enumerate(episodes),
        key=lambda pair: (
            float(pair[1].get("final_coverage", 0.0)),
            _episode_numeric_idx(pair[1], pair[0] + 1),
        ),
    )
    selected: list[dict] = []
    selected_rows: list[dict[str, object]] = []
    for rank, (source_pos, ep) in enumerate(ordered[: max(0, int(max_episodes))], start=1):
        item = dict(ep)
        item["_selection_rank"] = rank
        selected.append(item)
        selected_rows.append(
            {
                "episode_idx": _episode_numeric_idx(ep, source_pos + 1),
                "final_coverage": f"{float(ep.get('final_coverage', 0.0)):.4f}",
                "baseline_coverage": "",
                "score": "",
                "done_reason": str(ep.get("done_reason", "")),
            }
        )
    return selected, {
        "selection_mode": "lowest_coverage",
        "episodes_total": int(len(episodes)),
        "selected": selected_rows,
    }


def _select_train_low_spike_episodes(
    episodes: list[dict],
    max_episodes: int,
    *,
    gate_coverage: float = 0.80,
    gate_window: int = 20,
    low_percentile: float = 10.0,
    low_cap: float = 0.78,
    low_floor: float = 0.60,
    local_window: int = 21,
    local_drop_margin: float = 0.12,
    min_episode_gap: int = 12,
) -> tuple[list[dict], dict[str, object]]:
    train_eps = [dict(ep) for ep in episodes if str(ep.get("phase", "train")).strip().lower() == "train"]
    if len(train_eps) <= 0:
        return _select_lowest_coverage_episodes(episodes, max_episodes=max_episodes)

    coverages = [float(ep.get("final_coverage", 0.0)) for ep in train_eps]
    trailing = _trailing_mean(coverages, window=gate_window)
    local_baseline = _centered_rolling_mean(coverages, window=local_window)

    gate_start_pos = None
    for idx, mean_cov in enumerate(trailing):
        if (idx + 1) >= int(gate_window) and float(mean_cov) >= float(gate_coverage):
            gate_start_pos = idx
            break
    if gate_start_pos is None:
        gate_start_pos = 0
        selection_fallback = "gate_never_reached"
    else:
        selection_fallback = "none"

    post_coverages = coverages[gate_start_pos:]
    abs_threshold = float(low_cap)
    if len(post_coverages) > 0:
        percentile_value = float(np.percentile(np.asarray(post_coverages, dtype=np.float32), float(low_percentile)))
        abs_threshold = float(max(low_floor, min(low_cap, percentile_value + 0.02)))

    candidates: list[tuple[float, float, int, dict]] = []
    for idx in range(gate_start_pos, len(train_eps)):
        ep = train_eps[idx]
        cov = coverages[idx]
        baseline = local_baseline[idx]
        qualifies = bool((cov <= abs_threshold) or (cov <= (baseline - float(local_drop_margin))))
        if not qualifies:
            continue
        score = max(float(abs_threshold - cov), float(baseline - cov))
        item = dict(ep)
        item["_selection_rank"] = 0
        item["_selection_score"] = float(score)
        item["_selection_baseline_coverage"] = float(baseline)
        item["_selection_threshold_coverage"] = float(abs_threshold)
        candidates.append((float(score), float(cov), idx, item))

    if len(candidates) <= 0:
        selection_fallback = "lowest_coverage_after_gate"
        for idx in range(gate_start_pos, len(train_eps)):
            ep = train_eps[idx]
            cov = coverages[idx]
            item = dict(ep)
            item["_selection_rank"] = 0
            item["_selection_score"] = float(local_baseline[idx] - cov)
            item["_selection_baseline_coverage"] = float(local_baseline[idx])
            item["_selection_threshold_coverage"] = float(abs_threshold)
            candidates.append((float(item["_selection_score"]), float(cov), idx, item))

    candidates.sort(key=lambda item: (-item[0], item[1], _episode_numeric_idx(item[3], item[2] + 1)))
    selected: list[dict] = []
    selected_rows: list[dict[str, object]] = []
    chosen_episode_ids: list[int] = []

    for _, cov, idx, item in candidates:
        episode_id = _episode_numeric_idx(item, idx + 1)
        if any(abs(episode_id - prev) < int(min_episode_gap) for prev in chosen_episode_ids):
            continue
        item["_selection_rank"] = len(selected) + 1
        selected.append(item)
        chosen_episode_ids.append(episode_id)
        selected_rows.append(
            {
                "episode_idx": episode_id,
                "final_coverage": f"{cov:.4f}",
                "baseline_coverage": f"{float(item['_selection_baseline_coverage']):.4f}",
                "score": f"{float(item['_selection_score']):.4f}",
                "done_reason": str(item.get("done_reason", "")),
            }
        )
        if len(selected) >= int(max_episodes):
            break

    if len(selected) < int(max_episodes):
        fallback_order = sorted(
            enumerate(train_eps[gate_start_pos:], start=gate_start_pos),
            key=lambda pair: (
                float(pair[1].get("final_coverage", 0.0)),
                _episode_numeric_idx(pair[1], pair[0] + 1),
            ),
        )
        for idx, ep in fallback_order:
            episode_id = _episode_numeric_idx(ep, idx + 1)
            if episode_id in chosen_episode_ids:
                continue
            item = dict(ep)
            item["_selection_rank"] = len(selected) + 1
            item["_selection_score"] = float(local_baseline[idx] - coverages[idx])
            item["_selection_baseline_coverage"] = float(local_baseline[idx])
            item["_selection_threshold_coverage"] = float(abs_threshold)
            selected.append(item)
            chosen_episode_ids.append(episode_id)
            selected_rows.append(
                {
                    "episode_idx": episode_id,
                    "final_coverage": f"{float(ep.get('final_coverage', 0.0)):.4f}",
                    "baseline_coverage": f"{float(local_baseline[idx]):.4f}",
                    "score": f"{float(item['_selection_score']):.4f}",
                    "done_reason": str(ep.get("done_reason", "")),
                }
            )
            if len(selected) >= int(max_episodes):
                break

    gate_episode_idx = _episode_numeric_idx(train_eps[gate_start_pos], gate_start_pos + 1) if len(train_eps) > 0 else 0
    return selected, {
        "selection_mode": "train_low_spikes",
        "episodes_total": int(len(episodes)),
        "train_episode_count": int(len(train_eps)),
        "gate_start_position": int(gate_start_pos + 1),
        "gate_start_episode_idx": int(gate_episode_idx),
        "gate_coverage": float(gate_coverage),
        "gate_window": int(gate_window),
        "absolute_low_threshold": f"{abs_threshold:.4f}",
        "local_drop_margin": float(local_drop_margin),
        "selection_fallback": selection_fallback,
        "selected": selected_rows,
    }


def _select_train_postgate_failure_episodes(
    episodes: list[dict],
    *,
    gate_coverage: float = 0.80,
    gate_window: int = 100,
    coverage_target: float = 0.95,
) -> tuple[list[dict], dict[str, object]]:
    train_eps = [dict(ep) for ep in episodes if str(ep.get("phase", "train")).strip().lower() == "train"]
    if len(train_eps) <= 0:
        return [], {
            "selection_mode": "train_postgate_failures",
            "episodes_total": int(len(episodes)),
            "train_episode_count": 0,
            "gate_start_position": 0,
            "gate_start_episode_idx": 0,
            "gate_coverage": float(gate_coverage),
            "gate_window": int(gate_window),
            "coverage_target": float(coverage_target),
            "selection_fallback": "no_train_episodes",
            "selected": [],
        }

    coverages = [float(ep.get("final_coverage", 0.0)) for ep in train_eps]
    trailing = _trailing_mean(coverages, window=gate_window)
    gate_start_pos = None
    for idx, mean_cov in enumerate(trailing):
        if (idx + 1) >= int(gate_window) and float(mean_cov) >= float(gate_coverage):
            gate_start_pos = idx
            break

    selection_fallback = "none"
    if gate_start_pos is None:
        gate_start_pos = 0
        selection_fallback = "gate_never_reached"

    selected: list[dict] = []
    selected_rows: list[dict[str, object]] = []
    for idx in range(gate_start_pos, len(train_eps)):
        ep = train_eps[idx]
        cov = float(ep.get("final_coverage", 0.0))
        if cov >= float(coverage_target):
            continue
        item = dict(ep)
        item["_selection_rank"] = len(selected) + 1
        selected.append(item)
        selected_rows.append(
            {
                "episode_idx": _episode_numeric_idx(ep, idx + 1),
                "final_coverage": f"{cov:.4f}",
                "baseline_coverage": "",
                "score": "",
                "done_reason": str(ep.get("done_reason", "")),
            }
        )

    if len(selected) <= 0 and len(train_eps) > 0:
        selection_fallback = "no_failures_after_gate"

    gate_episode_idx = _episode_numeric_idx(train_eps[gate_start_pos], gate_start_pos + 1)
    return selected, {
        "selection_mode": "train_postgate_failures",
        "episodes_total": int(len(episodes)),
        "train_episode_count": int(len(train_eps)),
        "gate_start_position": int(gate_start_pos + 1),
        "gate_start_episode_idx": int(gate_episode_idx),
        "gate_coverage": float(gate_coverage),
        "gate_window": int(gate_window),
        "coverage_target": float(coverage_target),
        "selection_fallback": selection_fallback,
        "selected": selected_rows,
    }


def _select_train_highcov_timeout_episodes(
    episodes: list[dict],
    max_episodes: int,
    *,
    min_coverage: float = 0.85,
) -> tuple[list[dict], dict[str, object]]:
    train_eps = [dict(ep) for ep in episodes if str(ep.get("phase", "train")).strip().lower() == "train"]
    timeout_eps = [
        ep
        for ep in train_eps
        if str(ep.get("done_reason", "")).strip() == "max_episode_steps"
        and float(ep.get("final_coverage", 0.0)) >= float(min_coverage)
    ]
    timeout_eps.sort(
        key=lambda ep: (
            -float(ep.get("final_coverage", 0.0)),
            -float(ep.get("episode_length", 0.0)),
            _episode_numeric_idx(ep, 0),
        )
    )

    selected: list[dict] = []
    selected_rows: list[dict[str, object]] = []
    for rank, ep in enumerate(timeout_eps[: max(0, int(max_episodes))], start=1):
        item = dict(ep)
        item["_selection_rank"] = rank
        selected.append(item)
        selected_rows.append(
            {
                "episode_idx": _episode_numeric_idx(ep, rank),
                "final_coverage": f"{float(ep.get('final_coverage', 0.0)):.4f}",
                "baseline_coverage": "",
                "score": "",
                "done_reason": str(ep.get("done_reason", "")),
            }
        )

    return selected, {
        "selection_mode": "train_highcov_timeout",
        "episodes_total": int(len(episodes)),
        "train_episode_count": int(len(train_eps)),
        "timeout_episode_count": int(len(timeout_eps)),
        "min_coverage": float(min_coverage),
        "selection_fallback": ("no_matching_timeout_episode" if len(timeout_eps) <= 0 else "none"),
        "selected": selected_rows,
    }


def _select_train_long_success_episodes(
    episodes: list[dict],
    max_episodes: int,
    *,
    gate_coverage: float = 0.80,
    gate_window: int = 100,
    min_length: int = 350,
    percentile: float = 85.0,
) -> tuple[list[dict], dict[str, object]]:
    train_eps = [dict(ep) for ep in episodes if str(ep.get("phase", "train")).strip().lower() == "train"]
    if len(train_eps) <= 0:
        return [], {
            "selection_mode": "train_long_success",
            "episodes_total": int(len(episodes)),
            "train_episode_count": 0,
            "gate_start_position": 0,
            "gate_start_episode_idx": 0,
            "gate_coverage": float(gate_coverage),
            "gate_window": int(gate_window),
            "min_length": int(min_length),
            "length_percentile": float(percentile),
            "length_threshold": int(min_length),
            "selection_fallback": "no_train_episodes",
            "selected": [],
        }

    coverages = [float(ep.get("final_coverage", 0.0)) for ep in train_eps]
    trailing = _trailing_mean(coverages, window=gate_window)
    gate_start_pos = None
    for idx, mean_cov in enumerate(trailing):
        if (idx + 1) >= int(gate_window) and float(mean_cov) >= float(gate_coverage):
            gate_start_pos = idx
            break

    selection_fallback = "none"
    if gate_start_pos is None:
        gate_start_pos = 0
        selection_fallback = "gate_never_reached"

    successful_eps = [
        dict(train_eps[idx])
        for idx in range(gate_start_pos, len(train_eps))
        if int(train_eps[idx].get("success", 0)) == 1
        or str(train_eps[idx].get("done_reason", "")).strip() == "coverage_reached"
    ]
    success_lengths = [float(ep.get("episode_length", 0.0)) for ep in successful_eps]
    if len(success_lengths) > 0:
        length_threshold = max(
            int(min_length),
            int(math.ceil(float(np.percentile(np.asarray(success_lengths, dtype=np.float32), float(percentile))))),
        )
    else:
        length_threshold = int(min_length)

    candidates = [
        ep for ep in successful_eps
        if float(ep.get("episode_length", 0.0)) >= float(length_threshold)
    ]
    candidates.sort(
        key=lambda ep: (
            -float(ep.get("episode_length", 0.0)),
            -float(ep.get("final_coverage", 0.0)),
            _episode_numeric_idx(ep, 0),
        )
    )

    selected: list[dict] = []
    selected_rows: list[dict[str, object]] = []
    for rank, ep in enumerate(candidates[: max(0, int(max_episodes))], start=1):
        item = dict(ep)
        item["_selection_rank"] = rank
        selected.append(item)
        selected_rows.append(
            {
                "episode_idx": _episode_numeric_idx(ep, rank),
                "final_coverage": f"{float(ep.get('final_coverage', 0.0)):.4f}",
                "baseline_coverage": "",
                "score": f"{float(ep.get('episode_length', 0.0)):.1f}",
                "done_reason": str(ep.get("done_reason", "")),
            }
        )

    gate_episode_idx = _episode_numeric_idx(train_eps[gate_start_pos], gate_start_pos + 1) if len(train_eps) > 0 else 0
    if len(successful_eps) <= 0:
        selection_fallback = "no_success_after_gate"
    elif len(candidates) <= 0:
        selection_fallback = "no_long_success_above_threshold"
    return selected, {
        "selection_mode": "train_long_success",
        "episodes_total": int(len(episodes)),
        "train_episode_count": int(len(train_eps)),
        "success_episode_count": int(len(successful_eps)),
        "gate_start_position": int(gate_start_pos + 1),
        "gate_start_episode_idx": int(gate_episode_idx),
        "gate_coverage": float(gate_coverage),
        "gate_window": int(gate_window),
        "min_length": int(min_length),
        "length_percentile": float(percentile),
        "length_threshold": int(length_threshold),
        "selection_fallback": selection_fallback,
        "selected": selected_rows,
    }


def _select_train_special_low_coverage_episodes(
    episodes: list[dict],
    max_episodes: int,
    *,
    gate_coverage: float = 0.80,
    gate_window: int = 100,
    absolute_threshold: float = 0.75,
    local_drop_margin: float = 0.12,
) -> tuple[list[dict], dict[str, object]]:
    selected, meta = _select_train_low_spike_episodes(
        episodes,
        max_episodes=max_episodes,
        gate_coverage=gate_coverage,
        gate_window=gate_window,
        low_percentile=10.0,
        low_cap=float(absolute_threshold),
        low_floor=float(absolute_threshold),
        local_window=max(21, int(gate_window)),
        local_drop_margin=local_drop_margin,
        min_episode_gap=12,
    )
    meta["selection_mode"] = "train_special_low_coverage"
    meta["absolute_low_threshold"] = f"{float(absolute_threshold):.4f}"
    meta["local_drop_margin"] = float(local_drop_margin)
    return selected, meta


def _select_episodes(
    episodes: list[dict],
    *,
    selection_mode: str,
    max_episodes: int,
    gate_window: int = 100,
    coverage_target: float = 0.95,
    highcov_timeout_min_coverage: float = 0.85,
    long_success_gate_coverage: float = 0.80,
    long_success_gate_window: int = 100,
    long_success_min_length: int = 350,
    long_success_percentile: float = 85.0,
    lowcov_gate_coverage: float = 0.80,
    lowcov_gate_window: int = 100,
    lowcov_absolute_threshold: float = 0.75,
    lowcov_local_drop_margin: float = 0.12,
) -> tuple[list[dict], dict[str, object]]:
    mode = str(selection_mode).strip().lower()
    if mode == "lowest_coverage":
        return _select_lowest_coverage_episodes(episodes, max_episodes=max_episodes)
    if mode == "train_low_spikes":
        return _select_train_low_spike_episodes(episodes, max_episodes=max_episodes)
    if mode == "train_postgate_failures":
        return _select_train_postgate_failure_episodes(
            episodes,
            gate_window=gate_window,
            coverage_target=coverage_target,
        )
    if mode == "train_highcov_timeout":
        return _select_train_highcov_timeout_episodes(
            episodes,
            max_episodes=max_episodes,
            min_coverage=highcov_timeout_min_coverage,
        )
    if mode == "train_long_success":
        return _select_train_long_success_episodes(
            episodes,
            max_episodes=max_episodes,
            gate_coverage=long_success_gate_coverage,
            gate_window=long_success_gate_window,
            min_length=long_success_min_length,
            percentile=long_success_percentile,
        )
    if mode == "train_special_low_coverage":
        return _select_train_special_low_coverage_episodes(
            episodes,
            max_episodes=max_episodes,
            gate_coverage=lowcov_gate_coverage,
            gate_window=lowcov_gate_window,
            absolute_threshold=lowcov_absolute_threshold,
            local_drop_margin=lowcov_local_drop_margin,
        )
    return _select_first_episodes(episodes, max_episodes=max_episodes)


def _sanitize_token(text: object) -> str:
    token = str(text).strip().lower()
    if token == "":
        return "na"
    cleaned = []
    for ch in token:
        if ch.isalnum():
            cleaned.append(ch)
        else:
            cleaned.append("_")
    out = "".join(cleaned).strip("_")
    return out or "na"


def save_episode_trajectory_plots(
    run_dir: Path,
    episodes: list[dict],
    prefix: str,
    max_episodes: int = 1,
    selection_mode: str = "first",
    gate_window: int = 100,
    coverage_target: float = 0.95,
    output_subdir: str | Path | None = None,
    highcov_timeout_min_coverage: float = 0.85,
    long_success_gate_coverage: float = 0.80,
    long_success_gate_window: int = 100,
    long_success_min_length: int = 350,
    long_success_percentile: float = 85.0,
    lowcov_gate_coverage: float = 0.80,
    lowcov_gate_window: int = 100,
    lowcov_absolute_threshold: float = 0.75,
    lowcov_local_drop_margin: float = 0.12,
) -> list[Path]:
    if len(episodes) <= 0 or max_episodes <= 0:
        return []

    trajectories_dir = Path(run_dir) / "trajectories"
    if output_subdir is not None:
        trajectories_dir = trajectories_dir / Path(output_subdir)
    try:
        trajectories_dir.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        _warn(f"failed to create trajectory directory {trajectories_dir}: {exc}")
        return []

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
    except Exception as exc:
        _warn(f"matplotlib unavailable, skip trajectory plots: {exc}")
        return []

    selected, meta = _select_episodes(
        episodes,
        selection_mode=selection_mode,
        max_episodes=max_episodes,
        gate_window=gate_window,
        coverage_target=coverage_target,
        highcov_timeout_min_coverage=highcov_timeout_min_coverage,
        long_success_gate_coverage=long_success_gate_coverage,
        long_success_gate_window=long_success_gate_window,
        long_success_min_length=long_success_min_length,
        long_success_percentile=long_success_percentile,
        lowcov_gate_coverage=lowcov_gate_coverage,
        lowcov_gate_window=lowcov_gate_window,
        lowcov_absolute_threshold=lowcov_absolute_threshold,
        lowcov_local_drop_margin=lowcov_local_drop_margin,
    )
    _write_selection_summary(trajectories_dir, prefix, meta)

    generated: list[Path] = []
    for render_rank, ep in enumerate(selected, start=1):
        true_grid = ep.get("true_grid")
        trajectory = ep.get("trajectory_positions")
        if true_grid is None or trajectory is None:
            _warn(f"missing trajectory data for {prefix} rank{render_rank}")
            continue

        if len(trajectory) <= 0:
            _warn(f"empty trajectory for {prefix} rank{render_rank}")
            continue

        rows = [int(pos[0]) for pos in trajectory]
        cols = [int(pos[1]) for pos in trajectory]
        true_grid_arr = np.asarray(true_grid, dtype=np.int8)
        true_background = _format_background(true_grid_arr)
        belief_map = _align_dynamic_map_to_true_grid(
            ep.get("belief_map"),
            ep.get("belief_origin_world_rc"),
            true_grid_arr.shape,
            fill_value=INVISIBLE,
            dtype=np.int8,
        )
        semantic_viz_meta = _episode_semantic_visualization_meta(ep)

        fig = None
        try:
            height, width = true_background.shape
            fig_w = max(10.0, min(16.0, width / 4.5))
            fig_h = max(5.0, min(9.0, height / 6.0))

            fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(fig_w, fig_h), squeeze=False)
            ax_true = axes[0][0]
            ax_belief = axes[0][1]

            ax_true.imshow(true_background, cmap="gray", vmin=0.0, vmax=1.0, origin="upper")
            ax_true.plot(cols, rows, color="tab:blue", linewidth=2.0)
            ax_true.scatter([cols[0]], [rows[0]], c="tab:green", marker="o", s=45, label="start")
            ax_true.scatter([cols[-1]], [rows[-1]], c="tab:red", marker="x", s=55, label="end")
            ax_true.set_title("True Map + Trajectory")
            ax_true.legend(loc="upper right")

            if belief_map is not None:
                belief_background = _format_belief_background(belief_map)
                ax_belief.imshow(belief_background, cmap="gray", vmin=0.0, vmax=1.0, origin="upper")
                if semantic_viz_meta is not None:
                    blocks = semantic_viz_meta.get("blocks", [])
                    if isinstance(blocks, list) and len(blocks) > 0:
                        cmap = plt.cm.get_cmap("tab20", max(1, len(blocks)))
                        belief_origin = (
                            int(ep["belief_origin_world_rc"][0]),
                            int(ep["belief_origin_world_rc"][1]),
                        )
                        for draw_idx, block in enumerate(blocks):
                            if not isinstance(block, dict):
                                continue
                            block_mask = _coords_to_true_grid_mask(
                                np.asarray(block.get("rows", []), dtype=np.int32),
                                np.asarray(block.get("cols", []), dtype=np.int32),
                                belief_origin,
                                true_grid_arr.shape,
                            )
                            if np.any(block_mask):
                                overlay = np.zeros((height, width, 4), dtype=np.float32)
                                color = np.asarray(cmap(draw_idx % max(1, cmap.N))[:3], dtype=np.float32)
                                overlay[..., :3] = color
                                overlay[..., 3] = block_mask.astype(np.float32) * 0.16
                                ax_belief.imshow(overlay, origin="upper")

                                rr, cc = np.nonzero(block_mask)
                                rect = Rectangle(
                                    (float(np.min(cc)) - 0.5, float(np.min(rr)) - 0.5),
                                    float(np.max(cc) - np.min(cc) + 1),
                                    float(np.max(rr) - np.min(rr) + 1),
                                    fill=False,
                                    edgecolor=color,
                                    linewidth=1.1,
                                    alpha=0.45,
                                    linestyle="--",
                                )
                                ax_belief.add_patch(rect)

                            frontier_clusters = block.get("frontier_clusters", [])
                            if not isinstance(frontier_clusters, list):
                                continue
                            for frontier_cluster in frontier_clusters:
                                if not isinstance(frontier_cluster, dict):
                                    continue
                                frontier_mask = _coords_to_true_grid_mask(
                                    np.asarray(frontier_cluster.get("frontier_rows", []), dtype=np.int32),
                                    np.asarray(frontier_cluster.get("frontier_cols", []), dtype=np.int32),
                                    belief_origin,
                                    true_grid_arr.shape,
                                )
                                if not np.any(frontier_mask):
                                    continue
                                entry_overlay = np.zeros((height, width, 4), dtype=np.float32)
                                entry_overlay[..., 0] = 0.14
                                entry_overlay[..., 1] = 0.68
                                entry_overlay[..., 2] = 0.88
                                entry_overlay[..., 3] = frontier_mask.astype(np.float32) * 0.62
                                ax_belief.imshow(entry_overlay, origin="upper")

                observed_ratio = float(np.mean(belief_map != INVISIBLE))
                ax_belief.set_title(f"Final Belief + Semantic Blocks ({observed_ratio:.1%} observed)")
            else:
                ax_belief.imshow(true_background, cmap="gray", vmin=0.0, vmax=1.0, origin="upper")
                ax_belief.set_title("Belief Snapshot Unavailable")

            ax_belief.plot(cols, rows, color="tab:blue", linewidth=2.0)
            ax_belief.scatter([cols[0]], [rows[0]], c="tab:green", marker="o", s=45)
            ax_belief.scatter([cols[-1]], [rows[-1]], c="tab:red", marker="x", s=55)

            for ax in (ax_true, ax_belief):
                ax.set_xlim(-0.5, width - 0.5)
                ax.set_ylim(height - 0.5, -0.5)
                ax.set_aspect("equal")
                ax.set_xlabel("col")
                ax.set_ylabel("row")

            fig.suptitle(_episode_plot_title(prefix, ep, fallback_idx=render_rank))
            fig.tight_layout()
            fig.subplots_adjust(top=0.88)

            ep_label = _episode_file_label(ep, fallback_idx=render_rank)
            coverage = float(ep.get("final_coverage", 0.0))
            episode_length = int(ep.get("episode_length", 0))
            done_reason = _sanitize_token(ep.get("done_reason", ""))
            out_path = trajectories_dir / (
                f"{prefix}_{ep_label}_rank{render_rank:02d}_cov{coverage:.3f}"
                f"_len{episode_length:03d}_{done_reason}_trajectory.png"
            )
            fig.savefig(out_path, dpi=150)
            generated.append(out_path)
        except Exception as exc:
            _warn(f"failed to render {prefix} rank{render_rank}: {exc}")
        finally:
            if fig is not None:
                plt.close(fig)

    return generated


def save_train_special_trajectory_plots(
    run_dir: Path,
    episodes: list[dict],
    *,
    highcov_timeout_min_coverage: float = 0.85,
    highcov_timeout_max_plots: int = 5,
    long_success_gate_coverage: float = 0.80,
    long_success_gate_window: int = 100,
    long_success_min_length: int = 350,
    long_success_percentile: float = 85.0,
    long_success_max_plots: int = 5,
    lowcov_gate_coverage: float = 0.80,
    lowcov_gate_window: int = 100,
    lowcov_absolute_threshold: float = 0.75,
    lowcov_local_drop_margin: float = 0.12,
    lowcov_max_plots: int = 5,
) -> list[Path]:
    outputs: list[Path] = []
    outputs.extend(
        save_episode_trajectory_plots(
            run_dir,
            episodes,
            prefix="highcov_timeout",
            output_subdir=Path("train_special_episodes") / "highcov_timeout",
            max_episodes=max(0, int(highcov_timeout_max_plots)),
            selection_mode="train_highcov_timeout",
            highcov_timeout_min_coverage=highcov_timeout_min_coverage,
        )
    )
    outputs.extend(
        save_episode_trajectory_plots(
            run_dir,
            episodes,
            prefix="long_success",
            output_subdir=Path("train_special_episodes") / "long_success",
            max_episodes=max(0, int(long_success_max_plots)),
            selection_mode="train_long_success",
            long_success_gate_coverage=long_success_gate_coverage,
            long_success_gate_window=long_success_gate_window,
            long_success_min_length=long_success_min_length,
            long_success_percentile=long_success_percentile,
        )
    )
    outputs.extend(
        save_episode_trajectory_plots(
            run_dir,
            episodes,
            prefix="low_coverage",
            output_subdir=Path("train_special_episodes") / "low_coverage",
            max_episodes=max(0, int(lowcov_max_plots)),
            selection_mode="train_special_low_coverage",
            lowcov_gate_coverage=lowcov_gate_coverage,
            lowcov_gate_window=lowcov_gate_window,
            lowcov_absolute_threshold=lowcov_absolute_threshold,
            lowcov_local_drop_margin=lowcov_local_drop_margin,
        )
    )
    return outputs
