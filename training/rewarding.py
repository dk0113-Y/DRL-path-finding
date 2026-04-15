from __future__ import annotations

import math


REWARD_BREAKDOWN_FIELDS: tuple[str, ...] = (
    "info_reward_sum",
    "step_penalty_sum",
    "recent_revisit_penalty_sum",
    "stall_penalty_sum",
    "turn_penalty_sum",
    "timeout_penalty_sum",
    "terminal_bonus_sum",
)
REWARD_EVENT_SUMMARY_FIELDS: tuple[str, ...] = (
    "delta_empty_sum",
    "delta_obstacle_sum",
    "empty_info_gain_sum",
    "obstacle_info_gain_sum",
    "weighted_obstacle_info_gain_sum",
    "weighted_info_gain_sum",
    "empty_info_reward_sum",
    "obstacle_info_reward_sum",
    "obstacle_info_contribution_ratio",
    "recent_revisit_trigger_count",
    "stall_trigger_count",
    "zero_info_step_count",
    "turn_ge_90_count",
    "turn_135_count",
    "turn_180_count",
    "turn_penalty_weight_sum",
    "timeout_flag",
)


def zero_reward_breakdown() -> dict[str, float]:
    return {field: 0.0 for field in REWARD_BREAKDOWN_FIELDS}


def zero_reward_event_summary() -> dict[str, float]:
    return {field: 0.0 for field in REWARD_EVENT_SUMMARY_FIELDS}


def add_reward_breakdown(target: dict[str, float], delta: dict[str, float]) -> None:
    for field in REWARD_BREAKDOWN_FIELDS:
        target[field] = float(target.get(field, 0.0) + delta.get(field, 0.0))


def reward_from_breakdown(breakdown: dict[str, float]) -> float:
    return float(sum(float(breakdown.get(field, 0.0)) for field in REWARD_BREAKDOWN_FIELDS))


def fixed_half_perimeter_info_norm(scan_radius: int) -> float:
    r = int(scan_radius)
    if r < 1:
        raise ValueError("scan_radius must be >= 1")
    # Phase-1 reward cleanup fixes information normalization to the scan half-perimeter.
    return float(math.pi * float(r))


def info_gain_components(
    *,
    delta_empty: int,
    delta_obstacle: int,
    obstacle_weight: float,
    info_norm: float,
    reward_info_scale: float,
) -> dict[str, float]:
    denominator = max(1e-6, float(info_norm))
    empty_info_gain = float(delta_empty) / denominator
    obstacle_info_gain = float(delta_obstacle) / denominator
    weighted_obstacle_info_gain = float(obstacle_weight) * obstacle_info_gain
    weighted_info_gain_total = empty_info_gain + weighted_obstacle_info_gain
    empty_info_reward = float(reward_info_scale) * empty_info_gain
    obstacle_info_reward = float(reward_info_scale) * weighted_obstacle_info_gain
    total_info_reward = empty_info_reward + obstacle_info_reward
    obstacle_info_contribution_ratio = (
        float(obstacle_info_reward / total_info_reward) if total_info_reward > 1e-6 else 0.0
    )
    return {
        "empty_info_gain_sum": empty_info_gain,
        "obstacle_info_gain_sum": obstacle_info_gain,
        "weighted_obstacle_info_gain_sum": weighted_obstacle_info_gain,
        "weighted_info_gain_sum": weighted_info_gain_total,
        "empty_info_reward_sum": empty_info_reward,
        "obstacle_info_reward_sum": obstacle_info_reward,
        "info_reward_sum": total_info_reward,
        "obstacle_info_contribution_ratio": obstacle_info_contribution_ratio,
    }


def weighted_info_gain(
    *,
    delta_empty: int,
    delta_obstacle: int,
    obstacle_weight: float,
    info_norm: float,
) -> float:
    return float(
        info_gain_components(
            delta_empty=delta_empty,
            delta_obstacle=delta_obstacle,
            obstacle_weight=obstacle_weight,
            info_norm=info_norm,
            reward_info_scale=1.0,
        )["weighted_info_gain_sum"]
    )


def finalize_reward_event_summary(summary: dict[str, float]) -> dict[str, float]:
    finalized = {field: float(summary.get(field, 0.0)) for field in REWARD_EVENT_SUMMARY_FIELDS}
    total_info_reward = float(finalized.get("empty_info_reward_sum", 0.0)) + float(
        finalized.get("obstacle_info_reward_sum", 0.0)
    )
    finalized["obstacle_info_contribution_ratio"] = (
        float(finalized["obstacle_info_reward_sum"] / total_info_reward) if total_info_reward > 1e-6 else 0.0
    )
    return finalized


def turn_penalty_weight_from_steps(
    turn_steps: int,
    *,
    weight_45: float,
    weight_90: float,
    weight_135: float,
    weight_180: float,
) -> float:
    steps = int(turn_steps)
    if steps <= 0:
        return 0.0
    if steps == 1:
        return float(weight_45)
    if steps == 2:
        return float(weight_90)
    if steps == 3:
        return float(weight_135)
    return float(weight_180)


def valid_step_reward(
    cfg,
    *,
    delta_empty: int,
    delta_obstacle: int,
    info_norm: float,
    recent_revisit: bool,
    stall_triggered: bool,
    turn_penalty_weight: float = 0.0,
    success: bool,
) -> float:
    return reward_from_breakdown(
        valid_step_reward_breakdown(
            cfg,
            delta_empty=delta_empty,
            delta_obstacle=delta_obstacle,
            info_norm=info_norm,
            recent_revisit=recent_revisit,
            stall_triggered=stall_triggered,
            turn_penalty_weight=turn_penalty_weight,
            success=success,
        )
    )


def valid_step_reward_breakdown(
    cfg,
    *,
    delta_empty: int,
    delta_obstacle: int,
    info_norm: float,
    recent_revisit: bool,
    stall_triggered: bool,
    turn_penalty_weight: float = 0.0,
    success: bool,
) -> dict[str, float]:
    # Reward mainline:
    #   weighted information gain
    #   - fixed step cost
    #   - recent revisit penalty over the trajectory_history_steps horizon
    #   - stall penalty after consecutive zero-info steps
    #   - turn penalty = reward_turn_penalty_scale * explicit angle weight
    #   + success bonus
    breakdown = zero_reward_breakdown()
    info_metrics = info_gain_components(
        delta_empty=delta_empty,
        delta_obstacle=delta_obstacle,
        obstacle_weight=float(cfg.reward_obstacle_weight),
        info_norm=info_norm,
        reward_info_scale=float(cfg.reward_info_scale),
    )
    breakdown["info_reward_sum"] = float(info_metrics["info_reward_sum"])
    breakdown["step_penalty_sum"] = float(-cfg.reward_step_penalty)

    if recent_revisit:
        breakdown["recent_revisit_penalty_sum"] = float(-cfg.reward_revisit_penalty)
    if stall_triggered:
        breakdown["stall_penalty_sum"] = float(-cfg.reward_stall_penalty)
    if float(turn_penalty_weight) > 0.0:
        breakdown["turn_penalty_sum"] = float(-cfg.reward_turn_penalty_scale * float(turn_penalty_weight))
    if success:
        breakdown["terminal_bonus_sum"] = float(cfg.reward_terminal_bonus)

    return breakdown


def timeout_penalty_breakdown(cfg) -> dict[str, float]:
    breakdown = zero_reward_breakdown()
    breakdown["timeout_penalty_sum"] = float(-cfg.reward_timeout_penalty)
    return breakdown
