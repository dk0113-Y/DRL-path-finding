from __future__ import annotations

import math


REWARD_BREAKDOWN_FIELDS: tuple[str, ...] = (
    "info_reward_sum",
    "step_penalty_sum",
    "recent_revisit_penalty_sum",
    "stall_penalty_sum",
    "timeout_penalty_sum",
    "terminal_bonus_sum",
)


def zero_reward_breakdown() -> dict[str, float]:
    return {field: 0.0 for field in REWARD_BREAKDOWN_FIELDS}


def add_reward_breakdown(target: dict[str, float], delta: dict[str, float]) -> None:
    for field in REWARD_BREAKDOWN_FIELDS:
        target[field] = float(target.get(field, 0.0) + delta.get(field, 0.0))


def reward_from_breakdown(breakdown: dict[str, float]) -> float:
    return float(sum(float(breakdown.get(field, 0.0)) for field in REWARD_BREAKDOWN_FIELDS))


def _theoretical_visible_cells_from_scan_radius(scan_radius: int) -> int:
    r = int(scan_radius)
    if r < 1:
        raise ValueError("scan_radius must be >= 1")

    count = 0
    rr = r * r
    for dr in range(-r, r + 1):
        for dc in range(-r, r + 1):
            if dr * dr + dc * dc <= rr:
                count += 1

    if r >= 2:
        shoulder = (
            (-r, -1), (-r, 1),
            (r, -1), (r, 1),
            (-1, -r), (1, -r),
            (-1, r), (1, r),
        )
        disk_points = {
            (dr, dc)
            for dr in range(-r, r + 1)
            for dc in range(-r, r + 1)
            if dr * dr + dc * dc <= rr
        }
        count += sum(1 for p in shoulder if p not in disk_points)

    return int(count)


def _infer_scan_radius_from_visible_cells(theoretical_visible_cells: int) -> int | None:
    target = int(theoretical_visible_cells)
    if target <= 0:
        return None

    scan_radius = 1
    while scan_radius <= max(1, target):
        current = _theoretical_visible_cells_from_scan_radius(scan_radius)
        if current == target:
            return scan_radius
        if current > target:
            return None
        scan_radius += 1
    return None


def _resolve_info_norm_denominator(info_norm_override: float | str | None, theoretical_visible_cells: int) -> float:
    if info_norm_override is None:
        return max(1.0, float(theoretical_visible_cells) / 2.0)

    if isinstance(info_norm_override, str):
        mode = info_norm_override.strip().lower()
        if mode == "":
            return max(1.0, float(theoretical_visible_cells) / 2.0)
        if mode in {"half_area", "half_visible_area"}:
            return max(1.0, float(theoretical_visible_cells) / 2.0)
        if mode == "half_perimeter":
            scan_radius = _infer_scan_radius_from_visible_cells(theoretical_visible_cells)
            if scan_radius is None:
                raise ValueError(
                    "Unable to infer scan_radius from theoretical_visible_cells for reward_info_norm='half_perimeter'"
                )
            # Perimeter-scale normalization better matches that single-step new information
            # arrives near the scan boundary rather than scaling with total scan area.
            return max(1.0, math.pi * float(scan_radius))

        numeric_value = float(mode)
        if numeric_value > 0.0:
            return float(numeric_value)
        raise ValueError(f"reward_info_norm must be positive, got {info_norm_override!r}")

    if float(info_norm_override) > 0.0:
        return float(info_norm_override)
    raise ValueError(f"reward_info_norm must be positive, got {info_norm_override!r}")


def resolve_reward_info_norm(info_norm_override: float | str | None, theoretical_visible_cells: int) -> float:
    return _resolve_info_norm_denominator(info_norm_override, theoretical_visible_cells)


def weighted_info_gain(
    *,
    delta_empty: int,
    delta_obstacle: int,
    obstacle_weight: float,
    info_norm: float,
) -> float:
    weighted_delta = float(delta_empty) + (float(obstacle_weight) * float(delta_obstacle))
    return float(weighted_delta / max(1e-6, float(info_norm)))


def valid_step_reward(
    cfg,
    *,
    delta_empty: int,
    delta_obstacle: int,
    reward_info_norm: float,
    recent_revisit: bool,
    stall_triggered: bool,
    success: bool,
) -> float:
    return reward_from_breakdown(
        valid_step_reward_breakdown(
            cfg,
            delta_empty=delta_empty,
            delta_obstacle=delta_obstacle,
            reward_info_norm=reward_info_norm,
            recent_revisit=recent_revisit,
            stall_triggered=stall_triggered,
            success=success,
        )
    )


def valid_step_reward_breakdown(
    cfg,
    *,
    delta_empty: int,
    delta_obstacle: int,
    reward_info_norm: float,
    recent_revisit: bool,
    stall_triggered: bool,
    success: bool,
) -> dict[str, float]:
    # Reward mainline:
    #   weighted information gain
    #   - fixed step cost
    #   - recent-window revisit penalty
    #   - stall penalty after consecutive zero-info steps
    #   + success bonus
    breakdown = zero_reward_breakdown()
    info_gain = weighted_info_gain(
        delta_empty=delta_empty,
        delta_obstacle=delta_obstacle,
        obstacle_weight=float(cfg.reward_obstacle_weight),
        info_norm=reward_info_norm,
    )
    breakdown["info_reward_sum"] = float(cfg.reward_info_scale * info_gain)
    breakdown["step_penalty_sum"] = float(-cfg.reward_step_penalty)

    if recent_revisit:
        breakdown["recent_revisit_penalty_sum"] = float(-cfg.reward_revisit_penalty)
    if stall_triggered:
        breakdown["stall_penalty_sum"] = float(-cfg.reward_stall_penalty)
    if success:
        breakdown["terminal_bonus_sum"] = float(cfg.reward_terminal_bonus)

    return breakdown


def timeout_penalty_breakdown(cfg) -> dict[str, float]:
    breakdown = zero_reward_breakdown()
    breakdown["timeout_penalty_sum"] = float(-cfg.reward_timeout_penalty)
    return breakdown
