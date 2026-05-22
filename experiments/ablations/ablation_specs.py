from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class AblationSpec:
    ablation_id: str
    short_id: str
    group: str
    description: str
    zeroed_channels: tuple[str, ...] = ()
    reward_overrides: dict[str, float] = field(default_factory=dict)
    recommended: bool = True
    notes: tuple[str, ...] = ()


_SPECS: tuple[AblationSpec, ...] = (
    AblationSpec(
        ablation_id="no_frontier_channel",
        short_id="F1",
        group="channel_ablation",
        description="No frontier channel: frontier_block_area_map is zeroed.",
        zeroed_channels=("frontier_block_area_map",),
    ),
    AblationSpec(
        ablation_id="no_visit_count_channel",
        short_id="F2",
        group="channel_ablation",
        description="No visit count channel: visit_count_log_norm is zeroed.",
        zeroed_channels=("visit_count_log_norm",),
    ),
    AblationSpec(
        ablation_id="no_recent_trajectory_channel",
        short_id="F3",
        group="channel_ablation",
        description="No recent trajectory channel: recent_trajectory_decay is zeroed.",
        zeroed_channels=("recent_trajectory_decay",),
    ),
    AblationSpec(
        ablation_id="no_visit_traj_channels",
        short_id="F4",
        group="channel_ablation",
        description="No visit + trajectory channels: behavior-memory channels are zeroed.",
        zeroed_channels=("visit_count_log_norm", "recent_trajectory_decay"),
    ),
    AblationSpec(
        ablation_id="occupancy_only_canvas",
        short_id="F5",
        group="channel_ablation",
        description="Occupancy-only advantage canvas: keep free/obstacle, zero extra semantic channels.",
        zeroed_channels=("frontier_block_area_map", "visit_count_log_norm", "recent_trajectory_decay"),
    ),
    AblationSpec(
        ablation_id="no_step_penalty",
        short_id="R1",
        group="reward_ablation",
        description="No step penalty.",
        reward_overrides={"reward_step_penalty": 0.0},
    ),
    AblationSpec(
        ablation_id="no_revisit_penalty",
        short_id="R2",
        group="reward_ablation",
        description="No revisit penalty.",
        reward_overrides={"reward_revisit_penalty": 0.0},
    ),
    AblationSpec(
        ablation_id="no_turn_penalty",
        short_id="R3",
        group="reward_ablation",
        description="No turn penalty.",
        reward_overrides={"reward_turn_penalty_scale": 0.0},
    ),
    AblationSpec(
        ablation_id="no_timeout_penalty",
        short_id="R4",
        group="reward_ablation",
        description="No timeout penalty.",
        reward_overrides={"reward_timeout_penalty": 0.0},
    ),
    AblationSpec(
        ablation_id="no_efficiency_penalties",
        short_id="R5",
        group="reward_ablation",
        description="No efficiency penalties: step, revisit, turn, and timeout penalties are zeroed.",
        reward_overrides={
            "reward_step_penalty": 0.0,
            "reward_revisit_penalty": 0.0,
            "reward_turn_penalty_scale": 0.0,
            "reward_timeout_penalty": 0.0,
        },
    ),
    AblationSpec(
        ablation_id="sparse_reward_variant",
        short_id="R6",
        group="reward_ablation",
        description="Sparse reward variant: keep terminal success bonus only.",
        reward_overrides={
            "reward_info_scale": 0.0,
            "reward_obstacle_weight": 0.0,
            "reward_step_penalty": 0.0,
            "reward_revisit_penalty": 0.0,
            "reward_turn_penalty_scale": 0.0,
            "reward_timeout_penalty": 0.0,
        },
        recommended=False,
        notes=(
            "Keeps reward_terminal_bonus unchanged.",
            "Enhanced/unstable variant; not recommended for first formal batch.",
        ),
    ),
)


def list_ablation_specs() -> list[AblationSpec]:
    return list(_SPECS)


def ablation_slug(spec: AblationSpec) -> str:
    return f"{spec.short_id}_ablation_{spec.ablation_id}"


def is_channel_ablation(spec: AblationSpec) -> bool:
    return spec.group == "channel_ablation"


def is_reward_ablation(spec: AblationSpec) -> bool:
    return spec.group == "reward_ablation"


def _alias_map() -> dict[str, AblationSpec]:
    aliases: dict[str, AblationSpec] = {}
    for spec in _SPECS:
        aliases[spec.short_id.lower()] = spec
        aliases[spec.ablation_id.lower()] = spec
    return aliases


def get_ablation_spec(ablation_id_or_short_id: str) -> AblationSpec:
    key = str(ablation_id_or_short_id).strip().lower()
    aliases = _alias_map()
    if key in aliases:
        return aliases[key]
    available = ", ".join(
        f"{spec.short_id}/{spec.ablation_id}" for spec in _SPECS
    )
    raise ValueError(
        f"Unknown ablation id {ablation_id_or_short_id!r}. Available IDs: {available}"
    )
