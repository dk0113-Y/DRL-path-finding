from __future__ import annotations

from env.advantage_state_builder import (
    FRONTIER_CHANNEL_MODE_LOCAL_BINARY,
    FRONTIER_CHANNEL_MODE_LOCAL_GLOBAL_AREA,
    FRONTIER_CHANNEL_MODE_SEMANTIC_BLOCK_AREA_RASTER,
)


FRONTIER_CHANNEL_VARIANT_GROUP = "frontier_channel_variant"


_FRONTIER_CHANNEL_DESCRIPTORS: dict[str, dict[str, object]] = {
    FRONTIER_CHANNEL_MODE_SEMANTIC_BLOCK_AREA_RASTER: {
        "frontier_channel_source": "cumulative_belief_map + SharedSemanticSnapshot",
        "frontier_channel_index_rule": (
            "semantic frontier cluster geometry is projected into the agent-centered local canvas"
        ),
        "frontier_channel_value_rule": (
            "frontier cluster pixels receive block.block_area / total_accessible_unknown_area"
        ),
    },
    FRONTIER_CHANNEL_MODE_LOCAL_BINARY: {
        "frontier_channel_source": "cumulative_belief_map canonical frontier cache",
        "frontier_channel_index_rule": (
            "agent-centered local crop from cum_map.get_frontier_u8(refresh=False)"
        ),
        "frontier_channel_value_rule": "local frontier cells are 1.0; all other cells are 0.0",
    },
    FRONTIER_CHANNEL_MODE_LOCAL_GLOBAL_AREA: {
        "frontier_channel_source": "cumulative_belief_map canonical frontier cache + SharedSemanticSnapshot",
        "frontier_channel_index_rule": (
            "agent-centered local crop from cum_map.get_frontier_u8(refresh=False) is the primary spatial index"
        ),
        "frontier_channel_value_rule": (
            "local frontier cells receive their semantic unknown-block area ratio when matched; "
            "unmatched and non-frontier cells remain 0.0"
        ),
    },
}


def describe_frontier_channel_mode(mode: str) -> dict[str, object]:
    key = str(mode or FRONTIER_CHANNEL_MODE_SEMANTIC_BLOCK_AREA_RASTER).strip().lower()
    try:
        descriptor = _FRONTIER_CHANNEL_DESCRIPTORS[key]
    except KeyError as exc:
        available = ", ".join(sorted(_FRONTIER_CHANNEL_DESCRIPTORS))
        raise ValueError(f"Unknown frontier channel mode {mode!r}. Available modes: {available}") from exc
    return {
        "frontier_channel_mode": key,
        **dict(descriptor),
    }
