from __future__ import annotations


_BATCH_PRESETS: dict[str, list[str]] = {
    "recommended_first_batch": ["F1", "F4", "F5", "R5"],
    "structural_core_batch": ["D"],
    "structural_extended_batch": ["D", "E"],
    "semantic_core_batch": ["E"],
    "minimum_closure_batch": ["D", "F5", "R5"],
    "frontier_channel_variant_batch": ["F6", "F7"],
    "full_fr_batch": ["F1", "F2", "F3", "F4", "F5", "R1", "R2", "R3", "R4", "R5"],
    "extended_fr_batch": ["F1", "F2", "F3", "F4", "F5", "R1", "R2", "R3", "R4", "R5", "R6"],
}


def list_batch_presets() -> dict[str, list[str]]:
    return {name: list(ablation_ids) for name, ablation_ids in _BATCH_PRESETS.items()}


def get_batch_preset(name: str) -> list[str]:
    key = str(name).strip()
    if key in _BATCH_PRESETS:
        return list(_BATCH_PRESETS[key])
    available = ", ".join(sorted(_BATCH_PRESETS))
    raise ValueError(f"Unknown ablation batch preset {name!r}. Available presets: {available}")
