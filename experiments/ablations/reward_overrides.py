from __future__ import annotations

from dataclasses import fields, replace

from experiments.ablations.ablation_specs import AblationSpec, is_reward_ablation
from train_q_agent import TrainConfig


def apply_reward_overrides(cfg: TrainConfig, spec: AblationSpec) -> TrainConfig:
    reward_overrides = dict(spec.reward_overrides)
    if not reward_overrides:
        return cfg
    if not is_reward_ablation(spec):
        raise ValueError(
            f"Ablation {spec.ablation_id!r} has reward overrides but is in group {spec.group!r}"
        )

    config_fields = {field.name for field in fields(TrainConfig)}
    invalid = sorted(name for name in reward_overrides if name not in config_fields)
    if invalid:
        raise ValueError(
            "Reward override references fields that are not in TrainConfig: "
            + ", ".join(invalid)
        )
    non_reward = sorted(name for name in reward_overrides if not name.startswith("reward_"))
    if non_reward:
        raise ValueError(
            "Reward override may only target reward_* TrainConfig fields: "
            + ", ".join(non_reward)
        )

    return replace(
        cfg,
        **reward_overrides,
        reward_override=reward_overrides,
        ablation_group=spec.group,
        ablation_id=spec.ablation_id,
    )
