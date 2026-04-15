from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Dict, Optional

import torch


class CheckpointManager:
    """Manage the formal last checkpoint for final-probe evaluation."""

    def __init__(self, run_dir: Path):
        self.run_dir = Path(run_dir)
        self.ckpt_dir = self.run_dir / "checkpoints"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.last_path = self.ckpt_dir / "last.pt"

    @staticmethod
    def _serialize_config(cfg) -> object:
        if cfg is None:
            return None
        if is_dataclass(cfg):
            return asdict(cfg)
        return cfg

    def _build_payload(
        self,
        online_net,
        learner,
        env_steps: int,
        train_episode_idx: int | None = None,
        eval_metrics: Optional[Dict[str, object]] = None,
        train_config=None,
    ) -> Dict[str, object]:
        payload: Dict[str, object] = {
            "online_state_dict": online_net.state_dict(),
            "env_steps": int(env_steps),
            "learn_steps": int(getattr(learner, "learn_steps", 0)),
            "eval_metrics": eval_metrics,
            "train_config": self._serialize_config(train_config),
        }
        if train_episode_idx is not None:
            payload["train_episode_idx"] = int(train_episode_idx)
        if learner is not None and hasattr(learner, "optimizer"):
            payload["optimizer_state_dict"] = learner.optimizer.state_dict()
        return payload

    def save_last(
        self,
        online_net,
        learner,
        env_steps: int,
        train_episode_idx: int | None = None,
        eval_metrics: Optional[Dict[str, object]] = None,
        train_config=None,
    ) -> Path:
        payload = self._build_payload(
            online_net,
            learner,
            env_steps,
            train_episode_idx=train_episode_idx,
            eval_metrics=eval_metrics,
            train_config=train_config,
        )
        torch.save(payload, self.last_path)
        return self.last_path
