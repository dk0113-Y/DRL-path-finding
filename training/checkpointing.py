from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Dict, Optional

import torch


class CheckpointManager:
    """Manage last/best checkpoints for DDQN training."""

    def __init__(self, run_dir: Path):
        self.run_dir = Path(run_dir)
        self.ckpt_dir = self.run_dir / "checkpoints"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        self.last_path = self.ckpt_dir / "last.pt"
        self.best_path = self.ckpt_dir / "best.pt"

        self.best_success_rate = float("-inf")
        self.best_mean_coverage = float("-inf")

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
        if learner is not None and hasattr(learner, "optimizer"):
            payload["optimizer_state_dict"] = learner.optimizer.state_dict()
        return payload

    def save_last(
        self,
        online_net,
        learner,
        env_steps: int,
        eval_metrics: Optional[Dict[str, object]] = None,
        train_config=None,
    ) -> Path:
        payload = self._build_payload(online_net, learner, env_steps, eval_metrics=eval_metrics, train_config=train_config)
        torch.save(payload, self.last_path)
        return self.last_path

    def maybe_save_best(
        self,
        online_net,
        learner,
        env_steps: int,
        eval_metrics: Dict[str, object],
        train_config=None,
    ) -> bool:
        success = float(eval_metrics.get("eval_success_rate", 0.0))
        coverage = float(eval_metrics.get("eval_mean_coverage", 0.0))

        better = (success > self.best_success_rate) or (
            success == self.best_success_rate and coverage > self.best_mean_coverage
        )
        if not better:
            return False

        self.best_success_rate = success
        self.best_mean_coverage = coverage

        payload = self._build_payload(online_net, learner, env_steps, eval_metrics=eval_metrics, train_config=train_config)
        torch.save(payload, self.best_path)
        return True
