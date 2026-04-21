from __future__ import annotations

from dataclasses import asdict, is_dataclass
import shutil
from pathlib import Path
from typing import Dict

import torch


class CheckpointManager:
    """Manage formal checkpoints for training and post-hoc selection protocols."""

    def __init__(self, run_dir: Path):
        self.run_dir = Path(run_dir)
        self.ckpt_dir = self.run_dir / "checkpoints"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.model_select_dir = self.ckpt_dir / "model_select"
        self.last_path = self.ckpt_dir / "last.pt"
        self.best_path = self.ckpt_dir / "best.pt"

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
        train_config=None,
        selection_metadata: Dict[str, object] | None = None,
    ) -> Dict[str, object]:
        payload: Dict[str, object] = {
            "online_state_dict": online_net.state_dict(),
            "env_steps": int(env_steps),
            "learn_steps": int(getattr(learner, "learn_steps", 0)),
            "train_config": self._serialize_config(train_config),
        }
        if train_episode_idx is not None:
            payload["train_episode_idx"] = int(train_episode_idx)
        if selection_metadata is not None:
            payload["selection_metadata"] = dict(selection_metadata)
        if learner is not None and hasattr(learner, "optimizer"):
            payload["optimizer_state_dict"] = learner.optimizer.state_dict()
        return payload

    def save_last(
        self,
        online_net,
        learner,
        env_steps: int,
        train_episode_idx: int | None = None,
        train_config=None,
    ) -> Path:
        payload = self._build_payload(
            online_net,
            learner,
            env_steps,
            train_episode_idx=train_episode_idx,
            train_config=train_config,
        )
        torch.save(payload, self.last_path)
        return self.last_path

    def save_model_select_candidate(
        self,
        online_net,
        learner,
        env_steps: int,
        train_episode_idx: int | None = None,
        train_config=None,
        selection_metadata: Dict[str, object] | None = None,
    ) -> Path:
        """Save a checkpoint candidate that can later participate in top-k recheck."""

        self.model_select_dir.mkdir(parents=True, exist_ok=True)
        path = self.model_select_dir / f"env_{int(env_steps):09d}.pt"
        payload = self._build_payload(
            online_net,
            learner,
            env_steps,
            train_episode_idx=train_episode_idx,
            train_config=train_config,
            selection_metadata=selection_metadata,
        )
        torch.save(payload, path)
        return path

    def save_periodic_checkpoint(
        self,
        online_net,
        learner,
        env_steps: int,
        train_episode_idx: int | None = None,
        train_config=None,
        selection_metadata: Dict[str, object] | None = None,
    ) -> Path:
        """Save a train-only periodic checkpoint for post-hoc candidate selection."""

        path = self.ckpt_dir / f"ckpt_step_{int(env_steps)}.pt"
        payload = self._build_payload(
            online_net,
            learner,
            env_steps,
            train_episode_idx=train_episode_idx,
            train_config=train_config,
            selection_metadata=selection_metadata,
        )
        torch.save(payload, path)
        return path

    def save_best_from_checkpoint(
        self,
        checkpoint_path: Path,
        selection_metadata: Dict[str, object] | None = None,
    ) -> Path:
        """
        Promote an already-saved candidate to best.pt.

        The model-selection rule is enforced by the training loop. This method
        only copies the selected payload and attaches the final selection
        metadata so best.pt is the formal representative network.
        """

        source = Path(checkpoint_path)
        if not source.exists():
            raise FileNotFoundError(f"checkpoint candidate does not exist: {source}")
        if selection_metadata is None:
            shutil.copy2(source, self.best_path)
            return self.best_path

        payload = torch.load(source, map_location="cpu", weights_only=False)
        payload["selection_metadata"] = dict(selection_metadata)
        torch.save(payload, self.best_path)
        return self.best_path
