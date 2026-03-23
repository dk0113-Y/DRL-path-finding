from __future__ import annotations

import time
import warnings
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn.functional as F

from agents.q_value_agent import masked_q_values
from training.replay_buffer import ReplayBuffer


@dataclass(frozen=True)
class DDQNLearnerConfig:
    batch_size: int = 64
    min_replay_size: int = 2_000

    learning_rate: float = 1.0e-4
    weight_decay: float = 0.0
    grad_clip_norm: float = 10.0

    target_update_interval: int = 500
    enable_amp: bool = False
    amp_dtype: str = "fp16"
    enable_timing: bool = False
    return_debug_stats_every: int = 1


class DDQNLearner:
    """
    Double DQN learner with hard target sync.

    Target:
      a* = argmax_a Q_online(s', a) under next_action_mask
      y  = r_n + bootstrap_discount * Q_target(s', a*)
    """

    def __init__(
        self,
        online_net,
        target_net,
        cfg: Optional[DDQNLearnerConfig] = None,
        device: str = "cpu",
    ):
        self.cfg = cfg if cfg is not None else DDQNLearnerConfig()
        self.device = torch.device(device)
        self.amp_dtype = str(self.cfg.amp_dtype).lower()
        if self.amp_dtype not in {"fp16", "bf16"}:
            raise ValueError(f"Unsupported amp_dtype: {self.cfg.amp_dtype!r}; expected 'fp16' or 'bf16'")

        self.online_net = online_net.to(self.device)
        self.target_net = target_net.to(self.device)
        self.hard_update_target()

        self.target_net.eval()
        for p in self.target_net.parameters():
            p.requires_grad = False

        self.optimizer = torch.optim.Adam(
            self.online_net.parameters(),
            lr=float(self.cfg.learning_rate),
            weight_decay=float(self.cfg.weight_decay),
        )

        self.enable_amp = False
        self.autocast_dtype: Optional[torch.dtype] = None
        self.use_grad_scaler = False
        self.scaler = None
        if self.device.type == "cuda" and bool(self.cfg.enable_amp):
            requested_amp_dtype = torch.float16 if self.amp_dtype == "fp16" else torch.bfloat16
            if requested_amp_dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
                warnings.warn(
                    "DDQNLearner AMP bf16 was requested on CUDA, but bf16 is not supported on this device; "
                    "falling back to FP32 for correctness.",
                    stacklevel=2,
                )
            else:
                self.enable_amp = True
                self.autocast_dtype = requested_amp_dtype
                self.use_grad_scaler = requested_amp_dtype == torch.float16
                # GradScaler(enabled=False) acts as a clear no-op path for bf16 autocast.
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", FutureWarning)
                    self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_grad_scaler)

        self.learn_steps = 0
        self.sample_time = 0.0
        self.forward_backward_time = 0.0
        self.target_compute_time = 0.0
        self.optimizer_time = 0.0

    def hard_update_target(self) -> None:
        self.target_net.load_state_dict(self.online_net.state_dict())

    def _autocast_context(self):
        if not self.enable_amp or self.autocast_dtype is None:
            return nullcontext()
        return torch.amp.autocast(
            device_type=self.device.type,
            dtype=self.autocast_dtype,
            enabled=True,
        )

    def _should_return_debug_stats(self) -> bool:
        every = max(1, int(self.cfg.return_debug_stats_every))
        return every <= 1 or (self.learn_steps % every) == 0

    @torch.no_grad()
    def _compute_target_q(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        with self._autocast_context():
            q_next_online = self.online_net(
                batch["next_near_map"],
                batch["next_mid_map"],
                batch["next_frontier_tokens"],
                frontier_token_mask=batch["next_frontier_token_mask"],
                return_aux=False,
            )
            q_next_online = masked_q_values(q_next_online.float(), batch["next_action_mask"])
            next_action = torch.argmax(q_next_online, dim=1)

            q_next_target = self.target_net(
                batch["next_near_map"],
                batch["next_mid_map"],
                batch["next_frontier_tokens"],
                frontier_token_mask=batch["next_frontier_token_mask"],
                return_aux=False,
            ).float()
            next_q = q_next_target.gather(1, next_action.unsqueeze(1)).squeeze(1)

        target_q = batch["reward"] + batch["bootstrap_discount"] * next_q
        return target_q

    def train_step(self, replay: ReplayBuffer) -> Optional[Dict[str, float]]:
        if len(replay) < int(self.cfg.min_replay_size):
            return None
        if not replay.can_sample(int(self.cfg.batch_size)):
            return None

        timing_enabled = bool(self.cfg.enable_timing)

        t0 = time.perf_counter() if timing_enabled else 0.0
        batch = replay.sample(int(self.cfg.batch_size), device=self.device)
        if timing_enabled:
            self.sample_time += time.perf_counter() - t0

        self.online_net.train()

        with torch.no_grad():
            target_t0 = time.perf_counter() if timing_enabled else 0.0
            target_q = self._compute_target_q(batch)
            if timing_enabled:
                self.target_compute_time += time.perf_counter() - target_t0

        t0 = time.perf_counter() if timing_enabled else 0.0
        with self._autocast_context():
            q_all = self.online_net(
                batch["near_map"],
                batch["mid_map"],
                batch["frontier_tokens"],
                frontier_token_mask=batch["frontier_token_mask"],
                return_aux=False,
            )
            q_sa = q_all.gather(1, batch["action"].long().unsqueeze(1)).squeeze(1)

        # Keep the loss/TD path in float32 for stable scalar semantics regardless of autocast mode.
        q_sa_fp32 = q_sa.float()
        target_q_fp32 = target_q.float()
        td_err = target_q_fp32 - q_sa_fp32
        loss_each = F.smooth_l1_loss(q_sa_fp32, target_q_fp32, reduction="none")
        weights = batch.get("weights", torch.ones_like(loss_each))
        loss = (loss_each * weights.float()).mean()

        self.optimizer.zero_grad(set_to_none=True)
        if self.enable_amp and self.scaler is not None:
            scaled_loss = self.scaler.scale(loss)
            scaled_loss.backward()
            self.scaler.unscale_(self.optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.online_net.parameters(),
                max_norm=float(self.cfg.grad_clip_norm),
            )
        else:
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.online_net.parameters(),
                max_norm=float(self.cfg.grad_clip_norm),
            )
        if timing_enabled:
            self.forward_backward_time += time.perf_counter() - t0

        t0 = time.perf_counter() if timing_enabled else 0.0
        if self.enable_amp and self.scaler is not None:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
        if timing_enabled:
            self.optimizer_time += time.perf_counter() - t0

        self.online_net.eval()

        self.learn_steps += 1
        target_synced = 0.0
        if self.learn_steps % int(self.cfg.target_update_interval) == 0:
            self.hard_update_target()
            target_synced = 1.0

        replay.update_priorities(batch["indices"], td_err.detach().abs() + 1e-6)

        out: Dict[str, float] = {
            "learn_steps": float(self.learn_steps),
            "target_synced": float(target_synced),
        }
        if not self._should_return_debug_stats():
            return out

        out.update(
            {
                "loss": float(loss.item()),
                "q_mean": float(q_sa_fp32.mean().item()),
                "target_q_mean": float(target_q_fp32.mean().item()),
                "td_abs_mean": float(td_err.detach().abs().mean().item()),
                "grad_norm": float(grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm),
            }
        )
        if timing_enabled:
            out["sample_time"] = float(self.sample_time)
            out["target_compute_time"] = float(self.target_compute_time)
            out["forward_backward_time"] = float(self.forward_backward_time)
            out["optimizer_time"] = float(self.optimizer_time)
        return out

    def get_timing_stats(self) -> Dict[str, float]:
        return {
            "sample_time": float(self.sample_time),
            "target_compute_time": float(self.target_compute_time),
            "forward_backward_time": float(self.forward_backward_time),
            "optimizer_time": float(self.optimizer_time),
        }
