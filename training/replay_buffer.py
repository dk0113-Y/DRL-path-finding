from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, List, Mapping, Optional

import torch


TransitionDict = Dict[str, torch.Tensor | float | int | bool]


@dataclass(frozen=True)
class ReplayBufferConfig:
    capacity: int = 100_000
    prioritized: bool = False  # reserved extension point; uniform replay by default
    pin_memory: bool = True  # Performance-only data-path toggle; replay semantics stay unchanged.
    non_blocking_transfer: bool = True  # Performance-only H2D toggle; sampling semantics stay unchanged.
    channels_last_on_cuda: bool = False  # Tensor-layout toggle only; storage/sampling semantics stay unchanged.
    enable_timing: bool = False  # Performance-only profiling toggle; add/sample behavior is unchanged.


class NStepTransitionBuilder:
    """
    Build n-step transitions from 1-step transitions.

    Required step-transition keys:
      advantage_canvas, value_block_features, value_entry_features,
      value_block_mask, value_entry_mask, action_mask
      action, reward
      next_advantage_canvas, next_value_block_features, next_value_entry_features,
      next_value_block_mask, next_value_entry_mask, next_action_mask
      done

    Produced transition adds:
      bootstrap_discount (gamma ** k, or 0 when terminal is reached in window)
    """

    def __init__(self, n_step: int = 3, gamma: float = 0.99):
        if n_step < 1:
            raise ValueError("n_step must be >= 1")
        if not (0.0 <= gamma <= 1.0):
            raise ValueError("gamma must be in [0, 1]")
        self.n_step = int(n_step)
        self.gamma = float(gamma)
        self._queue: Deque[TransitionDict] = deque()

    def _build_from_prefix(self) -> TransitionDict:
        first = self._queue[0]

        reward_sum = 0.0
        discount = 1.0
        steps_used = 0
        terminal = False
        last = first

        for item in self._queue:
            reward_sum += discount * float(item["reward"])
            steps_used += 1
            last = item
            terminal = bool(item["done"])
            if terminal or steps_used >= self.n_step:
                break
            discount *= self.gamma

        bootstrap_discount = 0.0 if terminal else float(self.gamma ** steps_used)

        out: TransitionDict = {
            "advantage_canvas": first["advantage_canvas"],
            "value_block_features": first["value_block_features"],
            "value_entry_features": first["value_entry_features"],
            "value_block_mask": first["value_block_mask"],
            "value_entry_mask": first["value_entry_mask"],
            "action_mask": first["action_mask"],
            "action": int(first["action"]),
            "reward": float(reward_sum),
            "next_advantage_canvas": last["next_advantage_canvas"],
            "next_value_block_features": last["next_value_block_features"],
            "next_value_entry_features": last["next_value_entry_features"],
            "next_value_block_mask": last["next_value_block_mask"],
            "next_value_entry_mask": last["next_value_entry_mask"],
            "next_action_mask": last["next_action_mask"],
            "done": bool(terminal),
            "bootstrap_discount": float(bootstrap_discount),
        }
        return out

    def append(self, step_transition: TransitionDict) -> List[TransitionDict]:
        ready: List[TransitionDict] = []
        self._queue.append(step_transition)

        if bool(step_transition["done"]):
            while self._queue:
                ready.append(self._build_from_prefix())
                self._queue.popleft()
            return ready

        if len(self._queue) >= self.n_step:
            ready.append(self._build_from_prefix())
            self._queue.popleft()

        return ready

    def flush(self) -> List[TransitionDict]:
        ready: List[TransitionDict] = []
        while self._queue:
            ready.append(self._build_from_prefix())
            self._queue.popleft()
        return ready


class ReplayBuffer:
    """
    Uniform replay buffer for current tensor state structure.

    Stored keys:
      advantage_canvas, value_block_features, value_entry_features,
      value_block_mask, value_entry_mask, action_mask
      action, reward
      next_advantage_canvas, next_value_block_features, next_value_entry_features,
      next_value_block_mask, next_value_entry_mask, next_action_mask
      done, bootstrap_discount
    """

    _TENSOR_FIELDS = (
        ("advantage_canvas", torch.float32, 3),
        ("value_block_features", torch.float32, 2),
        ("value_entry_features", torch.float32, 3),
        ("value_block_mask", torch.bool, 1),
        ("value_entry_mask", torch.bool, 2),
        ("action_mask", torch.bool, 1),
        ("next_advantage_canvas", torch.float32, 3),
        ("next_value_block_features", torch.float32, 2),
        ("next_value_entry_features", torch.float32, 3),
        ("next_value_block_mask", torch.bool, 1),
        ("next_value_entry_mask", torch.bool, 2),
        ("next_action_mask", torch.bool, 1),
    )
    _SCALAR_FIELDS = (
        ("action", torch.long),
        ("reward", torch.float32),
        ("done", torch.bool),
        ("bootstrap_discount", torch.float32),
    )
    _REQUIRED_KEYS = tuple(
        [name for name, _, _ in _TENSOR_FIELDS] + [name for name, _ in _SCALAR_FIELDS]
    )
    _STORAGE_ORDER = _REQUIRED_KEYS
    _CHANNELS_LAST_FIELDS = (
        "advantage_canvas",
        "next_advantage_canvas",
    )

    def __init__(self, config: Optional[ReplayBufferConfig] = None):
        self.config = config if config is not None else ReplayBufferConfig()
        if self.config.capacity <= 0:
            raise ValueError("capacity must be > 0")

        self.capacity = int(self.config.capacity)
        self.size = 0
        self.pos = 0

        self._storage: Dict[str, torch.Tensor] = {}
        self._initialized = False
        self._pin_storage = bool(self.config.pin_memory) and torch.cuda.is_available()
        self._channels_last_on_cuda = bool(self.config.channels_last_on_cuda)
        self._timing_enabled = bool(self.config.enable_timing)
        self._sample_staging_cache: Dict[tuple[object, ...], list[dict[str, Any]]] = {}

        self.add_time = 0.0
        self.sample_time = 0.0
        self.h2d_time = 0.0

    def __len__(self) -> int:
        return self.size

    @staticmethod
    def _as_tensor(x, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        if isinstance(x, torch.Tensor):
            t = x.detach()
            if t.device.type != "cpu":
                # Replay storage is CPU-backed by design; non-CPU state tensors are
                # normalized back to CPU for compatibility, but the preferred caller
                # path is to insert CPU tensors directly to avoid extra transfers.
                target_dtype = dtype if dtype is not None else t.dtype
                t = t.to(device="cpu", dtype=target_dtype)
            elif dtype is not None and t.dtype != dtype:
                t = t.to(dtype=dtype)
        else:
            t = torch.as_tensor(x, device="cpu")
            if dtype is not None and t.dtype != dtype:
                t = t.to(dtype=dtype)

        if not t.is_contiguous():
            t = t.contiguous()
        return t

    @staticmethod
    def _squeeze_leading_batch(t: torch.Tensor, expect_dim: int) -> torch.Tensor:
        if t.dim() == expect_dim + 1 and t.shape[0] == 1:
            t = t.squeeze(0)
        if t.dim() != expect_dim:
            raise ValueError(f"tensor rank mismatch: expected {expect_dim}D, got {tuple(t.shape)}")
        if not t.is_contiguous():
            t = t.contiguous()
        return t

    def _normalize_transition(self, transition: Mapping[str, object]) -> Dict[str, torch.Tensor]:
        for k in self._REQUIRED_KEYS:
            if k not in transition:
                raise KeyError(f"missing transition key: {k}")

        norm: Dict[str, torch.Tensor] = {}
        for key, dtype, expect_dim in self._TENSOR_FIELDS:
            norm[key] = self._squeeze_leading_batch(self._as_tensor(transition[key], dtype=dtype), expect_dim)
        for key, dtype in self._SCALAR_FIELDS:
            norm[key] = self._as_tensor(transition[key], dtype=dtype).reshape(())
        return norm

    def _alloc_storage_tensor(self, shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
        t = torch.empty(shape, dtype=dtype, device="cpu")
        if self._pin_storage:
            t = t.pin_memory()
        return t

    def _alloc_batch_tensor(self, shape: tuple[int, ...], dtype: torch.dtype, pin_memory: bool) -> torch.Tensor:
        t = torch.empty(shape, dtype=dtype, device="cpu")
        if pin_memory:
            t = t.pin_memory()
        return t

    def _init_storage(self, sample: Dict[str, torch.Tensor]) -> None:
        self._storage = {
            "advantage_canvas": self._alloc_storage_tensor(
                (self.capacity, *sample["advantage_canvas"].shape), torch.float32
            ),
            "value_block_features": self._alloc_storage_tensor(
                (self.capacity, *sample["value_block_features"].shape), torch.float32
            ),
            "value_entry_features": self._alloc_storage_tensor(
                (self.capacity, *sample["value_entry_features"].shape), torch.float32
            ),
            "value_block_mask": self._alloc_storage_tensor(
                (self.capacity, *sample["value_block_mask"].shape), torch.bool
            ),
            "value_entry_mask": self._alloc_storage_tensor(
                (self.capacity, *sample["value_entry_mask"].shape), torch.bool
            ),
            "action_mask": self._alloc_storage_tensor((self.capacity, *sample["action_mask"].shape), torch.bool),
            "action": self._alloc_storage_tensor((self.capacity,), torch.long),
            "reward": self._alloc_storage_tensor((self.capacity,), torch.float32),
            "next_advantage_canvas": self._alloc_storage_tensor(
                (self.capacity, *sample["next_advantage_canvas"].shape), torch.float32
            ),
            "next_value_block_features": self._alloc_storage_tensor(
                (self.capacity, *sample["next_value_block_features"].shape), torch.float32
            ),
            "next_value_entry_features": self._alloc_storage_tensor(
                (self.capacity, *sample["next_value_entry_features"].shape), torch.float32
            ),
            "next_value_block_mask": self._alloc_storage_tensor(
                (self.capacity, *sample["next_value_block_mask"].shape), torch.bool
            ),
            "next_value_entry_mask": self._alloc_storage_tensor(
                (self.capacity, *sample["next_value_entry_mask"].shape), torch.bool
            ),
            "next_action_mask": self._alloc_storage_tensor(
                (self.capacity, *sample["next_action_mask"].shape), torch.bool
            ),
            "done": self._alloc_storage_tensor((self.capacity,), torch.bool),
            "bootstrap_discount": self._alloc_storage_tensor((self.capacity,), torch.float32),
        }
        self._initialized = True

    def _write_transition_to_slot(self, slot: int, transition: Dict[str, torch.Tensor]) -> None:
        for key in self._STORAGE_ORDER:
            self._storage[key][int(slot)].copy_(transition[key])

    def _normalize_transition_batch(self, transitions: List[Mapping[str, object]]) -> Dict[str, torch.Tensor]:
        if len(transitions) <= 0:
            raise ValueError("transitions must be non-empty")
        for transition in transitions:
            for key in self._REQUIRED_KEYS:
                if key not in transition:
                    raise KeyError(f"missing transition key: {key}")

        batch: Dict[str, torch.Tensor] = {}
        for key, dtype, expect_dim in self._TENSOR_FIELDS:
            values = [
                self._squeeze_leading_batch(self._as_tensor(transition[key], dtype=dtype), expect_dim)
                for transition in transitions
            ]
            batch[key] = torch.stack(values, dim=0)
        for key, dtype in self._SCALAR_FIELDS:
            values = [self._as_tensor(transition[key], dtype=dtype).reshape(()) for transition in transitions]
            batch[key] = torch.stack(values, dim=0)
        return batch

    def _write_batch_to_storage(self, start_slot: int, batch: Dict[str, torch.Tensor]) -> int:
        count = int(next(iter(batch.values())).shape[0])
        if count <= 0:
            return int(start_slot)

        write_start = int(start_slot)
        if count > self.capacity:
            keep = int(self.capacity)
            offset = count - keep
            batch = {key: value.narrow(0, offset, keep) for key, value in batch.items()}
            count = keep
            write_start = (int(start_slot) + offset) % self.capacity

        first_count = min(count, self.capacity - write_start)
        second_count = count - first_count
        for key in self._STORAGE_ORDER:
            source = batch[key]
            self._storage[key].narrow(0, write_start, first_count).copy_(source.narrow(0, 0, first_count))
            if second_count > 0:
                self._storage[key].narrow(0, 0, second_count).copy_(source.narrow(0, first_count, second_count))
        return (write_start + count) % self.capacity

    def add(self, transition: Mapping[str, object]) -> None:
        t0 = time.perf_counter() if self._timing_enabled else 0.0
        norm = self._normalize_transition(transition)
        if not self._initialized:
            self._init_storage(norm)

        i = int(self.pos)
        self._write_transition_to_slot(i, norm)

        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        if self._timing_enabled:
            self.add_time += time.perf_counter() - t0

    def add_many(self, transitions: List[Mapping[str, object]]) -> None:
        count = len(transitions)
        if count <= 0:
            return

        t0 = time.perf_counter() if self._timing_enabled else 0.0
        batch = self._normalize_transition_batch(transitions)
        if not self._initialized:
            sample = {key: value[0] for key, value in batch.items()}
            self._init_storage(sample)

        self.pos = self._write_batch_to_storage(int(self.pos), batch)
        self.size = min(self.size + int(count), self.capacity)
        if self._timing_enabled:
            self.add_time += time.perf_counter() - t0

    def can_sample(self, batch_size: int) -> bool:
        return self.size >= int(batch_size)

    def _get_sample_staging_tensor(
        self,
        key: str,
        shape: tuple[int, ...],
        dtype: torch.dtype,
        pin_memory: bool,
    ) -> tuple[torch.Tensor, tuple[object, ...], int]:
        spec = (str(key), tuple(int(v) for v in shape), dtype, bool(pin_memory))
        entries = self._sample_staging_cache.setdefault(spec, [])
        for idx, entry in enumerate(entries):
            event = entry.get("event")
            if event is None or bool(event.query()):
                entry["event"] = None
                return entry["tensor"], spec, idx

        tensor = self._alloc_batch_tensor(shape, dtype=dtype, pin_memory=pin_memory)
        entries.append({"tensor": tensor, "event": None})
        return tensor, spec, len(entries) - 1

    def _mark_staging_in_use(self, used_slots: list[tuple[tuple[object, ...], int]], device: torch.device) -> None:
        if len(used_slots) <= 0 or device.type != "cuda" or not torch.cuda.is_available():
            return
        stream = torch.cuda.current_stream(device)
        event = torch.cuda.Event()
        event.record(stream)
        for spec, slot_idx in used_slots:
            entries = self._sample_staging_cache.get(spec)
            if entries is None or slot_idx >= len(entries):
                continue
            entries[slot_idx]["event"] = event

    def sample(self, batch_size: int, device: Optional[torch.device] = None) -> Dict[str, torch.Tensor]:
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        if not self.can_sample(batch_size):
            raise ValueError(f"not enough samples: have {self.size}, need {batch_size}")

        batch_size = int(batch_size)
        target_device = None if device is None else torch.device(device)
        transfer_to_device = target_device is not None and target_device.type != "cpu"
        use_pinned_batch = transfer_to_device and self._pin_storage

        t0 = time.perf_counter() if self._timing_enabled else 0.0
        idx = torch.randint(0, self.size, (batch_size,), device="cpu", dtype=torch.long)

        out: Dict[str, torch.Tensor] = {}
        used_staging_slots: list[tuple[tuple[object, ...], int]] = []
        for key, storage_tensor in self._storage.items():
            if use_pinned_batch:
                gathered, spec, slot_idx = self._get_sample_staging_tensor(
                    key,
                    (batch_size, *storage_tensor.shape[1:]),
                    dtype=storage_tensor.dtype,
                    pin_memory=True,
                )
                used_staging_slots.append((spec, slot_idx))
                torch.index_select(storage_tensor, 0, idx, out=gathered)
            else:
                gathered = storage_tensor.index_select(0, idx)
            out[key] = gathered
        if self._timing_enabled:
            self.sample_time += time.perf_counter() - t0

        if transfer_to_device:
            non_blocking = bool(self.config.non_blocking_transfer) and target_device.type == "cuda"
            idx_for_device = idx.pin_memory() if use_pinned_batch else idx

            t0 = time.perf_counter() if self._timing_enabled else 0.0
            out = {key: value.to(target_device, non_blocking=non_blocking) for key, value in out.items()}
            if use_pinned_batch and non_blocking:
                self._mark_staging_in_use(used_staging_slots, target_device)
            if self._channels_last_on_cuda and target_device.type == "cuda":
                for key in self._CHANNELS_LAST_FIELDS:
                    out[key] = out[key].contiguous(memory_format=torch.channels_last)
            idx = idx_for_device.to(target_device, non_blocking=non_blocking)
            if self._timing_enabled:
                self.h2d_time += time.perf_counter() - t0

        out["indices"] = idx
        out["weights"] = torch.ones((batch_size,), dtype=torch.float32, device=idx.device)
        return out

    def get_timing_stats(self) -> Dict[str, float]:
        return {
            "add_time": float(self.add_time),
            "sample_time": float(self.sample_time),
            "h2d_time": float(self.h2d_time),
        }

    def update_priorities(self, indices: torch.Tensor, priorities: torch.Tensor) -> None:
        # Reserved extension point for prioritized replay; no-op in uniform replay.
        _ = indices
        _ = priorities
