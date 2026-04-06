from __future__ import annotations

"""
Frontier 增量更新策略 A/B 沙盒基准测试。

对比：
  - dirty_rect   : 当前正式默认实现
  - sparse_delta : 实验性“新增格子 + 老前沿删除”候选实现

运行：
  python tools/benchmark_frontier_update.py
"""

import argparse
import random
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from env.agent_version import LocalObservationModel
from env.block_random_g import RandomMapGenerator
from env.core_cummap import (
    CumulativeBeliefMap,
    FRONTIER_MIN_UNKNOWN_NEIGHBORS,
    FRONTIER_NEIGHBOR_CONNECTIVITY,
)
from env.core_radar import RadarSensor
from env.grid_topology import ACTIONS_8, EMPTY, INVISIBLE, GridTopology


POLICY_MODES = ("random_valid", "deterministic_greedy")


@dataclass(frozen=True)
class BenchmarkConfig:
    rows: int = 40
    cols: int = 60
    obs_size: int = 6
    scan_radius: int = 10
    obstacle_ratio: float = 0.20
    episodes: int = 30
    max_steps: int = 400
    seed_base: int = 20260405
    coverage_stop_threshold: Optional[float] = 0.98


@dataclass
class ModeAggregate:
    policy_mode: str
    episodes_run: int = 0
    total_steps: int = 0
    frontier_update_time_dirty: float = 0.0
    frontier_update_time_sparse: float = 0.0
    cummap_update_time_dirty: float = 0.0
    cummap_update_time_sparse: float = 0.0
    wall_time_sec: float = 0.0


class FrontierMismatchError(RuntimeError):
    pass


class TimedDirtyRectBeliefMap(CumulativeBeliefMap):
    def __init__(self, *args, **kwargs):
        self.frontier_update_time = 0.0
        self.frontier_update_calls = 0
        self.update_total_time = 0.0
        self.update_calls = 0
        super().__init__(*args, **kwargs)

    def _update_frontier_dirty_rects(self, dirty_rects) -> None:
        t0 = time.perf_counter()
        super()._update_frontier_dirty_rects(dirty_rects)
        self.frontier_update_time += time.perf_counter() - t0
        self.frontier_update_calls += 1

    def update(self, agent_state: tuple[int, int], local_snap: np.ndarray) -> tuple[int, int, int]:
        t0 = time.perf_counter()
        out = super().update(agent_state, local_snap)
        self.update_total_time += time.perf_counter() - t0
        self.update_calls += 1
        return out


class SparseDeltaBenchmarkBeliefMap(TimedDirtyRectBeliefMap):
    def __init__(self, *args, **kwargs):
        self._frontier_candidate_seed = np.zeros((0, 0), dtype=bool)
        self._frontier_candidate_mask = np.zeros((0, 0), dtype=bool)
        super().__init__(*args, **kwargs)

    def _ensure_candidate_buffers(self) -> tuple[np.ndarray, np.ndarray]:
        shape = tuple(self.map.shape)
        if self._frontier_candidate_seed.shape != shape:
            self._frontier_candidate_seed = np.zeros(shape, dtype=bool)
        if self._frontier_candidate_mask.shape != shape:
            self._frontier_candidate_mask = np.zeros(shape, dtype=bool)
        self._frontier_candidate_seed.fill(False)
        self._frontier_candidate_mask.fill(False)
        return self._frontier_candidate_seed, self._frontier_candidate_mask

    def _frontier_membership_for_coords(self, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
        rows_use = np.asarray(rows, dtype=np.int32).reshape(-1)
        cols_use = np.asarray(cols, dtype=np.int32).reshape(-1)
        if rows_use.size <= 0 or cols_use.size <= 0:
            return np.zeros((0,), dtype=bool)

        known_free = np.asarray(self.map[rows_use, cols_use] == EMPTY, dtype=bool)
        if not np.any(known_free):
            return known_free

        if int(FRONTIER_NEIGHBOR_CONNECTIVITY) != 4:
            raise ValueError(
                "sparse_delta benchmark currently assumes 4-neighbor frontier connectivity, "
                f"got {FRONTIER_NEIGHBOR_CONNECTIVITY}"
            )

        unknown_counts = np.zeros(rows_use.shape, dtype=np.uint8)
        valid = rows_use > 0
        if np.any(valid):
            unknown_counts[valid] += (self.map[rows_use[valid] - 1, cols_use[valid]] == INVISIBLE).astype(np.uint8)
        valid = rows_use + 1 < int(self.map.shape[0])
        if np.any(valid):
            unknown_counts[valid] += (self.map[rows_use[valid] + 1, cols_use[valid]] == INVISIBLE).astype(np.uint8)
        valid = cols_use > 0
        if np.any(valid):
            unknown_counts[valid] += (self.map[rows_use[valid], cols_use[valid] - 1] == INVISIBLE).astype(np.uint8)
        valid = cols_use + 1 < int(self.map.shape[1])
        if np.any(valid):
            unknown_counts[valid] += (self.map[rows_use[valid], cols_use[valid] + 1] == INVISIBLE).astype(np.uint8)
        return known_free & (unknown_counts >= int(FRONTIER_MIN_UNKNOWN_NEIGHBORS))

    def _update_frontier_sparse_delta(
        self,
        *,
        revealed_rows: Optional[np.ndarray] = None,
        revealed_cols: Optional[np.ndarray] = None,
        seam_dirty_rects=(),
    ) -> None:
        t0 = time.perf_counter()
        candidate_seed, candidate_mask = self._ensure_candidate_buffers()
        if revealed_rows is not None and revealed_cols is not None:
            rr = np.asarray(revealed_rows, dtype=np.int32).reshape(-1)
            cc = np.asarray(revealed_cols, dtype=np.int32).reshape(-1)
            if rr.size > 0 and cc.size > 0:
                candidate_seed[rr, cc] = True

        seam_rects = self._normalize_dirty_rects(seam_dirty_rects, tuple(self.map.shape))
        if len(seam_rects) > 0:
            for rect in seam_rects:
                candidate_seed[int(rect.r0):int(rect.r1), int(rect.c0):int(rect.c1)] = True

        if np.any(candidate_seed):
            candidate_mask[:, :] = candidate_seed
            candidate_mask[1:, :] |= candidate_seed[:-1, :]
            candidate_mask[:-1, :] |= candidate_seed[1:, :]
            candidate_mask[:, 1:] |= candidate_seed[:, :-1]
            candidate_mask[:, :-1] |= candidate_seed[:, 1:]

            candidate_rows, candidate_cols = np.nonzero(candidate_mask)
            membership = self._frontier_membership_for_coords(candidate_rows, candidate_cols)
            self.frontier_bool[candidate_rows, candidate_cols] = membership
            self.frontier_u8[candidate_rows, candidate_cols] = membership.astype(np.uint8) * 255
            self.frontier_revision += 1
            self._invalidate_frontier_stats_cache()

        self.frontier_update_time += time.perf_counter() - t0
        self.frontier_update_calls += 1

    def update(self, agent_state: tuple[int, int], local_snap: np.ndarray) -> tuple[int, int, int]:
        t_update = time.perf_counter()
        snap = np.asarray(local_snap, dtype=np.int8)
        if snap.shape != self.local_shape:
            raise ValueError(f"local_snap shape mismatch: expected {self.local_shape}, got {snap.shape}")

        self.step_count += 1

        gr, gc = self._project_local_world(agent_state)
        visible = (snap != INVISIBLE)
        dirty_rects = []

        ar, ac = int(agent_state[0]), int(agent_state[1])
        if np.any(visible):
            wr = gr[visible]
            wc = gc[visible]
            min_r = min(ar, int(wr.min()))
            max_r = max(ar, int(wr.max()))
            min_c = min(ac, int(wc.min()))
            max_c = max(ac, int(wc.max()))
        else:
            wr = np.empty((0,), dtype=np.int32)
            wc = np.empty((0,), dtype=np.int32)
            min_r = ar
            max_r = ar
            min_c = ac
            max_c = ac

        expansion = self._ensure_world_bounds(min_r, max_r, min_c, max_c)
        seam_dirty_rects = tuple() if expansion is None else expansion.seam_dirty_rects
        if expansion is not None:
            dirty_rects.extend(seam_dirty_rects)

        self._record_visit_in_bounds(agent_state)

        if not np.any(visible):
            self._update_frontier_sparse_delta(
                revealed_rows=None,
                revealed_cols=None,
                seam_dirty_rects=seam_dirty_rects,
            )
            self._refresh_coverage()
            self._update_analysis_box()
            self.update_total_time += time.perf_counter() - t_update
            self.update_calls += 1
            return 0, 0, 0

        vv = snap[visible]
        ir = wr - int(self.origin_world_rc[0])
        ic = wc - int(self.origin_world_rc[1])

        unseen = (self.map[ir, ic] == INVISIBLE)
        if not np.any(unseen):
            self._update_frontier_sparse_delta(
                revealed_rows=None,
                revealed_cols=None,
                seam_dirty_rects=seam_dirty_rects,
            )
            self._refresh_coverage()
            self._update_analysis_box()
            self.update_total_time += time.perf_counter() - t_update
            self.update_calls += 1
            return 0, 0, 0

        wir = ir[unseen]
        wic = ic[unseen]
        wvv = vv[unseen]
        self.map[wir, wic] = wvv
        self.kpm_count += self._count_coverage_hits(wr[unseen], wc[unseen])
        self._invalidate_map_state_caches()
        reveal_dirty = self._expand_dirty_rect(self._dirty_rect_from_points(wir, wic), radius=1)
        if reveal_dirty is not None:
            dirty_rects.append(reveal_dirty)
        self._update_frontier_sparse_delta(
            revealed_rows=wir,
            revealed_cols=wic,
            seam_dirty_rects=seam_dirty_rects,
        )

        updated = int(wvv.size)
        delta_empty = int((wvv == EMPTY).sum())
        delta_obstacle = updated - delta_empty
        self._refresh_coverage()
        self._update_analysis_box()
        self.update_total_time += time.perf_counter() - t_update
        self.update_calls += 1
        return updated, delta_empty, delta_obstacle


@contextmanager
def seeded_rng(seed: int):
    py_state = random.getstate()
    np_state = np.random.get_state()
    random.seed(int(seed))
    np.random.seed(int(seed))
    try:
        yield
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)


def parse_args() -> BenchmarkConfig:
    parser = argparse.ArgumentParser(
        description="A/B benchmark for frontier incremental update strategies.",
    )
    parser.add_argument("--rows", type=int, default=40)
    parser.add_argument("--cols", type=int, default=60)
    parser.add_argument("--obs-size", type=int, default=6)
    parser.add_argument("--scan-radius", type=int, default=10)
    parser.add_argument("--obstacle-ratio", type=float, default=0.20)
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--seed-base", type=int, default=20260405)
    parser.add_argument(
        "--coverage-stop-threshold",
        type=float,
        default=0.98,
        help="Stop episode early once coverage reaches this threshold; set negative to disable.",
    )
    args = parser.parse_args()
    coverage_stop_threshold: Optional[float]
    if float(args.coverage_stop_threshold) < 0.0:
        coverage_stop_threshold = None
    else:
        coverage_stop_threshold = float(args.coverage_stop_threshold)
    return BenchmarkConfig(
        rows=int(args.rows),
        cols=int(args.cols),
        obs_size=int(args.obs_size),
        scan_radius=int(args.scan_radius),
        obstacle_ratio=float(args.obstacle_ratio),
        episodes=int(args.episodes),
        max_steps=int(args.max_steps),
        seed_base=int(args.seed_base),
        coverage_stop_threshold=coverage_stop_threshold,
    )


def make_generator(cfg: BenchmarkConfig) -> RandomMapGenerator:
    return RandomMapGenerator(
        rows=int(cfg.rows),
        cols=int(cfg.cols),
        obs_size=int(cfg.obs_size),
        obstacle_ratio=float(cfg.obstacle_ratio),
    )


def generate_episode_case(cfg: BenchmarkConfig, episode_seed: int) -> tuple[np.ndarray, tuple[int, int]]:
    with seeded_rng(episode_seed):
        generator = make_generator(cfg)
        grid, start = generator.generate_map()
    return np.asarray(grid, dtype=np.int8), (int(start[0]), int(start[1]))


def choose_random_valid_action(
    *,
    rng: random.Random,
    valid_actions: tuple[int, ...],
    agent_state: tuple[int, int],
    visit_count: np.ndarray,
) -> int:
    _ = agent_state, visit_count
    return int(rng.choice(list(valid_actions)))


def choose_deterministic_greedy_action(
    *,
    rng: random.Random,
    valid_actions: tuple[int, ...],
    agent_state: tuple[int, int],
    visit_count: np.ndarray,
) -> int:
    _ = rng
    priority_order = (0, 2, 4, 6, 1, 3, 5, 7)
    priority_rank = {action_idx: rank for rank, action_idx in enumerate(priority_order)}

    best_action = None
    best_key = None
    for action_idx in valid_actions:
        dr, dc = ACTIONS_8[int(action_idx)]
        nr = int(agent_state[0] + dr)
        nc = int(agent_state[1] + dc)
        key = (
            int(visit_count[nr, nc]),
            priority_rank.get(int(action_idx), 999),
            int(action_idx),
        )
        if best_key is None or key < best_key:
            best_key = key
            best_action = int(action_idx)
    if best_action is None:
        raise RuntimeError("deterministic policy received empty valid action set")
    return best_action


def get_policy_fn(policy_mode: str) -> Callable[..., int]:
    if policy_mode == "random_valid":
        return choose_random_valid_action
    if policy_mode == "deterministic_greedy":
        return choose_deterministic_greedy_action
    raise ValueError(f"unsupported policy mode: {policy_mode!r}")


def compare_frontier_states(
    *,
    dirty_map: CumulativeBeliefMap,
    sparse_map: CumulativeBeliefMap,
    policy_mode: str,
    episode_idx: int,
    step_idx: int,
) -> None:
    if dirty_map.frontier_bool.shape != sparse_map.frontier_bool.shape:
        raise FrontierMismatchError(
            "frontier shape mismatch: "
            f"policy_mode={policy_mode} episode_idx={episode_idx} step_idx={step_idx} "
            f"dirty_shape={tuple(dirty_map.frontier_bool.shape)} "
            f"sparse_shape={tuple(sparse_map.frontier_bool.shape)}"
        )

    bool_equal = np.array_equal(dirty_map.frontier_bool, sparse_map.frontier_bool)
    u8_equal = np.array_equal(dirty_map.frontier_u8, sparse_map.frontier_u8)
    if bool_equal and u8_equal:
        return

    bool_mismatch = int(np.count_nonzero(dirty_map.frontier_bool != sparse_map.frontier_bool))
    u8_mismatch = int(np.count_nonzero(dirty_map.frontier_u8 != sparse_map.frontier_u8))
    dirty_stats = dirty_map.debug_frontier_consistency_stats()
    sparse_stats = sparse_map.debug_frontier_consistency_stats()
    raise FrontierMismatchError(
        "frontier cache mismatch: "
        f"policy_mode={policy_mode} episode_idx={episode_idx} step_idx={step_idx} "
        f"bool_mismatch={bool_mismatch} u8_mismatch={u8_mismatch} "
        f"map_shape={tuple(dirty_map.frontier_bool.shape)} "
        f"dirty_full_consistent={dirty_stats.consistent} "
        f"sparse_full_consistent={sparse_stats.consistent} "
        f"dirty_frontier_revision={dirty_stats.frontier_revision} "
        f"sparse_frontier_revision={sparse_stats.frontier_revision}"
    )


def run_policy_mode(cfg: BenchmarkConfig, policy_mode: str) -> ModeAggregate:
    sensor = RadarSensor(scan_radius=int(cfg.scan_radius))
    policy_fn = get_policy_fn(policy_mode)
    aggregate = ModeAggregate(policy_mode=policy_mode)
    bench_t0 = time.perf_counter()

    for episode_idx in range(int(cfg.episodes)):
        episode_seed = int(cfg.seed_base) + episode_idx
        action_seed = int(cfg.seed_base) + (100000 * (POLICY_MODES.index(policy_mode) + 1)) + episode_idx
        action_rng = random.Random(action_seed)

        grid, start_state = generate_episode_case(cfg, episode_seed)
        free_mask = GridTopology.free_mask(grid)
        obs_model = LocalObservationModel(grid, start_state, sensor=sensor)
        first_local_snap = np.asarray(obs_model.local_snap, dtype=np.int8).copy()

        dirty_map = TimedDirtyRectBeliefMap(
            grid,
            start_state,
            first_local_snap,
            enable_timing=True,
        )
        sparse_map = SparseDeltaBenchmarkBeliefMap(
            grid,
            start_state,
            first_local_snap,
            enable_timing=True,
        )
        compare_frontier_states(
            dirty_map=dirty_map,
            sparse_map=sparse_map,
            policy_mode=policy_mode,
            episode_idx=episode_idx,
            step_idx=0,
        )

        agent_state = (int(start_state[0]), int(start_state[1]))
        visit_count = np.zeros(grid.shape, dtype=np.int32)
        visit_count[int(agent_state[0]), int(agent_state[1])] += 1

        episode_steps = 0
        for step_idx in range(1, int(cfg.max_steps) + 1):
            valid_actions = GridTopology.valid_action_indices_fast(free_mask, agent_state)
            if len(valid_actions) <= 0:
                break

            action_idx = policy_fn(
                rng=action_rng,
                valid_actions=valid_actions,
                agent_state=agent_state,
                visit_count=visit_count,
            )
            dr, dc = ACTIONS_8[int(action_idx)]
            agent_state = (int(agent_state[0] + dr), int(agent_state[1] + dc))
            visit_count[int(agent_state[0]), int(agent_state[1])] += 1

            local_snap = np.asarray(obs_model.observe_fast(agent_state), dtype=np.int8)
            dirty_update = dirty_map.update(agent_state, local_snap)
            sparse_update = sparse_map.update(agent_state, local_snap)
            if dirty_update != sparse_update:
                raise FrontierMismatchError(
                    "belief update tuple mismatch: "
                    f"policy_mode={policy_mode} episode_idx={episode_idx} step_idx={step_idx} "
                    f"dirty_update={dirty_update} sparse_update={sparse_update}"
                )

            if not np.isclose(float(dirty_map.coverage_rate), float(sparse_map.coverage_rate)):
                raise FrontierMismatchError(
                    "coverage_rate mismatch: "
                    f"policy_mode={policy_mode} episode_idx={episode_idx} step_idx={step_idx} "
                    f"dirty_coverage={dirty_map.coverage_rate:.6f} "
                    f"sparse_coverage={sparse_map.coverage_rate:.6f}"
                )

            compare_frontier_states(
                dirty_map=dirty_map,
                sparse_map=sparse_map,
                policy_mode=policy_mode,
                episode_idx=episode_idx,
                step_idx=step_idx,
            )
            episode_steps += 1

            if (
                cfg.coverage_stop_threshold is not None
                and float(dirty_map.coverage_rate) >= float(cfg.coverage_stop_threshold)
            ):
                break

        aggregate.episodes_run += 1
        aggregate.total_steps += int(episode_steps)
        aggregate.frontier_update_time_dirty += float(dirty_map.frontier_update_time)
        aggregate.frontier_update_time_sparse += float(sparse_map.frontier_update_time)
        aggregate.cummap_update_time_dirty += float(dirty_map.update_total_time)
        aggregate.cummap_update_time_sparse += float(sparse_map.update_total_time)

    aggregate.wall_time_sec = time.perf_counter() - bench_t0
    return aggregate


def pct_change(candidate: float, baseline: float) -> float:
    if abs(float(baseline)) <= 1e-12:
        return float("nan")
    return ((float(candidate) - float(baseline)) / float(baseline)) * 100.0


def avg_time_ms(total_time: float, total_steps: int) -> float:
    if int(total_steps) <= 0:
        return float("nan")
    return (float(total_time) / float(total_steps)) * 1000.0


def print_mode_summary(result: ModeAggregate) -> None:
    frontier_delta_pct = pct_change(result.frontier_update_time_sparse, result.frontier_update_time_dirty)
    cummap_delta_pct = pct_change(result.cummap_update_time_sparse, result.cummap_update_time_dirty)
    print("")
    print(f"[mode] {result.policy_mode}")
    print(f"episodes_run={result.episodes_run}")
    print(f"total_steps={result.total_steps}")
    print(
        "dirty_rect: "
        f"frontier_total_sec={result.frontier_update_time_dirty:.6f} "
        f"frontier_avg_ms={avg_time_ms(result.frontier_update_time_dirty, result.total_steps):.6f} "
        f"cummap_total_sec={result.cummap_update_time_dirty:.6f} "
        f"cummap_avg_ms={avg_time_ms(result.cummap_update_time_dirty, result.total_steps):.6f}"
    )
    print(
        "sparse_delta: "
        f"frontier_total_sec={result.frontier_update_time_sparse:.6f} "
        f"frontier_avg_ms={avg_time_ms(result.frontier_update_time_sparse, result.total_steps):.6f} "
        f"cummap_total_sec={result.cummap_update_time_sparse:.6f} "
        f"cummap_avg_ms={avg_time_ms(result.cummap_update_time_sparse, result.total_steps):.6f}"
    )
    print(
        "delta_vs_dirty_rect: "
        f"frontier_pct={frontier_delta_pct:.3f}% "
        f"cummap_pct={cummap_delta_pct:.3f}% "
        f"wall_time_sec={result.wall_time_sec:.3f}"
    )


def print_overall_summary(results: list[ModeAggregate]) -> None:
    total_steps = int(sum(item.total_steps for item in results))
    dirty_frontier = float(sum(item.frontier_update_time_dirty for item in results))
    sparse_frontier = float(sum(item.frontier_update_time_sparse for item in results))
    dirty_cummap = float(sum(item.cummap_update_time_dirty for item in results))
    sparse_cummap = float(sum(item.cummap_update_time_sparse for item in results))
    print("")
    print("[overall]")
    print(f"policy_modes={','.join(item.policy_mode for item in results)}")
    print(f"total_steps={total_steps}")
    print(
        "dirty_rect: "
        f"frontier_total_sec={dirty_frontier:.6f} "
        f"frontier_avg_ms={avg_time_ms(dirty_frontier, total_steps):.6f} "
        f"cummap_total_sec={dirty_cummap:.6f} "
        f"cummap_avg_ms={avg_time_ms(dirty_cummap, total_steps):.6f}"
    )
    print(
        "sparse_delta: "
        f"frontier_total_sec={sparse_frontier:.6f} "
        f"frontier_avg_ms={avg_time_ms(sparse_frontier, total_steps):.6f} "
        f"cummap_total_sec={sparse_cummap:.6f} "
        f"cummap_avg_ms={avg_time_ms(sparse_cummap, total_steps):.6f}"
    )
    print(
        "delta_vs_dirty_rect: "
        f"frontier_pct={pct_change(sparse_frontier, dirty_frontier):.3f}% "
        f"cummap_pct={pct_change(sparse_cummap, dirty_cummap):.3f}%"
    )


def main() -> int:
    cfg = parse_args()
    print("frontier incremental benchmark")
    print(
        f"config rows={cfg.rows} cols={cfg.cols} obs_size={cfg.obs_size} "
        f"scan_radius={cfg.scan_radius} obstacle_ratio={cfg.obstacle_ratio:.3f} "
        f"episodes={cfg.episodes} max_steps={cfg.max_steps} seed_base={cfg.seed_base} "
        f"coverage_stop_threshold={cfg.coverage_stop_threshold}"
    )
    print("baseline=dirty_rect candidate=sparse_delta")

    results: list[ModeAggregate] = []
    try:
        for policy_mode in POLICY_MODES:
            result = run_policy_mode(cfg, policy_mode)
            results.append(result)
            print_mode_summary(result)
    except FrontierMismatchError as exc:
        print("")
        print("[mismatch]")
        print(str(exc))
        return 1

    print("")
    print("consistency_check=passed")
    print("mismatch_detected=False")
    print_overall_summary(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
