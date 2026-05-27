from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from env.core_radar import RadarSensor
from env.grid_topology import ACTIONS_8, EMPTY, INVISIBLE, OBSTACLE, GridTopology


ACTION_ORDER_NAMES = ("N", "NE", "E", "SE", "S", "SW", "W", "NW")


@dataclass(frozen=True)
class FrontierGreedyDecision:
    action_idx: int | None
    decision_mode: str
    target_array_rc: tuple[int, int] | None = None
    path_cost: int | None = None
    fallback_score: float | None = None


class ClassicalFrontierGreedyPolicy:
    def __init__(self, *, scan_radius: int = 10):
        self.sensor = RadarSensor(scan_radius=int(scan_radius))
        self._action_rank = {int(action_idx): int(action_idx) for action_idx in range(len(ACTIONS_8))}

    def policy_summary(self) -> dict[str, object]:
        return {
            "policy_id": "Anew_B_classical_frontier_greedy",
            "policy_type": "traditional_non_learning",
            "target_selection_rule": (
                "Legacy B rule: select the reachable frontier cluster anchor with the lowest known-free BFS cost; "
                "fall back to raw frontier cells from the cumulative belief frontier cache."
            ),
            "frontier_source": "shared_semantic_snapshot_then_cumulative_frontier_cache",
            "step_selection_rule": (
                "Legacy B rule: after selecting a frontier target, choose the valid next action by squared "
                "Euclidean distance to the target, recent-trajectory revisit flag, visit count, then fixed "
                "ACTIONS_8 order. The action is not chosen by following a BFS shortest-path gradient."
            ),
            "fallback_rule": (
                "When no reachable frontier exists, choose the valid action with the largest belief-only "
                "expected immediate information gain using radar line-of-sight over the current belief."
            ),
            "tie_break_rule": (
                "target_distance_or_gain, recent_trajectory_revisit, visit_count, fixed action order "
                "(N, NE, E, SE, S, SW, W, NW)"
            ),
            "decision_input_contract": (
                "The public decision path uses only belief-derived map state, the current "
                "array-space pose, valid action indices, visit counts, recent trajectory, frontier cache, "
                "and optional shared semantic snapshot."
            ),
            "action_order": list(ACTION_ORDER_NAMES),
            "legacy_behavior_source": "DRL_PF baseline classical_frontier_greedy_v1",
        }

    @staticmethod
    def _normalize_valid(valid_action_indices: Sequence[int]) -> list[int]:
        valid = sorted({int(action_idx) for action_idx in valid_action_indices})
        return [action_idx for action_idx in valid if 0 <= action_idx < len(ACTIONS_8)]

    @staticmethod
    def _target_from_snapshot(
        *,
        distance_map: np.ndarray,
        agent_array_rc: tuple[int, int],
        semantic_snapshot,
    ) -> tuple[tuple[int, int] | None, int | None]:
        if semantic_snapshot is None:
            return None, None
        best_key: tuple[int, float, int, int, int, int] | None = None
        best_anchor: tuple[int, int] | None = None
        for block_order, block in enumerate(tuple(getattr(semantic_snapshot, "accessible_blocks", tuple()) or tuple())):
            block_index = int(getattr(block, "block_index", block_order))
            for cluster_order, cluster in enumerate(tuple(getattr(block, "frontier_clusters", tuple()) or tuple())):
                anchor = getattr(cluster, "frontier_anchor_rc", None)
                if anchor is None:
                    continue
                row, col = int(anchor[0]), int(anchor[1])
                if not GridTopology.in_bounds(distance_map.shape, row, col):
                    continue
                cost = int(distance_map[row, col])
                if cost < 0:
                    continue
                euclidean = float((row - int(agent_array_rc[0])) ** 2 + (col - int(agent_array_rc[1])) ** 2)
                key = (
                    cost,
                    euclidean,
                    block_index,
                    int(getattr(cluster, "frontier_index", cluster_order)),
                    row,
                    col,
                )
                if best_key is None or key < best_key:
                    best_key = key
                    best_anchor = (row, col)
        if best_key is None:
            return None, None
        return best_anchor, int(best_key[0])

    @staticmethod
    def _target_from_frontier_cache(
        *,
        frontier_u8: np.ndarray,
        distance_map: np.ndarray,
        agent_array_rc: tuple[int, int],
    ) -> tuple[tuple[int, int] | None, int | None]:
        frontier = np.asarray(frontier_u8, dtype=np.uint8) > 0
        if frontier.shape != distance_map.shape or not np.any(frontier):
            return None, None
        rows, cols = np.nonzero(frontier)
        best_key: tuple[int, float, int, int] | None = None
        best_anchor: tuple[int, int] | None = None
        for row, col in zip(rows.tolist(), cols.tolist()):
            cost = int(distance_map[int(row), int(col)])
            if cost < 0:
                continue
            euclidean = float((int(row) - int(agent_array_rc[0])) ** 2 + (int(col) - int(agent_array_rc[1])) ** 2)
            key = (cost, euclidean, int(row), int(col))
            if best_key is None or key < best_key:
                best_key = key
                best_anchor = (int(row), int(col))
        if best_key is None:
            return None, None
        return best_anchor, int(best_key[0])

    @staticmethod
    def _best_reachable_target(
        *,
        known_free_grid: np.ndarray,
        agent_array_rc: tuple[int, int],
        belief_map: np.ndarray,
        semantic_snapshot,
        frontier_u8: np.ndarray | None,
    ) -> tuple[tuple[int, int] | None, int | None]:
        dist = GridTopology.bfs_distance_map(known_free_grid, agent_array_rc)
        target, cost = ClassicalFrontierGreedyPolicy._target_from_snapshot(
            distance_map=dist,
            agent_array_rc=agent_array_rc,
            semantic_snapshot=semantic_snapshot,
        )
        if target is not None:
            return target, cost
        if frontier_u8 is None:
            frontier_u8 = GridTopology.frontier_mask(belief_map).astype(np.uint8) * 255
        return ClassicalFrontierGreedyPolicy._target_from_frontier_cache(
            frontier_u8=frontier_u8,
            distance_map=dist,
            agent_array_rc=agent_array_rc,
        )

    @staticmethod
    def _select_step_toward_target(
        *,
        visit_count: np.ndarray,
        agent_array_rc: tuple[int, int],
        target_array_rc: tuple[int, int],
        valid_actions: list[int],
        recent_trajectory_positions: Sequence[tuple[int, int]],
    ) -> int | None:
        if tuple(agent_array_rc) == tuple(target_array_rc):
            return None
        recent = {(int(row), int(col)) for row, col in recent_trajectory_positions}
        scored: list[tuple[int, int, int, int]] = []
        ar, ac = int(agent_array_rc[0]), int(agent_array_rc[1])
        for action_idx in valid_actions:
            dr, dc = ACTIONS_8[int(action_idx)]
            nr, nc = ar + int(dr), ac + int(dc)
            if not GridTopology.in_bounds(visit_count.shape, nr, nc):
                continue
            dist2 = (nr - int(target_array_rc[0])) ** 2 + (nc - int(target_array_rc[1])) ** 2
            scored.append(
                (
                    int(dist2),
                    int((nr, nc) in recent),
                    int(visit_count[nr, nc]),
                    int(action_idx),
                )
            )
        if len(scored) <= 0:
            return None
        return int(min(scored)[3])

    @staticmethod
    def _cell_value(
        belief_map: np.ndarray,
        array_rc: tuple[int, int],
    ) -> int:
        row, col = int(array_rc[0]), int(array_rc[1])
        if not GridTopology.in_bounds(belief_map.shape, row, col):
            return INVISIBLE
        return int(belief_map[row, col])

    def _belief_corner_occluded(
        self,
        *,
        belief_map: np.ndarray,
        candidate_rc: tuple[int, int],
        prev_rel: tuple[int, int] | None,
        cur_rel: tuple[int, int],
    ) -> bool:
        if not bool(getattr(self.sensor, "block_corner_peeking", True)):
            return False
        if prev_rel is None:
            return False
        dr = int(cur_rel[0]) - int(prev_rel[0])
        dc = int(cur_rel[1]) - int(prev_rel[1])
        if abs(dr) != 1 or abs(dc) != 1:
            return False
        side_a = (int(candidate_rc[0]) + int(cur_rel[0]), int(candidate_rc[1]) + int(prev_rel[1]))
        side_b = (int(candidate_rc[0]) + int(prev_rel[0]), int(candidate_rc[1]) + int(cur_rel[1]))
        return self._cell_value(belief_map, side_a) == OBSTACLE and self._cell_value(belief_map, side_b) == OBSTACLE

    def _expected_immediate_unknown_gain(
        self,
        *,
        belief_map: np.ndarray,
        candidate_rc: tuple[int, int],
    ) -> float:
        visible_unknown: set[tuple[int, int]] = set()
        for ray in self.sensor.local_ray_templates:
            prev_rel: tuple[int, int] | None = None
            for rel_r, rel_c, _, _ in ray:
                rel = (int(rel_r), int(rel_c))
                if self._belief_corner_occluded(
                    belief_map=belief_map,
                    candidate_rc=candidate_rc,
                    prev_rel=prev_rel,
                    cur_rel=rel,
                ):
                    break
                cell = (int(candidate_rc[0]) + rel[0], int(candidate_rc[1]) + rel[1])
                value = self._cell_value(belief_map, cell)
                if value == OBSTACLE:
                    break
                if value == INVISIBLE:
                    visible_unknown.add(cell)
                prev_rel = rel
        return float(len(visible_unknown))

    def _fallback_action(
        self,
        *,
        belief_map: np.ndarray,
        agent_array_rc: tuple[int, int],
        valid_actions: list[int],
        visit_count: np.ndarray,
        recent_trajectory_positions: Sequence[tuple[int, int]],
    ) -> FrontierGreedyDecision:
        if len(valid_actions) <= 0:
            return FrontierGreedyDecision(action_idx=None, decision_mode="no_valid_action")
        ar, ac = int(agent_array_rc[0]), int(agent_array_rc[1])
        recent = {(int(row), int(col)) for row, col in recent_trajectory_positions}
        scored: list[tuple[float, int, int, int]] = []
        for action_idx in valid_actions:
            dr, dc = ACTIONS_8[int(action_idx)]
            next_rc = (ar + int(dr), ac + int(dc))
            score = self._expected_immediate_unknown_gain(
                belief_map=belief_map,
                candidate_rc=next_rc,
            )
            visit = int(visit_count[next_rc[0], next_rc[1]]) if GridTopology.in_bounds(visit_count.shape, *next_rc) else 0
            scored.append((-float(score), int(next_rc in recent), visit, int(action_idx)))
        best_score_neg, _, _, best_action = min(scored)
        best_score = float(-best_score_neg)
        mode = "fallback_information_gain" if best_score > 0.0 else "safe_fallback"
        return FrontierGreedyDecision(
            action_idx=int(best_action),
            decision_mode=mode,
            fallback_score=best_score,
        )

    def decide(
        self,
        *,
        belief_map: np.ndarray,
        agent_array_rc: tuple[int, int],
        valid_action_indices: Sequence[int],
        semantic_snapshot=None,
        visit_count: np.ndarray | None = None,
        frontier_u8: np.ndarray | None = None,
        recent_trajectory_positions: Sequence[tuple[int, int]] = (),
        scan_radius: int = 10,
    ) -> FrontierGreedyDecision:
        belief = np.asarray(belief_map, dtype=np.int8)
        if belief.ndim != 2:
            raise ValueError("belief_map must be a 2D array")
        visits = (
            np.zeros_like(belief, dtype=np.int32)
            if visit_count is None
            else np.asarray(visit_count, dtype=np.int32)
        )
        if visits.shape != belief.shape:
            raise ValueError(f"visit_count shape mismatch: expected {belief.shape}, got {visits.shape}")
        if int(scan_radius) != int(self.sensor.scan_r):
            self.sensor = RadarSensor(scan_radius=int(scan_radius))
        valid_actions = self._normalize_valid(valid_action_indices)
        if len(valid_actions) <= 0:
            return FrontierGreedyDecision(action_idx=None, decision_mode="no_valid_action")

        agent_arr = (int(agent_array_rc[0]), int(agent_array_rc[1]))
        known_free_grid = np.asarray(belief == EMPTY, dtype=bool)

        target, path_cost = self._best_reachable_target(
            known_free_grid=known_free_grid,
            agent_array_rc=agent_arr,
            belief_map=belief,
            semantic_snapshot=semantic_snapshot,
            frontier_u8=frontier_u8,
        )
        if target is not None and path_cost is not None:
            action_idx = self._select_step_toward_target(
                visit_count=visits,
                agent_array_rc=agent_arr,
                target_array_rc=target,
                valid_actions=valid_actions,
                recent_trajectory_positions=recent_trajectory_positions,
            )
            if action_idx is not None:
                return FrontierGreedyDecision(
                    action_idx=int(action_idx),
                    decision_mode="frontier_target",
                    target_array_rc=target,
                    path_cost=int(path_cost),
                )

        return self._fallback_action(
            belief_map=belief,
            agent_array_rc=agent_arr,
            valid_actions=valid_actions,
            visit_count=visits,
            recent_trajectory_positions=recent_trajectory_positions,
        )
