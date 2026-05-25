from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from env.grid_topology import ACTIONS_8, EMPTY, INVISIBLE, GridTopology


ACTION_ORDER_NAMES = ("N", "NE", "E", "SE", "S", "SW", "W", "NW")


@dataclass(frozen=True)
class FrontierGreedyDecision:
    action_idx: int | None
    decision_mode: str
    target_array_rc: tuple[int, int] | None = None
    path_cost: int | None = None
    fallback_score: float | None = None


class ClassicalFrontierGreedyPolicy:
    def policy_summary(self) -> dict[str, object]:
        return {
            "policy_id": "Anew_B_classical_frontier_greedy",
            "policy_type": "traditional_non_learning",
            "target_selection_rule": (
                "Select the reachable frontier anchor with the lowest BFS path cost "
                "over currently known free belief cells."
            ),
            "path_cost_rule": (
                "BFS uses the repository 8-action grid topology and its diagonal "
                "corner-cut prevention over known free belief cells only."
            ),
            "fallback_rule": (
                "If no reachable frontier target exists, choose the valid next action "
                "with the largest count of currently unknown belief cells in the scan "
                "footprint; if all scores tie at zero, choose the first valid action."
            ),
            "tie_break_rule": (
                "Frontier target ties are resolved by cost, row, column, block index, "
                "and cluster index; action ties use ACTIONS_8 order N, NE, E, SE, S, SW, W, NW."
            ),
            "decision_input_contract": (
                "The public decision path uses only belief-derived map state, the current "
                "array-space pose, valid action indices, and optional shared semantic snapshot."
            ),
            "action_order": list(ACTION_ORDER_NAMES),
        }

    @staticmethod
    def _normalize_valid(valid_action_indices: Sequence[int]) -> list[int]:
        valid = sorted({int(action_idx) for action_idx in valid_action_indices})
        return [action_idx for action_idx in valid if 0 <= action_idx < len(ACTIONS_8)]

    @staticmethod
    def _candidate_cells_from_snapshot(semantic_snapshot) -> list[tuple[int, int, int, int]]:
        candidates: list[tuple[int, int, int, int]] = []
        if semantic_snapshot is None:
            return candidates
        blocks = tuple(getattr(semantic_snapshot, "accessible_blocks", tuple()) or tuple())
        for block_order, block in enumerate(blocks):
            block_index = int(getattr(block, "block_index", block_order))
            clusters = tuple(getattr(block, "frontier_clusters", tuple()) or tuple())
            for cluster_order, cluster in enumerate(clusters):
                cluster_index = int(getattr(cluster, "frontier_index", cluster_order))
                rows = np.asarray(getattr(cluster, "rows", np.zeros((0,), dtype=np.int32)), dtype=np.int32)
                cols = np.asarray(getattr(cluster, "cols", np.zeros((0,), dtype=np.int32)), dtype=np.int32)
                for row, col in zip(rows.tolist(), cols.tolist()):
                    candidates.append((int(row), int(col), block_index, cluster_index))
                if len(rows) <= 0:
                    anchor = getattr(cluster, "frontier_anchor_rc", None)
                    if anchor is not None:
                        candidates.append((int(anchor[0]), int(anchor[1]), block_index, cluster_index))
        return candidates

    @staticmethod
    def _candidate_cells_from_belief(belief_map: np.ndarray) -> list[tuple[int, int, int, int]]:
        frontier = GridTopology.frontier_mask(belief_map)
        rows, cols = np.nonzero(frontier)
        return [(int(row), int(col), 0, 0) for row, col in zip(rows.tolist(), cols.tolist())]

    @staticmethod
    def _best_reachable_target(
        *,
        known_free_grid: np.ndarray,
        agent_array_rc: tuple[int, int],
        belief_map: np.ndarray,
        semantic_snapshot,
    ) -> tuple[tuple[int, int] | None, int | None]:
        dist = GridTopology.bfs_distance_map(known_free_grid, agent_array_rc)
        candidates = ClassicalFrontierGreedyPolicy._candidate_cells_from_snapshot(semantic_snapshot)
        if len(candidates) <= 0:
            candidates = ClassicalFrontierGreedyPolicy._candidate_cells_from_belief(belief_map)

        best_key: tuple[int, int, int, int, int] | None = None
        best_cell: tuple[int, int] | None = None
        for row, col, block_index, cluster_index in candidates:
            if not GridTopology.in_bounds(known_free_grid.shape, row, col):
                continue
            if not bool(known_free_grid[row, col]):
                continue
            cost = int(dist[row, col])
            if cost < 0:
                continue
            key = (cost, int(row), int(col), int(block_index), int(cluster_index))
            if best_key is None or key < best_key:
                best_key = key
                best_cell = (int(row), int(col))
        if best_key is None:
            return None, None
        return best_cell, int(best_key[0])

    @staticmethod
    def _select_step_toward_target(
        *,
        known_free_grid: np.ndarray,
        agent_array_rc: tuple[int, int],
        target_array_rc: tuple[int, int],
        valid_actions: list[int],
    ) -> int | None:
        if tuple(agent_array_rc) == tuple(target_array_rc):
            return None
        to_target = GridTopology.bfs_distance_map(known_free_grid, target_array_rc)
        scored: list[tuple[int, int]] = []
        ar, ac = int(agent_array_rc[0]), int(agent_array_rc[1])
        for action_idx in valid_actions:
            dr, dc = ACTIONS_8[int(action_idx)]
            nr, nc = ar + int(dr), ac + int(dc)
            if not GridTopology.in_bounds(known_free_grid.shape, nr, nc):
                continue
            cost = int(to_target[nr, nc])
            if cost < 0:
                continue
            scored.append((cost, int(action_idx)))
        if len(scored) <= 0:
            return None
        return int(min(scored)[1])

    @staticmethod
    def _unknown_count_near(
        *,
        belief_map: np.ndarray,
        center_rc: tuple[int, int],
        scan_radius: int,
    ) -> int:
        radius = max(1, int(scan_radius))
        cr, cc = int(center_rc[0]), int(center_rc[1])
        r0 = max(0, cr - radius)
        r1 = min(int(belief_map.shape[0]), cr + radius + 1)
        c0 = max(0, cc - radius)
        c1 = min(int(belief_map.shape[1]), cc + radius + 1)
        if r0 >= r1 or c0 >= c1:
            return 0
        patch = belief_map[r0:r1, c0:c1]
        rr, cc_grid = np.ogrid[r0:r1, c0:c1]
        disk = ((rr - cr) ** 2 + (cc_grid - cc) ** 2) <= (radius * radius)
        return int(np.count_nonzero((patch == INVISIBLE) & disk))

    def _fallback_action(
        self,
        *,
        belief_map: np.ndarray,
        agent_array_rc: tuple[int, int],
        valid_actions: list[int],
        scan_radius: int,
    ) -> FrontierGreedyDecision:
        if len(valid_actions) <= 0:
            return FrontierGreedyDecision(action_idx=None, decision_mode="no_valid_action")
        ar, ac = int(agent_array_rc[0]), int(agent_array_rc[1])
        scored: list[tuple[int, int]] = []
        for action_idx in valid_actions:
            dr, dc = ACTIONS_8[int(action_idx)]
            score = self._unknown_count_near(
                belief_map=belief_map,
                center_rc=(ar + int(dr), ac + int(dc)),
                scan_radius=scan_radius,
            )
            scored.append((-int(score), int(action_idx)))
        best_score_neg, best_action = min(scored)
        best_score = float(-best_score_neg)
        mode = "immediate_info_gain" if best_score > 0.0 else "safe_fallback"
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
        scan_radius: int = 10,
    ) -> FrontierGreedyDecision:
        belief = np.asarray(belief_map, dtype=np.int8)
        if belief.ndim != 2:
            raise ValueError("belief_map must be a 2D array")
        valid_actions = self._normalize_valid(valid_action_indices)
        if len(valid_actions) <= 0:
            return FrontierGreedyDecision(action_idx=None, decision_mode="no_valid_action")

        agent_arr = (int(agent_array_rc[0]), int(agent_array_rc[1]))
        known_free_grid = np.asarray(belief == EMPTY, dtype=bool)
        if GridTopology.in_bounds(known_free_grid.shape, agent_arr[0], agent_arr[1]):
            known_free_grid[agent_arr[0], agent_arr[1]] = True

        target, path_cost = self._best_reachable_target(
            known_free_grid=known_free_grid,
            agent_array_rc=agent_arr,
            belief_map=belief,
            semantic_snapshot=semantic_snapshot,
        )
        if target is not None and path_cost is not None:
            action_idx = self._select_step_toward_target(
                known_free_grid=known_free_grid,
                agent_array_rc=agent_arr,
                target_array_rc=target,
                valid_actions=valid_actions,
            )
            if action_idx is not None:
                return FrontierGreedyDecision(
                    action_idx=int(action_idx),
                    decision_mode="frontier_greedy",
                    target_array_rc=target,
                    path_cost=int(path_cost),
                )

        return self._fallback_action(
            belief_map=belief,
            agent_array_rc=agent_arr,
            valid_actions=valid_actions,
            scan_radius=int(scan_radius),
        )
