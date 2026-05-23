from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional, Sequence

import numpy as np

from env.core_radar import RadarSensor
from env.grid_topology import ACTIONS_8, EMPTY, INVISIBLE, OBSTACLE, GridTopology


BASELINE_NAME = "classical_frontier_greedy_v1"


@dataclass(frozen=True)
class FrontierGreedyPolicyConfig:
    baseline_name: str = BASELINE_NAME
    scan_radius: int = 10
    use_shared_semantic_snapshot: bool = True
    action_tie_break_order: tuple[int, ...] = tuple(range(len(ACTIONS_8)))
    target_selection_rule: str = (
        "Select the reachable frontier cluster anchor with the lowest known-free BFS cost; "
        "fall back to raw frontier cells from the cumulative belief frontier cache."
    )
    frontier_source: str = "shared_semantic_snapshot_then_cumulative_frontier_cache"
    fallback_rule: str = (
        "When no reachable frontier exists, choose the legal action with the largest "
        "belief-only expected immediate information gain."
    )
    tie_break_rule: str = (
        "target_distance_or_gain, recent_trajectory_revisit, visit_count, fixed action order "
        "(N, NE, E, SE, S, SW, W, NW)"
    )
    no_training: bool = True
    no_q_network: bool = True
    no_checkpoint: bool = True
    no_ground_truth_map_for_decision: bool = True

    def __post_init__(self) -> None:
        order = tuple(int(v) for v in self.action_tie_break_order)
        if sorted(order) != list(range(len(ACTIONS_8))):
            raise ValueError(
                "action_tie_break_order must be a permutation of "
                f"0..{len(ACTIONS_8) - 1}, got {self.action_tie_break_order!r}"
            )
        if int(self.scan_radius) < 1:
            raise ValueError("scan_radius must be >= 1")
        object.__setattr__(self, "action_tie_break_order", order)


@dataclass(frozen=True)
class FrontierGreedyBeliefView:
    """Policy-facing belief snapshot that deliberately excludes simulator truth."""

    belief_map: np.ndarray
    visit_count: np.ndarray
    frontier_u8: np.ndarray
    origin_world_rc: tuple[int, int]

    @classmethod
    def from_cumulative_map(
        cls,
        cum_map,
        *,
        frontier_u8: Optional[np.ndarray] = None,
    ) -> "FrontierGreedyBeliefView":
        return cls(
            belief_map=np.asarray(cum_map.map, dtype=np.int8),
            visit_count=np.asarray(cum_map.visit_count, dtype=np.int32),
            frontier_u8=np.asarray(
                cum_map.get_frontier_u8(refresh=False) if frontier_u8 is None else frontier_u8,
                dtype=np.uint8,
            ),
            origin_world_rc=(int(cum_map.origin_world_rc[0]), int(cum_map.origin_world_rc[1])),
        )

    @property
    def shape(self) -> tuple[int, int]:
        return int(self.belief_map.shape[0]), int(self.belief_map.shape[1])

    def world_to_array(self, world_rc: tuple[int, int]) -> tuple[int, int]:
        return (
            int(world_rc[0]) - int(self.origin_world_rc[0]),
            int(world_rc[1]) - int(self.origin_world_rc[1]),
        )

    def array_to_world(self, array_rc: tuple[int, int]) -> tuple[int, int]:
        return (
            int(array_rc[0]) + int(self.origin_world_rc[0]),
            int(array_rc[1]) + int(self.origin_world_rc[1]),
        )

    def in_array_bounds(self, array_rc: tuple[int, int]) -> bool:
        r, c = int(array_rc[0]), int(array_rc[1])
        return 0 <= r < int(self.belief_map.shape[0]) and 0 <= c < int(self.belief_map.shape[1])

    def cell_value_array(self, array_rc: tuple[int, int]) -> int:
        if not self.in_array_bounds(array_rc):
            return INVISIBLE
        return int(self.belief_map[int(array_rc[0]), int(array_rc[1])])

    def cell_value_world(self, world_rc: tuple[int, int]) -> int:
        return self.cell_value_array(self.world_to_array(world_rc))

    def visit_count_world(self, world_rc: tuple[int, int]) -> int:
        array_rc = self.world_to_array(world_rc)
        if not self.in_array_bounds(array_rc):
            return 0
        return int(self.visit_count[int(array_rc[0]), int(array_rc[1])])


@dataclass(frozen=True)
class FrontierGreedyDecision:
    action_idx: int
    mode: str
    target_source: str
    target_anchor_array_rc: Optional[tuple[int, int]]
    fallback_expected_gain: Optional[float]


class FrontierGreedyPolicy:
    """Deterministic, non-learning frontier-greedy exploration policy."""

    def __init__(self, cfg: Optional[FrontierGreedyPolicyConfig] = None):
        self.cfg = cfg if cfg is not None else FrontierGreedyPolicyConfig()
        self.sensor = RadarSensor(scan_radius=int(self.cfg.scan_radius))
        self._action_rank = {
            int(action_idx): rank
            for rank, action_idx in enumerate(self.cfg.action_tie_break_order)
        }
        self.last_decision: Optional[FrontierGreedyDecision] = None

    def reset_episode(self) -> None:
        self.last_decision = None

    def policy_summary(self) -> dict[str, object]:
        return {
            "baseline_name": str(self.cfg.baseline_name),
            "target_selection_rule": str(self.cfg.target_selection_rule),
            "frontier_source": str(self.cfg.frontier_source),
            "fallback_rule": str(self.cfg.fallback_rule),
            "tie_break_order": [
                {
                    "action_idx": int(action_idx),
                    "delta_rc": [int(ACTIONS_8[int(action_idx)][0]), int(ACTIONS_8[int(action_idx)][1])],
                }
                for action_idx in self.cfg.action_tie_break_order
            ],
            "tie_break_rule": str(self.cfg.tie_break_rule),
            "use_shared_semantic_snapshot": bool(self.cfg.use_shared_semantic_snapshot),
            "no_training": True,
            "no_q_network": True,
            "no_checkpoint": True,
            "no_ground_truth_map_for_decision": True,
        }

    def select_action(
        self,
        *,
        belief: FrontierGreedyBeliefView,
        agent_state: tuple[int, int],
        valid_actions: Sequence[int],
        shared_semantic_snapshot=None,
        recent_trajectory_positions: Sequence[tuple[int, int]] = (),
    ) -> int:
        ordered_valid = self._ordered_valid_actions(valid_actions)
        if len(ordered_valid) <= 0:
            raise RuntimeError("FrontierGreedyPolicy received an empty valid action set")

        agent_arr = belief.world_to_array(agent_state)
        distance_map = self._known_free_distance_map(belief, agent_arr)
        target = self._select_frontier_target(
            belief=belief,
            agent_arr=agent_arr,
            distance_map=distance_map,
            shared_semantic_snapshot=shared_semantic_snapshot,
        )
        if target is not None:
            anchor, source = target
            action_idx = self._action_toward_target(
                belief=belief,
                agent_state=agent_state,
                target_anchor_array_rc=anchor,
                ordered_valid=ordered_valid,
                recent_trajectory_positions=recent_trajectory_positions,
            )
            self.last_decision = FrontierGreedyDecision(
                action_idx=int(action_idx),
                mode="frontier_target",
                target_source=source,
                target_anchor_array_rc=(int(anchor[0]), int(anchor[1])),
                fallback_expected_gain=None,
            )
            return int(action_idx)

        action_idx, expected_gain = self._fallback_information_gain_action(
            belief=belief,
            agent_state=agent_state,
            ordered_valid=ordered_valid,
            recent_trajectory_positions=recent_trajectory_positions,
        )
        self.last_decision = FrontierGreedyDecision(
            action_idx=int(action_idx),
            mode="fallback_information_gain",
            target_source="none",
            target_anchor_array_rc=None,
            fallback_expected_gain=float(expected_gain),
        )
        return int(action_idx)

    def _ordered_valid_actions(self, valid_actions: Sequence[int]) -> list[int]:
        valid = {int(action_idx) for action_idx in valid_actions}
        ordered = [int(action_idx) for action_idx in self.cfg.action_tie_break_order if int(action_idx) in valid]
        ordered.extend(sorted(valid.difference(ordered)))
        return ordered

    @staticmethod
    def _known_free_distance_map(
        belief: FrontierGreedyBeliefView,
        agent_arr: tuple[int, int],
    ) -> np.ndarray:
        free = np.asarray(belief.belief_map == EMPTY, dtype=bool)
        if not belief.in_array_bounds(agent_arr):
            return np.full(belief.shape, -1, dtype=np.int32)
        return GridTopology.bfs_distance_map(free, agent_arr, unreachable_value=-1)

    def _select_frontier_target(
        self,
        *,
        belief: FrontierGreedyBeliefView,
        agent_arr: tuple[int, int],
        distance_map: np.ndarray,
        shared_semantic_snapshot,
    ) -> Optional[tuple[tuple[int, int], str]]:
        if bool(self.cfg.use_shared_semantic_snapshot) and shared_semantic_snapshot is not None:
            target = self._select_target_from_semantic_snapshot(
                agent_arr=agent_arr,
                distance_map=distance_map,
                shared_semantic_snapshot=shared_semantic_snapshot,
            )
            if target is not None:
                return target, "shared_semantic_snapshot_frontier_cluster"

        target = self._select_target_from_frontier_cache(
            belief=belief,
            agent_arr=agent_arr,
            distance_map=distance_map,
        )
        if target is not None:
            return target, "cumulative_frontier_cache"
        return None

    @staticmethod
    def _select_target_from_semantic_snapshot(
        *,
        agent_arr: tuple[int, int],
        distance_map: np.ndarray,
        shared_semantic_snapshot,
    ) -> Optional[tuple[int, int]]:
        best_key = None
        best_anchor = None
        for block in getattr(shared_semantic_snapshot, "accessible_blocks", ()):
            for cluster in getattr(block, "frontier_clusters", ()):
                anchor = tuple(int(v) for v in getattr(cluster, "frontier_anchor_rc"))
                ar, ac = int(anchor[0]), int(anchor[1])
                if ar < 0 or ar >= int(distance_map.shape[0]) or ac < 0 or ac >= int(distance_map.shape[1]):
                    continue
                distance = int(distance_map[ar, ac])
                if distance < 0:
                    continue
                euclidean = float((ar - int(agent_arr[0])) ** 2 + (ac - int(agent_arr[1])) ** 2)
                key = (
                    distance,
                    euclidean,
                    int(getattr(block, "block_index", 0)),
                    int(getattr(cluster, "frontier_index", 0)),
                    ar,
                    ac,
                )
                if best_key is None or key < best_key:
                    best_key = key
                    best_anchor = anchor
        return best_anchor

    @staticmethod
    def _select_target_from_frontier_cache(
        *,
        belief: FrontierGreedyBeliefView,
        agent_arr: tuple[int, int],
        distance_map: np.ndarray,
    ) -> Optional[tuple[int, int]]:
        frontier = np.asarray(belief.frontier_u8, dtype=np.uint8) > 0
        if frontier.shape != distance_map.shape or not np.any(frontier):
            return None

        rows, cols = np.nonzero(frontier)
        best_key = None
        best_anchor = None
        for ar, ac in zip(rows.tolist(), cols.tolist()):
            distance = int(distance_map[int(ar), int(ac)])
            if distance < 0:
                continue
            euclidean = float((int(ar) - int(agent_arr[0])) ** 2 + (int(ac) - int(agent_arr[1])) ** 2)
            key = (distance, euclidean, int(ar), int(ac))
            if best_key is None or key < best_key:
                best_key = key
                best_anchor = (int(ar), int(ac))
        return best_anchor

    def _action_toward_target(
        self,
        *,
        belief: FrontierGreedyBeliefView,
        agent_state: tuple[int, int],
        target_anchor_array_rc: tuple[int, int],
        ordered_valid: Sequence[int],
        recent_trajectory_positions: Sequence[tuple[int, int]],
    ) -> int:
        recent = {(int(r), int(c)) for r, c in recent_trajectory_positions}
        best_key = None
        best_action = int(ordered_valid[0])
        for action_idx in ordered_valid:
            dr, dc = ACTIONS_8[int(action_idx)]
            next_world = (int(agent_state[0]) + int(dr), int(agent_state[1]) + int(dc))
            next_arr = belief.world_to_array(next_world)
            dist2 = (
                (int(next_arr[0]) - int(target_anchor_array_rc[0])) ** 2
                + (int(next_arr[1]) - int(target_anchor_array_rc[1])) ** 2
            )
            key = (
                int(dist2),
                int(next_world in recent),
                int(belief.visit_count_world(next_world)),
                int(self._action_rank.get(int(action_idx), len(ACTIONS_8))),
            )
            if best_key is None or key < best_key:
                best_key = key
                best_action = int(action_idx)
        return best_action

    def _fallback_information_gain_action(
        self,
        *,
        belief: FrontierGreedyBeliefView,
        agent_state: tuple[int, int],
        ordered_valid: Sequence[int],
        recent_trajectory_positions: Sequence[tuple[int, int]],
    ) -> tuple[int, float]:
        recent = {(int(r), int(c)) for r, c in recent_trajectory_positions}
        best_key = None
        best_action = int(ordered_valid[0])
        best_gain = 0.0
        for action_idx in ordered_valid:
            dr, dc = ACTIONS_8[int(action_idx)]
            next_world = (int(agent_state[0]) + int(dr), int(agent_state[1]) + int(dc))
            expected_gain = self._expected_immediate_unknown_gain(belief, next_world)
            key = (
                -float(expected_gain),
                int(next_world in recent),
                int(belief.visit_count_world(next_world)),
                int(self._action_rank.get(int(action_idx), len(ACTIONS_8))),
            )
            if best_key is None or key < best_key:
                best_key = key
                best_action = int(action_idx)
                best_gain = float(expected_gain)
        return best_action, best_gain

    def _expected_immediate_unknown_gain(
        self,
        belief: FrontierGreedyBeliefView,
        candidate_world: tuple[int, int],
    ) -> float:
        visible_unknown: set[tuple[int, int]] = set()
        for ray in self.sensor.local_ray_templates:
            prev_rel: Optional[tuple[int, int]] = None
            for rel_r, rel_c, _, _ in ray:
                rel = (int(rel_r), int(rel_c))
                if self._belief_corner_occluded(belief, candidate_world, prev_rel, rel):
                    break
                world_rc = (int(candidate_world[0]) + rel[0], int(candidate_world[1]) + rel[1])
                value = belief.cell_value_world(world_rc)
                if value == OBSTACLE:
                    break
                if value == INVISIBLE:
                    visible_unknown.add(world_rc)
                prev_rel = rel
        return float(len(visible_unknown))

    def _belief_corner_occluded(
        self,
        belief: FrontierGreedyBeliefView,
        candidate_world: tuple[int, int],
        prev_rel: Optional[tuple[int, int]],
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
        side_a = (int(candidate_world[0]) + int(cur_rel[0]), int(candidate_world[1]) + int(prev_rel[1]))
        side_b = (int(candidate_world[0]) + int(prev_rel[0]), int(candidate_world[1]) + int(cur_rel[1]))
        return belief.cell_value_world(side_a) == OBSTACLE and belief.cell_value_world(side_b) == OBSTACLE


def baseline_policy_config_dict(cfg: FrontierGreedyPolicyConfig) -> dict[str, object]:
    payload = asdict(cfg)
    payload["action_tie_break_order"] = [int(v) for v in cfg.action_tie_break_order]
    return payload
