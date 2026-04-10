from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .decision_rules import verdict_rank


@dataclass(frozen=True)
class TrialSpec:
    turn_penalty_scale: float
    revisit_penalty: float
    note: str

    def run_name(self, entry_cap: int) -> str:
        turn_part = int(round(self.turn_penalty_scale * 100))
        revisit_part = int(round(self.revisit_penalty * 100))
        return f"sched_turn{turn_part:03d}_revisit{revisit_part:03d}_entry{entry_cap}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "reward_turn_penalty_scale": self.turn_penalty_scale,
            "reward_revisit_penalty": self.revisit_penalty,
            "note": self.note,
        }


class RecipeBase:
    name = "base"

    def initial_trial(self) -> TrialSpec:
        raise NotImplementedError

    def next_trial(self, first_verdict: str) -> TrialSpec:
        raise NotImplementedError

    def branch_preview(self) -> dict[str, Any]:
        raise NotImplementedError

    def finalize_recommendation(
        self,
        baseline_result: Any,
        trial_records: list[dict[str, Any]],
        entry_cap: int,
    ) -> dict[str, Any]:
        best_record: dict[str, Any] | None = None
        best_choice = {
            "source": "baseline",
            "parameters": {
                "reward_turn_penalty_scale": 0.03,
                "reward_revisit_penalty": 0.12,
                "max_entries_per_block": entry_cap,
            },
            "reason": "No executed trial beat the baseline gate.",
            "verdict": "baseline",
        }
        best_rank = -1

        for record in trial_records:
            verdict = ((record.get("comparison") or {}).get("verdict")) or "not_better"
            rank = verdict_rank(verdict)
            if rank < 1:
                continue
            if rank > best_rank:
                best_record = record
                best_choice = {
                    "source": record.get("trial_id"),
                    "parameters": {
                        "reward_turn_penalty_scale": record["params"]["reward_turn_penalty_scale"],
                        "reward_revisit_penalty": record["params"]["reward_revisit_penalty"],
                        "max_entries_per_block": entry_cap,
                    },
                    "reason": f"{record.get('trial_id')} scored {verdict} against baseline.",
                    "verdict": verdict,
                }
                best_rank = rank
                continue
            if rank == best_rank and self._is_candidate_preferred(record, best_record, baseline_result):
                best_record = record
                best_choice = {
                    "source": record.get("trial_id"),
                    "parameters": {
                        "reward_turn_penalty_scale": record["params"]["reward_turn_penalty_scale"],
                        "reward_revisit_penalty": record["params"]["reward_revisit_penalty"],
                        "max_entries_per_block": entry_cap,
                    },
                    "reason": (
                        f"{record.get('trial_id')} tied on verdict but had stronger final_probe reward/"
                        "success/coverage trade-off."
                    ),
                    "verdict": verdict,
                }

        return best_choice

    def _is_candidate_preferred(
        self,
        candidate_record: dict[str, Any],
        current_record: dict[str, Any] | None,
        baseline_result: Any,
    ) -> bool:
        candidate_probe = (candidate_record.get("result") or {}).get("final_probe") or {}
        if current_record is None:
            current_probe = getattr(baseline_result, "final_probe", {}) or {}
        else:
            current_probe = (current_record.get("result") or {}).get("final_probe") or {}

        def metric(block: dict[str, Any], key: str) -> float | None:
            value = block.get(key)
            return float(value) if isinstance(value, (int, float)) else None

        def score(block: dict[str, Any]) -> tuple[float, float, float, float, float]:
            reward = metric(block, "eval_mean_reward")
            success = metric(block, "eval_success_rate")
            coverage = metric(block, "eval_mean_coverage")
            episode_length = metric(block, "eval_mean_episode_length")
            repeat_ratio = metric(block, "eval_mean_repeat_visit_ratio")
            return (
                reward if reward is not None else float("-inf"),
                success if success is not None else float("-inf"),
                coverage if coverage is not None else float("-inf"),
                -episode_length if episode_length is not None else float("-inf"),
                -repeat_ratio if repeat_ratio is not None else float("-inf"),
            )

        candidate_score = score(candidate_probe)
        current_score = score(current_probe)
        return candidate_score > current_score


class TurnRevisitTreeV1Recipe(RecipeBase):
    name = "turn_revisit_tree_v1"

    def initial_trial(self) -> TrialSpec:
        return TrialSpec(
            turn_penalty_scale=0.07,
            revisit_penalty=0.10,
            note="Step 1: stronger turn penalty plus slightly lighter revisit penalty.",
        )

    def next_trial(self, first_verdict: str) -> TrialSpec:
        if first_verdict in {"better", "slightly_better"}:
            return TrialSpec(
                turn_penalty_scale=0.07,
                revisit_penalty=0.08,
                note="Aggressive branch: keep turn=0.07 and relax revisit penalty further.",
            )
        return TrialSpec(
            turn_penalty_scale=0.05,
            revisit_penalty=0.10,
            note="Conservative branch: soften turn penalty while keeping revisit=0.10.",
        )

    def branch_preview(self) -> dict[str, Any]:
        return {
            "baseline_reference": {
                "reward_turn_penalty_scale": 0.03,
                "reward_revisit_penalty": 0.12,
            },
            "step_1": self.initial_trial().to_dict(),
            "if_better_or_slightly_better": self.next_trial("better").to_dict(),
            "if_not_better": self.next_trial("not_better").to_dict(),
        }


def get_recipe(name: str) -> RecipeBase:
    recipes = {
        TurnRevisitTreeV1Recipe.name: TurnRevisitTreeV1Recipe(),
    }
    try:
        return recipes[name]
    except KeyError as exc:
        raise ValueError(f"Unknown recipe '{name}'. Available: {', '.join(sorted(recipes))}") from exc
