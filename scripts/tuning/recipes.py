from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .decision_rules import verdict_rank


GOOD_VERDICTS = {"better", "slightly_better"}


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


@dataclass(frozen=True)
class TrialPlan:
    trial_spec: TrialSpec
    compare_to: str
    branch_position: str
    decision_note: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "trial_spec": self.trial_spec.to_dict(),
            "compare_to": self.compare_to,
            "branch_position": self.branch_position,
            "decision_note": self.decision_note,
        }


class RecipeBase:
    name = "base"

    def baseline_reference(self, entry_cap: int) -> dict[str, Any]:
        raise NotImplementedError

    def max_new_trials(self) -> int:
        raise NotImplementedError

    def initial_trial_plan(self, entry_cap: int) -> TrialPlan:
        raise NotImplementedError

    def next_trial_plan(self, entry_cap: int, trial_records: list[dict[str, Any]]) -> TrialPlan | None:
        raise NotImplementedError

    def branch_preview(self, entry_cap: int) -> dict[str, Any]:
        raise NotImplementedError

    def possible_trial_plans(self, entry_cap: int) -> dict[str, TrialPlan]:
        raise NotImplementedError

    def finalize_recommendation(
        self,
        baseline_result: Any,
        trial_records: list[dict[str, Any]],
        entry_cap: int,
    ) -> dict[str, Any]:
        baseline_reference = self.baseline_reference(entry_cap)
        best_record: dict[str, Any] | None = None
        best_choice = {
            "source": "baseline",
            "parameters": baseline_reference,
            "comparison_reference": "baseline",
            "reason": "No executed trial beat the decision-tree comparison gate.",
            "verdict": "baseline",
            "recommendation_basis": "decision_tree_internal_comparisons",
        }
        best_rank = -1

        for record in trial_records:
            verdict = ((record.get("comparison") or {}).get("verdict")) or "not_better"
            rank = verdict_rank(verdict)
            if rank < 1:
                continue
            reference_label = ((record.get("comparison_reference") or {}).get("label")) or "baseline"
            if rank > best_rank:
                best_record = record
                best_choice = {
                    "source": record.get("trial_id"),
                    "parameters": {
                        "reward_turn_penalty_scale": record["params"]["reward_turn_penalty_scale"],
                        "reward_revisit_penalty": record["params"]["reward_revisit_penalty"],
                        "max_entries_per_block": entry_cap,
                    },
                    "comparison_reference": reference_label,
                    "reason": f"{record.get('trial_id')} scored {verdict} against {reference_label}.",
                    "verdict": verdict,
                    "recommendation_basis": "decision_tree_internal_comparisons",
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
                    "comparison_reference": reference_label,
                    "reason": (
                        f"{record.get('trial_id')} tied on verdict and had the stronger final_probe "
                        f"tie-break against current recommendation candidates."
                    ),
                    "verdict": verdict,
                    "recommendation_basis": "decision_tree_internal_comparisons",
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

        return score(candidate_probe) > score(current_probe)

    def _verdict_for(self, trial_records: list[dict[str, Any]], trial_id: str) -> str:
        for record in trial_records:
            if record.get("trial_id") == trial_id:
                return ((record.get("comparison") or {}).get("verdict")) or "not_better"
        raise ValueError(f"Missing verdict for {trial_id}")


class TurnRevisitTreeV1Recipe(RecipeBase):
    name = "turn_revisit_tree_v1"

    def baseline_reference(self, entry_cap: int) -> dict[str, Any]:
        return {
            "reward_turn_penalty_scale": 0.03,
            "reward_revisit_penalty": 0.12,
            "max_entries_per_block": entry_cap,
        }

    def max_new_trials(self) -> int:
        return 2

    def initial_trial_plan(self, entry_cap: int) -> TrialPlan:
        return TrialPlan(
            trial_spec=TrialSpec(
                turn_penalty_scale=0.07,
                revisit_penalty=0.10,
                note="Step 1: stronger turn penalty plus slightly lighter revisit penalty.",
            ),
            compare_to="baseline",
            branch_position="trial_01",
            decision_note="First probe against baseline reference.",
        )

    def next_trial_plan(self, entry_cap: int, trial_records: list[dict[str, Any]]) -> TrialPlan | None:
        if not trial_records:
            return self.initial_trial_plan(entry_cap)
        if len(trial_records) >= self.max_new_trials():
            return None

        first_verdict = self._verdict_for(trial_records, "trial_01")
        if first_verdict in GOOD_VERDICTS:
            return TrialPlan(
                trial_spec=TrialSpec(
                    turn_penalty_scale=0.07,
                    revisit_penalty=0.08,
                    note="Aggressive branch: keep turn=0.07 and relax revisit penalty further.",
                ),
                compare_to="baseline",
                branch_position="trial_02.after_trial_01_good",
                decision_note="trial_01 cleared the baseline gate, so test revisit=0.08 next.",
            )

        return TrialPlan(
            trial_spec=TrialSpec(
                turn_penalty_scale=0.05,
                revisit_penalty=0.10,
                note="Conservative branch: soften turn penalty while keeping revisit=0.10.",
            ),
            compare_to="baseline",
            branch_position="trial_02.after_trial_01_not_better",
            decision_note="trial_01 did not clear the baseline gate, so soften turn to 0.05.",
        )

    def branch_preview(self, entry_cap: int) -> dict[str, Any]:
        return {
            "baseline_reference": self.baseline_reference(entry_cap),
            "trial_01": self.initial_trial_plan(entry_cap).to_dict(),
            "after_trial_01_better_or_slightly_better": {
                "trial_02": self.possible_trial_plans(entry_cap)["trial_02_if_trial_01_good"].to_dict(),
            },
            "after_trial_01_not_better": {
                "trial_02": self.possible_trial_plans(entry_cap)["trial_02_if_trial_01_not_better"].to_dict(),
            },
        }

    def possible_trial_plans(self, entry_cap: int) -> dict[str, TrialPlan]:
        return {
            "trial_01": self.initial_trial_plan(entry_cap),
            "trial_02_if_trial_01_good": TrialPlan(
                trial_spec=TrialSpec(
                    turn_penalty_scale=0.07,
                    revisit_penalty=0.08,
                    note="Aggressive branch: keep turn=0.07 and relax revisit penalty further.",
                ),
                compare_to="baseline",
                branch_position="trial_02.after_trial_01_good",
                decision_note="trial_01 cleared the baseline gate, so test revisit=0.08 next.",
            ),
            "trial_02_if_trial_01_not_better": TrialPlan(
                trial_spec=TrialSpec(
                    turn_penalty_scale=0.05,
                    revisit_penalty=0.10,
                    note="Conservative branch: soften turn penalty while keeping revisit=0.10.",
                ),
                compare_to="baseline",
                branch_position="trial_02.after_trial_01_not_better",
                decision_note="trial_01 did not clear the baseline gate, so soften turn to 0.05.",
            ),
        }


class TurnRevisitTreeV2ThreeStageRecipe(RecipeBase):
    name = "turn_revisit_tree_v2_three_stage"

    def baseline_reference(self, entry_cap: int) -> dict[str, Any]:
        return {
            "reward_turn_penalty_scale": 0.03,
            "reward_revisit_penalty": 0.12,
            "max_entries_per_block": entry_cap,
        }

    def max_new_trials(self) -> int:
        return 3

    def initial_trial_plan(self, entry_cap: int) -> TrialPlan:
        return TrialPlan(
            trial_spec=TrialSpec(
                turn_penalty_scale=0.07,
                revisit_penalty=0.10,
                note="Stage 1: first strong-turn / lighter-revisit probe.",
            ),
            compare_to="baseline",
            branch_position="trial_01",
            decision_note="First probe against baseline reference.",
        )

    def next_trial_plan(self, entry_cap: int, trial_records: list[dict[str, Any]]) -> TrialPlan | None:
        if not trial_records:
            return self.initial_trial_plan(entry_cap)
        if len(trial_records) >= self.max_new_trials():
            return None

        first_verdict = self._verdict_for(trial_records, "trial_01")
        if len(trial_records) == 1:
            if first_verdict in GOOD_VERDICTS:
                return TrialPlan(
                    trial_spec=TrialSpec(
                        turn_penalty_scale=0.07,
                        revisit_penalty=0.08,
                        note="Stage 2A: keep turn=0.07, lower revisit to 0.08.",
                    ),
                    compare_to="trial_01",
                    branch_position="trial_02.branch_A",
                    decision_note="trial_01 beat baseline, so compare a lower revisit directly against trial_01.",
                )
            return TrialPlan(
                trial_spec=TrialSpec(
                    turn_penalty_scale=0.05,
                    revisit_penalty=0.10,
                    note="Stage 2B: soften turn to 0.05 while keeping revisit=0.10.",
                ),
                compare_to="baseline",
                branch_position="trial_02.branch_B",
                decision_note="trial_01 did not beat baseline, so compare a softer turn setting against baseline.",
            )

        if len(trial_records) == 2:
            second_verdict = self._verdict_for(trial_records, "trial_02")
            if first_verdict in GOOD_VERDICTS:
                if second_verdict in GOOD_VERDICTS:
                    return TrialPlan(
                        trial_spec=TrialSpec(
                            turn_penalty_scale=0.05,
                            revisit_penalty=0.08,
                            note="Stage 3A1: keep revisit=0.08 and lower turn to 0.05.",
                        ),
                        compare_to="trial_02",
                        branch_position="trial_03.branch_A1",
                        decision_note=(
                            "trial_02 still improved over trial_01, so lower turn against trial_02 "
                            "to isolate whether revisit=0.08 remains helpful without turn=0.07."
                        ),
                    )
                return TrialPlan(
                    trial_spec=TrialSpec(
                        turn_penalty_scale=0.05,
                        revisit_penalty=0.10,
                        note="Stage 3A2: lower turn to 0.05 at revisit=0.10.",
                    ),
                    compare_to="trial_01",
                    branch_position="trial_03.branch_A2",
                    decision_note=(
                        "trial_02 did not improve over trial_01, so compare turn=0.05 directly against "
                        "trial_01 to test whether turn=0.07 was too strong."
                    ),
                )

            if second_verdict in GOOD_VERDICTS:
                return TrialPlan(
                    trial_spec=TrialSpec(
                        turn_penalty_scale=0.05,
                        revisit_penalty=0.08,
                        note="Stage 3B1: keep turn=0.05 and lower revisit to 0.08.",
                    ),
                    compare_to="trial_02",
                    branch_position="trial_03.branch_B1",
                    decision_note=(
                        "trial_02 beat baseline, so compare a lower revisit directly against trial_02 "
                        "to see whether revisit can go lower under turn=0.05."
                    ),
                )
            return TrialPlan(
                trial_spec=TrialSpec(
                    turn_penalty_scale=0.03,
                    revisit_penalty=0.10,
                    note="Stage 3B2: revert turn to 0.03 and lower revisit to 0.10.",
                ),
                compare_to="baseline",
                branch_position="trial_03.branch_B2",
                decision_note=(
                    "trial_02 still did not beat baseline, so compare revisit-only reduction "
                    "against the baseline turn strength."
                ),
            )

        return None

    def branch_preview(self, entry_cap: int) -> dict[str, Any]:
        plans = self.possible_trial_plans(entry_cap)
        return {
            "baseline_reference": self.baseline_reference(entry_cap),
            "trial_01": plans["trial_01"].to_dict(),
            "after_trial_01_better_or_slightly_better": {
                "trial_02": plans["trial_02_branch_A"].to_dict(),
                "if_trial_02_better_or_slightly_better": {
                    "trial_03": plans["trial_03_branch_A1"].to_dict(),
                },
                "if_trial_02_not_better": {
                    "trial_03": plans["trial_03_branch_A2"].to_dict(),
                },
            },
            "after_trial_01_not_better": {
                "trial_02": plans["trial_02_branch_B"].to_dict(),
                "if_trial_02_better_or_slightly_better": {
                    "trial_03": plans["trial_03_branch_B1"].to_dict(),
                },
                "if_trial_02_not_better": {
                    "trial_03": plans["trial_03_branch_B2"].to_dict(),
                },
            },
        }

    def possible_trial_plans(self, entry_cap: int) -> dict[str, TrialPlan]:
        return {
            "trial_01": self.initial_trial_plan(entry_cap),
            "trial_02_branch_A": TrialPlan(
                trial_spec=TrialSpec(
                    turn_penalty_scale=0.07,
                    revisit_penalty=0.08,
                    note="Stage 2A: keep turn=0.07, lower revisit to 0.08.",
                ),
                compare_to="trial_01",
                branch_position="trial_02.branch_A",
                decision_note="trial_01 beat baseline, so compare a lower revisit directly against trial_01.",
            ),
            "trial_02_branch_B": TrialPlan(
                trial_spec=TrialSpec(
                    turn_penalty_scale=0.05,
                    revisit_penalty=0.10,
                    note="Stage 2B: soften turn to 0.05 while keeping revisit=0.10.",
                ),
                compare_to="baseline",
                branch_position="trial_02.branch_B",
                decision_note="trial_01 did not beat baseline, so compare a softer turn setting against baseline.",
            ),
            "trial_03_branch_A1": TrialPlan(
                trial_spec=TrialSpec(
                    turn_penalty_scale=0.05,
                    revisit_penalty=0.08,
                    note="Stage 3A1: keep revisit=0.08 and lower turn to 0.05.",
                ),
                compare_to="trial_02",
                branch_position="trial_03.branch_A1",
                decision_note=(
                    "trial_02 still improved over trial_01, so lower turn against trial_02 "
                    "to isolate whether revisit=0.08 remains helpful without turn=0.07."
                ),
            ),
            "trial_03_branch_A2": TrialPlan(
                trial_spec=TrialSpec(
                    turn_penalty_scale=0.05,
                    revisit_penalty=0.10,
                    note="Stage 3A2: lower turn to 0.05 at revisit=0.10.",
                ),
                compare_to="trial_01",
                branch_position="trial_03.branch_A2",
                decision_note=(
                    "trial_02 did not improve over trial_01, so compare turn=0.05 directly against "
                    "trial_01 to test whether turn=0.07 was too strong."
                ),
            ),
            "trial_03_branch_B1": TrialPlan(
                trial_spec=TrialSpec(
                    turn_penalty_scale=0.05,
                    revisit_penalty=0.08,
                    note="Stage 3B1: keep turn=0.05 and lower revisit to 0.08.",
                ),
                compare_to="trial_02",
                branch_position="trial_03.branch_B1",
                decision_note=(
                    "trial_02 beat baseline, so compare a lower revisit directly against trial_02 "
                    "to see whether revisit can go lower under turn=0.05."
                ),
            ),
            "trial_03_branch_B2": TrialPlan(
                trial_spec=TrialSpec(
                    turn_penalty_scale=0.03,
                    revisit_penalty=0.10,
                    note="Stage 3B2: revert turn to 0.03 and lower revisit to 0.10.",
                ),
                compare_to="baseline",
                branch_position="trial_03.branch_B2",
                decision_note=(
                    "trial_02 still did not beat baseline, so compare revisit-only reduction "
                    "against the baseline turn strength."
                ),
            ),
        }


def get_recipe(name: str) -> RecipeBase:
    recipes = {
        TurnRevisitTreeV1Recipe.name: TurnRevisitTreeV1Recipe(),
        TurnRevisitTreeV2ThreeStageRecipe.name: TurnRevisitTreeV2ThreeStageRecipe(),
    }
    try:
        return recipes[name]
    except KeyError as exc:
        raise ValueError(f"Unknown recipe '{name}'. Available: {', '.join(sorted(recipes))}") from exc
