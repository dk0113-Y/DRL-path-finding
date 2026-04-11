from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class ComparisonOutcome:
    verdict: str
    reasons: list[str]
    details: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict,
            "reasons": self.reasons,
            "details": self.details,
        }


def verdict_rank(verdict: str) -> int:
    return {
        "not_better": 0,
        "slightly_better": 1,
        "better": 2,
    }.get(verdict, -1)


def _final_probe_block(summary: Any) -> dict[str, Any]:
    if hasattr(summary, "final_probe"):
        return getattr(summary, "final_probe") or {}
    if isinstance(summary, dict):
        return summary.get("final_probe") or {}
    return {}


def _value(metrics: dict[str, Any], key: str) -> float | None:
    value = metrics.get(key)
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _reduction_ratio(candidate: float | None, baseline: float | None) -> float | None:
    if candidate is None or baseline is None:
        return None
    if baseline == 0:
        return 0.0 if candidate == 0 else None
    return (baseline - candidate) / baseline


def _improved_lower_is_better(candidate: float | None, baseline: float | None) -> bool:
    if candidate is None or baseline is None:
        return False
    return candidate < baseline


def compare_candidate_to_reference(candidate_summary: Any, reference_summary: Any) -> ComparisonOutcome:
    candidate = _final_probe_block(candidate_summary)
    reference = _final_probe_block(reference_summary)
    if not candidate or not reference:
        raise ValueError("Both candidate and reference require final_probe metrics for comparison.")

    success_candidate = _value(candidate, "eval_success_rate")
    success_reference = _value(reference, "eval_success_rate")
    coverage_candidate = _value(candidate, "eval_mean_coverage")
    coverage_reference = _value(reference, "eval_mean_coverage")
    turn180_candidate = _value(candidate, "eval_mean_turn_180_count")
    turn180_reference = _value(reference, "eval_mean_turn_180_count")
    turn90_candidate = _value(candidate, "eval_mean_turn_ge_90_count")
    turn90_reference = _value(reference, "eval_mean_turn_ge_90_count")
    timeout_candidate = _value(candidate, "eval_mean_timeout_flag")
    timeout_reference = _value(reference, "eval_mean_timeout_flag")
    info_candidate = _value(candidate, "eval_mean_weighted_info_gain_sum")
    info_reference = _value(reference, "eval_mean_weighted_info_gain_sum")

    details = {
        "success_delta": None
        if success_candidate is None or success_reference is None
        else success_candidate - success_reference,
        "coverage_delta": None
        if coverage_candidate is None or coverage_reference is None
        else coverage_candidate - coverage_reference,
        "turn_180_reduction_ratio": _reduction_ratio(turn180_candidate, turn180_reference),
        "turn_ge_90_reduction_ratio": _reduction_ratio(turn90_candidate, turn90_reference),
        "timeout_delta": None
        if timeout_candidate is None or timeout_reference is None
        else timeout_candidate - timeout_reference,
        "weighted_info_gain_delta": None
        if info_candidate is None or info_reference is None
        else info_candidate - info_reference,
        "weighted_info_gain_ratio": None
        if info_candidate is None or info_reference in (None, 0)
        else info_candidate / info_reference,
    }

    direct_bad_reasons: list[str] = []
    if (
        success_candidate is not None
        and success_reference is not None
        and success_candidate < success_reference - 0.05
    ):
        direct_bad_reasons.append(
            f"final_probe success_rate dropped by {success_reference - success_candidate:.4f} (> 0.05)"
        )
    if (
        coverage_candidate is not None
        and coverage_reference is not None
        and coverage_candidate < coverage_reference - 0.005
    ):
        direct_bad_reasons.append(
            f"final_probe coverage dropped by {coverage_reference - coverage_candidate:.4f} (> 0.005)"
        )

    if direct_bad_reasons:
        return ComparisonOutcome(
            verdict="not_better",
            reasons=direct_bad_reasons,
            details=details,
        )

    better_checks = {
        "turn_180_reduced_10pct": (
            turn180_candidate is not None
            and turn180_reference is not None
            and (
                (turn180_reference == 0 and turn180_candidate <= 0)
                or (turn180_reference > 0 and turn180_candidate <= turn180_reference * 0.90)
            )
        ),
        "turn_ge_90_reduced_8pct": (
            turn90_candidate is not None
            and turn90_reference is not None
            and (
                (turn90_reference == 0 and turn90_candidate <= 0)
                or (turn90_reference > 0 and turn90_candidate <= turn90_reference * 0.92)
            )
        ),
        "timeout_not_higher": (
            timeout_candidate is not None
            and timeout_reference is not None
            and timeout_candidate <= timeout_reference
        ),
        "weighted_info_gain_not_down_over_1pct": (
            info_candidate is not None
            and info_reference is not None
            and (
                (info_reference == 0 and info_candidate >= 0)
                or (info_reference != 0 and info_candidate >= info_reference * 0.99)
            )
        ),
    }
    details["better_checks"] = better_checks

    if all(better_checks.values()):
        return ComparisonOutcome(
            verdict="better",
            reasons=[
                "final_probe turn_180_count decreased by at least 10%",
                "final_probe turn_ge_90_count decreased by at least 8%",
                "final_probe timeout_flag did not increase",
                "final_probe weighted_info_gain_sum stayed within 1% of baseline",
            ],
            details=details,
        )

    auxiliary_checks = {
        "episode_length_down": _improved_lower_is_better(
            _value(candidate, "eval_mean_episode_length"),
            _value(reference, "eval_mean_episode_length"),
        ),
        "repeat_visit_ratio_down": _improved_lower_is_better(
            _value(candidate, "eval_mean_repeat_visit_ratio"),
            _value(reference, "eval_mean_repeat_visit_ratio"),
        ),
        "recent_revisit_count_down": _improved_lower_is_better(
            _value(candidate, "eval_mean_recent_revisit_count"),
            _value(reference, "eval_mean_recent_revisit_count"),
        ),
        "zero_info_step_count_down": _improved_lower_is_better(
            _value(candidate, "eval_mean_zero_info_step_count"),
            _value(reference, "eval_mean_zero_info_step_count"),
        ),
    }
    details["auxiliary_checks"] = auxiliary_checks

    improved_aux_count = sum(1 for improved in auxiliary_checks.values() if improved)
    if improved_aux_count >= 3:
        return ComparisonOutcome(
            verdict="slightly_better",
            reasons=[
                "final_probe did not trigger the direct-bad success/coverage gate",
                f"{improved_aux_count} of 4 auxiliary efficiency metrics improved",
            ],
            details=details,
        )

    reasons = [
        "final_probe did not clear the strict positive gate",
        f"{improved_aux_count} of 4 auxiliary efficiency metrics improved",
    ]
    return ComparisonOutcome(
        verdict="not_better",
        reasons=reasons,
        details=details,
    )


def compare_to_baseline(candidate_summary: Any, baseline_summary: Any) -> ComparisonOutcome:
    return compare_candidate_to_reference(candidate_summary, baseline_summary)
