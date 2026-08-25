"""Aggregate evaluator scores into a session report."""

from __future__ import annotations

import time
from typing import Optional

from src.evals.deterministic import _mission_status
from src.evals.types import EvalContext, EvalScore, SessionEvalReport

# Weights for overall score (deterministic evaluators).
_DETERMINISTIC_WEIGHTS: dict[str, float] = {
    "mission_outcome": 0.25,
    "milestone_pass_rate": 0.20,
    "spec_gaming": 0.15,
    "validation_pass_rate": 0.15,
    "retry_burden": 0.10,
    "worker_reliability": 0.10,
    "tool_efficiency": 0.05,
    "replan_stability": 0.05,
}

_LLM_JUDGE_WEIGHT = 0.15  # each LLM judge contributes equally when present


def aggregate_scores(
    ctx: EvalContext,
    scores: list[EvalScore],
    *,
    deterministic_only: bool,
) -> SessionEvalReport:
    """Compute weighted overall score and pass/fail."""
    by_name = {s.name: s for s in scores}

    det_total = 0.0
    det_weight = 0.0
    for name, weight in _DETERMINISTIC_WEIGHTS.items():
        if name in by_name:
            det_total += by_name[name].score * weight
            det_weight += weight

    llm_scores = [s for s in scores if s.kind == "llm_judge"]
    llm_total = sum(s.score for s in llm_scores)
    llm_count = len(llm_scores)

    if deterministic_only or llm_count == 0:
        overall = det_total / det_weight if det_weight else 0.0
        weights = dict(_DETERMINISTIC_WEIGHTS)
    else:
        # Blend: 70% deterministic bundle, 30% LLM judge average
        det_norm = det_total / det_weight if det_weight else 0.0
        llm_norm = llm_total / llm_count if llm_count else 0.0
        overall = 0.7 * det_norm + 0.3 * llm_norm
        weights = {**_DETERMINISTIC_WEIGHTS, "llm_judge_blend": 0.3}

    overall_passed = overall >= 0.7 and (
        by_name["spec_gaming"].passed if "spec_gaming" in by_name else True
    )

    return SessionEvalReport(
        session_id=ctx.session_id,
        evaluated_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
        mission_status=_mission_status(ctx),
        overall_score=round(overall, 4),
        overall_passed=overall_passed,
        scores=scores,
        event_count=len(ctx.events),
        deterministic_only=deterministic_only,
        weights=weights,
    )
