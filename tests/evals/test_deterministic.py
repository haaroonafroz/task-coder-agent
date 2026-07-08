"""Unit tests for deterministic session evaluators."""

from __future__ import annotations

import pytest

from src.evals.deterministic import run_deterministic_evals
from src.evals.aggregator import aggregate_scores
from src.evals.types import EvalContext


def _ctx(events: list[dict], plan: dict | None = None) -> EvalContext:
    return EvalContext(
        session_id="test123",
        events=events,
        plan=plan or {"milestones": [{"id": "M1"}, {"id": "M2"}]},
        handoffs=[],
        session_meta={"status": "completed"},
        user_request="Build a REST API",
    )


def test_mission_completed_scores_high():
    events = [
        {"type": "session.started", "data": {"request": "Build API"}},
        {"type": "milestone.started", "data": {"milestone_id": "M1"}},
        {"type": "milestone.passed", "data": {"milestone_id": "M1"}},
        {"type": "mission.complete", "data": {"status": "completed"}},
    ]
    ctx = _ctx(events)
    scores = run_deterministic_evals(ctx)
    report = aggregate_scores(ctx, scores, deterministic_only=True)
    by_name = {s.name: s for s in scores}
    assert by_name["mission_outcome"].score == 1.0
    assert by_name["spec_gaming"].score == 1.0
    assert report.overall_passed is True


def test_spec_gaming_fails_overall():
    events = [
        {"type": "milestone.started", "data": {}},
        {"type": "validation.spec_gaming", "data": {"unauthorized_edits": ["tests/test_x.py"]}},
        {"type": "mission.complete", "data": {"status": "partial"}},
    ]
    ctx = _ctx(events)
    scores = run_deterministic_evals(ctx)
    report = aggregate_scores(ctx, scores, deterministic_only=True)
    by_name = {s.name: s for s in scores}
    assert by_name["spec_gaming"].score == 0.0
    assert by_name["spec_gaming"].passed is False
    assert report.overall_passed is False


def test_retry_burden_penalizes_retries():
    events = [
        {"type": "milestone.started", "data": {}},
        {"type": "milestone.retry", "data": {"retry": 1}},
        {"type": "milestone.retry", "data": {"retry": 2}},
        {"type": "milestone.retries_exhausted", "data": {}},
        {"type": "mission.complete", "data": {"status": "failed"}},
    ]
    ctx = _ctx(events)
    scores = run_deterministic_evals(ctx)
    by_name = {s.name: s for s in scores}
    assert by_name["retry_burden"].score < 0.5
    assert by_name["retry_burden"].passed is False
