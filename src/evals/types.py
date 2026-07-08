"""Shared types for session-level evaluations (Phase 6)."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Optional


@dataclass
class EvalScore:
    """Result from a single evaluator."""

    name: str
    score: float          # 0.0 – 1.0
    passed: bool
    summary: str
    details: dict[str, Any] = field(default_factory=dict)
    evidence: list[str] = field(default_factory=list)
    kind: str = "deterministic"  # deterministic | llm_judge

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SessionEvalReport:
    """Aggregated evaluation report for one session."""

    session_id: str
    evaluated_at: str
    mission_status: Optional[str]
    overall_score: float
    overall_passed: bool
    scores: list[EvalScore]
    event_count: int
    deterministic_only: bool = True
    weights: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["scores"] = [s.to_dict() if isinstance(s, EvalScore) else s for s in self.scores]
        return d


@dataclass
class EvalContext:
    """All artifacts consumed by evaluators for one session."""

    session_id: str
    events: list[dict[str, Any]]
    plan: dict[str, Any]
    handoffs: list[dict[str, Any]]
    session_meta: dict[str, Any]
    user_request: str = ""
