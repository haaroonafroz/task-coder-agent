"""Session eval endpoints (Phase 6)."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from src.api.deps import get_session_manager, require_session
from src.api.schemas import EvalScoreResponse, SessionEvalReportResponse
from src.evals.runner import get_cached_report, run_session_evals
from src.evals.types import SessionEvalReport
from src.session import SessionContext, SessionManager

router = APIRouter(prefix="/sessions/{sid}/evals", tags=["evals"])


def _to_response(report: SessionEvalReport) -> SessionEvalReportResponse:
    return SessionEvalReportResponse(
        session_id=report.session_id,
        evaluated_at=report.evaluated_at,
        mission_status=report.mission_status,
        overall_score=report.overall_score,
        overall_passed=report.overall_passed,
        event_count=report.event_count,
        deterministic_only=report.deterministic_only,
        weights=report.weights,
        scores=[
            EvalScoreResponse(
                name=s.name,
                score=s.score,
                passed=s.passed,
                summary=s.summary,
                details=s.details,
                evidence=s.evidence,
                kind=s.kind,
            )
            for s in report.scores
        ],
    )


@router.get("", response_model=SessionEvalReportResponse)
async def get_evals(
    sid: str,
    manager: SessionManager = Depends(get_session_manager),
) -> SessionEvalReportResponse:
    """Return the cached eval report for a session."""
    ctx = require_session(sid, manager)
    report = get_cached_report(ctx)
    if report is None:
        raise HTTPException(
            status_code=404,
            detail=(
                f"No eval report for session '{sid}'. "
                "Run a mission first, then POST /evals/run."
            ),
        )
    return _to_response(report)


@router.post("/run", response_model=SessionEvalReportResponse)
async def run_evals(
    sid: str,
    manager: SessionManager = Depends(get_session_manager),
) -> SessionEvalReportResponse:
    """Recompute evals from session artifacts and persist the report."""
    ctx = require_session(sid, manager)
    try:
        report = run_session_evals(
            ctx,
            persist=True,
            model=ctx.selected_model,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return _to_response(report)
