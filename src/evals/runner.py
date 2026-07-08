"""Orchestrate session-level eval execution."""

from __future__ import annotations

import os
from typing import Optional, TYPE_CHECKING

from src.evals.aggregator import aggregate_scores
from src.evals.deterministic import run_deterministic_evals
from src.evals.llm_judge import llm_judges_enabled, run_llm_judge_evals
from src.evals.loader import load_eval_context
from src.evals.phoenix_export import export_eval_report_to_phoenix
from src.evals.store import load_report, save_report
from src.evals.types import SessionEvalReport
from src.llm_client import ModelChoice
from src.session import SessionContext
from src.telemetry import telemetry_context_from_session

if TYPE_CHECKING:
    from src.events import EventEmitter


def auto_eval_enabled() -> bool:
    return os.getenv("MISSIONS_AUTO_EVAL", "false").strip().lower() == "true"


def phoenix_eval_export_enabled() -> bool:
    return os.getenv("MISSIONS_EVAL_PHOENIX_EXPORT", "true").strip().lower() == "true"


def run_session_evals(
    session: SessionContext,
    *,
    persist: bool = True,
    phoenix_export: Optional[bool] = None,
    include_llm_judges: Optional[bool] = None,
    model: ModelChoice = "auto",
    emitter: Optional["EventEmitter"] = None,
) -> SessionEvalReport:
    """
    Run all evaluators for a session and optionally persist + export.

    Args:
        session:           Session to evaluate.
        persist:           Write ``sessions/<id>/evals/report.json``.
        phoenix_export:    Export OTel eval spans (default from env).
        include_llm_judges: Run LLM judges (default from MISSIONS_EVAL_LLM_JUDGE).
        model:             Backend for LLM judges.
        emitter:           Optional emitter for ``eval.completed`` event.
    """
    ctx = load_eval_context(session)
    if not ctx.events:
        raise ValueError(
            f"Session '{session.session_id}' has no events — run a mission first."
        )

    scores = run_deterministic_evals(ctx)

    run_llm = include_llm_judges if include_llm_judges is not None else llm_judges_enabled()
    if run_llm:
        try:
            scores.extend(run_llm_judge_evals(ctx, model=model))
        except Exception as exc:
            print(f"[Evals] LLM judge evals failed: {exc}")

    report = aggregate_scores(ctx, scores, deterministic_only=not run_llm)

    if persist:
        save_report(session, report)

    do_export = phoenix_export if phoenix_export is not None else phoenix_eval_export_enabled()
    if do_export:
        tcx = telemetry_context_from_session(session)
        export_eval_report_to_phoenix(report, session=tcx)

    if emitter:
        emitter.emit(
            "eval.completed",
            overall_score=report.overall_score,
            overall_passed=report.overall_passed,
            deterministic_only=report.deterministic_only,
            evaluator_count=len(report.scores),
        )

    return report


def get_cached_report(session: SessionContext) -> Optional[SessionEvalReport]:
    return load_report(session)
