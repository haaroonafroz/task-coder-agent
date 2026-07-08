"""Export eval results to Phoenix as OTel spans."""

from __future__ import annotations

from __future__ import annotations

from src.evals.types import SessionEvalReport
from src.telemetry import get_tracer, TelemetryContext


def export_eval_report_to_phoenix(
    report: SessionEvalReport,
    *,
    session: TelemetryContext | None = None,
) -> None:
    """
    Record eval results as a dedicated span tree in Phoenix.

    Harness-native export (separate from Phoenix's ``arize-phoenix-evals``
    trace-annotation workflow, which can be added later to annotate existing
    LLM spans in the Phoenix UI).
    """
    tracer = get_tracer()
    attrs: dict[str, str | float | bool] = {
        "session.id": report.session_id,
        "eval.overall_score": report.overall_score,
        "eval.overall_passed": report.overall_passed,
        "eval.event_count": report.event_count,
        "eval.deterministic_only": report.deterministic_only,
    }
    if session:
        attrs.update({k: v for k, v in session.span_attributes().items() if k != "session.id"})
    if report.mission_status:
        attrs["eval.mission_status"] = report.mission_status

    with tracer.start_as_current_span("eval.session", attributes=attrs):
        for score in report.scores:
            with tracer.start_as_current_span(
                f"eval.{score.name}",
                attributes={
                    "eval.name": score.name,
                    "eval.score": score.score,
                    "eval.passed": score.passed,
                    "eval.kind": score.kind,
                    "eval.summary": score.summary[:200],
                },
            ):
                pass
