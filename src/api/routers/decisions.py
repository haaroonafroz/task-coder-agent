"""HITL decision endpoints for paused runs (review/hotfix/verifier gates)."""

from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException

from src.api.deps import (
    get_message_store,
    get_run_queue,
    get_session_manager,
    require_session,
)
from src.api.messages import MessageStore
from src.api.run_queue import RunQueue
from src.api.schemas import (
    DecisionCreate,
    DecisionResolveResponse,
    PendingDecisionResponse,
)
from src.decisions import (
    clear_pending_decision,
    load_pending_decision,
    save_resolution,
)
from src.session import SessionContext, SessionManager

router = APIRouter(prefix="/sessions/{sid}/decisions", tags=["decisions"])


@router.get("", response_model=PendingDecisionResponse)
async def get_decision(
    sid: str,
    manager: SessionManager = Depends(get_session_manager),
) -> PendingDecisionResponse:
    ctx = require_session(sid, manager)
    pending = load_pending_decision(ctx)
    if pending is None:
        raise HTTPException(status_code=404, detail="No pending decision")
    return PendingDecisionResponse(**pending)


@router.post("", response_model=DecisionResolveResponse, status_code=202)
async def resolve_decision(
    sid: str,
    body: DecisionCreate,
    manager: SessionManager = Depends(get_session_manager),
    message_store: MessageStore = Depends(get_message_store),
    run_queue: RunQueue = Depends(get_run_queue),
) -> DecisionResolveResponse:
    ctx = require_session(sid, manager)
    pending = load_pending_decision(ctx)
    if pending is None:
        raise HTTPException(status_code=404, detail="No pending decision")
    if body.action not in pending.get("options", []):
        raise HTTPException(
            status_code=400,
            detail=(
                f"Action '{body.action}' is not offered "
                f"(options: {pending.get('options', [])})"
            ),
        )

    payload: dict[str, Any] = pending.get("payload", {})
    # Clear first: decisions resolve exactly once (double-submit safe).
    clear_pending_decision(ctx)

    run_id: Optional[str] = None
    if body.action == "dismiss":
        detail = "Decision dismissed — no further work started."
    elif body.action == "escalate_mission":
        request = _escalation_request(pending, payload)
        rec = run_queue.enqueue(
            ctx, request, run_kind="new", execution_route="mission"
        )
        run_id = rec.run_id
        detail = f"Escalated to a planned mission (run {run_id})."
    else:  # apply_fix | run_smoke | setup_env → packet follow-up runs
        packet = _packet_for_action(body.action, pending, payload)
        request = _packet_request(body.action, pending, payload)
        rec = run_queue.enqueue_packet(ctx, packet, request)
        run_id = rec.run_id
        detail = f"Follow-up '{body.action}' started (run {run_id})."

    save_resolution(ctx, run_id, body.action)
    try:
        message_store.append(
            ctx, "assistant",
            f"Decision resolved ({body.action}): {detail}",
            run_id=run_id,
        )
    except Exception:
        pass
    return DecisionResolveResponse(
        action=body.action, session_id=sid, run_id=run_id, detail=detail
    )


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------

def _packet_for_action(
    action: str,
    pending: dict[str, Any],
    payload: dict[str, Any],
) -> dict[str, Any]:
    report = payload.get("review_report") or {}
    touched = (
        payload.get("touched_files")
        or payload.get("target_files")
        or []
    )
    if action == "apply_fix":
        milestone = payload.get("milestone")
        if not isinstance(milestone, dict) or not milestone:
            raise HTTPException(
                status_code=422,
                detail="This decision carries no stored fix packet to apply.",
            )
        return {
            "kind": "review_fix",
            "milestone": milestone,
            "target_files": touched,
            "report": report,
        }
    if action == "run_smoke":
        return {
            "kind": "smoke",
            "touched_files": touched,
            "target_files": touched,
            "report": report,
        }
    if action == "setup_env":
        packages = payload.get("packages") or payload.get("missing_checks") or []
        return {
            "kind": "setup_env",
            "packages": packages,
            "touched_files": touched,
            "target_files": touched,
            "report": report,
        }
    raise HTTPException(status_code=400, detail=f"Unknown action '{action}'")


def _packet_request(
    action: str,
    pending: dict[str, Any],
    payload: dict[str, Any],
) -> str:
    title = str(pending.get("title", "follow-up"))
    touched = payload.get("touched_files") or payload.get("target_files") or []
    if action == "setup_env":
        missing = payload.get("missing_checks") or payload.get("packages") or []
        return (
            f"Approved follow-up (set up test env) for '{title}'. "
            f"Missing checks: {missing}. Touched files: {touched}."
        )
    if action == "run_smoke":
        return (
            f"Approved follow-up (smoke checks) for '{title}'. "
            f"Touched files: {touched}."
        )
    return (
        f"Approved follow-up fix for '{title}'. Targets: {touched}."
    )


def _escalation_request(
    pending: dict[str, Any],
    payload: dict[str, Any],
) -> str:
    """Compose a mission request carrying prior focused-run evidence."""
    lines = [
        "Escalate to a planned mission.",
        "",
        f"Prior context: {pending.get('title', '')}",
        str(pending.get("summary", "")).strip(),
    ]
    verdict = payload.get("verdict")
    if verdict:
        lines.append(f"Prior focused attempt verdict: {verdict}.")
    errors = payload.get("errors") or []
    if errors:
        lines.append("Prior errors:")
        lines.extend(f"- {err}" for err in errors[-4:])
    verification = payload.get("verification") or {}
    if isinstance(verification, dict) and verification.get("rationale"):
        lines.append(f"Fix verification: {verification.get('rationale')}")
    report = payload.get("review_report") or {}
    if isinstance(report, dict) and report.get("summary"):
        lines.append(f"Review summary: {report.get('summary')}")
    targets = payload.get("target_files") or payload.get("touched_files") or []
    if targets:
        lines.append(f"Known-affected files: {list(targets)}")
    lines.append(
        "Start from this evidence; do not re-derive what is already established."
    )
    return "\n".join(line for line in lines if line.strip())
