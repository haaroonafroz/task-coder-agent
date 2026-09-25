"""Human-in-the-loop pending decisions for review/hotfix flows.

A run never blocks waiting for a user. When a decision point is reached
(review has actionable findings, fix is UNVERIFIED, rework needed), the run
persists a ``pending_decision.json`` beside the session state and finishes
with an ``awaiting_decision`` status. The user resolves it via the decisions
API, which enqueues a follow-up run.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Optional

from src.session import SessionContext

PENDING_FILENAME = "pending_decision.json"
RESOLUTION_FILENAME = "decision_resolution.json"


def pending_path(session: SessionContext) -> Path:
    return session.state_root / PENDING_FILENAME


def save_pending_decision(
    session: SessionContext,
    decision: dict[str, Any],
) -> dict[str, Any]:
    payload = {
        **decision,
        "session_id": session.session_id,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "resolved": False,
    }
    pending_path(session).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def load_pending_decision(session: SessionContext) -> Optional[dict[str, Any]]:
    path = pending_path(session)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def clear_pending_decision(session: SessionContext) -> None:
    try:
        pending_path(session).unlink()
    except OSError:
        pass


def save_resolution(
    session: SessionContext, run_id: Optional[str], action: str
) -> dict[str, Any]:
    payload = {
        "session_id": session.session_id,
        "run_id": run_id,
        "action": action,
        "resolved_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    path = session.state_root / RESOLUTION_FILENAME
    try:
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except OSError:
        pass
    return payload
