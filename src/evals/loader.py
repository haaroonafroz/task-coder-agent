"""Load session artifacts for evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.session import SessionContext
from src.evals.types import EvalContext


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _load_events(path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    if not path.exists():
        return events
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    except OSError:
        pass
    return events


def _load_handoffs(handoffs_dir: Path) -> list[dict[str, Any]]:
    handoffs: list[dict[str, Any]] = []
    if not handoffs_dir.exists():
        return handoffs
    for path in sorted(handoffs_dir.glob("*.json")):
        try:
            handoffs.append(json.loads(path.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, OSError):
            continue
    return handoffs


def load_eval_context(session: SessionContext) -> EvalContext:
    """Load events, plan, handoffs, and metadata for evaluators."""
    events = _load_events(session.events_path)
    plan = _load_json(session.plan_path)
    handoffs = _load_handoffs(session.handoffs_dir)
    meta = session.to_meta_dict()

    user_request = ""
    for ev in events:
        if ev.get("type") == "session.started":
            user_request = ev.get("data", {}).get("request", "") or ""
            break

    return EvalContext(
        session_id=session.session_id,
        events=events,
        plan=plan,
        handoffs=handoffs,
        session_meta=meta,
        user_request=user_request,
    )
