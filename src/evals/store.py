"""Persist and load session eval reports."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

from src.session import SessionContext
from src.evals.types import EvalScore, SessionEvalReport


def evals_dir(session: SessionContext) -> Path:
    d = session.root / "evals"
    d.mkdir(parents=True, exist_ok=True)
    return d


def report_path(session: SessionContext) -> Path:
    return evals_dir(session) / "report.json"


def save_report(session: SessionContext, report: SessionEvalReport) -> Path:
    """Write report.json and a timestamped snapshot."""
    path = report_path(session)
    data = report.to_dict()
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    ts = time.strftime("%Y%m%dT%H%M%S")
    snapshot = evals_dir(session) / f"report_{ts}.json"
    snapshot.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return path


def load_report(session: SessionContext) -> Optional[SessionEvalReport]:
    path = report_path(session)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None

    scores = [
        EvalScore(**s) if isinstance(s, dict) else s
        for s in data.get("scores", [])
    ]
    return SessionEvalReport(
        session_id=data.get("session_id", session.session_id),
        evaluated_at=data.get("evaluated_at", ""),
        mission_status=data.get("mission_status"),
        overall_score=float(data.get("overall_score", 0.0)),
        overall_passed=bool(data.get("overall_passed", False)),
        scores=scores,
        event_count=int(data.get("event_count", 0)),
        deterministic_only=bool(data.get("deterministic_only", True)),
        weights=data.get("weights", {}),
    )
