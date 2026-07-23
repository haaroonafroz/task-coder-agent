"""Unit tests for run cancellation."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.api.run_cancellation import RunCancellation
from src.api.run_queue import RunQueue, RunRegistry
from src.api.messages import MessageStore
from src.run_control import RunCancelledError, ensure_not_cancelled
from src.session import SessionContext


def test_run_cancellation_request_and_clear() -> None:
    reg = RunCancellation()
    assert not reg.is_cancelled("abc")
    reg.request("abc")
    assert reg.is_cancelled("abc")
    reg.clear("abc")
    assert not reg.is_cancelled("abc")


def test_ensure_not_cancelled_raises() -> None:
    with pytest.raises(RunCancelledError):
        ensure_not_cancelled(lambda: True)
    ensure_not_cancelled(lambda: False)
    ensure_not_cancelled(None)


def _session(tmp_path: Path) -> SessionContext:
    sid = "cancel01"
    root = tmp_path / "sessions" / sid
    ws = root / "workspace"
    for d in (root, ws, root / "handoffs", root / "uploads", root / "parsed_requirements"):
        d.mkdir(parents=True)
    return SessionContext(
        session_id=sid,
        title="cancel test",
        root=root,
        workspace_root=ws,
        plan_path=root / "plan.json",
        handoffs_dir=root / "handoffs",
        memory_store_path=root / "memory_store.json",
        events_path=root / "events.jsonl",
        uploads_dir=root / "uploads",
        parsed_requirements_dir=root / "parsed_requirements",
        meta_path=root / "session.json",
        selected_model="auto",
        created_at="now",
        status="created",
    )


def test_cancel_queued_run(tmp_path: Path) -> None:
    sessions_root = tmp_path / "sessions"
    registry = RunRegistry(sessions_root)
    message_store = MessageStore(sessions_root)
    runtime = MagicMock()
    session_manager = MagicMock()

    queue = RunQueue(runtime, registry, session_manager, message_store)
    ctx = _session(tmp_path)

    rec = queue.enqueue(ctx, "build something")
    cancelled = queue.cancel(rec.run_id)

    assert cancelled.status == "cancelled"
    assert cancelled.finished_at is not None

    queue.shutdown(wait=False)
