"""
Append-only message log per session.

Each session has ``sessions/<id>/messages.jsonl`` with one JSON object per
line::

    {"id", "role", "content", "ts", "run_id?"}

The store is the substrate for the N:M chat model: user messages may or may
not trigger a run, and the run executor appends an assistant summary message
on completion so the conversation log stays coherent.
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Optional

from src.session import SessionContext


class MessageStore:
    """Thread-safe append/read log for session messages."""

    def __init__(self, sessions_root: Path) -> None:
        self._root = sessions_root
        self._locks: dict[str, threading.Lock] = {}
        self._registry_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _path(self, session_id: str) -> Path:
        return self._root / session_id / "messages.jsonl"

    def _lock(self, session_id: str) -> threading.Lock:
        with self._registry_lock:
            lock = self._locks.get(session_id)
            if lock is None:
                lock = threading.Lock()
                self._locks[session_id] = lock
            return lock

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def append(
        self,
        ctx: SessionContext,
        role: str,
        content: str,
        run_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """Append a message and return the stored dict."""
        msg = {
            "id": uuid.uuid4().hex[:12],
            "role": role,
            "content": content,
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "run_id": run_id,
        }
        path = self._path(ctx.session_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(msg, default=str)
        with self._lock(ctx.session_id):
            with path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
        return msg

    def list(self, ctx: SessionContext, limit: Optional[int] = None) -> list[dict[str, Any]]:
        """Return all (or the most recent N) messages, oldest first."""
        path = self._path(ctx.session_id)
        if not path.exists():
            return []
        msgs: list[dict[str, Any]] = []
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            return []
        for line in text.splitlines():
            if not line.strip():
                continue
            try:
                msgs.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        if limit is not None and limit > 0:
            return msgs[-limit:]
        return msgs
