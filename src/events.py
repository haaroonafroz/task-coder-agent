"""
Event stream for the Missions Runtime.

Every phase transition, tool call, and validation outcome is recorded as a
single JSON object appended to ``sessions/<id>/events.jsonl``. This file is
the substrate for:

  - Phase 3 (FastAPI): SSE/WebSocket endpoints replay history then stream
    live events to the frontend.
  - Phase 6 (Phoenix evals): session-level evals consume the event timeline.
  - Phase 7 (Reflections): the trajectory fed to the reflection engine is
    reconstructed from this stream.

Event schema (one JSON object per line)::

    {
      "ts": "2026-07-07T15:00:00",
      "type": "milestone.started",
      "session_id": "1eb6c8ba75f8",
      "data": { "milestone_id": "M2", "title": "..." }
    }

The emitter is thread-safe. A module-level registry maps session_id to the
active emitter so the API layer (Phase 3) can look up the live emitter for a
session and subscribe to events without holding a reference to the runtime.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any, Callable, Optional

# ---------------------------------------------------------------------------
# Active emitter registry — session_id -> EventEmitter
# ---------------------------------------------------------------------------

_EMITTERS: dict[str, "EventEmitter"] = {}
_REGISTRY_LOCK = threading.Lock()


def register_emitter(emitter: "EventEmitter") -> None:
    """Track an emitter as the active one for its session."""
    with _REGISTRY_LOCK:
        _EMITTERS[emitter.session_id] = emitter


def unregister_emitter(session_id: str) -> None:
    """Remove an emitter from the active registry (e.g. after run completes)."""
    with _REGISTRY_LOCK:
        _EMITTERS.pop(session_id, None)


def get_emitter(session_id: str) -> Optional["EventEmitter"]:
    """Return the active emitter for a session, or None if no run is active."""
    with _REGISTRY_LOCK:
        return _EMITTERS.get(session_id)


def emitter_for_session(session_id: str, events_path: Path) -> "EventEmitter":
    """
    Get or create an emitter for a session.

    If a run is active, returns the live emitter (so subscribers see real-time
    events). Otherwise creates a read-only emitter bound to the existing
    events file (so the API can replay history for a completed session).
    """
    live = get_emitter(session_id)
    if live is not None:
        return live
    return EventEmitter(events_path=events_path, session_id=session_id)


# ---------------------------------------------------------------------------
# EventEmitter
# ---------------------------------------------------------------------------

EventCallback = Callable[[dict[str, Any]], None]


class EventEmitter:
    """
    Append-only JSONL event log with in-process pub/sub.

    Each :meth:`emit` call:
      1. Builds a structured event dict with timestamp, type, session_id, data.
      2. Appends it as one JSON line to ``events_path``.
      3. Notifies all subscribers synchronously.

    The file append and subscriber notification are guarded by a lock so the
    emitter is safe to use from the runtime thread and the API server thread
    concurrently.
    """

    def __init__(self, events_path: Path, session_id: str) -> None:
        self._path = events_path
        self.session_id = session_id
        self._subscribers: list[EventCallback] = []
        self._lock = threading.Lock()
        self._path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Emission
    # ------------------------------------------------------------------

    def emit(self, event_type: str, **data: Any) -> dict[str, Any]:
        """
        Record an event and notify subscribers.

        Args:
            event_type: Dotted event name (e.g. ``"milestone.started"``).
            **data:     Event payload fields.

        Returns:
            The full event dict that was written.
        """
        event: dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "type": event_type,
            "session_id": self.session_id,
            "data": data,
        }
        line = json.dumps(event, default=str)
        with self._lock:
            with self._path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
            subs = list(self._subscribers)
        for callback in subs:
            try:
                callback(event)
            except Exception:
                pass
        return event

    # ------------------------------------------------------------------
    # Subscription
    # ------------------------------------------------------------------

    def subscribe(self, callback: EventCallback) -> Callable[[], None]:
        """
        Register a callback invoked on every emitted event.

        Returns an unsubscribe function.
        """
        with self._lock:
            self._subscribers.append(callback)

        def _unsubscribe() -> None:
            with self._lock:
                try:
                    self._subscribers.remove(callback)
                except ValueError:
                    pass

        return _unsubscribe

    # ------------------------------------------------------------------
    # History
    # ------------------------------------------------------------------

    def history(self, since_index: int = 0) -> list[dict[str, Any]]:
        """
        Read all (or a tail of) past events from the JSONL file.

        Args:
            since_index: Skip the first N events (for incremental replay).

        Returns:
            List of parsed event dicts.
        """
        events: list[dict[str, Any]] = []
        if not self._path.exists():
            return events
        try:
            lines = self._path.read_text(encoding="utf-8").splitlines()
        except OSError:
            return events
        for line in lines:
            if not line.strip():
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return events[since_index:] if since_index > 0 else events

    def event_count(self) -> int:
        """Return the number of events written so far."""
        if not self._path.exists():
            return 0
        try:
            return sum(
                1 for line in self._path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            )
        except OSError:
            return 0
