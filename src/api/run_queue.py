"""
Serial run executor for the Control API.

The Missions runtime is serial by design: ``src.tools.paths`` holds a
module-global workspace root mutated by ``set_workspace_root`` before each
run. If the API launched two runs concurrently in threads, they would
clobber each other's workspace root and silently write into the wrong
session.

RunQueue solves this by routing all runs through a single daemon worker
thread consuming a ``queue.Queue``. ``POST /runs`` returns immediately
with a 202 + ``run_id``; the worker executes ``runtime.run()`` one at a
time. The worker is a *daemon* thread so the server (and tests) can shut
down cleanly even if a run is blocked on a slow LLM call.

RunRegistry tracks every run in-memory and persists each run record to
``sessions/<id>/runs/<run_id>.json`` so history survives restarts. The run
executor also appends an assistant summary message to the session's
``messages.jsonl`` on completion so the conversation log stays coherent.
"""

from __future__ import annotations

import json
import queue
import threading
import time
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional

from src.llm_client import ModelChoice
from src.main import MissionsRuntime
from src.run_control import RunCancelledError
from src.session import SessionContext, SessionManager

from src.api.messages import MessageStore
from src.api.run_cancellation import RunCancellation


class CancelNotAllowedError(Exception):
    """Raised when a run cannot be cancelled (already terminal)."""


# ---------------------------------------------------------------------------
# Run record
# ---------------------------------------------------------------------------

@dataclass
class RunRecord:
    run_id: str
    session_id: str
    request: str
    status: str  # queued | running | completed | partial | failed | error | cancelled
    model: str
    queued_at: str
    run_kind: str = "auto"
    plan_id: Optional[str] = None
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    result: Optional[dict[str, Any]] = None
    error: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# RunRegistry
# ---------------------------------------------------------------------------

class RunRegistry:
    """In-memory + on-disk registry of runs, keyed by run_id."""

    def __init__(self, sessions_root: Path) -> None:
        self._sessions_root = sessions_root
        self._runs: dict[str, RunRecord] = {}
        self._by_session: dict[str, list[str]] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _run_path(self, session_id: str, run_id: str) -> Path:
        return self._sessions_root / session_id / "runs" / f"{run_id}.json"

    def _persist(self, rec: RunRecord) -> None:
        path = self._run_path(rec.session_id, rec.run_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(rec.to_dict(), indent=2), encoding="utf-8")

    def _load_existing(self, session_id: str) -> None:
        """Load run records from disk for a session (idempotent)."""
        runs_dir = self._sessions_root / session_id / "runs"
        if not runs_dir.exists():
            return
        for entry in runs_dir.glob("*.json"):
            try:
                data = json.loads(entry.read_text(encoding="utf-8"))
                rec = RunRecord(**data)
                if rec.run_id not in self._runs:
                    self._runs[rec.run_id] = rec
                    self._by_session.setdefault(session_id, []).append(rec.run_id)
            except (json.JSONDecodeError, OSError, TypeError):
                continue

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def create(
        self,
        session_id: str,
        request: str,
        model: str,
        run_kind: str = "auto",
    ) -> RunRecord:
        run_id = uuid.uuid4().hex[:12]
        rec = RunRecord(
            run_id=run_id,
            session_id=session_id,
            request=request,
            status="queued",
            model=model,
            queued_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
            run_kind=run_kind,
        )
        with self._lock:
            self._runs[run_id] = rec
            self._by_session.setdefault(session_id, []).append(run_id)
            self._persist(rec)
        return rec

    def get(self, run_id: str) -> Optional[RunRecord]:
        with self._lock:
            rec = self._runs.get(run_id)
            if rec is not None:
                return rec
        if not self._sessions_root.exists():
            return None
        for session_dir in self._sessions_root.iterdir():
            if not session_dir.is_dir():
                continue
            path = session_dir / "runs" / f"{run_id}.json"
            if not path.exists():
                continue
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                rec = RunRecord(**data)
                with self._lock:
                    self._runs[run_id] = rec
                    self._by_session.setdefault(rec.session_id, []).append(run_id)
                return rec
            except (json.JSONDecodeError, OSError, TypeError):
                return None
        return None

    def list_for_session(self, session_id: str) -> list[RunRecord]:
        with self._lock:
            ids = list(self._by_session.get(session_id, []))
            # Lazy-load from disk if not in memory (e.g. after restart).
            if not ids:
                self._load_existing(session_id)
                ids = list(self._by_session.get(session_id, []))
            return [self._runs[i] for i in ids if i in self._runs]

    def update(self, rec: RunRecord) -> None:
        with self._lock:
            self._runs[rec.run_id] = rec
            self._persist(rec)

    def reconcile_stale_runs(self) -> list[RunRecord]:
        """
        Mark orphaned queued/running runs as cancelled after a server restart.

        Returns the list of runs that were reconciled.
        """
        reconciled: list[RunRecord] = []
        if not self._sessions_root.exists():
            return reconciled

        for session_dir in sorted(self._sessions_root.iterdir()):
            if not session_dir.is_dir():
                continue
            runs_dir = session_dir / "runs"
            if not runs_dir.exists():
                continue
            for entry in runs_dir.glob("*.json"):
                try:
                    data = json.loads(entry.read_text(encoding="utf-8"))
                    if data.get("status") not in ("queued", "running"):
                        continue
                    rec = RunRecord(**data)
                    rec.status = "cancelled"
                    rec.finished_at = time.strftime("%Y-%m-%dT%H:%M:%S")
                    rec.error = "Run interrupted by server restart"
                    with self._lock:
                        self._runs[rec.run_id] = rec
                        self._by_session.setdefault(rec.session_id, []).append(rec.run_id)
                        self._persist(rec)
                    reconciled.append(rec)
                except (json.JSONDecodeError, OSError, TypeError):
                    continue

            session_meta = session_dir / "session.json"
            if session_meta.exists():
                try:
                    meta = json.loads(session_meta.read_text(encoding="utf-8"))
                    if meta.get("status") == "running":
                        meta["status"] = "paused"
                        session_meta.write_text(
                            json.dumps(meta, indent=2), encoding="utf-8"
                        )
                except (json.JSONDecodeError, OSError):
                    pass

        return reconciled


# ---------------------------------------------------------------------------
# RunQueue — serial executor (single daemon worker thread)
# ---------------------------------------------------------------------------

_SENTINEL = object()


class RunQueue:
    """
    Serial executor wrapping a single shared MissionsRuntime.

    All runs are funneled through one daemon worker thread so the
    module-global workspace root is never raced on. Construction is cheap;
    the heavy MissionsRuntime (Qdrant, sentence-transformers) is built once
    at app startup and shared.
    """

    def __init__(
        self,
        runtime: MissionsRuntime,
        registry: RunRegistry,
        session_manager: SessionManager,
        message_store: MessageStore,
        cancellation: Optional[RunCancellation] = None,
    ) -> None:
        self._runtime = runtime
        self._registry = registry
        self._session_manager = session_manager
        self._message_store = message_store
        self._cancellation = cancellation or RunCancellation()
        self._queue: "queue.Queue[Any]" = queue.Queue()
        self._shutdown = False
        self._active_run_id: Optional[str] = None

        stale = self._registry.reconcile_stale_runs()
        if stale:
            print(
                f"[RunQueue] Reconciled {len(stale)} stale run(s) "
                "after server restart."
            )

        self._worker = threading.Thread(
            target=self._run_loop,
            name="missions-run-worker",
            daemon=True,
        )
        self._worker.start()

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def enqueue(
        self,
        ctx: SessionContext,
        request: str,
        model: Optional[ModelChoice] = None,
        run_kind: str = "auto",
    ) -> RunRecord:
        """Create a run record and submit it to the serial worker."""
        chosen_model = model or ctx.selected_model or "auto"
        rec = self._registry.create(
            ctx.session_id,
            request,
            str(chosen_model),
            run_kind=run_kind,
        )
        self._queue.put((rec, ctx, chosen_model))
        return rec

    def cancel(self, run_id: str) -> RunRecord:
        """
        Request cancellation for a queued or running run.

        Queued runs are marked cancelled immediately. Running runs are
        cooperatively stopped at the next runtime checkpoint.
        """
        rec = self._registry.get(run_id)
        if rec is None:
            raise KeyError(run_id)
        if rec.status not in ("queued", "running"):
            raise CancelNotAllowedError(
                f"Run '{run_id}' has status '{rec.status}' and cannot be cancelled"
            )

        self._cancellation.request(run_id)

        if rec.status == "queued":
            rec.status = "cancelled"
            rec.finished_at = time.strftime("%Y-%m-%dT%H:%M:%S")
            rec.error = "Cancelled before execution started"
            self._registry.update(rec)
            try:
                ctx = self._session_manager.load_session(rec.session_id)
                if ctx:
                    self._message_store.append(
                        ctx,
                        "assistant",
                        "Run cancelled before it started.",
                        run_id=rec.run_id,
                    )
            except Exception:
                pass
        elif rec.status == "running" and self._active_run_id != run_id:
            # Zombie run — worker died (e.g. API reload) but disk still says running.
            rec.status = "cancelled"
            rec.finished_at = time.strftime("%Y-%m-%dT%H:%M:%S")
            rec.error = "Run interrupted (no active worker)"
            self._registry.update(rec)
            try:
                ctx = self._session_manager.load_session(rec.session_id)
                if ctx:
                    self._session_manager.update_status(ctx, "paused")
                    self._message_store.append(
                        ctx,
                        "assistant",
                        "Run cancelled — worker was no longer active.",
                        run_id=rec.run_id,
                    )
            except Exception:
                pass

        return rec

    def shutdown(self, wait: bool = False) -> None:
        """Signal the worker to stop. Does not interrupt a running run."""
        self._shutdown = True
        self._queue.put(_SENTINEL)
        if wait:
            self._worker.join(timeout=5.0)

    # ------------------------------------------------------------------
    # Worker loop (runs on the single daemon thread)
    # ------------------------------------------------------------------

    def _run_loop(self) -> None:
        while not self._shutdown:
            item = self._queue.get()
            if item is _SENTINEL:
                break
            rec, ctx, model = item
            self._execute(rec, ctx, model)

    def _execute(
        self,
        rec: RunRecord,
        ctx: SessionContext,
        model: ModelChoice,
    ) -> None:
        """Runs on the single worker thread — no concurrent runs possible."""
        if self._cancellation.is_cancelled(rec.run_id):
            rec.status = "cancelled"
            rec.finished_at = time.strftime("%Y-%m-%dT%H:%M:%S")
            rec.error = "Cancelled while queued"
            self._registry.update(rec)
            self._cancellation.clear(rec.run_id)
            return

        rec.status = "running"
        rec.started_at = time.strftime("%Y-%m-%dT%H:%M:%S")
        self._registry.update(rec)
        self._active_run_id = rec.run_id
        cancel_check = lambda: self._cancellation.is_cancelled(rec.run_id)

        try:
            fresh = self._session_manager.load_session(ctx.session_id) or ctx
            self._runtime.model = model
            result = self._runtime.run(
                rec.request,
                session=fresh,
                cancel_check=cancel_check,
                run_kind=rec.run_kind,
            )
            rec.plan_id = result.plan_id or rec.plan_id or result.mission_id
            rec.run_kind = result.run_kind
            if result.status == "cancelled" or self._cancellation.is_cancelled(rec.run_id):
                rec.status = "cancelled"
                rec.error = "Run cancelled by user"
                rec.result = {
                    "mission_id": result.mission_id,
                    "title": result.title,
                    "status": result.status,
                    "milestones_passed": result.milestones_passed,
                    "milestones_total": result.milestones_total,
                    "total_elapsed_ms": result.total_elapsed_ms,
                    "model_used": result.model_used,
                    "session_id": result.session_id,
                    "run_kind": result.run_kind,
                    "plan_id": result.plan_id,
                }
                try:
                    self._message_store.append(
                        fresh,
                        "assistant",
                        "Run cancelled by user.",
                        run_id=rec.run_id,
                    )
                except Exception:
                    pass
            else:
                rec.result = {
                    "mission_id": result.mission_id,
                    "title": result.title,
                    "status": result.status,
                    "milestones_passed": result.milestones_passed,
                    "milestones_total": result.milestones_total,
                    "total_elapsed_ms": result.total_elapsed_ms,
                    "model_used": result.model_used,
                    "session_id": result.session_id,
                    "run_kind": result.run_kind,
                    "plan_id": result.plan_id,
                    "summary_text": result.summary_text,
                    "failure_reason": result.failure_reason,
                }
                rec.status = result.status
        except RunCancelledError as exc:
            rec.status = "cancelled"
            rec.error = str(exc)
            try:
                fresh = self._session_manager.load_session(ctx.session_id) or ctx
                self._message_store.append(
                    fresh,
                    "assistant",
                    "Run cancelled by user.",
                    run_id=rec.run_id,
                )
            except Exception:
                pass
        except Exception as exc:  # noqa: BLE001 - surface any failure to the client
            rec.status = "error"
            rec.error = f"{type(exc).__name__}: {exc}"
            try:
                self._message_store.append(
                    ctx, "assistant",
                    f"Run failed with error: {rec.error}",
                    run_id=rec.run_id,
                )
            except Exception:
                pass
        finally:
            if rec.status == "cancelled":
                pass
            elif rec.status not in ("error",) and rec.result:
                summary = rec.result.get("summary_text") or (
                    f"Run finished — status: {rec.status}, "
                    f"milestones {rec.result.get('milestones_passed', 0)}/"
                    f"{rec.result.get('milestones_total', 0)} passed, "
                    f"elapsed {rec.result.get('total_elapsed_ms', 0) / 1000:.1f}s."
                )
                try:
                    fresh = self._session_manager.load_session(ctx.session_id) or ctx
                    self._message_store.append(
                        fresh, "assistant", summary, run_id=rec.run_id
                    )
                except Exception:
                    pass
            rec.finished_at = time.strftime("%Y-%m-%dT%H:%M:%S")
            self._registry.update(rec)
            self._active_run_id = None
            self._cancellation.clear(rec.run_id)
