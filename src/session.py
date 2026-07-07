"""
Session control plane for the Missions Runtime.

Replaces the global ``active_mission/`` + root ``workspace/`` layout with
isolated, per-session directory trees under ``sessions/<session_id>/``.

Layout created for each session::

    sessions/<session_id>/
        session.json              # Session metadata (model, status, phoenix binding, ...)
        plan.json                 # Live milestone plan
        memory_store.json         # JSON fallback memory
        events.jsonl              # Append-only event stream (Phase 2)
        handoffs/                 # Per-milestone handoff telemetry
        uploads/                  # Raw user-uploaded requirements documents
        parsed_requirements/      # LlamaParse output (Phase 8)
        workspace/                # Sandboxed code generation area

The runtime is serial, so only one session's workspace is "active" at a time.
``set_workspace_root()`` (in ``src.tools.paths``) is called before each session
run to point all file/shell tools at the session's workspace.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional

from src.llm_client import ModelChoice

_ROOT = Path(__file__).parent.parent
_SESSIONS_ROOT = _ROOT / "sessions"


@dataclass
class SessionContext:
    """
    All paths and metadata for one isolated Missions session.

    Created by :meth:`SessionManager.create_session` and loaded by
    :meth:`SessionManager.load_session`. Passed through the runtime so every
    phase writes into the session's own tree instead of a global directory.
    """

    session_id: str
    title: str
    root: Path                              # sessions/<id>/
    workspace_root: Path                    # sessions/<id>/workspace/
    plan_path: Path                         # sessions/<id>/plan.json
    handoffs_dir: Path                      # sessions/<id>/handoffs/
    memory_store_path: Path                 # sessions/<id>/memory_store.json
    events_path: Path                       # sessions/<id>/events.jsonl
    uploads_dir: Path                       # sessions/<id>/uploads/
    parsed_requirements_dir: Path           # sessions/<id>/parsed_requirements/
    meta_path: Path                         # sessions/<id>/session.json
    selected_model: ModelChoice
    created_at: str
    status: str                             # "created" | "planning" | "running" | "completed" | "failed" | "paused"
    phoenix_session_id: Optional[str] = None
    phoenix_project: Optional[str] = None
    thinking_profile: str = "auto"
    reflection_memory_ids_used: list[str] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def ensure_dirs(self) -> None:
        """Create every directory in the session tree (idempotent)."""
        self.root.mkdir(parents=True, exist_ok=True)
        self.workspace_root.mkdir(parents=True, exist_ok=True)
        (self.workspace_root / "tests").mkdir(parents=True, exist_ok=True)
        self.handoffs_dir.mkdir(parents=True, exist_ok=True)
        self.uploads_dir.mkdir(parents=True, exist_ok=True)
        self.parsed_requirements_dir.mkdir(parents=True, exist_ok=True)

    def to_meta_dict(self) -> dict[str, Any]:
        """Serialise metadata for session.json (paths as strings)."""
        d = asdict(self)
        for k, v in list(d.items()):
            if isinstance(v, Path):
                d[k] = str(v)
        return d


# ---------------------------------------------------------------------------
# SessionManager
# ---------------------------------------------------------------------------

class SessionManager:
    """
    Create, load, list, and persist Missions sessions.

    Each session is a self-contained directory under ``sessions/``. The
    manager only touches filesystem metadata; the runtime is responsible for
    execution.
    """

    def __init__(self, sessions_root: Path = _SESSIONS_ROOT) -> None:
        self.sessions_root = sessions_root

    # ------------------------------------------------------------------
    # Creation / loading
    # ------------------------------------------------------------------

    def create_session(
        self,
        title: str,
        model: ModelChoice = "auto",
        thinking_profile: str = "auto",
        phoenix_project: Optional[str] = None,
    ) -> SessionContext:
        """
        Create a new session directory tree and write its metadata.

        Args:
            title:           Human-readable session title.
            model:           LLM backend choice for this session.
            thinking_profile: Thinking-mode profile ("auto" | "on" | "off").
            phoenix_project: Optional Phoenix project name to bind evals to.

        Returns:
            A fully initialised SessionContext with dirs on disk.
        """
        session_id = uuid.uuid4().hex[:12]
        now = time.strftime("%Y-%m-%dT%H:%M:%S")
        ctx = SessionContext(
            session_id=session_id,
            title=title,
            root=self.sessions_root / session_id,
            workspace_root=self.sessions_root / session_id / "workspace",
            plan_path=self.sessions_root / session_id / "plan.json",
            handoffs_dir=self.sessions_root / session_id / "handoffs",
            memory_store_path=self.sessions_root / session_id / "memory_store.json",
            events_path=self.sessions_root / session_id / "events.jsonl",
            uploads_dir=self.sessions_root / session_id / "uploads",
            parsed_requirements_dir=self.sessions_root / session_id / "parsed_requirements",
            meta_path=self.sessions_root / session_id / "session.json",
            selected_model=model,
            created_at=now,
            status="created",
            thinking_profile=thinking_profile,
            phoenix_project=phoenix_project,
            phoenix_session_id=session_id,  # bind 1:1 by default
        )
        ctx.ensure_dirs()
        self._save_meta(ctx)
        return ctx

    def load_session(self, session_id: str) -> Optional[SessionContext]:
        """
        Load an existing session by id.

        Returns None if the session directory or metadata is missing.
        """
        meta_path = self.sessions_root / session_id / "session.json"
        if not meta_path.exists():
            return None
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None
        return self._ctx_from_meta(meta)

    def list_sessions(self) -> list[dict[str, Any]]:
        """
        Return metadata for all sessions, newest first.

        Each entry is the raw session.json dict plus the session_id.
        """
        sessions: list[dict[str, Any]] = []
        if not self.sessions_root.exists():
            return sessions
        for entry in self.sessions_root.iterdir():
            if not entry.is_dir():
                continue
            meta_path = entry / "session.json"
            if not meta_path.exists():
                continue
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                sessions.append(meta)
            except (json.JSONDecodeError, OSError):
                continue
        sessions.sort(key=lambda m: m.get("created_at", ""), reverse=True)
        return sessions

    # ------------------------------------------------------------------
    # Status / persistence
    # ------------------------------------------------------------------

    def update_status(self, ctx: SessionContext, status: str) -> None:
        """Persist a new session status to session.json."""
        ctx.status = status
        self._save_meta(ctx)

    def _save_meta(self, ctx: SessionContext) -> None:
        ctx.meta_path.parent.mkdir(parents=True, exist_ok=True)
        ctx.meta_path.write_text(
            json.dumps(ctx.to_meta_dict(), indent=2), encoding="utf-8"
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _ctx_from_meta(meta: dict[str, Any]) -> SessionContext:
        """Rebuild a SessionContext from a persisted meta dict."""
        return SessionContext(
            session_id=meta["session_id"],
            title=meta.get("title", "Untitled"),
            root=Path(meta["root"]),
            workspace_root=Path(meta["workspace_root"]),
            plan_path=Path(meta["plan_path"]),
            handoffs_dir=Path(meta["handoffs_dir"]),
            memory_store_path=Path(meta["memory_store_path"]),
            events_path=Path(meta["events_path"]),
            uploads_dir=Path(meta["uploads_dir"]),
            parsed_requirements_dir=Path(meta["parsed_requirements_dir"]),
            meta_path=Path(meta["meta_path"]),
            selected_model=meta.get("selected_model", "auto"),
            created_at=meta.get("created_at", ""),
            status=meta.get("status", "created"),
            phoenix_session_id=meta.get("phoenix_session_id"),
            phoenix_project=meta.get("phoenix_project"),
            thinking_profile=meta.get("thinking_profile", "auto"),
            reflection_memory_ids_used=meta.get("reflection_memory_ids_used", []),
        )
