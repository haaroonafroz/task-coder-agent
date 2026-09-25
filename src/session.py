"""
Session control plane for the Missions Runtime.

Sessions own harness state and reference either a managed or external code
workspace. External code is never placed inside or deleted with session state.

Layout created for each session::

    sessions/<session_id>/
        session.json              # Session metadata (model, status, phoenix binding, ...)
        plan.json                 # Live milestone plan
        memory_store.json         # JSON fallback memory
        events.jsonl              # Append-only event stream (Phase 2)
        handoffs/                 # Per-milestone handoff telemetry
        uploads/                  # Raw user-uploaded requirements documents
        parsed_requirements/      # LlamaParse output (Phase 8)
        # code lives separately in managed-workspaces/<id>/ or an external path
        .venv/                    # Session-local Python environment (Phase 9)
        .tmp/ .home/ .cache/      # Sandbox tmp/home/pip cache (Phase 9)

The runtime is serial, so only one bound workspace is "active" at a time.
``set_workspace_root()`` (in ``src.tools.paths``) is called before each session
run to point all file/shell tools at the session's workspace.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from src.llm_client import ModelChoice
from src.settings import resolve_home
from src.workspace.models import WorkspaceBinding
from src.workspace.service import WorkspaceService, default_managed_root

_ROOT = Path(__file__).parent.parent


def get_default_sessions_root() -> Path:
    """Session state lives under ``$TASK_CODER_HOME/sessions``."""
    return resolve_home() / "sessions"


_SESSIONS_ROOT = get_default_sessions_root()


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
    state_root: Path                        # Harness-owned sessions/<id>/
    workspace: WorkspaceBinding             # Managed or attached code root
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
    project_profile: Optional[dict[str, Any]] = None
    git_preflight: Optional[dict[str, Any]] = None

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def ensure_dirs(self) -> None:
        """Create harness-owned state directories without touching external code."""
        self.state_root.mkdir(parents=True, exist_ok=True)
        self.handoffs_dir.mkdir(parents=True, exist_ok=True)
        self.uploads_dir.mkdir(parents=True, exist_ok=True)
        self.parsed_requirements_dir.mkdir(parents=True, exist_ok=True)

    @property
    def root(self) -> Path:
        """Compatibility alias for older call sites."""
        return self.state_root

    @property
    def workspace_root(self) -> Path:
        return self.workspace.root

    def to_meta_dict(self) -> dict[str, Any]:
        """Serialise metadata for session.json (paths as strings)."""
        return {
            "session_id": self.session_id,
            "title": self.title,
            "state_root": str(self.state_root),
            "root": str(self.state_root),  # compatibility for existing clients
            "workspace": self.workspace.to_dict(),
            "workspace_root": str(self.workspace_root),
            "plan_path": str(self.plan_path),
            "handoffs_dir": str(self.handoffs_dir),
            "memory_store_path": str(self.memory_store_path),
            "events_path": str(self.events_path),
            "uploads_dir": str(self.uploads_dir),
            "parsed_requirements_dir": str(self.parsed_requirements_dir),
            "meta_path": str(self.meta_path),
            "selected_model": self.selected_model,
            "created_at": self.created_at,
            "status": self.status,
            "phoenix_session_id": self.phoenix_session_id,
            "phoenix_project": self.phoenix_project,
            "thinking_profile": self.thinking_profile,
            "reflection_memory_ids_used": self.reflection_memory_ids_used,
            "project_profile": self.project_profile,
            "git_preflight": self.git_preflight,
        }


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

    def __init__(
        self,
        sessions_root: Path = _SESSIONS_ROOT,
        workspace_service: Optional[WorkspaceService] = None,
    ) -> None:
        self.sessions_root = sessions_root
        self.workspace_service = workspace_service or WorkspaceService(
            default_managed_root(sessions_root)
        )

    # ------------------------------------------------------------------
    # Creation / loading
    # ------------------------------------------------------------------

    def create_session(
        self,
        title: str,
        model: ModelChoice = "auto",
        thinking_profile: str = "auto",
        phoenix_project: Optional[str] = None,
        workspace_kind: str = "managed",
        workspace_path: Optional[str] = None,
        workspace_access_mode: str = "read_write",
        environment_strategy: str = "auto",
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
        if workspace_kind == "external":
            if not workspace_path:
                raise ValueError("External sessions require a workspace path")
            workspace = self.workspace_service.attach(
                workspace_path,
                access_mode=workspace_access_mode,  # type: ignore[arg-type]
                environment_strategy=environment_strategy,  # type: ignore[arg-type]
            )
        elif workspace_kind == "managed":
            workspace = self.workspace_service.create_managed(
                session_id,
                access_mode=workspace_access_mode,  # type: ignore[arg-type]
                environment_strategy=(
                    "harness" if environment_strategy == "auto" else environment_strategy
                ),  # type: ignore[arg-type]
            )
        else:
            raise ValueError(f"Unknown workspace kind: {workspace_kind}")
        profile = self.workspace_service.inspect(workspace).to_dict()
        state_root = self.sessions_root / session_id
        ctx = SessionContext(
            session_id=session_id,
            title=title,
            state_root=state_root,
            workspace=workspace,
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
            project_profile=profile,
            git_preflight=profile.get("git"),
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
        # Atomic write: write to a temp file then os.replace so concurrent
        # readers (e.g. the API server thread) never see a partial file.
        import os
        import tempfile
        data = json.dumps(ctx.to_meta_dict(), indent=2)
        fd, tmp = tempfile.mkstemp(
            dir=str(ctx.meta_path.parent), suffix=".tmp", prefix="session_"
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(data)
            os.replace(tmp, ctx.meta_path)
        except Exception:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _ctx_from_meta(meta: dict[str, Any]) -> SessionContext:
        """Rebuild a SessionContext from a persisted meta dict."""
        state_root = Path(meta.get("state_root") or meta["root"])
        raw_workspace = meta.get("workspace")
        if isinstance(raw_workspace, dict):
            workspace = WorkspaceBinding.from_dict(raw_workspace)
        else:
            # Transparent migration for sessions written before workspace
            # bindings existed.
            workspace_root = Path(meta["workspace_root"])
            workspace = WorkspaceBinding(
                kind="managed",
                root=workspace_root,
                access_mode="read_write",
                environment_strategy="harness",
                sandbox_required=False,
            )
        return SessionContext(
            session_id=meta["session_id"],
            title=meta.get("title", "Untitled"),
            state_root=state_root,
            workspace=workspace,
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
            project_profile=meta.get("project_profile"),
            git_preflight=meta.get("git_preflight"),
        )
