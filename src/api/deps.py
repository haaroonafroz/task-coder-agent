"""
Shared singletons and FastAPI dependencies for the Control API.

All heavy singletons (MissionsRuntime, DynamicToolRouter, SessionManager)
are constructed once at app startup and stored in app.state. Routers
retrieve them via the Depends() functions defined here.
"""

from __future__ import annotations

from fastapi import Depends, FastAPI, Request

from src.events import EventEmitter, emitter_for_session, get_emitter
from src.main import MissionsRuntime
from src.session import SessionContext, SessionManager
from src.tool_registry import DynamicToolRouter

# Config path for the skills index (same constant used by the runtime).
import pathlib
_ROOT = pathlib.Path(__file__).parent.parent.parent
_SKILLS_PATH = _ROOT / "config" / "skills.md"


# ---------------------------------------------------------------------------
# Singleton accessors
# ---------------------------------------------------------------------------

def get_session_manager(request: Request) -> SessionManager:
    return request.app.state.session_manager


def get_runtime(request: Request) -> MissionsRuntime:
    return request.app.state.runtime


def get_router(request: Request) -> DynamicToolRouter:
    return request.app.state.router


def get_run_queue(request: Request):
    from src.api.run_queue import RunQueue
    return request.app.state.run_queue


def get_message_store(request: Request):
    from src.api.messages import MessageStore
    return request.app.state.message_store


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def require_session(sid: str, manager: SessionManager) -> SessionContext:
    ctx = manager.load_session(sid)
    if ctx is None:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail=f"Session '{sid}' not found")
    return ctx


def resolve_emitter(ctx: SessionContext) -> EventEmitter:
    """Return the live emitter if a run is active, else a read-only one."""
    live = get_emitter(ctx.session_id)
    if live is not None:
        return live
    return emitter_for_session(ctx.session_id, ctx.events_path)
