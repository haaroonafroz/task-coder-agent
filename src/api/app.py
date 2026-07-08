"""
Missions Control API — FastAPI application factory.

Exposes ``create_app()`` which builds the FastAPI instance, wires up
shared singletons in ``app.state``, and mounts all routers under
``/api/v1``.

Run with::

    python -m src.api            # uvicorn on :8088
    python -m src.main --serve   # convenience alias
"""

from __future__ import annotations

import pathlib
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.run_queue import RunQueue, RunRegistry
from src.api.messages import MessageStore
from src.main import MissionsRuntime
from src.session import SessionManager
from src.tool_registry import DynamicToolRouter

_ROOT = pathlib.Path(__file__).parent.parent.parent
_SESSIONS_ROOT = _ROOT / "sessions"
_SKILLS_PATH = _ROOT / "config" / "skills.md"

# Frontend dev server (Phase 4) — allowed CORS origins.
_CORS_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:8088",
    "http://127.0.0.1:8088",
]


@asynccontextmanager
async def _lifespan(app: FastAPI):
    """Construct heavy singletons once at startup; tear down on shutdown."""
    # Single shared runtime — Qdrant + sentence-transformers load happens once.
    runtime = MissionsRuntime(model="auto", telemetry=False, memory=True)
    router = DynamicToolRouter(_SKILLS_PATH)
    session_manager = SessionManager(sessions_root=_SESSIONS_ROOT)
    message_store = MessageStore(sessions_root=_SESSIONS_ROOT)
    run_registry = RunRegistry(sessions_root=_SESSIONS_ROOT)
    run_queue = RunQueue(runtime, run_registry, session_manager, message_store)

    app.state.runtime = runtime
    app.state.router = router
    app.state.session_manager = session_manager
    app.state.message_store = message_store
    app.state.run_registry = run_registry
    app.state.run_queue = run_queue

    try:
        yield
    finally:
        run_queue.shutdown(wait=False)


def create_app() -> FastAPI:
    """Build and return the configured FastAPI application."""
    app = FastAPI(
        title="Missions Control API",
        version="1.0.0",
        description="HTTP control plane for the Missions multi-agent runtime.",
        lifespan=_lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=_CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Routers are mounted under /api/v1.
    _register_routers(app)

    return app


def _register_routers(app: FastAPI) -> None:
    """Import and include all API routers."""
    from src.api.routers import (
        sessions,
        messages,
        runs,
        events,
        workspace,
        models,
        tools,
        skills,
        health,
        uploads,
    )

    prefix = "/api/v1"
    app.include_router(health.router, prefix=prefix)
    app.include_router(sessions.router, prefix=prefix)
    app.include_router(messages.router, prefix=prefix)
    app.include_router(runs.router, prefix=prefix)
    app.include_router(events.router, prefix=prefix)
    app.include_router(workspace.router, prefix=prefix)
    app.include_router(models.router, prefix=prefix)
    app.include_router(tools.router, prefix=prefix)
    app.include_router(skills.router, prefix=prefix)
    app.include_router(uploads.router, prefix=prefix)
