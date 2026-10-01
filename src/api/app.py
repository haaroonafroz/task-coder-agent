"""
Missions Control API — FastAPI application factory.

Exposes ``create_app()`` which builds the FastAPI instance, wires up
shared singletons in ``app.state``, and mounts all routers under
``/api/v1``. The built React UI is served from the same origin when
``frontend/dist`` exists.

Run with::

    missions serve
    python -m src.api
    python -m src.main --serve
"""

from __future__ import annotations

import pathlib
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from src.api.run_queue import RunQueue, RunRegistry
from src.api.messages import MessageStore
from src.main import MissionsRuntime
from src.session import SessionManager, get_default_sessions_root
from src.settings import get_settings
from src.tool_registry import DynamicToolRouter

_ROOT = pathlib.Path(__file__).parent.parent.parent
_SKILLS_PATH = _ROOT / "config" / "skills.md"
_FRONTEND_DIST = _ROOT / "frontend" / "dist"

_CORS_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:8088",
    "http://127.0.0.1:8088",
]


def _api_telemetry_enabled() -> bool:
    return bool(get_settings().observability.api_telemetry)


@asynccontextmanager
async def _lifespan(app: FastAPI):
    """Construct heavy singletons once at startup; tear down on shutdown."""
    get_settings()
    sessions_root = get_default_sessions_root()
    runtime = MissionsRuntime(
        model="auto", telemetry=_api_telemetry_enabled(), memory=True
    )
    router = DynamicToolRouter(_SKILLS_PATH)
    session_manager = SessionManager(sessions_root=sessions_root)
    message_store = MessageStore(sessions_root=sessions_root)
    run_registry = RunRegistry(sessions_root=sessions_root)
    run_queue = RunQueue(runtime, run_registry, session_manager, message_store)

    app.state.runtime = runtime
    app.state.router = router
    app.state.session_manager = session_manager
    app.state.message_store = message_store
    app.state.run_registry = run_registry
    app.state.run_queue = run_queue
    app.state.skills_path = _SKILLS_PATH

    try:
        yield
    finally:
        run_queue.shutdown(wait=False)
        router = getattr(app.state, "router", None)
        if router is not None and hasattr(router, "close"):
            try:
                router.close()
            except Exception:
                pass


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

    _register_routers(app)
    _mount_frontend(app)
    return app


def _register_routers(app: FastAPI) -> None:
    """Import and include all API routers."""
    from src.api.routers import (
        sessions,
        messages,
        runs,
        decisions,
        events,
        workspace,
        models,
        tools,
        skills,
        health,
        uploads,
        evals,
        settings,
    )

    prefix = "/api/v1"
    app.include_router(health.router, prefix=prefix)
    app.include_router(sessions.router, prefix=prefix)
    app.include_router(messages.router, prefix=prefix)
    app.include_router(runs.router, prefix=prefix)
    app.include_router(decisions.router, prefix=prefix)
    app.include_router(events.router, prefix=prefix)
    app.include_router(workspace.router, prefix=prefix)
    app.include_router(models.router, prefix=prefix)
    app.include_router(tools.router, prefix=prefix)
    app.include_router(skills.router, prefix=prefix)
    app.include_router(uploads.router, prefix=prefix)
    app.include_router(evals.router, prefix=prefix)
    app.include_router(settings.router, prefix=prefix)


def _mount_frontend(app: FastAPI) -> None:
    """Serve the built SPA from FastAPI when frontend/dist exists."""
    dist = _FRONTEND_DIST
    if not dist.is_dir() or not (dist / "index.html").is_file():
        return

    assets = dist / "assets"
    if assets.is_dir():
        app.mount("/assets", StaticFiles(directory=assets), name="frontend-assets")

    @app.get("/")
    async def spa_index() -> FileResponse:
        return FileResponse(dist / "index.html")

    @app.get("/{full_path:path}")
    async def spa_fallback(full_path: str) -> FileResponse:
        if full_path.startswith("api"):
            raise HTTPException(status_code=404, detail="Not found")
        candidate = dist / full_path
        if candidate.is_file():
            return FileResponse(candidate)
        return FileResponse(dist / "index.html")
