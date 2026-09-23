"""Session CRUD endpoints."""

from __future__ import annotations

import shutil
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException

from src.api.deps import get_session_manager, require_session
from src.api.schemas import SessionCreate, SessionResponse, SessionUpdate
from src.session import SessionManager

router = APIRouter(prefix="/sessions", tags=["sessions"])


def _to_response(meta: dict) -> SessionResponse:
    """Build a SessionResponse from a raw session.json meta dict."""
    workspace = meta.get("workspace") or {
        "kind": "managed",
        "path": meta.get("workspace_root", ""),
        "root": meta.get("workspace_root", ""),
        "access_mode": "read_write",
        "environment_strategy": "harness",
        "sandbox_required": False,
    }
    workspace_response = {
        "kind": workspace.get("kind", "managed"),
        "path": workspace.get("path") or workspace.get("root", ""),
        "access_mode": workspace.get("access_mode", "read_write"),
        "environment_strategy": workspace.get("environment_strategy", "auto"),
        "git_root": workspace.get("git_root"),
        "sandbox_required": workspace.get("sandbox_required", False),
    }
    return SessionResponse(
        session_id=meta.get("session_id", ""),
        title=meta.get("title", "Untitled"),
        status=meta.get("status", "created"),
        selected_model=meta.get("selected_model", "auto"),
        thinking_profile=meta.get("thinking_profile", "auto"),
        created_at=meta.get("created_at", ""),
        phoenix_session_id=meta.get("phoenix_session_id"),
        phoenix_project=meta.get("phoenix_project"),
        workspace_root=meta.get("workspace_root", ""),
        plan_path=meta.get("plan_path", ""),
        events_path=meta.get("events_path", ""),
        workspace=workspace_response,
        project_profile=meta.get("project_profile"),
    )


@router.post("", response_model=SessionResponse, status_code=201)
async def create_session(
    body: SessionCreate,
    manager: SessionManager = Depends(get_session_manager),
) -> SessionResponse:
    try:
        ctx = manager.create_session(
            title=body.title,
            model=body.model,
            thinking_profile=body.thinking_profile,
            phoenix_project=body.phoenix_project,
            workspace_kind=body.workspace.kind,
            workspace_path=body.workspace.path,
            workspace_access_mode=body.workspace.access_mode,
            environment_strategy=body.workspace.environment_strategy,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return _to_response(ctx.to_meta_dict())


@router.get("", response_model=list[SessionResponse])
async def list_sessions(
    status: Optional[str] = None,
    limit: int = 100,
    manager: SessionManager = Depends(get_session_manager),
) -> list[SessionResponse]:
    sessions = manager.list_sessions()
    if status:
        sessions = [s for s in sessions if s.get("status") == status]
    return [_to_response(s) for s in sessions[:limit]]


@router.get("/{sid}", response_model=SessionResponse)
async def get_session(
    sid: str,
    manager: SessionManager = Depends(get_session_manager),
) -> SessionResponse:
    ctx = require_session(sid, manager)
    return _to_response(ctx.to_meta_dict())


@router.patch("/{sid}", response_model=SessionResponse)
async def update_session(
    sid: str,
    body: SessionUpdate,
    manager: SessionManager = Depends(get_session_manager),
) -> SessionResponse:
    ctx = require_session(sid, manager)
    if body.title is not None:
        ctx.title = body.title
    if body.status is not None:
        manager.update_status(ctx, body.status)
    if body.selected_model is not None:
        ctx.selected_model = body.selected_model
    if body.thinking_profile is not None:
        ctx.thinking_profile = body.thinking_profile
    manager._save_meta(ctx)  # persist any field changes
    return _to_response(ctx.to_meta_dict())


@router.delete("/{sid}", status_code=204)
async def delete_session(
    sid: str,
    manager: SessionManager = Depends(get_session_manager),
) -> None:
    ctx = manager.load_session(sid)
    if ctx is None:
        raise HTTPException(status_code=404, detail=f"Session '{sid}' not found")
    if ctx.workspace.kind == "managed" and ctx.workspace_root.exists():
        shutil.rmtree(ctx.workspace_root, ignore_errors=True)
    if ctx.root.exists():
        shutil.rmtree(ctx.root, ignore_errors=True)
    return None
