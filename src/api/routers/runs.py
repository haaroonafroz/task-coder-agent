"""Runs endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from src.api.deps import (
    get_run_queue,
    get_session_manager,
    require_session,
)
from src.api.run_queue import CancelNotAllowedError, RunQueue
from src.api.schemas import RunCreate, RunResponse
from src.session import SessionManager

router = APIRouter(prefix="/sessions/{sid}/runs", tags=["runs"])


@router.post("", response_model=RunResponse, status_code=202)
async def create_run(
    sid: str,
    body: RunCreate,
    manager: SessionManager = Depends(get_session_manager),
    run_queue: RunQueue = Depends(get_run_queue),
) -> RunResponse:
    ctx = require_session(sid, manager)
    rec = run_queue.enqueue(ctx, body.request, model=body.model)
    return RunResponse(**rec.to_dict())


@router.get("", response_model=list[RunResponse])
async def list_runs(
    sid: str,
    manager: SessionManager = Depends(get_session_manager),
    run_queue: RunQueue = Depends(get_run_queue),
) -> list[RunResponse]:
    require_session(sid, manager)
    records = run_queue._registry.list_for_session(sid)
    return [RunResponse(**r.to_dict()) for r in records]


@router.get("/{rid}", response_model=RunResponse)
async def get_run(
    sid: str,
    rid: str,
    manager: SessionManager = Depends(get_session_manager),
    run_queue: RunQueue = Depends(get_run_queue),
) -> RunResponse:
    require_session(sid, manager)
    rec = run_queue._registry.get(rid)
    if rec is None or rec.session_id != sid:
        raise HTTPException(status_code=404, detail=f"Run '{rid}' not found in session '{sid}'")
    return RunResponse(**rec.to_dict())


@router.post("/{rid}/cancel", response_model=RunResponse)
async def cancel_run(
    sid: str,
    rid: str,
    manager: SessionManager = Depends(get_session_manager),
    run_queue: RunQueue = Depends(get_run_queue),
) -> RunResponse:
    require_session(sid, manager)
    rec = run_queue._registry.get(rid)
    if rec is None or rec.session_id != sid:
        raise HTTPException(status_code=404, detail=f"Run '{rid}' not found in session '{sid}'")
    try:
        updated = run_queue.cancel(rid)
    except CancelNotAllowedError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return RunResponse(**updated.to_dict())
