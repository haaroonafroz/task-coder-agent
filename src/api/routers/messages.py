"""Messages endpoints (chat-style N:M message log)."""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException

from src.api.deps import (
    get_message_store,
    get_run_queue,
    get_session_manager,
    require_session,
)
from src.api.messages import MessageStore
from src.api.run_queue import RunQueue
from src.api.schemas import MessageCreate, MessageResponse
from src.session import SessionManager

router = APIRouter(prefix="/sessions/{sid}/messages", tags=["messages"])


@router.post("", response_model=MessageResponse, status_code=201)
async def create_message(
    sid: str,
    body: MessageCreate,
    manager: SessionManager = Depends(get_session_manager),
    message_store: MessageStore = Depends(get_message_store),
    run_queue: RunQueue = Depends(get_run_queue),
) -> MessageResponse:
    ctx = require_session(sid, manager)
    run_id: Optional[str] = None
    if body.trigger_run:
        rec = run_queue.enqueue(
            ctx,
            body.content,
            model=body.model,
            run_kind=body.run_kind,
        )
        run_id = rec.run_id
    msg = message_store.append(ctx, "user", body.content, run_id=run_id)
    msg["run_kind"] = body.run_kind
    return MessageResponse(**msg)


@router.get("", response_model=list[MessageResponse])
async def list_messages(
    sid: str,
    limit: Optional[int] = None,
    manager: SessionManager = Depends(get_session_manager),
    message_store: MessageStore = Depends(get_message_store),
) -> list[MessageResponse]:
    ctx = require_session(sid, manager)
    msgs = message_store.list(ctx, limit=limit)
    return [MessageResponse(**m) for m in msgs]
