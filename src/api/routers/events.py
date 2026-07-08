"""Events endpoints — SSE stream + non-streaming history."""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, Header, HTTPException
from fastapi.responses import StreamingResponse

from src.api.deps import get_session_manager, require_session, resolve_emitter
from src.api.schemas import EventResponse
from src.api.sse import event_stream
from src.session import SessionManager

router = APIRouter(prefix="/sessions/{sid}/events", tags=["events"])


def _parse_since(
    last_event_id: Optional[str],
    since: Optional[int],
) -> int:
    """Prefer explicit query param, then Last-Event-ID header."""
    if since is not None and since > 0:
        return since
    if last_event_id is not None:
        try:
            return int(last_event_id) + 1  # resume after the last seen
        except ValueError:
            return 0
    return 0


@router.get("", response_class=StreamingResponse)
async def stream_events(
    sid: str,
    since: Optional[int] = None,
    last_event_id: Optional[str] = Header(default=None, alias="Last-Event-ID"),
    manager: SessionManager = Depends(get_session_manager),
):
    """SSE stream: replay history then stream live events."""
    ctx = require_session(sid, manager)
    emitter = resolve_emitter(ctx)
    start = _parse_since(last_event_id, since)

    async def gen():
        async for frame in event_stream(emitter, since=start):
            yield frame

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/history", response_model=list[EventResponse])
async def event_history(
    sid: str,
    since: Optional[int] = None,
    manager: SessionManager = Depends(get_session_manager),
) -> list[EventResponse]:
    ctx = require_session(sid, manager)
    emitter = resolve_emitter(ctx)
    start = since or 0
    history = emitter.history(since_index=start)
    out = []
    for idx, event in enumerate(history, start=start):
        out.append(EventResponse(
            ts=event.get("ts", ""),
            type=event.get("type", ""),
            session_id=event.get("session_id", sid),
            data=event.get("data", {}),
            index=idx,
        ))
    return out
