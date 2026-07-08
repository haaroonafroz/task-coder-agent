"""
SSE bridge: replay history then stream live EventEmitter events as
Server-Sent Events.

The EventEmitter is synchronous (callback-based pub/sub). FastAPI's
StreamingResponse needs an async generator. We bridge the two with an
asyncio.Queue: a synchronous subscriber callback enqueues events, and the
async generator awaits them.

The subscriber callback may be invoked from any thread (the runtime worker
thread, or the API thread). ``asyncio.Queue.put_nowait`` is NOT safe to
call from another thread — it won't wake the event loop's waiter. We use
``loop.call_soon_threadsafe`` to schedule the put on the event loop
thread, which correctly notifies any awaiting ``queue.get()``.

Each SSE event carries an ``id: <index>`` line so clients can reconnect
with ``Last-Event-ID`` (or ``?since=N``) and resume from where they left
off.
"""

from __future__ import annotations

import asyncio
import json
from typing import AsyncGenerator, Optional

from src.events import EventEmitter


def _format_sse(event: dict, index: int) -> str:
    """Serialize one event dict as an SSE frame."""
    payload = json.dumps(event, default=str)
    return f"id: {index}\nevent: {event.get('type', 'message')}\ndata: {payload}\n\n"


async def event_stream(
    emitter: EventEmitter,
    since: int = 0,
) -> AsyncGenerator[str, None]:
    """
    Yield SSE frames: first replay history (since `since`), then live.

    Cancellation-safe: when the client disconnects, the generator's
    ``finally`` block unsubscribes the callback so we don't leak.
    """
    loop = asyncio.get_running_loop()
    queue: "asyncio.Queue[Optional[dict]]" = asyncio.Queue()

    def _on_event(event: dict) -> None:
        # Called from the emitter's thread (may be the runtime worker thread).
        # Must be scheduled on the event loop thread to safely wake awaiters.
        try:
            loop.call_soon_threadsafe(queue.put_nowait, event)
        except RuntimeError:
            # Event loop may be closed during shutdown — ignore.
            pass

    unsubscribe = emitter.subscribe(_on_event)

    try:
        # 1. Replay history (events already on disk).
        history = emitter.history(since_index=since)
        for idx, event in enumerate(history, start=since):
            yield _format_sse(event, idx)

        # 2. Stream live events.
        while True:
            event = await queue.get()
            if event is None:
                # Sentinel used on shutdown (not currently triggered).
                break
            # Compute the global index from the on-disk count so reconnects line up.
            idx = emitter.event_count() - 1
            yield _format_sse(event, idx)
    finally:
        unsubscribe()
