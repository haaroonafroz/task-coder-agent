"""Health and readiness endpoints."""

from __future__ import annotations

import os
import socket

from fastapi import APIRouter, Depends, Request

from src.api.deps import get_router, get_runtime
from src.api.schemas import HealthResponse, ReadyResponse
from src.llm_client import _ENDPOINTS

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(status="ok")


@router.get("/ready", response_model=ReadyResponse)
async def ready(
    runtime=Depends(get_runtime),
    router_obj=Depends(get_router),
) -> ReadyResponse:
    """Check LLM endpoint reachability and Qdrant connection."""
    checks: dict[str, dict] = {}

    # LLM backends — TCP probe of each base_url host:port.
    for key, cfg in _ENDPOINTS.items():
        url = cfg.get("base_url", "")
        reachable = False
        err = None
        try:
            from urllib.parse import urlparse
            parsed = urlparse(url)
            host = parsed.hostname or "localhost"
            port = parsed.port or (443 if parsed.scheme == "https" else 80)
            with socket.create_connection((host, port), timeout=2):
                reachable = True
        except Exception as exc:  # noqa: BLE001
            err = str(exc)
        checks[key] = {"reachable": reachable, "error": err}

    # Qdrant — router exposes the client.
    qdrant_ok = False
    qdrant_err = None
    try:
        client = getattr(router_obj, "_client", None)
        if client is not None:
            client.get_collections()
            qdrant_ok = True
    except Exception as exc:  # noqa: BLE001
        qdrant_err = str(exc)
    checks["qdrant"] = {"reachable": qdrant_ok, "error": qdrant_err}

    ready_all = all(c.get("reachable") for c in checks.values())
    return ReadyResponse(ready=ready_all, checks=checks)
