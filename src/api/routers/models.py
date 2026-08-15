"""Models catalog endpoints."""

from __future__ import annotations

import socket
from urllib.parse import urlparse

from fastapi import APIRouter, HTTPException

from src.api.schemas import ModelInfo
from src.llm_client import get_model_catalog

router = APIRouter(prefix="/models", tags=["models"])


def _probe(url: str) -> tuple[bool, str | None]:
    """TCP probe a base_url; return (reachable, error)."""
    if not url:
        return True, None
    try:
        parsed = urlparse(url)
        host = parsed.hostname or "localhost"
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        with socket.create_connection((host, port), timeout=2):
            return True, None
    except Exception as exc:  # noqa: BLE001
        return False, str(exc)


def _to_model_info(entry: dict) -> ModelInfo:
    reachable, err = _probe(entry.get("base_url", ""))
    return ModelInfo(
        key=entry["key"],
        model=entry.get("model", ""),
        base_url=entry.get("base_url", ""),
        available=reachable if entry["key"] != "auto" else True,
        error=err,
        models_by_role=entry.get("models_by_role", {}),
        thinking_by_role=entry.get("thinking_by_role", {}),
        context_length=entry.get("context_length"),
    )


@router.get("", response_model=list[ModelInfo])
async def list_models() -> list[ModelInfo]:
    return [_to_model_info(entry) for entry in get_model_catalog()]


@router.get("/{key}", response_model=ModelInfo)
async def get_model(key: str) -> ModelInfo:
    for entry in get_model_catalog():
        if entry["key"] == key:
            return _to_model_info(entry)
    raise HTTPException(status_code=404, detail=f"Unknown model '{key}'")
