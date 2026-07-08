"""Models catalog endpoints."""

from __future__ import annotations

import socket
from urllib.parse import urlparse

from fastapi import APIRouter, HTTPException

from src.api.schemas import ModelInfo
from src.llm_client import _ENDPOINTS

router = APIRouter(prefix="/models", tags=["models"])


def _probe(url: str) -> tuple[bool, str | None]:
    """TCP probe a base_url; return (reachable, error)."""
    try:
        parsed = urlparse(url)
        host = parsed.hostname or "localhost"
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        with socket.create_connection((host, port), timeout=2):
            return True, None
    except Exception as exc:  # noqa: BLE001
        return False, str(exc)


@router.get("", response_model=list[ModelInfo])
async def list_models() -> list[ModelInfo]:
    out = []
    for key, cfg in _ENDPOINTS.items():
        reachable, err = _probe(cfg.get("base_url", ""))
        out.append(ModelInfo(
            key=key,
            model=cfg.get("model", ""),
            base_url=cfg.get("base_url", ""),
            available=reachable,
            error=err,
        ))
    # Add the synthetic "auto" entry.
    out.insert(0, ModelInfo(
        key="auto",
        model="(auto — local → gemini → gpt4o)",
        base_url="",
        available=True,
        error=None,
    ))
    return out


@router.get("/{key}", response_model=ModelInfo)
async def get_model(key: str) -> ModelInfo:
    if key == "auto":
        return ModelInfo(
            key="auto",
            model="(auto — local → gemini → gpt4o)",
            base_url="",
            available=True,
            error=None,
        )
    cfg = _ENDPOINTS.get(key)
    if cfg is None:
        raise HTTPException(status_code=404, detail=f"Unknown model '{key}'")
    reachable, err = _probe(cfg.get("base_url", ""))
    return ModelInfo(
        key=key,
        model=cfg.get("model", ""),
        base_url=cfg.get("base_url", ""),
        available=reachable,
        error=err,
    )
