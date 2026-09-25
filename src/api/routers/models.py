"""Models catalog endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from src.api.schemas import ModelInfo
from src.llm_client import get_model_catalog, probe_provider
from src.settings import get_settings

router = APIRouter(prefix="/models", tags=["models"])


def _to_model_info(entry: dict, *, probe: bool = True) -> ModelInfo:
    discovered: list[str] = []
    err = None
    available = True
    if entry["key"] == "auto":
        available = bool(get_settings().fallback_ids())
        if not available:
            err = "No providers configured"
    elif not entry.get("enabled", True):
        available = False
        err = "disabled in Settings"
    elif probe:
        provider = get_settings().provider(entry["key"])
        api_key = provider.api_key if provider else ""
        ok, discovered, probe_err = probe_provider(
            entry.get("base_url", ""), api_key, timeout=3.0
        )
        available = ok
        err = probe_err
    return ModelInfo(
        key=entry["key"],
        model=entry.get("model", ""),
        base_url=entry.get("base_url", ""),
        available=available,
        error=err,
        models_by_role=entry.get("models_by_role", {}),
        thinking_by_role=entry.get("thinking_by_role", {}),
        context_length=entry.get("context_length"),
        label=entry.get("label"),
        adapter=entry.get("adapter"),
        compat=entry.get("compat"),
        enabled=entry.get("enabled", True),
        api_key_set=entry.get("api_key_set", False),
        discovered_models=discovered,
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
