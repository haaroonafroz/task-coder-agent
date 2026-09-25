"""Settings and first-run configuration endpoints."""

from __future__ import annotations

import os
from typing import Any

from fastapi import APIRouter, HTTPException, Request

from src.api.schemas import (
    ProviderProbeRequest,
    ProviderProbeResponse,
    SettingsResponse,
    SettingsUpdate,
)
from src.llm_client import probe_provider
from src.settings import (
    PROVIDER_PRESETS,
    get_settings,
    replace_settings,
    resolve_home,
    update_settings,
)
from src.settings.schema import MissionsSettings
from src.settings.store import migrated_from_env
from src.tool_registry import DynamicToolRouter

router = APIRouter(prefix="/settings", tags=["settings"])


def _in_container() -> bool:
    return PathExists("/.dockerenv") or os.getenv("MISSIONS_CONTAINER", "").strip() == "true"


def PathExists(path: str) -> bool:
    from pathlib import Path
    return Path(path).exists()


def _preserve_secrets(current: MissionsSettings, patch: dict[str, Any]) -> dict[str, Any]:
    llm = patch.get("llm")
    if isinstance(llm, dict):
        providers = llm.get("providers")
        if isinstance(providers, list):
            for item in providers:
                if not isinstance(item, dict):
                    continue
                if item.get("api_key"):
                    continue
                existing = current.provider(str(item.get("id") or ""))
                if existing and existing.api_key:
                    item["api_key"] = existing.api_key
    qdrant = patch.get("qdrant")
    if isinstance(qdrant, dict) and not qdrant.get("api_key") and current.qdrant.api_key:
        qdrant["api_key"] = current.qdrant.api_key
    return patch


def _rebuild_router(request: Request) -> None:
    skills = getattr(request.app.state, "skills_path", None)
    if skills is None:
        from pathlib import Path
        skills = Path(__file__).resolve().parents[3] / "config" / "skills.md"
    old = getattr(request.app.state, "router", None)
    if old is not None and hasattr(old, "close"):
        try:
            old.close()
        except Exception:
            pass
    try:
        request.app.state.router = DynamicToolRouter(skills)
    except Exception as exc:  # noqa: BLE001
        print(f"[Settings] Failed to rebuild tool router: {exc}")


@router.get("", response_model=SettingsResponse)
async def read_settings() -> SettingsResponse:
    settings = get_settings()
    return SettingsResponse(
        settings=settings.redacted_dict(),
        home=str(resolve_home()),
        presets=PROVIDER_PRESETS,
        migrated_from_env=migrated_from_env(),
        container=_in_container(),
    )


@router.patch("", response_model=SettingsResponse)
async def patch_settings(body: SettingsUpdate, request: Request) -> SettingsResponse:
    if not body.settings:
        raise HTTPException(status_code=400, detail="Empty settings patch")
    current = get_settings()
    merged_patch = _preserve_secrets(current, body.settings)
    try:
        update_settings(merged_patch)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    _rebuild_router(request)
    settings = get_settings()
    return SettingsResponse(
        settings=settings.redacted_dict(),
        home=str(resolve_home()),
        presets=PROVIDER_PRESETS,
        migrated_from_env=migrated_from_env(),
        container=_in_container(),
    )


@router.put("", response_model=SettingsResponse)
async def put_settings(body: SettingsUpdate, request: Request) -> SettingsResponse:
    current = get_settings()
    payload = _preserve_secrets(current, body.settings or {})
    try:
        settings = MissionsSettings.model_validate(payload)
        replace_settings(settings, persist=True)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    _rebuild_router(request)
    settings = get_settings()
    return SettingsResponse(
        settings=settings.redacted_dict(),
        home=str(resolve_home()),
        presets=PROVIDER_PRESETS,
        migrated_from_env=migrated_from_env(),
        container=_in_container(),
    )


@router.post("/test-provider", response_model=ProviderProbeResponse)
async def test_provider(body: ProviderProbeRequest) -> ProviderProbeResponse:
    key = body.api_key or ""
    if not key and body.provider_id:
        provider = get_settings().provider(body.provider_id)
        if provider:
            key = provider.api_key
    ok, models, error = probe_provider(body.base_url, key)
    return ProviderProbeResponse(ok=ok, models=models, error=error)
