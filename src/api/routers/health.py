"""Health and readiness endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from src.api.deps import get_router, get_runtime
from src.api.schemas import HealthResponse, ReadyResponse
from src.llm_client import probe_provider
from src.sandbox.executor import _bwrap_available
from src.settings import get_settings, resolve_home

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health(router_obj=Depends(get_router)) -> HealthResponse:
    settings = get_settings()
    qdrant = {}
    encoder = "none"
    if hasattr(router_obj, "status"):
        qdrant = router_obj.status()
        encoder = qdrant.get("encoder", "none")
    return HealthResponse(
        status="ok",
        home=str(resolve_home()),
        qdrant=qdrant,
        embeddings=encoder,
        sandbox={
            "executor": settings.runtime.sandbox.executor,
            "mode": settings.runtime.sandbox.mode,
            "bwrap": _bwrap_available(),
            "require_bwrap": settings.runtime.sandbox.require_bwrap,
            "jail": "bwrap" if _bwrap_available() else "policy-only",
        },
        providers_configured=len(settings.enabled_providers()),
    )


@router.get("/ready", response_model=ReadyResponse)
async def ready(
    runtime=Depends(get_runtime),
    router_obj=Depends(get_router),
) -> ReadyResponse:
    """Check configured LLM providers and Qdrant — not every vendor on earth."""
    checks: dict[str, dict] = {}
    settings = get_settings()

    for provider in settings.enabled_providers():
        ok, models, err = probe_provider(provider.base_url, provider.api_key, timeout=3.0)
        checks[provider.id] = {
            "reachable": ok,
            "error": err,
            "models": models[:8],
        }

    qdrant_ok = False
    qdrant_err = None
    qdrant_mode = settings.qdrant.mode
    try:
        if qdrant_mode == "off":
            qdrant_ok = True
            qdrant_err = "disabled (keyword fallback)"
        else:
            client = getattr(router_obj, "_client", None)
            if client is not None:
                client.get_collections()
                qdrant_ok = True
            else:
                qdrant_err = getattr(router_obj, "_qdrant_error", None) or "not connected"
    except Exception as exc:  # noqa: BLE001
        qdrant_err = str(exc)
    checks["qdrant"] = {
        "reachable": qdrant_ok,
        "error": qdrant_err,
        "mode": qdrant_mode,
    }

    llm_ok = True
    if settings.enabled_providers():
        llm_ok = any(c.get("reachable") for k, c in checks.items() if k != "qdrant")
    ready_all = llm_ok and qdrant_ok
    return ReadyResponse(ready=ready_all, checks=checks)
