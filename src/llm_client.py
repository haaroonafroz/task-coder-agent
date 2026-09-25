"""
OpenAI-compatible LLM client for the Missions harness.

Providers are named connections (base_url + api_key + model) loaded from
settings. Vendor quirks live in adapters:

  generic / openai     — stock Chat Completions
  gemini_openai        — Gemini's OpenAI-compat endpoint + thinking_config
  llamacpp_qwen        — llama.cpp chat_template_kwargs for Qwen thinking
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Optional, Literal
from urllib.parse import urlparse

from openai import OpenAI, APIConnectionError, APIStatusError

from src.settings import ADAPTER_UNSUPPORTED_PARAMS, get_settings
from src.settings.schema import ProviderConfig, RoleSettings

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------
ModelChoice = str  # "auto" or a provider id
AgentRole = Literal[
    "triage", "orchestrator", "worker", "hotfix", "reviewer", "validator"
]
ThinkingLevel = Literal["off", "minimal", "low", "medium", "high", "xhigh"]


@dataclass(frozen=True)
class ResolvedModelConfig:
    """Concrete model + thinking settings for one (provider, role) pair."""

    backend: str
    role: AgentRole
    base_url: str
    api_key: str
    model_name: str
    thinking_level: ThinkingLevel
    adapter: str = "generic"


@dataclass
class LLMResult:
    text: str
    model_used: str
    prefill_ms: float
    decode_ms: float
    total_ms: float
    tokens_generated: int
    tokens_prompt: int = 0
    fallback_used: bool = False
    thinking_level: Optional[str] = None
    thinking_text: str = ""


def get_context_length() -> int:
    settings = get_settings()
    local = settings.provider("local")
    if local and local.context_length:
        return max(1024, int(local.context_length))
    return max(1024, int(settings.llm.context_length or 32768))


def _role_settings(role: AgentRole) -> RoleSettings:
    return get_settings().roles.for_role(role)


def _normalize_thinking(role_cfg: RoleSettings, adapter: str) -> ThinkingLevel:
    effort = (role_cfg.thinking or "medium").strip().lower()
    enabled = role_cfg.thinking_enabled
    if effort == "off" or not enabled:
        return "off"
    aliases = {"minimal": "low", "high": "xhigh", "max": "xhigh"}
    effort = aliases.get(effort, effort)
    if adapter == "llamacpp_qwen":
        if effort not in ("low", "medium", "xhigh"):
            effort = "medium"
        return effort  # type: ignore[return-value]
    if adapter == "gemini_openai":
        if effort not in ("off", "minimal", "low", "medium", "high"):
            effort = "medium"
        return effort  # type: ignore[return-value]
    return "off"


def resolve_model_config(backend: str, role: AgentRole) -> ResolvedModelConfig:
    """Resolve provider + role into a concrete chat-completions config."""
    if backend == "auto":
        raise KeyError("resolve_model_config does not accept backend='auto'")
    settings = get_settings()
    provider = settings.provider(backend)
    if provider is None or not provider.enabled:
        raise KeyError(f"Unknown or disabled provider: {backend}")
    role_cfg = _role_settings(role)
    adapter = provider.adapter or "generic"
    return ResolvedModelConfig(
        backend=backend,
        role=role,
        base_url=provider.base_url,
        api_key=provider.api_key or "EMPTY",
        model_name=provider.model_for_role(role),
        thinking_level=_normalize_thinking(role_cfg, adapter),
        adapter=adapter,
    )


def get_model_catalog() -> list[dict[str, Any]]:
    """Return the configured provider catalog, with a synthetic ``auto`` entry."""
    settings = get_settings()
    roles = (
        "triage", "orchestrator", "worker", "hotfix", "reviewer", "validator"
    )
    entries: list[dict[str, Any]] = []
    for provider in settings.llm.providers:
        models_by_role = {role: provider.model_for_role(role) for role in roles}
        thinking_by_role = {
            role: _normalize_thinking(settings.roles.for_role(role), provider.adapter)
            for role in roles
        }
        entries.append({
            "key": provider.id,
            "label": provider.display_label(),
            "base_url": provider.base_url,
            "adapter": provider.adapter,
            "enabled": provider.enabled,
            "model": provider.model_for_role("orchestrator"),
            "models_by_role": models_by_role,
            "thinking_by_role": thinking_by_role,
            "context_length": provider.context_length or (
                get_context_length() if provider.adapter == "llamacpp_qwen" else None
            ),
            "api_key_set": bool(provider.api_key),
        })

    fallback = settings.fallback_ids()
    auto_label = (
        "(auto — " + " → ".join(fallback) + ")"
        if fallback
        else "(auto — no providers configured)"
    )
    entries.insert(0, {
        "key": "auto",
        "label": "Auto",
        "base_url": "",
        "adapter": "generic",
        "enabled": bool(fallback),
        "model": auto_label,
        "models_by_role": {role: "(fallback chain)" for role in roles},
        "thinking_by_role": {role: "per-provider" for role in roles},
        "context_length": get_context_length(),
        "api_key_set": False,
    })
    return entries


def _candidate_ids(model: ModelChoice) -> list[str]:
    settings = get_settings()
    if model and model != "auto":
        provider = settings.provider(model)
        if provider is None:
            raise RuntimeError(
                f"Unknown provider '{model}'. Add it in Settings or pick Auto."
            )
        if not provider.enabled:
            raise RuntimeError(f"Provider '{model}' is disabled in Settings.")
        return [model]
    ids = settings.fallback_ids()
    if not ids:
        raise RuntimeError(
            "No LLM providers configured. Open Settings and add a local "
            "OpenAI-compatible URL or a cloud API key."
        )
    return ids


def _gemini_extra_body(thinking_level: ThinkingLevel) -> dict[str, Any]:
    return {
        "extra_body": {
            "google": {
                "thinking_config": {
                    "thinking_level": thinking_level,
                    "include_thoughts": False,
                }
            }
        }
    }


def _build_client(cfg: ResolvedModelConfig) -> OpenAI:
    return OpenAI(api_key=cfg.api_key or "EMPTY", base_url=cfg.base_url)


def probe_provider(
    base_url: str,
    api_key: str = "",
    timeout: float = 5.0,
) -> tuple[bool, list[str], Optional[str]]:
    """Call GET {base_url}/models and return (ok, model ids, error)."""
    url = (base_url or "").rstrip("/")
    if not url:
        return False, [], "base_url is empty"
    if not url.endswith("/models"):
        url = url + "/models"
    headers = {"Accept": "application/json"}
    if api_key and api_key != "EMPTY":
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(url, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return False, [], f"HTTP {exc.code}: {exc.reason}"
    except Exception as exc:  # noqa: BLE001
        return False, [], str(exc)

    models: list[str] = []
    data = payload.get("data", payload) if isinstance(payload, dict) else payload
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict) and item.get("id"):
                models.append(str(item["id"]))
            elif isinstance(item, str):
                models.append(item)
    return True, models, None


def tcp_reachable(url: str, timeout: float = 2.0) -> tuple[bool, Optional[str]]:
    if not url:
        return True, None
    try:
        parsed = urlparse(url)
        host = parsed.hostname or "localhost"
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        import socket
        with socket.create_connection((host, port), timeout=timeout):
            return True, None
    except Exception as exc:  # noqa: BLE001
        return False, str(exc)


def call_llm(
    prompt: Optional[str] = None,
    model: ModelChoice = "auto",
    max_tokens: int = 2048,
    system_prompt: Optional[str] = None,
    json_mode: bool = False,
    role: AgentRole = "worker",
    enable_thinking: Optional[bool] = None,
    messages: Optional[list[dict[str, Any]]] = None,
    stream_context: Any = None,
) -> LLMResult:
    """
    Send a prompt to the selected LLM and return the result with precise timing.

    ``model`` is ``auto`` or a configured provider id. ``auto`` walks
    ``settings.llm.fallback_order``.
    """
    candidates = _candidate_ids(model)
    last_exc: Optional[Exception] = None

    for attempt, model_key in enumerate(candidates):
        try:
            return _call_single(
                prompt=prompt,
                model_key=model_key,
                role=role,
                max_tokens=max_tokens,
                system_prompt=system_prompt,
                json_mode=json_mode,
                fallback_used=(attempt > 0),
                enable_thinking_override=enable_thinking,
                messages=messages,
                stream_context=stream_context,
            )
        except (APIConnectionError, ConnectionRefusedError, OSError) as exc:
            print(f"[LLMClient] {model_key} unreachable: {exc}. Trying next.")
            last_exc = exc
        except APIStatusError as exc:
            if exc.status_code in (400, 429) or exc.status_code >= 500:
                print(f"[LLMClient] {model_key} HTTP {exc.status_code} — trying next.")
                last_exc = exc
            elif exc.status_code in (401, 403):
                print(f"[LLMClient] {model_key} auth error ({exc.status_code}) — trying next.")
                last_exc = exc
            else:
                raise
        except Exception as exc:
            print(f"[LLMClient] {model_key} error: {exc}. Trying next.")
            last_exc = exc

    raise RuntimeError(
        f"All LLM backends failed. Last error: {last_exc}"
    ) from last_exc


def _call_single(
    prompt: Optional[str],
    model_key: str,
    role: AgentRole,
    max_tokens: int,
    system_prompt: Optional[str],
    json_mode: bool,
    fallback_used: bool,
    enable_thinking_override: Optional[bool],
    messages: Optional[list[dict[str, Any]]] = None,
    stream_context: Any = None,
) -> LLMResult:
    """Execute a single streaming call to the given provider."""
    cfg = resolve_model_config(model_key, role)
    role_cfg = _role_settings(role)
    settings = get_settings()

    thinking_level = cfg.thinking_level
    if enable_thinking_override is not None and cfg.adapter == "llamacpp_qwen":
        thinking_level = "medium" if enable_thinking_override else "off"

    model_used = f"{model_key}/{cfg.model_name}"
    if stream_context is not None:
        stream_context.start(model_used=model_used, thinking_level=thinking_level)

    client = _build_client(cfg)

    chat_messages: list[dict[str, Any]] = []
    if system_prompt:
        chat_messages.append({"role": "system", "content": system_prompt})

    if messages is not None:
        chat_messages.extend(
            {"role": m["role"], "content": m.get("content", "")}
            for m in messages
        )
    else:
        chat_messages.append({"role": "user", "content": prompt or ""})

    kwargs: dict[str, Any] = dict(
        model=cfg.model_name or "default",
        messages=chat_messages,
        temperature=role_cfg.temperature,
        top_p=role_cfg.top_p,
        seed=settings.llm.seed,
        max_tokens=max_tokens,
        stream=True,
        stream_options={"include_usage": True},
    )
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    if cfg.adapter == "llamacpp_qwen":
        template_kwargs: dict[str, Any] = {
            "enable_thinking": thinking_level != "off",
        }
        if thinking_level != "off":
            template_kwargs["reasoning_effort"] = thinking_level
        kwargs["extra_body"] = {"chat_template_kwargs": template_kwargs}

    if cfg.adapter == "gemini_openai" and thinking_level not in ("off",):
        kwargs["extra_body"] = _gemini_extra_body(thinking_level)

    for param in ADAPTER_UNSUPPORTED_PARAMS.get(cfg.adapter, set()):
        kwargs.pop(param, None)

    t0 = time.perf_counter()
    t_first: Optional[float] = None
    content_chunks: list[str] = []
    thinking_chunks: list[str] = []
    pending_thinking = ""
    pending_content = ""
    prompt_tokens = 0
    completion_tokens = 0
    delta_flush = 48

    def _flush(channel: str, pending: str) -> str:
        if stream_context is not None and pending:
            stream_context.delta(channel, pending)
        return ""

    stream = client.chat.completions.create(**kwargs)
    for chunk in stream:
        if not chunk.choices:
            if chunk.usage is not None:
                prompt_tokens = chunk.usage.prompt_tokens or 0
                completion_tokens = chunk.usage.completion_tokens or 0
            continue

        delta = chunk.choices[0].delta
        reasoning = getattr(delta, "reasoning_content", None)
        content = getattr(delta, "content", None)

        if t_first is None and (reasoning or content):
            t_first = time.perf_counter()

        if reasoning:
            thinking_chunks.append(reasoning)
            pending_thinking += reasoning
            if len(pending_thinking) >= delta_flush:
                pending_thinking = _flush("thinking", pending_thinking)

        if content:
            content_chunks.append(content)
            pending_content += content
            if len(pending_content) >= delta_flush:
                pending_content = _flush("content", pending_content)

        if chunk.usage is not None:
            prompt_tokens = chunk.usage.prompt_tokens or 0
            completion_tokens = chunk.usage.completion_tokens or 0

    pending_thinking = _flush("thinking", pending_thinking)
    pending_content = _flush("content", pending_content)

    t2 = time.perf_counter()
    if t_first is None:
        t_first = t2

    prefill_ms = (t_first - t0) * 1000.0
    decode_ms = (t2 - t_first) * 1000.0
    total_ms = (t2 - t0) * 1000.0

    if fallback_used:
        print(
            f"[LLMClient] Using fallback: {model_key}/{cfg.model_name} "
            f"(role={role}, thinking={thinking_level})"
        )

    thinking_text = "".join(thinking_chunks)
    text = "".join(content_chunks)

    result = LLMResult(
        text=text,
        model_used=model_used,
        prefill_ms=round(prefill_ms, 2),
        decode_ms=round(decode_ms, 2),
        total_ms=round(total_ms, 2),
        tokens_generated=completion_tokens,
        tokens_prompt=prompt_tokens,
        fallback_used=fallback_used,
        thinking_level=thinking_level,
        thinking_text=thinking_text,
    )

    if stream_context is not None:
        stream_context.finish(
            result,
            thinking_text=thinking_text,
            output_text=text,
        )

    return result
