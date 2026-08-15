"""
Multi-model LLM client — Missions Framework.

Priority chain (when model="auto"):
  1. Local  — Qwen/Gemma via llama.cpp server
  2. Gemini — gemini-3.1-flash-lite via OpenAI-compatible Gemini endpoint
  3. GPT-4o — gpt-4o / gpt-4o-mini via OpenAI API

The frontend selects a backend key (``local`` | ``gemini`` | ``gpt4o`` | ``auto``).
Each agent role resolves to a concrete model + thinking profile via the registry
(Phase 11):

  Backend   Orchestrator / Validator     Worker
  --------  -------------------------     ------
  local     TARGET_MODEL, thinking=medium TARGET_MODEL, thinking=off (/no_think)
  gemini    GEMINI_MODEL, thinking=max   GEMINI_MODEL, thinking=default
  gpt4o     OPENAI_LLM_MODEL             OPENAI_LLM_MODEL_ENRICHMENT

Environment variables (can be set in .env):
  LLM_SPECULATIVE_URL          — llama.cpp server base URL
  TARGET_MODEL                 — local model alias
  GEMINI_API_KEY, GEMINI_MODEL
  GEMINI_THINKING_LEVEL_DEFAULT, GEMINI_THINKING_LEVEL_MAX
  OPENAI_API_KEY, OPENAI_LLM_MODEL, OPENAI_LLM_MODEL_ENRICHMENT
  LLM_TEMPERATURE, LLM_TEMPERATURE_<ROLE>, LLM_TOP_P, LLM_TOP_P_<ROLE>, LLM_SEED
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Literal

from openai import OpenAI, APIConnectionError, APIStatusError

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / ".env", override=False)
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------
ModelChoice = Literal["local", "gemini", "gpt4o", "auto"]
AgentRole = Literal["triage", "orchestrator", "worker", "validator"]
ThinkingLevel = Literal["off", "minimal", "low", "medium", "high"]

# Parameters not accepted by each backend's OpenAI-compatible endpoint.
_UNSUPPORTED_PARAMS: dict[str, set[str]] = {
    "local": set(),
    "gemini": {"seed", "stream_options"},
    "gpt4o": set(),
}

_FALLBACK_CHAIN: list[str] = ["local", "gemini", "gpt4o"]

SEED = int(os.getenv("LLM_SEED", "42"))


def _env_float(name: str) -> float:
    raw = os.getenv(name, "").strip()
    if not raw:
        raise RuntimeError(f"Missing required environment variable: {name}")
    try:
        return float(raw)
    except ValueError as exc:
        raise RuntimeError(f"Invalid float in {name}={raw!r}") from exc


def _role_sampling(role: AgentRole, base_var: str) -> float:
    """Resolve a per-role float from env.

    Precedence: ``{base_var}_<ROLE>`` > ``{base_var}`` (both must be set in .env).
    """
    role_var = f"{base_var}_{role.upper()}"
    raw = os.getenv(role_var, "").strip()
    if raw:
        try:
            return float(raw)
        except ValueError as exc:
            raise RuntimeError(f"Invalid float in {role_var}={raw!r}") from exc
    return _env_float(base_var)


def _role_temperature(role: AgentRole) -> float:
    return _role_sampling(role, "LLM_TEMPERATURE")


def _role_top_p(role: AgentRole) -> float:
    return _role_sampling(role, "LLM_TOP_P")


@dataclass(frozen=True)
class ResolvedModelConfig:
    """Concrete model + thinking settings for one (backend, role) pair."""

    backend: str
    role: AgentRole
    base_url: str
    api_key: str
    model_name: str
    thinking_level: ThinkingLevel


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
    """Return configured llama.cpp context length for UI metrics."""
    try:
        return max(1024, int(os.getenv("CONTEXT_LEN", "32768")))
    except ValueError:
        return 32768


# ---------------------------------------------------------------------------
# Registry — env-backed endpoint metadata (no per-role model baked in)
# ---------------------------------------------------------------------------

def _endpoint_meta(backend: str) -> dict[str, str]:
    """Return base_url + api_key for a backend key."""
    if backend == "local":
        return {
            "base_url": os.getenv("LLM_SPECULATIVE_URL", "http://localhost:8001/v1"),
            "api_key": "EMPTY",
        }
    if backend == "gemini":
        return {
            "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
            "api_key": os.getenv("GEMINI_API_KEY", ""),
        }
    if backend == "gpt4o":
        return {
            "base_url": "https://api.openai.com/v1",
            "api_key": os.getenv("OPENAI_API_KEY", ""),
        }
    raise KeyError(f"Unknown backend: {backend}")


def _role_thinking_level(backend: str, role: AgentRole) -> ThinkingLevel:
    """
    Resolve the thinking level for a backend + role.

    Gemini reads GEMINI_THINKING_LEVEL_DEFAULT (worker) and
    GEMINI_THINKING_LEVEL_MAX (orchestrator/validator).
    Local uses medium for triage/planning/validation, off for worker (/no_think).
    GPT has no thinking knob — level is always off (model choice handles capability).
    """
    if backend == "gemini":
        if role == "worker":
            raw = os.getenv("GEMINI_THINKING_LEVEL_DEFAULT", "minimal")
        else:
            raw = os.getenv("GEMINI_THINKING_LEVEL_MAX", "medium")
        level = raw.strip().lower()
        if level in ("off", "minimal", "low", "medium", "high"):
            return level  # type: ignore[return-value]
        return "minimal" if role == "worker" else "medium"

    if backend == "local":
        if role == "worker":
            return "off"
        # llama.cpp templates only understand thinking on/off; any non-"off"
        # level means enabled. Measured: orchestrator medium thinking is the
        # dominant mission cost (~80%), so these are env-tunable.
        raw = os.getenv(f"LOCAL_THINKING_LEVEL_{role.upper()}", "medium").strip().lower()
        if raw in ("off", "minimal", "low", "medium", "high"):
            return raw  # type: ignore[return-value]
        return "medium"

    return "off"


def _role_model_name(backend: str, role: AgentRole) -> str:
    """Resolve the concrete model name for a backend + role."""
    if backend == "local":
        return os.getenv("TARGET_MODEL", "qwen3-27b")
    if backend == "gemini":
        return os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")
    if backend == "gpt4o":
        if role == "worker":
            return os.getenv("OPENAI_LLM_MODEL_ENRICHMENT", "gpt-4o-mini")
        return os.getenv("OPENAI_LLM_MODEL", "gpt-4o")
    raise KeyError(f"Unknown backend: {backend}")


def resolve_model_config(backend: str, role: AgentRole) -> ResolvedModelConfig:
    """
    Resolve env-backed model name + thinking level for a backend and agent role.

    Args:
        backend: ``local`` | ``gemini`` | ``gpt4o`` (not ``auto``).
        role:    ``triage`` | ``orchestrator`` | ``worker`` | ``validator``.

    Returns:
        A :class:`ResolvedModelConfig` with everything needed for one LLM call.
    """
    meta = _endpoint_meta(backend)
    return ResolvedModelConfig(
        backend=backend,
        role=role,
        base_url=meta["base_url"],
        api_key=meta["api_key"],
        model_name=_role_model_name(backend, role),
        thinking_level=_role_thinking_level(backend, role),
    )


def get_model_catalog() -> list[dict[str, Any]]:
    """
    Return the full model registry for API exposure.

    Each entry describes one selectable backend with per-role model names and
    thinking levels. Includes a synthetic ``auto`` entry at the front.
    """
    entries: list[dict[str, Any]] = []
    for key in _FALLBACK_CHAIN:
        meta = _endpoint_meta(key)
        models_by_role = {
            role: _role_model_name(key, role)
            for role in ("triage", "orchestrator", "worker", "validator")
        }
        thinking_by_role = {
            role: _role_thinking_level(key, role)
            for role in ("triage", "orchestrator", "worker", "validator")
        }
        entries.append({
            "key": key,
            "base_url": meta["base_url"],
            "model": models_by_role["orchestrator"],
            "models_by_role": models_by_role,
            "thinking_by_role": thinking_by_role,
            "context_length": get_context_length() if key == "local" else None,
        })

    entries.insert(0, {
        "key": "auto",
        "base_url": "",
        "model": "(auto — local → gemini → gpt4o)",
        "models_by_role": {
            "triage": "(fallback chain)",
            "orchestrator": "(fallback chain)",
            "worker": "(fallback chain)",
            "validator": "(fallback chain)",
        },
        "thinking_by_role": {
            "triage": "per-backend",
            "orchestrator": "per-backend",
            "worker": "per-backend",
            "validator": "per-backend",
        },
        "context_length": get_context_length(),
    })
    return entries


# Backward-compatible flat endpoint dict (orchestrator model as primary label).
def _build_legacy_endpoints() -> dict[str, dict]:
    out: dict[str, dict] = {}
    for key in _FALLBACK_CHAIN:
        meta = _endpoint_meta(key)
        out[key] = {
            "base_url": meta["base_url"],
            "api_key": meta["api_key"],
            "model": _role_model_name(key, "orchestrator"),
        }
    return out


_ENDPOINTS: dict[str, dict] = _build_legacy_endpoints()


# ---------------------------------------------------------------------------
# Provider-specific thinking adapters
# ---------------------------------------------------------------------------

def _local_user_content(
    prompt: Any,
    thinking_level: ThinkingLevel,
) -> Any:
    """Add the local thinking directive without corrupting image parts."""
    if thinking_level in ("off", "minimal"):
        if isinstance(prompt, str):
            return "/no_think\n\n" + prompt
        if isinstance(prompt, list):
            parts = [
                dict(part) if isinstance(part, dict) else part
                for part in prompt
            ]
            for part in parts:
                if isinstance(part, dict) and part.get("type") == "text":
                    part["text"] = "/no_think\n\n" + str(part.get("text", ""))
                    break
            return parts
    return prompt


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


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

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

    Args:
        prompt:        User-turn prompt text. Ignored when ``messages`` is given.
        model:         Backend key: ``local`` | ``gemini`` | ``gpt4o`` | ``auto``.
        max_tokens:    Maximum completion tokens.
        system_prompt: Optional system-turn text prepended to messages.
        json_mode:     If True, request JSON object output.
        role:          Agent role — selects per-role model + thinking from registry.
        enable_thinking: Deprecated. Maps to role thinking when provided without
                         changing role defaults (True → medium, False → off for local).
        messages:      Optional native chat message list (``role``/``content``
                       dicts). Content may be a string or an OpenAI-compatible
                       multimodal content-part list. Prefer this over flattened
                       single-turn prompts:
                       it preserves role semantics and keeps the rendered prompt
                       append-only so server-side prefix caches stay hot.

    Returns:
        LLMResult with text, timing, token counts, and resolved model id.

    Raises:
        RuntimeError: If all models in the chain fail.
    """
    candidates = _FALLBACK_CHAIN if model == "auto" else [model]
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
    """Execute a single streaming call to the given model backend."""
    cfg = resolve_model_config(model_key, role)

    # Legacy enable_thinking override (local only meaningful path)
    thinking_level = cfg.thinking_level
    if enable_thinking_override is not None and model_key == "local":
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

    if model_key == "local" and thinking_level in ("off", "minimal"):
        for msg in chat_messages:
            if msg["role"] == "user":
                msg["content"] = _local_user_content(msg["content"], thinking_level)
                break

    kwargs: dict[str, Any] = dict(
        model=cfg.model_name,
        messages=chat_messages,
        temperature=_role_temperature(role),
        top_p=_role_top_p(role),
        seed=SEED,
        max_tokens=max_tokens,
        stream=True,
        stream_options={"include_usage": True},
    )
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    # llama.cpp: /no_think prompt prefixes are NOT honoured by Qwen jinja
    # chat templates — the model still emits hidden reasoning_content that
    # consumes the completion budget (~270 tokens for a 7-token answer in
    # measurement). enable_thinking must be set as a chat template kwarg.
    if model_key == "local" and thinking_level == "off":
        kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}

    if model_key == "gemini" and thinking_level not in ("off",):
        kwargs["extra_body"] = _gemini_extra_body(thinking_level)

    for param in _UNSUPPORTED_PARAMS.get(model_key, set()):
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
