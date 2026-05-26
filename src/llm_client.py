"""
Multi-model LLM client — Missions Framework.

Priority chain:
  1. Local  — Qwen3.6-27B-GGUF via llama.cpp server (primary, hardware-optimal)
  2. Gemini — google/gemini-2.5-pro via OpenAI-compatible Gemini endpoint
  3. GPT-4o — gpt-4o via OpenAI API

Only one model is active per call.  The runtime selects the model based on the
`model` parameter.  "local" attempts the llama.cpp server; if it is unreachable
the call automatically falls back to "gemini", then "gpt4o".

Environment variables (can be set in .env):
  LOCAL_LLM_URL    — base URL for llama.cpp server  (default: http://localhost:8000/v1)
  LOCAL_MODEL_NAME — model alias set in llama-server (default: qwen3-27b)
  GEMINI_API_KEY   — Google AI Studio key
  OPENAI_API_KEY   — OpenAI API key
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Literal

from openai import OpenAI, APIConnectionError, APIStatusError

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / ".env", override=False)
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Endpoint + model configuration
# ---------------------------------------------------------------------------
ModelChoice = Literal["local", "gemini", "gpt4o", "auto"]

_ENDPOINTS: dict[str, dict] = {
    "local": {
        "base_url": os.getenv("LLM_SPECULATIVE_URL", "http://localhost:8001/v1"),
        "api_key": "EMPTY",
        "model": os.getenv("TARGET_MODEL", "qwen3-27b"),
    },
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "api_key": os.getenv("GEMINI_API_KEY", ""),
        "model": os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite"),
    },
    "gpt4o": {
        "base_url": "https://api.openai.com/v1",
        "api_key": os.getenv("OPENAI_API_KEY", ""),
        "model": os.getenv("OPENAI_LLM_MODEL", "gpt-4o"),
    },
}

# Parameters not accepted by each backend's OpenAI-compatible endpoint.
_UNSUPPORTED_PARAMS: dict[str, set[str]] = {
    "local":  set(),
    "gemini": {"seed", "stream_options"},
    "gpt4o":  set(),
}

# Fallback order when model="auto" or local is unreachable
_FALLBACK_CHAIN: list[str] = ["local", "gemini", "gpt4o"]

TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.8"))
TOP_P = float(os.getenv("LLM_TOP_P", "0.95"))
SEED = int(os.getenv("LLM_SEED", "42"))


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


def _build_client(model_key: str) -> tuple[OpenAI, str]:
    """Return an (OpenAI client, model_name) pair for the given model key."""
    cfg = _ENDPOINTS[model_key]
    client = OpenAI(api_key=cfg["api_key"] or "EMPTY", base_url=cfg["base_url"])
    return client, cfg["model"]


def call_llm(
    prompt: str,
    model: ModelChoice = "auto",
    max_tokens: int = 2048,
    system_prompt: Optional[str] = None,
    json_mode: bool = False,
    enable_thinking: bool = False,
) -> LLMResult:
    """
    Send a prompt to the selected LLM and return the result with precise timing.

    Timing uses the streaming API to separate prefill from decode latency:
      T0 = before request
      T1 = time of first token  → prefill = T1 - T0
      T2 = time of last token   → decode  = T2 - T1

    Args:
        prompt:        User-turn prompt text.
        model:         "local" | "gemini" | "gpt4o" | "auto".
                       "auto" tries the fallback chain in order.
        max_tokens:    Maximum completion tokens.
        system_prompt: Optional system-turn text prepended to messages.
        json_mode:     If True, request JSON object output (supported by all three).

    Returns:
        LLMResult with text, timing, and token counts.

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
                max_tokens=max_tokens,
                system_prompt=system_prompt,
                json_mode=json_mode,
                fallback_used=(attempt > 0),
                enable_thinking=enable_thinking,
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
    prompt: str,
    model_key: str,
    max_tokens: int,
    system_prompt: Optional[str],
    json_mode: bool,
    fallback_used: bool,
    enable_thinking: bool,
) -> LLMResult:
    """Execute a single streaming call to the given model backend."""
    client, model_name = _build_client(model_key)

    # Qwen3.6 thinking mode implementation
    # separate implementation for Gemma4 models
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    user_content = prompt
    if model_key == "local":
        if enable_thinking:
            user_content = prompt
        else:
            user_content = "/no_think\n\n" + prompt
    messages.append({"role": "user", "content": user_content})

    kwargs: dict = dict(
        model=model_name,
        messages=messages,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        seed=SEED,
        max_tokens=max_tokens,
        stream=True,
        stream_options={"include_usage": True},
    )
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    # pop unsupported params
    for param in _UNSUPPORTED_PARAMS.get(model_key, set()):
        kwargs.pop(param, None)

    t0 = time.perf_counter()
    t_first: Optional[float] = None
    chunks: list[str] = []
    prompt_tokens = 0
    completion_tokens = 0

    stream = client.chat.completions.create(**kwargs)
    for chunk in stream:
        if t_first is None and chunk.choices and chunk.choices[0].delta.content:
            t_first = time.perf_counter()
        if chunk.choices and chunk.choices[0].delta.content:
            chunks.append(chunk.choices[0].delta.content)
        if chunk.usage is not None:
            prompt_tokens = chunk.usage.prompt_tokens or 0
            completion_tokens = chunk.usage.completion_tokens or 0

    t2 = time.perf_counter()
    if t_first is None:
        t_first = t2

    prefill_ms = (t_first - t0) * 1000.0
    decode_ms = (t2 - t_first) * 1000.0
    total_ms = (t2 - t0) * 1000.0

    if fallback_used:
        print(f"[LLMClient] Using fallback model: {model_key} ({model_name})")

    return LLMResult(
        text="".join(chunks),
        model_used=f"{model_key}/{model_name}",
        prefill_ms=round(prefill_ms, 2),
        decode_ms=round(decode_ms, 2),
        total_ms=round(total_ms, 2),
        tokens_generated=completion_tokens,
        tokens_prompt=prompt_tokens,
        fallback_used=fallback_used,
    )
