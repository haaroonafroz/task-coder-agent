"""
LLM client abstraction with precise per-phase timing.

Wraps any OpenAI-compatible endpoint (llama-server or vLLM) and captures:
  - prefill_ms       : time from request submission to first token
  - decode_ms        : time from first token to last token
  - total_ms         : prefill + decode
  - tokens_generated : completion_tokens from usage
  - text             : the generated string

Both baseline and speculative modes share this single interface.
The only difference between modes is which endpoint URL is used.

Endpoint URLs are read from .env:
  LLM_BASELINE_URL    (default: http://localhost:8000/v1)
  LLM_SPECULATIVE_URL (default: http://localhost:8001/v1)

TARGET_MODEL must match the --alias set in the llama-server launch script.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from openai import OpenAI

# ---------------------------------------------------------------------------
# Load .env from repo root (silently a no-op if file is absent)
# ---------------------------------------------------------------------------
try:
    from dotenv import load_dotenv
    _env_path = Path(__file__).parent.parent / ".env"
    load_dotenv(dotenv_path=_env_path, override=False)
except ImportError:
    pass  # python-dotenv not installed; rely on shell-exported HF_TOKEN

_hf_token = os.getenv("HF_TOKEN", "")
if _hf_token:
    os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", _hf_token)
    os.environ.setdefault("HF_TOKEN", _hf_token)

# ---------------------------------------------------------------------------
# Endpoint registry – overridable via .env
# LLM_BASELINE_URL / LLM_SPECULATIVE_URL are the canonical names.
# VLLM_BASELINE_URL / VLLM_SPECULATIVE_URL are accepted as fallbacks for
# backward compatibility with the vLLM branch .env files.
# ---------------------------------------------------------------------------
ENDPOINTS: dict[str, str] = {
    "baseline": (
        os.getenv("LLM_BASELINE_URL")
        or os.getenv("VLLM_BASELINE_URL")
        or "http://localhost:8000/v1"
    ),
    "speculative": (
        os.getenv("LLM_SPECULATIVE_URL")
        or os.getenv("VLLM_SPECULATIVE_URL")
        or "http://localhost:8001/v1"
    ),
}

# Model IDs (overridable via .env)
TARGET_MODEL: str = os.getenv("TARGET_MODEL", "google/gemma-4-E4B-it")

# Frozen sampling parameters (identical across both modes, per Section 7.3)
TEMPERATURE = 0.0
TOP_P = 0.95  # as mentioned in the gemma official docs
SEED = 42


@dataclass
class LLMResult:
    text: str
    prefill_ms: float
    decode_ms: float
    total_ms: float
    tokens_generated: int


def call_llm(
    prompt: str,
    mode: str,
    max_tokens: int,
    model_name: str = TARGET_MODEL,
    system_prompt: Optional[str] = None,
) -> LLMResult:
    """
    Submit a prompt to the vLLM server and return the result with timing.

    Timing strategy (external measurement):
      T0 = before request
      T1 = time of first token   → prefill = T1 - T0
      T2 = time of last token    → decode  = T2 - T1

    We use the streaming API to detect the first token precisely.
    Non-streaming requests only give total wall time, which conflates
    prefill with decode — unacceptable for this analysis.
    """
    base_url = ENDPOINTS[mode]
    client = OpenAI(api_key="EMPTY", base_url=base_url)

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    t0 = time.perf_counter()
    t_first_token: Optional[float] = None
    chunks: list[str] = []
    total_completion_tokens = 0

    stream = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        seed=SEED,
        max_tokens=max_tokens,
        stream=True,
        stream_options={"include_usage": True},
    )

    for chunk in stream:
        if t_first_token is None and chunk.choices and chunk.choices[0].delta.content:
            t_first_token = time.perf_counter()
        if chunk.choices and chunk.choices[0].delta.content:
            chunks.append(chunk.choices[0].delta.content)
        # vLLM emits usage in the final chunk via stream_options
        if chunk.usage is not None:
            total_completion_tokens = chunk.usage.completion_tokens

    t2 = time.perf_counter()

    # Fall back if first token was never recorded (empty output)
    if t_first_token is None:
        t_first_token = t2

    prefill_ms = (t_first_token - t0) * 1000.0
    decode_ms = (t2 - t_first_token) * 1000.0
    total_ms = (t2 - t0) * 1000.0
    text = "".join(chunks)

    return LLMResult(
        text=text,
        prefill_ms=round(prefill_ms, 2),
        decode_ms=round(decode_ms, 2),
        total_ms=round(total_ms, 2),
        tokens_generated=total_completion_tokens,
    )
