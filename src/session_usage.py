"""Running per-session LLM token totals (prefill + generated)."""

from __future__ import annotations

import json
import os
import tempfile
import threading
from pathlib import Path
from typing import Any

_LOCK = threading.Lock()
META_LOCK = _LOCK


def empty_token_usage() -> dict[str, int]:
    return {
        "prompt": 0,
        "generated": 0,
        "calls": 0,
        "estimated_calls": 0,
    }


def merge_token_usage(*parts: dict[str, Any] | None) -> dict[str, int]:
    """Keep the higher running totals when combining in-memory and on-disk values."""
    merged = empty_token_usage()
    for part in parts:
        if not part:
            continue
        for key in merged:
            try:
                merged[key] = max(int(merged[key]), int(part.get(key) or 0))
            except (TypeError, ValueError):
                continue
    return merged


def record_session_tokens(
    meta_path: Path,
    *,
    prompt: int,
    generated: int,
    estimated: bool = False,
) -> dict[str, int]:
    """Atomically add one LLM call's tokens to ``session.json``."""
    prompt = max(0, int(prompt or 0))
    generated = max(0, int(generated or 0))
    with _LOCK:
        current = empty_token_usage()
        payload: dict[str, Any] = {}
        if meta_path.is_file():
            try:
                payload = json.loads(meta_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                payload = {}
            current = merge_token_usage(payload.get("token_usage"))
        current["prompt"] += prompt
        current["generated"] += generated
        current["calls"] += 1
        if estimated:
            current["estimated_calls"] += 1
        payload["token_usage"] = current
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(
            dir=str(meta_path.parent), suffix=".tmp", prefix="session_tokens_"
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, indent=2) + "\n")
            os.replace(tmp, meta_path)
        except Exception:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
        return current
