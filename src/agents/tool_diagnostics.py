"""Bounded, redacted diagnostics and failure fingerprints for tool calls."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

_MAX_OUTPUT = 1600
_SECRET_RE = re.compile(
    r"(?i)(bearer\s+|(?:api[_-]?key|token|secret|password|authorization)"
    r"\s*[=:]\s*|(?:sk|hf)_[A-Za-z0-9_-]{6,})[^\s,;\"']+"
)
_SESSION_PATH_RE = re.compile(r"/sessions/[^/\s]+")
_LINE_NUMBER_RE = re.compile(r"(?<=:)\d+(?=[:)])")


def redact_text(value: Any, limit: int = _MAX_OUTPUT) -> str:
    """Convert tool output to bounded text while masking common secrets."""
    text = str(value or "")
    text = _SECRET_RE.sub("[REDACTED]", text)
    if len(text) > limit:
        return text[-limit:]
    return text


def classify_tool_result(result: dict[str, Any]) -> str:
    """Classify a failed result without asking an LLM to interpret it."""
    if result.get("success", False):
        return "success"
    if result.get("policy_denied"):
        return "policy_denied"
    if result.get("timed_out"):
        return "timeout"
    category = str(result.get("error_category", "")).strip()
    if category:
        return category
    error = str(result.get("error", "")).lower()
    if "no active sandbox" in error:
        return "sandbox_unavailable"
    if "no such file" in error or "not found" in error:
        return "path_or_executable"
    if result.get("returncode") is not None:
        return "nonzero_exit"
    return "tool_error"


def _normalise_for_signature(value: Any) -> str:
    text = redact_text(value, limit=1000).lower()
    text = _SESSION_PATH_RE.sub("<session>", text)
    text = _LINE_NUMBER_RE.sub("<line>", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tool_failure_signature(
    tool_name: str,
    args: Any,
    result: dict[str, Any],
) -> str:
    """Return a stable short signature for repeated identical tool failures."""
    command = (
        args.get("script") or args.get("command") or args
        if isinstance(args, dict)
        else args
    )
    error = result.get("error") or result.get("stderr") or result.get("stdout")
    basis = "|".join((
        tool_name,
        classify_tool_result(result),
        _normalise_for_signature(command),
        _normalise_for_signature(error),
        str(result.get("returncode", "")),
    ))
    return hashlib.sha1(basis.encode("utf-8")).hexdigest()[:12]


def event_diagnostics(
    tool_name: str,
    args: dict[str, Any],
    result: dict[str, Any],
    duration_ms: float,
) -> dict[str, Any]:
    """Build a safe diagnostic payload for ``events.jsonl``."""
    payload: dict[str, Any] = {
        "error_category": classify_tool_result(result),
        "duration_ms": round(duration_ms, 2),
    }
    for key in (
        "returncode",
        "timed_out",
        "policy_denied",
        "execution_mode",
        "executor",
        "cwd",
    ):
        if key in result:
            payload[key] = result[key]

    if tool_name in {"run_shellscript", "run_pytest", "run_linter"}:
        payload["stdout_tail"] = redact_text(result.get("stdout", ""))
        payload["stderr_tail"] = redact_text(result.get("stderr", ""))

    if not result.get("success", False):
        error = result.get("error") or result.get("stderr") or result.get("stdout")
        if error:
            payload["error"] = redact_text(error)
        payload["failure_signature"] = tool_failure_signature(tool_name, args, result)

    return payload


def compact_tool_result(result: dict[str, Any], limit: int = 4000) -> str:
    """Serialize a result for the Worker while preserving useful diagnostics."""
    return json.dumps(result, indent=2, default=str)[:limit]
