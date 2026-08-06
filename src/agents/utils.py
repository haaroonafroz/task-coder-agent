"""
Shared utilities for all agent phases.

Contains JSON parsing, conversation management, and filesystem helpers
used by orchestrator, worker, and validator.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Optional


# ---------------------------------------------------------------------------
# JSON parsing
# ---------------------------------------------------------------------------

def _looks_like_tool_call_attempt(text: str) -> bool:
    """Heuristic: model tried a tool call but JSON did not parse."""
    lowered = text.lower()
    return (
        '"tool"' in lowered
        or '"write_file"' in lowered
        or '"patch_file"' in lowered
        or '"read_file"' in lowered
    )


def _load_json_candidate(text: str, *, repair: bool = False) -> Optional[Any]:
    """
    Parse one JSON candidate string.

    If repair=True, attempts recovery via json-repair before giving up.
    json-repair is an optional dependency; failure is silently ignored.
    """
    if repair:
        try:
            import json_repair
            return json_repair.loads(text)
        except Exception:
            return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def parse_json_from_text(text: str) -> Optional[dict]:
    """
    Extract and parse the first JSON object from a text string.

    Tries three candidate extraction strategies in order:
      1. The full stripped text (model emitted clean JSON)
      2. A fenced ```json ... ``` block
      3. A greedy {…} match

    For each candidate, first tries strict json.loads, then json-repair.
    """
    candidates: list[str] = []
    stripped = text.strip()
    if stripped:
        candidates.append(stripped)

    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence_match:
        candidates.append(fence_match.group(1))

    greedy_match = re.search(r"\{.*\}", text, re.DOTALL)
    if greedy_match:
        candidates.append(greedy_match.group())

    seen: set[str] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)

        parsed = _load_json_candidate(candidate, repair=False)
        if isinstance(parsed, dict):
            return parsed

        parsed = _load_json_candidate(candidate, repair=True)
        if isinstance(parsed, dict):
            print("    [Parser] Recovered malformed JSON via json-repair.")
            return parsed

    return None


# ---------------------------------------------------------------------------
# Conversation helpers
# ---------------------------------------------------------------------------

def flatten_conversation(conversation: list[dict]) -> str:
    """Serialise a multi-turn conversation into a single prompt string."""
    parts = []
    for msg in conversation:
        role = msg["role"].upper()
        parts.append(f"[{role}]\n{msg['content']}")
    return "\n\n".join(parts)


def trim_conversation(conversation: list[dict], max_turns: int = 12) -> list[dict]:
    """
    Trim conversation history to prevent prompt token bloat.

    Always preserves the first message (full milestone brief) and retains
    the most recent (max_turns - 1) messages so immediate context is intact.
    """
    if len(conversation) <= max_turns:
        return conversation
    return [conversation[0]] + conversation[-(max_turns - 1):]


# ---------------------------------------------------------------------------
# Failure fingerprinting (replan dedup / repeated-error circuit breaker)
# ---------------------------------------------------------------------------

_ERROR_LINE_KEYS = (
    "error", "assert", "failed", "exception", "not found", "denied",
    "no module", "traceback", "exit code", "returncode",
)


def failure_signature(
    milestone_id: str,
    command: str,
    contract_output: str,
    returncode: Optional[int],
) -> str:
    """
    Stable fingerprint of a validation failure.

    LLM verdicts paraphrase their guidance every time, so text comparison
    cannot dedup repeated replan loops. This fingerprint keys on the
    deterministic facts — milestone, contract command, exit code, and the
    first error-looking output line — normalised to strip file paths, line
    numbers, and digits so the SAME underlying failure always maps to the
    SAME signature across retries and replans.
    """
    err_line = ""
    for line in (contract_output or "").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if any(key in stripped.lower() for key in _ERROR_LINE_KEYS):
            err_line = stripped
            break

    norm = re.sub(r"[\w./\\-]+\.py", "<file>", err_line.lower())
    norm = re.sub(r"\d+", "<n>", norm)
    norm = re.sub(r"\s+", " ", norm).strip()

    base = (
        f"{milestone_id}|rc={returncode}|"
        f"{(command or '').strip().lower()}|{norm}"
    )
    return hashlib.sha1(base.encode()).hexdigest()[:12]


# ---------------------------------------------------------------------------
# Filesystem helpers
# ---------------------------------------------------------------------------

def target_files_exist(target_files: list[str]) -> tuple[bool, list[str]]:
    """
    Check which target files are present on disk.

    Returns:
        (all_exist, missing_list) — missing_list is empty when all_exist is True.
    """
    from src.tools.paths import resolve_workspace_path
    missing = []
    for rel in target_files:
        p = resolve_workspace_path(rel)
        if not p.exists():
            missing.append(rel)
    return (len(missing) == 0, missing)
