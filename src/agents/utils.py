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
        or "<tool_call>" in lowered
        or "<function=" in lowered
    )


def _strip_thinking_markers(text: str) -> str:
    """Remove leaked Qwen thinking blocks before parsing agent output."""
    cleaned = re.sub(
        r"<(?:think|redacted_reasoning)>.*?</(?:think|redacted_reasoning)>",
        "",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    return cleaned.strip()


def _normalize_tool_call_dict(
    tool: str,
    args: Any,
    *,
    reasoning: str = "",
) -> dict[str, Any]:
    return {
        "tool": tool.strip(),
        "args": args if isinstance(args, dict) else {},
        "reasoning": reasoning.strip(),
    }


def _parse_xml_parameters(inner: str) -> dict[str, Any]:
    args: dict[str, Any] = {}
    if not inner:
        return args
    for match in re.finditer(
        r"<parameter=([^>\s]+)>\s*(.*?)\s*</parameter>",
        inner,
        flags=re.DOTALL | re.IGNORECASE,
    ):
        args[match.group(1).strip()] = match.group(2).strip()
    if args:
        return args
    stripped = inner.strip()
    if stripped.startswith("{"):
        payload = _load_json_candidate(stripped, repair=True)
        if isinstance(payload, dict):
            value = payload.get("arguments", payload.get("args", payload))
            return value if isinstance(value, dict) else {}
    return args


def parse_xml_tool_calls(text: str) -> Optional[dict[str, Any]]:
    """
    Parse Qwen-style XML tool blocks into the harness JSON tool-call shape.

    Supports:
      - ``<tool_call><function=git_diff></function></tool_call>``
      - ``<tool_call>{"name":"read_file","arguments":{...}}</tool_call>``
      - ``<tool_call><function=read_file><parameter=file_path>app.py</parameter></function></tool_call>``
    """
    calls: list[dict[str, Any]] = []
    blocks = re.findall(
        r"<tool_call>\s*(.*?)\s*</tool_call>",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if blocks:
        for block in blocks:
            block = block.strip()
            if not block:
                continue
            if block.startswith("{"):
                payload = _load_json_candidate(block, repair=True)
                if isinstance(payload, dict):
                    tool = payload.get("tool") or payload.get("name")
                    args = payload.get("args", payload.get("arguments", {}))
                    if tool:
                        calls.append(
                            _normalize_tool_call_dict(
                                str(tool),
                                args,
                            )
                        )
                        continue
            fn_match = re.search(
                r"<function=([^>\s/]+)(?:\s[^>]*)?>\s*(.*?)\s*</function>",
                block,
                flags=re.DOTALL | re.IGNORECASE,
            )
            if fn_match:
                calls.append(
                    _normalize_tool_call_dict(
                        fn_match.group(1),
                        _parse_xml_parameters(fn_match.group(2)),
                    )
                )
                continue
            bare = re.match(r"<function=([^>/\s]+)\s*/?>", block, re.IGNORECASE)
            if bare:
                calls.append(_normalize_tool_call_dict(bare.group(1), {}))
    else:
        for match in re.finditer(
            r"<function=([^>\s/]+)(?:\s[^>]*)?>\s*(.*?)\s*</function>",
            text,
            flags=re.DOTALL | re.IGNORECASE,
        ):
            calls.append(
                _normalize_tool_call_dict(
                    match.group(1),
                    _parse_xml_parameters(match.group(2)),
                )
            )

    if not calls:
        return None
    if len(calls) == 1:
        return calls[0]
    return {"calls": calls}


def parse_agent_turn(text: str) -> Optional[dict]:
    """
    Parse one agent turn from JSON or local-model XML tool-call output.

    Tries strict/recovered JSON first, then Qwen-style ``<tool_call>`` blocks.
    """
    cleaned = _strip_thinking_markers(text)
    parsed = parse_json_from_text(cleaned)
    if parsed is not None:
        return parsed
    xml_parsed = parse_xml_tool_calls(cleaned)
    if xml_parsed is not None:
        print("    [Parser] Recovered XML/Qwen tool-call format.")
        return xml_parsed
    return None


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


def validate_plan_payload(parsed: dict) -> Optional[str]:
    """Return an error string when a plan dict is invalid, else None."""
    milestones = parsed.get("milestones")
    if not isinstance(milestones, list) or not milestones:
        return "plan must contain a non-empty 'milestones' list"
    for i, ms in enumerate(milestones):
        if not isinstance(ms, dict) or not ms.get("id"):
            return f"milestone #{i + 1} is missing an 'id'"
        if not ms.get("target_files"):
            return f"milestone '{ms.get('id')}' must list 'target_files'"
        criteria = ms.get("acceptance_criteria")
        if (
            not isinstance(criteria, list)
            or not criteria
            or any(not isinstance(item, str) or not item.strip() for item in criteria)
        ):
            return (
                f"milestone '{ms.get('id')}' must include a non-empty "
                "'acceptance_criteria' list of non-empty strings"
            )
        profile = str(ms.get("validation_profile", "auto")).lower()
        if profile not in {"auto", "ui", "python", "lint", "structural"}:
            return (
                f"milestone '{ms.get('id')}' has unsupported "
                f"validation_profile '{profile}'"
            )
        if "validation_contract" in ms:
            return (
                f"milestone '{ms.get('id')}' must not include "
                "'validation_contract'; provide high-level intent only"
            )
    return None


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
