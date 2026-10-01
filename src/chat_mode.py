"""User-facing chat modes — orthogonal to execution_route.

``execution_route`` (mission / hotfix / review) is *how* a build executes.
``chat_mode`` (ask / plan / build) is *whether* to converse, plan, or execute.

The model cannot switch modes from inside a prompt; only the API field can.
"""

from __future__ import annotations

import re
from typing import Any, Literal, Optional

ChatMode = Literal["ask", "plan", "build"]

CHAT_MODES: tuple[str, ...] = ("ask", "plan", "build")
DEFAULT_API_CHAT_MODE: ChatMode = "build"
DEFAULT_SESSION_CHAT_MODE: ChatMode = "ask"

# Whole-message matches only — "I want to build a todo app" is NOT an approval.
_APPROVE_PHRASES = frozenset({
    "build",
    "build it",
    "build this",
    "go",
    "go ahead",
    "lgtm",
    "looks good",
    "sounds good",
    "proceed",
    "do it",
    "yes",
    "ok",
    "okay",
    "ship it",
    "approve",
    "approved",
    "let's go",
    "lets go",
    "start",
    "start building",
    "run it",
})

_PUNCT_RE = re.compile(r"[.!?]+$")
_SPACE_RE = re.compile(r"\s+")


def normalize_chat_mode(value: Optional[str], default: str = DEFAULT_API_CHAT_MODE) -> ChatMode:
    raw = str(value or "").strip().lower()
    if raw in CHAT_MODES:
        return raw  # type: ignore[return-value]
    return default if default in CHAT_MODES else DEFAULT_API_CHAT_MODE  # type: ignore[return-value]


def is_plan_approval_phrase(text: str) -> bool:
    """True when the user message is a short, explicit approval of a draft plan."""
    cleaned = _SPACE_RE.sub(" ", _PUNCT_RE.sub("", (text or "").strip().lower())).strip()
    return cleaned in _APPROVE_PHRASES


def plan_awaiting_approval(plan: Optional[dict[str, Any]]) -> bool:
    """Pending milestones that have not been approved for execution yet."""
    if not isinstance(plan, dict):
        return False
    milestones = plan.get("milestones") or []
    if not isinstance(milestones, list) or not milestones:
        return False
    pending = any(
        str(m.get("status", "")).lower() != "completed"
        for m in milestones
        if isinstance(m, dict)
    )
    if not pending:
        return False
    state = str(plan.get("approval_state") or "").strip().lower()
    if state in ("approved", "executing", "completed", "dismissed"):
        return False
    return True


def format_plan_card_summary(plan: dict[str, Any]) -> str:
    """Short, user-facing plan listing rendered as markdown in the UI."""
    title = str(plan.get("title") or "Untitled plan").strip()
    lines = [f"**{title}**", ""]
    for ms in plan.get("milestones") or []:
        if not isinstance(ms, dict):
            continue
        ms_id = str(ms.get("id") or "?").strip()
        ms_title = str(ms.get("title") or "").strip()
        status = str(ms.get("status") or "pending").strip()
        lines.append(f"- **{ms_id}** — {ms_title} ({status})")
        description = str(ms.get("description") or "").strip()
        if description:
            lines.append(f"  {description[:240]}")
    constraints = plan.get("global_constraints") or []
    if constraints:
        lines.append("")
        lines.append("**Constraints**")
        for item in constraints[:8]:
            lines.append(f"- {item}")
    return "\n".join(lines).strip()


def plan_milestones_payload(plan: dict[str, Any]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for ms in plan.get("milestones") or []:
        if not isinstance(ms, dict):
            continue
        out.append({
            "id": str(ms.get("id") or ""),
            "title": str(ms.get("title") or ""),
            "description": str(ms.get("description") or "")[:400],
            "status": str(ms.get("status") or "pending"),
        })
    return out
