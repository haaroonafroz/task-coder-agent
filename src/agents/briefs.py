"""Session-scoped Ask/Plan/Build briefs — compact memory, not transcripts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from src.session import SessionContext

MAX_BRIEF_CHARS = 4000
MAX_LIST_ITEMS = 12
MAX_ITEM_CHARS = 280

CONVERSATION_BRIEF_NAME = "conversation_brief.json"
MISSION_BRIEF_NAME = "mission_brief.json"

_LIST_KEYS = ("preferences", "constraints", "decisions", "open_questions")


def empty_conversation_brief() -> dict[str, Any]:
    return {
        "user_goal": "",
        "preferences": [],
        "constraints": [],
        "decisions": [],
        "open_questions": [],
    }


def conversation_brief_path(session: SessionContext) -> Path:
    return session.state_root / CONVERSATION_BRIEF_NAME


def mission_brief_path(session: SessionContext) -> Path:
    return session.state_root / MISSION_BRIEF_NAME


def _cap_text(value: Any, limit: int) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "…"


def _cap_list(value: Any) -> list[str]:
    if isinstance(value, str) and value.strip():
        value = [value]
    if not isinstance(value, list):
        return []
    items: list[str] = []
    for raw in value:
        text = _cap_text(raw, MAX_ITEM_CHARS)
        if text and text not in items:
            items.append(text)
        if len(items) >= MAX_LIST_ITEMS:
            break
    return items


def normalize_conversation_brief(payload: Optional[dict[str, Any]]) -> dict[str, Any]:
    src = payload if isinstance(payload, dict) else {}
    brief = empty_conversation_brief()
    brief["user_goal"] = _cap_text(src.get("user_goal"), 600)
    for key in _LIST_KEYS:
        brief[key] = _cap_list(src.get(key))
    return enforce_brief_budget(brief)


def enforce_brief_budget(brief: dict[str, Any]) -> dict[str, Any]:
    """Drop oldest list tails until the JSON stays under MAX_BRIEF_CHARS."""
    out = dict(brief)
    encoded = json.dumps(out, ensure_ascii=False)
    while len(encoded) > MAX_BRIEF_CHARS:
        trimmed = False
        for key in _LIST_KEYS:
            items = list(out.get(key) or [])
            if items:
                out[key] = items[:-1]
                trimmed = True
                break
        if not trimmed:
            out["user_goal"] = _cap_text(out.get("user_goal"), max(40, len(str(out.get("user_goal", ""))) // 2))
            encoded = json.dumps(out, ensure_ascii=False)
            if len(encoded) <= MAX_BRIEF_CHARS:
                break
            out["user_goal"] = _cap_text(out.get("user_goal"), 80)
            break
        encoded = json.dumps(out, ensure_ascii=False)
    return out


def load_conversation_brief(session: SessionContext) -> dict[str, Any]:
    path = conversation_brief_path(session)
    if not path.exists():
        return empty_conversation_brief()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return empty_conversation_brief()
    return normalize_conversation_brief(data if isinstance(data, dict) else None)


def save_conversation_brief(session: SessionContext, brief: dict[str, Any]) -> dict[str, Any]:
    normalized = normalize_conversation_brief(brief)
    path = conversation_brief_path(session)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(normalized, indent=2, ensure_ascii=False), encoding="utf-8")
    return normalized


def load_mission_brief(session: SessionContext) -> dict[str, Any]:
    path = mission_brief_path(session)
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def save_mission_brief(session: SessionContext, brief: dict[str, Any]) -> dict[str, Any]:
    path = mission_brief_path(session)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = brief if isinstance(brief, dict) else {}
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


def conversation_brief_is_empty(brief: dict[str, Any]) -> bool:
    if str(brief.get("user_goal") or "").strip():
        return False
    return not any(brief.get(key) for key in _LIST_KEYS)


def format_conversation_brief(brief: dict[str, Any]) -> str:
    if conversation_brief_is_empty(brief):
        return ""
    lines: list[str] = []
    goal = str(brief.get("user_goal") or "").strip()
    if goal:
        lines.append(f"Goal: {goal}")
    for label, key in (
        ("Preferences", "preferences"),
        ("Constraints", "constraints"),
        ("Decisions", "decisions"),
        ("Open questions", "open_questions"),
    ):
        items = brief.get(key) or []
        if not items:
            continue
        lines.append(f"{label}:")
        lines.extend(f"  - {item}" for item in items)
    return "\n".join(lines)


def format_mission_brief(brief: dict[str, Any]) -> str:
    if not brief:
        return ""
    lines: list[str] = []
    title = str(brief.get("title") or "").strip()
    status = str(brief.get("status") or "").strip()
    if title:
        header = title if not status else f"{title} ({status})"
        lines.append(header)
    summary = str(brief.get("summary_text") or "").strip()
    if summary:
        lines.append(summary[:2000])
    files = brief.get("files_modified") or []
    if files:
        lines.append("Files touched: " + ", ".join(str(p) for p in files[:40]))
    return "\n".join(lines)


def build_mission_brief_from_run(
    *,
    plan: Optional[dict[str, Any]],
    status: str,
    summary_text: str,
    files_modified: Optional[list[str]] = None,
) -> dict[str, Any]:
    plan = plan if isinstance(plan, dict) else {}
    milestones = []
    for ms in plan.get("milestones") or []:
        if not isinstance(ms, dict):
            continue
        milestones.append({
            "id": str(ms.get("id") or ""),
            "title": str(ms.get("title") or ""),
            "status": str(ms.get("status") or ""),
        })
    return {
        "plan_id": str(plan.get("plan_id") or plan.get("mission_id") or ""),
        "title": str(plan.get("title") or ""),
        "status": status,
        "milestones": milestones,
        "files_modified": list(files_modified or []),
        "summary_text": _cap_text(summary_text, 2000),
    }
