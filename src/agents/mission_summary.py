"""Deterministic mission recap for chat and run history."""

from __future__ import annotations

from typing import Any, Mapping, Optional


def _field(handoff: Any, name: str, default: Any = None) -> Any:
    if isinstance(handoff, Mapping):
        return handoff.get(name, default)
    return getattr(handoff, name, default)


def _truncate(text: str, limit: int = 400) -> str:
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "…"


def _last_actionable_error(error_log: list[str]) -> str:
    if not error_log:
        return ""
    raw = error_log[-1]
    if "Guidance:" in raw:
        _, guidance = raw.split("Guidance:", 1)
        return _truncate(guidance.strip())
    if raw.startswith("REPLAN requested:"):
        return _truncate(raw.removeprefix("REPLAN requested:").strip())
    if raw.startswith("Worker blocked:"):
        return _truncate(raw.removeprefix("Worker blocked:").strip())
    return _truncate(raw)


def build_mission_summary(
    *,
    title: str,
    status: str,
    milestones_passed: int,
    milestones_total: int,
    total_elapsed_ms: float,
    handoffs: list[Any],
    incomplete_milestone_ids: Optional[list[str]] = None,
) -> dict[str, Any]:
    """Aggregate milestone handoffs into a chat-friendly mission recap."""
    incomplete_milestone_ids = incomplete_milestone_ids or []
    all_files: set[str] = set()
    milestone_lines: list[str] = []

    for handoff in handoffs:
        files = _field(handoff, "files_modified", []) or []
        for path in files:
            if path:
                all_files.add(str(path))

        ms_id = _field(handoff, "milestone_id", "?")
        ms_title = _field(handoff, "title", "")
        verdict = _field(handoff, "verdict", "?")
        worker_summary = _field(handoff, "worker_summary", "")
        error_log = _field(handoff, "error_log", []) or []

        icon = {"PASS": "✓", "FAIL": "✗", "REPLAN": "↻", "BLOCKED": "⊘"}.get(str(verdict), "?")
        milestone_lines.append(f"[{icon}] {ms_id}: {ms_title} — {verdict}")

        if worker_summary and worker_summary not in {
            "Exhausted retries",
            "Validator requested plan negotiation.",
        }:
            milestone_lines.append(f"    Worker: {_truncate(str(worker_summary), 240)}")

        if verdict != "PASS":
            detail = _last_actionable_error(list(error_log or []))
            if detail:
                milestone_lines.append(f"    Detail: {detail}")

    failure_reason = _failure_reason(
        status=status,
        handoffs=handoffs,
        incomplete_milestone_ids=incomplete_milestone_ids,
    )

    lines = [
        f"{title}",
        f"Status: {status} · {milestones_passed}/{milestones_total} milestones passed · "
        f"{total_elapsed_ms / 1000:.1f}s",
    ]

    if all_files:
        lines.extend(["", "Files touched:"])
        lines.extend(f"  • {path}" for path in sorted(all_files))

    if milestone_lines:
        lines.extend(["", "What happened:"])
        lines.extend(f"  {line}" if not line.startswith("[") else f"  {line}" for line in milestone_lines)

    if failure_reason:
        lines.extend(["", "Why the mission stopped:", f"  {failure_reason}"])
    elif status == "completed":
        lines.extend(["", "Outcome:", "  All milestones passed validation."])

    return {
        "summary_text": "\n".join(lines),
        "failure_reason": failure_reason,
        "files_modified": sorted(all_files),
    }


def _failure_reason(
    *,
    status: str,
    handoffs: list[Any],
    incomplete_milestone_ids: list[str],
) -> str:
    if status == "completed":
        return ""

    if not handoffs:
        return "The mission ended before any milestone completed."

    last = handoffs[-1]
    verdict = _field(last, "verdict", "")
    ms_id = _field(last, "milestone_id", "?")
    ms_title = _field(last, "title", "")
    error_log = _field(last, "error_log", []) or []

    if verdict == "FAIL":
        detail = _last_actionable_error(list(error_log or []))
        base = f"Milestone {ms_id} ({ms_title}) failed after retries were exhausted."
        return f"{base} {detail}".strip()

    if verdict == "REPLAN":
        detail = _last_actionable_error(list(error_log or []))
        base = f"Milestone {ms_id} ({ms_title}) requested a plan change the runtime could not resolve."
        return f"{base} {detail}".strip()

    if verdict == "BLOCKED":
        worker_summary = _field(last, "worker_summary", "")
        return f"Milestone {ms_id} ({ms_title}) was blocked: {_truncate(str(worker_summary), 320)}"

    if incomplete_milestone_ids:
        remaining = ", ".join(incomplete_milestone_ids)
        return f"The mission halted with unfinished milestones: {remaining}."

    if status == "partial":
        return "Some milestones passed, but the mission did not finish the full plan."

    return "The mission did not complete successfully."
