"""Read-only defect triage for repair runs."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Optional

from src.agents.llm_stream_events import stream_context_for
from src.agents.utils import parse_json_from_text
from src.events import EventEmitter
from src.llm_client import ModelChoice, call_llm, resolve_model_config
from src.telemetry import TelemetryContext, span_llm_call

_ROOT = Path(__file__).parent.parent.parent
_TRIAGE_MD = (_ROOT / "config" / "triage.md").read_text(encoding="utf-8")
_MAX_TOKENS_TRIAGE = 8192
_MAX_SNAPSHOT_CHARS = 32000
_TEXT_SUFFIXES = {
    ".c",
    ".css",
    ".csv",
    ".go",
    ".html",
    ".ini",
    ".java",
    ".js",
    ".json",
    ".jsx",
    ".md",
    ".py",
    ".rs",
    ".sh",
    ".sql",
    ".toml",
    ".ts",
    ".tsx",
    ".txt",
    ".xml",
    ".yaml",
    ".yml",
}
_SKIP_DIRS = {
    ".git",
    ".pytest_cache",
    ".venv",
    "__pycache__",
    "node_modules",
}


def run_triage(
    user_request: str,
    *,
    workspace_root: Path,
    session_root: Path,
    model: ModelChoice,
    previous_plan: Optional[dict[str, Any]] = None,
    session: Optional[TelemetryContext] = None,
    emitter: Optional[EventEmitter] = None,
) -> dict[str, Any]:
    """Create a read-only, evidence-backed report for a repair run."""
    if emitter:
        emitter.emit("triage.started")

    prompt = _build_prompt(
        user_request,
        workspace_root=workspace_root,
        session_root=session_root,
        previous_plan=previous_plan,
    )
    span_model = (
        resolve_model_config(model, "triage").model_name
        if model != "auto"
        else model
    )
    last_error = "unknown"
    result_text = ""

    for attempt in range(2):
        with span_llm_call("triage", "repair", span_model, session=session):
            result = call_llm(
                prompt,
                model=model,
                max_tokens=_MAX_TOKENS_TRIAGE,
                json_mode=True,
                role="triage",
                stream_context=stream_context_for(
                    emitter,
                    "triage",
                    output_kind="json",
                ),
            )
        result_text = result.text
        parsed = parse_json_from_text(result.text)
        validation_error = _validate_report(parsed)
        if validation_error is None:
            report = parsed
            if emitter:
                emitter.emit(
                    "triage.completed",
                    confidence=report.get("confidence", "low"),
                    affected_files=report.get("affected_files", []),
                )
            return report

        last_error = validation_error
        prompt = (
            f"{prompt}\n\n"
            "## CORRECTION REQUIRED\n"
            f"Your previous response was rejected: {last_error}.\n"
            f"Previous response (truncated): {result_text[:1500]}\n"
            "Return only a valid JSON object matching the required triage schema."
        )

    if emitter:
        emitter.emit("triage.failed", error=last_error)
    raise RuntimeError(f"Triage produced an invalid report: {last_error}")


def _validate_report(parsed: Optional[dict[str, Any]]) -> Optional[str]:
    if not isinstance(parsed, dict):
        return "response was not a JSON object"
    if not isinstance(parsed.get("summary"), str) or not parsed["summary"].strip():
        return "summary must be a non-empty string"
    if not isinstance(parsed.get("evidence"), list):
        return "evidence must be a list"
    if not isinstance(parsed.get("affected_files"), list):
        return "affected_files must be a list"
    if not isinstance(parsed.get("regression_requirements"), list):
        return "regression_requirements must be a list"
    if parsed.get("confidence") not in {"high", "medium", "low"}:
        return "confidence must be high, medium, or low"
    return None


def _build_prompt(
    user_request: str,
    *,
    workspace_root: Path,
    session_root: Path,
    previous_plan: Optional[dict[str, Any]],
) -> str:
    return (
        f"{_TRIAGE_MD}\n\n---\n\n"
        f"## User's Repair Request\n{user_request}\n\n"
        f"## Session Conversation\n```\n"
        f"{_read_messages(session_root)}\n```\n\n"
        f"## Previous Plan\n```json\n"
        f"{json.dumps(previous_plan or {}, indent=2)[:12000]}\n```\n\n"
        f"## Recent Session Events\n```\n"
        f"{_read_recent_events(session_root)}\n```\n\n"
        f"## Current Workspace Snapshot\n"
        f"{_workspace_snapshot(workspace_root, previous_plan, user_request)}\n\n"
        "Produce the triage JSON now."
    )


def _read_recent_events(session_root: Path) -> str:
    path = session_root / "events.jsonl"
    try:
        return "\n".join(path.read_text(encoding="utf-8").splitlines()[-80:])[-12000:]
    except OSError:
        return "(events unavailable)"


def _read_messages(session_root: Path) -> str:
    path = session_root / "messages.jsonl"
    try:
        return "\n".join(path.read_text(encoding="utf-8").splitlines()[-20:])[-10000:]
    except OSError:
        return "(messages unavailable)"


def _workspace_snapshot(
    workspace_root: Path,
    previous_plan: Optional[dict[str, Any]],
    user_request: str,
) -> str:
    candidates = _candidate_paths(workspace_root, previous_plan, user_request)
    if not candidates:
        candidates = _workspace_files(workspace_root)[:20]

    sections: list[str] = []
    total = 0
    for path in candidates:
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        rel = path.relative_to(workspace_root)
        section = f"### {rel}\n```\n{text[:8000]}\n```\n"
        if total + len(section) > _MAX_SNAPSHOT_CHARS:
            break
        sections.append(section)
        total += len(section)

    files = "\n".join(
        f"- {path.relative_to(workspace_root)}"
        for path in _workspace_files(workspace_root)[:80]
    )
    return (
        "### Workspace files\n"
        f"{files or '(empty)'}\n\n"
        "### Selected file contents\n"
        f"{''.join(sections) or '(no readable text files)'}"
    )


def _candidate_paths(
    workspace_root: Path,
    previous_plan: Optional[dict[str, Any]],
    user_request: str,
) -> list[Path]:
    candidates: list[Path] = []
    plan = previous_plan or {}
    for raw in re.findall(r"""File\s+["']([^"']+)["']""", user_request):
        path = _safe_workspace_path(workspace_root, raw)
        if path is not None and path.exists() and path not in candidates:
            candidates.append(path)
    for milestone in plan.get("milestones", []):
        for raw in milestone.get("target_files", []):
            path = _safe_workspace_path(workspace_root, str(raw))
            if path is not None and path.exists() and path not in candidates:
                candidates.append(path)

    # Tracebacks commonly contain absolute workspace paths. Include those
    # files without allowing a report to escape the session workspace.
    return candidates + [
        path
        for path in _workspace_files(workspace_root)
        if path not in candidates
    ]


def _workspace_files(workspace_root: Path) -> list[Path]:
    if not workspace_root.exists():
        return []
    paths: list[Path] = []
    for path in workspace_root.rglob("*"):
        if not path.is_file() or any(part in _SKIP_DIRS for part in path.parts):
            continue
        if path.suffix.lower() in _TEXT_SUFFIXES:
            paths.append(path)
    return sorted(paths, key=lambda path: str(path))


def _safe_workspace_path(workspace_root: Path, raw: str) -> Optional[Path]:
    candidate = Path(raw)
    if candidate.is_absolute():
        try:
            candidate.relative_to(workspace_root)
        except ValueError:
            return None
        return candidate
    try:
        resolved = (workspace_root / candidate).resolve()
        resolved.relative_to(workspace_root.resolve())
    except ValueError:
        return None
    return resolved
