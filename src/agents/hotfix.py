"""Focused hotfix profile backed by the hardened implementation tool loop."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable, Optional

from src.agents.worker import run_worker
from src.agents.contracts import normalize_hotfix_result
from src.events import EventEmitter
from src.llm_client import ModelChoice
from src.telemetry import TelemetryContext

_CONFIG_DIR = Path(__file__).parent.parent.parent / "config"
_HOTFIX_MD = (_CONFIG_DIR / "hotfix.md").read_text(encoding="utf-8")

MAX_HOTFIX_TOOL_CALLS = int(os.getenv("MAX_HOTFIX_TOOL_CALLS", "10"))
MAX_TOKENS_HOTFIX = int(os.getenv("MAX_TOKENS_HOTFIX", "12288"))


def run_hotfix(
    *,
    milestone: dict[str, Any],
    plan: dict[str, Any],
    curated_tools_md: str,
    error_feedback: Optional[str],
    retry_count: int,
    model: ModelChoice,
    memory: Any,
    emitter: Optional[EventEmitter] = None,
    session: Optional[TelemetryContext] = None,
    cancel_check: Optional[Callable[[], bool]] = None,
    prior_conversation: Optional[list[dict]] = None,
    initial_tool_names: Optional[set[str]] = None,
    prior_active_tools: Optional[set[str]] = None,
    tool_searcher: Optional[Callable[[str, int], dict[str, Any]]] = None,
    prior_failure_state: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Run the Hotfix prompt/profile while reusing Worker safety machinery."""
    if emitter:
        emitter.emit(
            "hotfix.started",
            milestone_id=milestone.get("id", "HOTFIX"),
            retry=retry_count,
        )
    result = run_worker(
        milestone=milestone,
        plan=plan,
        curated_tools_md=curated_tools_md,
        error_feedback=error_feedback,
        retry_count=retry_count,
        model=model,
        memory=memory,
        emitter=emitter,
        session=session,
        cancel_check=cancel_check,
        prior_conversation=prior_conversation,
        initial_tool_names=initial_tool_names,
        prior_active_tools=prior_active_tools,
        tool_searcher=tool_searcher,
        prior_failure_state=prior_failure_state,
        agent_role="hotfix",
        system_prompt=_HOTFIX_MD,
        max_tool_calls=MAX_HOTFIX_TOOL_CALLS,
        max_tokens=MAX_TOKENS_HOTFIX,
    )
    result["hotfix_result"] = normalize_hotfix_result(result).to_dict()
    if emitter:
        emitter.emit(
            "hotfix.finished",
            milestone_id=milestone.get("id", "HOTFIX"),
            status=result.get("status", "blocked"),
            files_modified=result.get("files_modified", []),
        )
    return result
