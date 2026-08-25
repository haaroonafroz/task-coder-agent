"""Read-only Code Review agent profile."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Callable, Optional

from src.agents.contracts import ReviewReport, normalize_review_report
from src.agents.llm_stream_events import stream_context_for
from src.agents.orchestrator_explore import build_workspace_orientation
from src.agents.tool_diagnostics import compact_tool_result, event_diagnostics
from src.agents.utils import parse_agent_turn, trim_conversation
from src.events import EventEmitter
from src.llm_client import ModelChoice, call_llm, resolve_model_config
from src.run_control import ensure_not_cancelled
from src.telemetry import TelemetryContext, span_llm_call, span_tool_call
from src.tools import dispatch
from src.tools.tool_contracts import validate_tool_call

_CONFIG_DIR = Path(__file__).parent.parent.parent / "config"
_CODE_REVIEW_MD = (_CONFIG_DIR / "code_review.md").read_text(encoding="utf-8")

_REVIEW_TOOLS = frozenset({
    "project_info",
    "list_directory",
    "search_grep",
    "read_file",
    "git_diff",
    "view_git_log",
    "run_checks",
})

MAX_REVIEW_TOOL_CALLS = int(os.getenv("MAX_REVIEW_TOOL_CALLS", "12"))
MAX_TOKENS_REVIEWER = int(os.getenv("MAX_TOKENS_REVIEWER", "16384"))
_NON_JSON_RETRIES = 4
_MAX_BATCH = 3

_TOOLS_MD = """\
## Read-only tools

Use only: project_info, list_directory, search_grep, read_file, git_diff,
view_git_log, run_checks.

Search before reading large files. You cannot write, patch, install, commit, or
start services.

### Tool call format (required)

Emit EXACTLY ONE JSON object per turn. No XML tags, no markdown fences, no prose.

Single tool:
{"tool":"git_diff","args":{},"reasoning":"Inspect current changes first."}

Batch (up to 3):
{"calls":[
  {"tool":"git_diff","args":{},"reasoning":"Inspect current changes."},
  {"tool":"read_file","args":{"file_path":"app.py"},"reasoning":"Read changed file."}
]}

Finish with:
{"action":"review","report":{"verdict":"clean","summary":"...","scope":"...","findings":[]}}
"""


def run_code_review(
    *,
    user_request: str,
    workspace_root: Path,
    model: ModelChoice,
    previous_plan: Optional[dict[str, Any]] = None,
    verification_of: Optional[dict[str, Any]] = None,
    emitter: Optional[EventEmitter] = None,
    session: Optional[TelemetryContext] = None,
    cancel_check: Optional[Callable[[], bool]] = None,
) -> ReviewReport:
    """Inspect code with read-only tools and return a normalized report."""
    orientation = build_workspace_orientation(
        workspace_root, user_request, previous_plan
    )
    phase = "review_verify" if verification_of else "review"
    prompt = (
        f"{orientation}\n\n"
        f"## User request\n{user_request}\n\n"
        f"{_TOOLS_MD}\n"
    )
    if verification_of:
        prompt += (
            "## Findings to verify after an attempted fix\n"
            f"```json\n{json.dumps(verification_of, indent=2)[:12000]}\n```\n"
            "Re-check only these findings. Report clean only if their fix criteria "
            "are now satisfied.\n"
        )
    prompt += (
        'Begin with read-only inspection. Finish with '
        '{"action":"review","report":{...}}.'
    )
    conversation: list[dict[str, Any]] = [{"role": "user", "content": prompt}]
    tool_calls = 0
    invalid_json = 0

    if emitter:
        emitter.emit("review.started", phase=phase)

    while tool_calls < MAX_REVIEW_TOOL_CALLS:
        ensure_not_cancelled(cancel_check)
        span_model = (
            resolve_model_config(model, "reviewer").model_name
            if model != "auto"
            else model
        )
        with span_llm_call("reviewer", phase, span_model, session=session):
            result = call_llm(
                messages=trim_conversation(conversation, max_turns=16),
                model=model,
                max_tokens=MAX_TOKENS_REVIEWER,
                system_prompt=_CODE_REVIEW_MD,
                json_mode=True,
                role="reviewer",
                stream_context=stream_context_for(
                    emitter,
                    "reviewer",
                    phase=phase,
                    output_kind="json",
                ),
            )

        raw = result.text.strip()
        parsed = parse_agent_turn(raw)
        if parsed is None:
            invalid_json += 1
            if invalid_json >= _NON_JSON_RETRIES:
                raise RuntimeError(
                    "Code Review agent repeatedly returned an unparseable response."
                )
            conversation.extend([
                {"role": "assistant", "content": raw},
                {
                    "role": "user",
                    "content": (
                        "Invalid output. Emit EXACTLY ONE JSON object — no XML "
                        "`<tool_call>` tags, no markdown fences, no prose.\n\n"
                        'Tool example: {"tool":"git_diff","args":{},"reasoning":"..."}\n'
                        'Finish example: {"action":"review","report":{...}}'
                    ),
                },
            ])
            continue
        invalid_json = 0

        if parsed.get("action") == "review":
            report = normalize_review_report(
                parsed.get("report"),
                workspace_root=workspace_root,
                tool_calls=tool_calls,
            )
            if emitter:
                emitter.emit(
                    "review.completed",
                    phase=phase,
                    verdict=report.verdict,
                    findings=len(report.findings),
                    actionable_findings=len(report.actionable_findings),
                )
            return report

        calls, error, note = _extract_calls(parsed)
        if error:
            conversation.extend([
                {"role": "assistant", "content": raw},
                {"role": "user", "content": error},
            ])
            continue

        outputs: list[str] = []
        for call in calls:
            tool_name = str(call.get("tool", "") or "")
            args = call.get("args", {}) or {}
            reasoning = str(call.get("reasoning", "") or "")
            if tool_name not in _REVIEW_TOOLS:
                tool_result = {
                    "success": False,
                    "error_category": "review_tool_denied",
                    "error": (
                        f"`{tool_name}` denied: Code Review is read-only. "
                        f"Allowed tools: {sorted(_REVIEW_TOOLS)}"
                    ),
                }
                duration_ms = 0.0
            else:
                contract_error = validate_tool_call(tool_name, args)
                if contract_error:
                    tool_result = contract_error
                    duration_ms = 0.0
                else:
                    if emitter:
                        emitter.emit(
                            "tool.called",
                            role="reviewer",
                            phase=phase,
                            tool=tool_name,
                            args_keys=list(args.keys()),
                            reasoning=reasoning,
                            call_index=tool_calls + 1,
                        )
                    started = time.perf_counter()
                    with span_tool_call(tool_name, phase, session=session):
                        tool_result = dispatch(tool_name, args)
                    duration_ms = (time.perf_counter() - started) * 1000.0
            tool_calls += 1
            if emitter:
                emitter.emit(
                    "tool.result",
                    role="reviewer",
                    phase=phase,
                    tool=tool_name,
                    success=tool_result.get("success", False),
                    call_index=tool_calls,
                    **event_diagnostics(tool_name, args, tool_result, duration_ms),
                )
            outputs.append(
                f"Tool result for `{tool_name}`:\n```json\n"
                f"{compact_tool_result(tool_result)}\n```"
            )
            if tool_calls >= MAX_REVIEW_TOOL_CALLS:
                break

        feedback = "\n\n".join(outputs)
        if note:
            feedback = f"{note}\n\n{feedback}"
        conversation.extend([
            {"role": "assistant", "content": raw},
            {
                "role": "user",
                "content": (
                    f"{feedback}\n\nContinue reviewing or emit the final review report."
                ),
            },
        ])

    raise RuntimeError(
        f"Code Review exhausted {MAX_REVIEW_TOOL_CALLS} read-only tool calls "
        "without returning a report."
    )


def _extract_calls(
    parsed: dict[str, Any],
) -> tuple[list[dict[str, Any]], Optional[str], Optional[str]]:
    raw_calls = parsed.get("calls")
    if isinstance(raw_calls, list) and raw_calls:
        calls = [item for item in raw_calls if isinstance(item, dict)]
        if not calls:
            return [], "calls must contain tool-call objects", None
        if len(calls) > _MAX_BATCH:
            return calls[:_MAX_BATCH], None, f"Batch truncated to {_MAX_BATCH} calls."
        return calls, None, None
    if parsed.get("tool"):
        return [parsed], None, None
    return [], (
        'Emit {"tool":"...","args":{},"reasoning":"..."} or '
        '{"action":"review","report":{...}}.'
    ), None
