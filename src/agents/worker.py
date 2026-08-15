"""
Worker Agent — Phase 3.

    Runs a multi-turn tool-call conversation loop until the worker signals
"complete", "blocked", or "request_scope", or the per-milestone tool call
budget is exhausted.

Protocol notes (small-model optimised):

  - Native chat messages are sent to the LLM (system + alternating
    user/assistant), never a flattened single-turn blob. This preserves role
    semantics and keeps the rendered prompt append-only so llama.cpp's
    prefix cache stays hot across turns.
  - json_mode is ON: llama.cpp applies a GBNF grammar so every response is
    guaranteed parseable JSON, eliminating the invalid-JSON retry class.
  - The worker may emit a BATCH of up to MAX_BATCH_CALLS tool calls per turn
    ({"calls": [...]}) — one LLM round trip instead of three.
  - After every successful write/patch the HARNESS auto-runs the milestone's
    validation contract and appends the result, so test feedback costs zero
    LLM turns.
  - On validator FAIL the conversation is RESUMED (not restarted cold): the
    worker keeps everything it already learned and only receives the new
    failure feedback.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Callable, Optional

from src.llm_client import call_llm, ModelChoice, resolve_model_config
from src.telemetry import span_llm_call, span_tool_call, TelemetryContext
from src.tools import dispatch
from src.tools.file_ops import get_created_files
from src.tools.tool_contracts import validate_tool_call
from src.tools.paths import normalize_workspace_path, get_workspace_root
from src.events import EventEmitter
from src.run_control import RunCancelledError, ensure_not_cancelled
from src.agents.utils import (
    parse_json_from_text,
    trim_conversation,
    target_files_exist,
    _looks_like_tool_call_attempt,
)
from src.sandbox.commands import execute_contract
from src.sandbox.context import get_sandbox_context
from src.sandbox.process_manager import stop_server
from src.agents.llm_stream_events import stream_context_for
from src.agents.validation_compiler import compile_validation_contract
from src.sandbox.dependency_check import (
    check_target_file_dependencies,
    format_missing_dependency_message,
    planned_module_names,
)
from src.agents.tool_diagnostics import (
    compact_tool_result,
    event_diagnostics,
    tool_failure_signature,
)

# ---------------------------------------------------------------------------
# Paths and configuration
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).parent.parent.parent
_CONFIG_DIR = _ROOT / "config"

_WORKER_MD = (_CONFIG_DIR / "worker.md").read_text()

MAX_TOKENS_WORKER     = int(os.getenv("MAX_TOKENS_WORKER", "12288"))
MAX_WORKER_TOOL_CALLS = int(os.getenv("MAX_WORKER_TOOL_CALLS", "20"))
MAX_BATCH_CALLS       = int(os.getenv("MAX_WORKER_BATCH_CALLS", "3"))
MAX_HISTORY_TURNS     = int(os.getenv("MAX_WORKER_HISTORY_TURNS", "16"))
AUTORUN_MAX           = int(os.getenv("WORKER_CONTRACT_AUTORUN_MAX", "8"))
AUTORUN_STDOUT_CHARS  = int(os.getenv("WORKER_AUTORUN_STDOUT_CHARS", "600"))
AUTORUN_STDERR_CHARS  = int(os.getenv("WORKER_AUTORUN_STDERR_CHARS", "400"))
MAX_SAME_TOOL_FAILURES = int(os.getenv("MAX_SAME_TOOL_FAILURES", "2"))
MAX_CONSECUTIVE_TOOL_FAILURES = int(
    os.getenv("MAX_CONSECUTIVE_TOOL_FAILURES", "5")
)
_NON_JSON_RETRIES     = 3
UI_NUDGE_AFTER = int(os.getenv("WORKER_UI_NUDGE_AFTER", "4"))
UI_STRONG_NUDGE_AFTER = int(os.getenv("WORKER_UI_STRONG_NUDGE_AFTER", "8"))

SEARCH_TOOLS_MD = """\
## Control tool: search_tools
Use this when the currently available operational tools do not cover the next
action. Retrieval is deterministic and returns up to 3 additional tools.
Newly discovered tools become callable on the NEXT turn, not in the same batch.
```json
{"tool": "search_tools", "args": {"query": "<capability needed>", "limit": 3},
 "reasoning": "The current tool set does not include this capability."}
```
"""


# ---------------------------------------------------------------------------
# Phase 3 — Worker loop
# ---------------------------------------------------------------------------

def run_worker(
    milestone: dict,
    plan: dict,
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
    """
    Execute the worker agent loop for a single milestone.

    The worker is given the milestone brief, workspace context, curated tools,
    and (on retries) error feedback from the Validator. It emits JSON tool
    calls (single or batched) until it signals complete, request_scope,
    blocked, or exhausts its budget.

    Args:
        milestone:        The milestone dict from the plan.
        plan:             The full mission plan (for title context).
        curated_tools_md: Markdown of top-k skill blocks from the router.
        error_feedback:   Validator error message from the previous retry, or None.
        retry_count:      Current retry attempt (0-indexed).
        model:            LLM backend to use.
        memory:           MissionMemory instance, or None if disabled.
        emitter:          Optional EventEmitter for streaming tool/worker events.
        session:          Optional telemetry context used to bind LLM/tool spans
                          to a Phoenix session (Phase 5).
        cancel_check:     Optional callable returning True when the run should stop.
        prior_conversation: Conversation from the previous attempt of THIS
                          milestone. When provided, the worker resumes its own
                          thread instead of re-deriving context from scratch.
        initial_tool_names: Tools selected for the first Worker turn.
        prior_active_tools: Discovered/active tools preserved across retries.
        tool_searcher: Deterministic callback used by the search_tools meta-tool.
        prior_failure_state: Failure counts preserved across Worker retries.

    Returns:
        dict with keys: status, summary, files_modified, tool_calls, conversation.
    """
    ms_id = milestone.get("id", "?")
    print(f"\n  [Phase 3] WORKER — milestone {ms_id} (attempt {retry_count + 1})")

    if emitter:
        emitter.emit("worker.started", milestone_id=ms_id, retry=retry_count)

    ms = _normalize_milestone_for_worker(milestone)
    target_files = ms.get("target_files", [])
    contract, _compile_error = compile_validation_contract(milestone)
    contract = contract or {}
    is_ui_milestone = _is_ui_milestone(milestone, contract)

    active_tools = set(prior_active_tools or initial_tool_names or ())
    active_tools.add("search_tools")
    discovered_tools: set[str] = set(active_tools) - {"search_tools"}

    if prior_conversation:
        conversation: list[dict] = list(prior_conversation)
        conversation.append({
            "role": "user",
            "content": _retry_entry_message(error_feedback, memory, milestone, target_files),
        })
    else:
        grounding = _fetch_memory_grounding(memory, milestone)
        memory_constraints = _memory_constraints_block(memory, milestone)
        if memory_constraints.strip():
            print("    [Worker] Injecting memory constraints from prior failures.")

        user_turn = (
            f"{_workspace_context_block(target_files)}\n"
            f"{_worker_milestone_brief(ms)}\n"
            f"## Current Mission\n{plan.get('title', '')}\n\n"
            f"## Milestone to Implement\n"
            f"**ID**: {ms_id}\n"
            f"**Title**: {milestone.get('title', '')}\n"
            f"**Description**: {milestone.get('description', '')}\n"
            f"{grounding}"
            f"{memory_constraints}"
            + (
                f"\n## Error Feedback from Validator (latest retry)\n{error_feedback}\n"
                if error_feedback else ""
            )
            + f"\n## Available Tools\n{curated_tools_md}\n\n"
            + (_ui_worker_guidance_block() if is_ui_milestone else "")
            + SEARCH_TOOLS_MD
            + "Start now. Emit ONE JSON object (a tool call, a batch of up to "
            f"{MAX_BATCH_CALLS} calls, complete, request_scope, or blocked)."
        )
        conversation = [{"role": "user", "content": user_turn}]

    tool_call_count = 0
    non_json_retries = 0
    autorun_count = 0
    files_modified: list[str] = []
    failure_state = prior_failure_state or {}
    failure_counts: dict[str, int] = dict(failure_state.get("counts", {}))
    consecutive_failures = int(failure_state.get("consecutive", 0))
    breaker_reason: Optional[str] = None
    started_server_ids: set[str] = set()
    ui_calls_since_write = 0
    total_ui_inspect_calls = 0

    def _cleanup_started_servers() -> None:
        for server_id in list(started_server_ids):
            stop_server(server_id)
            started_server_ids.discard(server_id)

    def _finish(result: dict) -> dict:
        _cleanup_started_servers()
        result["conversation"] = conversation
        result["created_files"] = get_created_files()
        result["active_tools"] = sorted(active_tools)
        result["discovered_tools"] = sorted(discovered_tools)
        result["failure_state"] = {
            "counts": failure_counts,
            "consecutive": consecutive_failures,
        }
        return result

    while tool_call_count < MAX_WORKER_TOOL_CALLS:
        try:
            ensure_not_cancelled(cancel_check)
        except RunCancelledError:
            if emitter:
                emitter.emit("worker.cancelled", milestone_id=ms_id)
            return _finish({
                "status": "cancelled", "summary": "Run cancelled",
                "tool_calls": tool_call_count,
            })

        # Safety valve only — the common path never trims, keeping the
        # rendered prompt append-only (prefix-cache friendly).
        messages = trim_conversation(conversation, max_turns=MAX_HISTORY_TURNS)

        span_model = (
            resolve_model_config(model, "worker").model_name
            if model != "auto" else model
        )
        with span_llm_call("worker", ms_id, span_model, session=session):
            llm_result = call_llm(
                messages=messages,
                model=model,
                max_tokens=MAX_TOKENS_WORKER,
                system_prompt=_WORKER_MD,
                json_mode=True,
                role="worker",
                stream_context=stream_context_for(
                    emitter,
                    "worker",
                    milestone_id=ms_id,
                    output_kind="json",
                ),
            )

        raw = llm_result.text.strip()
        parsed = parse_json_from_text(raw)

        if parsed is None:
            non_json_retries += 1
            print(f"    [Worker] Non-JSON response ({non_json_retries}/{_NON_JSON_RETRIES}).")
            if emitter:
                emitter.emit(
                    "worker.invalid_json",
                    milestone_id=ms_id,
                    attempt=non_json_retries,
                    max_attempts=_NON_JSON_RETRIES,
                )
            if non_json_retries >= _NON_JSON_RETRIES:
                if emitter:
                    emitter.emit("worker.blocked", milestone_id=ms_id, reason="invalid_json")
                return _finish({
                    "status": "blocked",
                    "reason": f"Worker failed to emit valid JSON after {_NON_JSON_RETRIES} attempts.",
                    "tool_calls": tool_call_count,
                })
            conversation.append({"role": "assistant", "content": raw})
            conversation.append({
                "role": "user",
                "content": _worker_invalid_json_message(raw, target_files),
            })
            continue

        non_json_retries = 0
        status = parsed.get("status")

        if status == "request_scope":
            requested = parsed.get("requested_paths", parsed.get("requested_files", []))
            if not isinstance(requested, list):
                requested = []
            reason = str(parsed.get("reason", "Additional workspace scope is required."))
            print(f"    [Worker] REQUEST_SCOPE after {tool_call_count} tool call(s).")
            if emitter:
                emitter.emit(
                    "worker.request_scope",
                    milestone_id=ms_id,
                    reason=reason,
                    requested_paths=[str(path) for path in requested],
                    tool_calls=tool_call_count,
                )
            return _finish({
                "status": "request_scope",
                "reason": reason,
                "requested_paths": [str(path) for path in requested],
                "requested_capabilities": parsed.get("requested_capabilities", []),
                "tool_calls": tool_call_count,
            })

        if status == "complete":
            files_modified.extend(parsed.get("files_modified", []))
            print(f"    [Worker] Signalled COMPLETE after {tool_call_count} tool call(s).")
            ok, missing = target_files_exist(target_files)
            if target_files and not ok:
                print(f"    [Worker] Rejected premature COMPLETE — missing: {missing}")
                if emitter:
                    emitter.emit(
                        "worker.complete_rejected",
                        milestone_id=ms_id,
                        missing=missing,
                    )
                conversation.append({"role": "assistant", "content": raw})
                conversation.append({
                    "role": "user",
                    "content": (
                        f"REJECTED. These required files are still missing: {missing}. "
                        "Use write_file to create them, then signal complete again."
                    ),
                })
                continue

            dep_report = check_target_file_dependencies(
                target_files,
                planned_modules=planned_module_names(plan),
                phase=(
                    "test_scaffold"
                    if str(contract.get("type", "")).lower() == "test_scaffold"
                    else "implementation"
                ),
            )
            if not dep_report.ok:
                message = format_missing_dependency_message(dep_report)
                print(f"    [Worker] Rejected COMPLETE — {message}")
                if emitter:
                    emitter.emit(
                        "dependency.missing",
                        milestone_id=ms_id,
                        packages=dep_report.missing_packages,
                        imports=dep_report.missing_imports,
                        checked_files=dep_report.checked_files,
                        source="worker",
                    )
                    emitter.emit(
                        "worker.complete_rejected",
                        milestone_id=ms_id,
                        reason="missing_dependencies",
                        packages=dep_report.missing_packages,
                    )
                conversation.append({"role": "assistant", "content": raw})
                conversation.append({
                    "role": "user",
                    "content": (
                        f"REJECTED. {message} "
                        'Use install_dependency, e.g. '
                        '{"tool": "install_dependency", "args": {"package_name": "pygame"}, '
                        '"reasoning": "Required by main.py imports."}'
                    ),
                })
                continue

            if emitter:
                emitter.emit(
                    "worker.complete",
                    milestone_id=ms_id,
                    tool_calls=tool_call_count,
                    files_modified=list(set(files_modified)),
                )
            return _finish({
                "status": "complete",
                "summary": parsed.get("summary", ""),
                "files_modified": list(set(files_modified)),
                "tool_calls": tool_call_count,
            })

        if status == "blocked":
            reason = parsed.get("reason", "Unknown block")
            clarification = parsed.get("needs_clarification", "")
            print(f"    [Worker] BLOCKED after {tool_call_count} tool call(s).")
            print(f"    [Worker] Reason: {reason}")
            if clarification:
                print(f"    [Worker] Needs clarification: {clarification}")
            if emitter:
                emitter.emit(
                    "worker.blocked",
                    milestone_id=ms_id,
                    reason=reason,
                    clarification=clarification,
                    tool_calls=tool_call_count,
                )
            return _finish({
                "status": "blocked", "reason": reason,
                "clarification": clarification,
                "tool_calls": tool_call_count,
            })

        # --- Tool calls: single object or batch ------------------------------
        calls, call_error, call_note = _extract_tool_calls(parsed)
        if call_error:
            conversation.append({"role": "assistant", "content": raw})
            conversation.append({"role": "user", "content": call_error})
            continue

        batch_results: list[str] = []
        wrote_files = False
        discovery_in_batch = False
        for call in calls:
            tool_name = call.get("tool", "")
            if not isinstance(tool_name, str):
                tool_name = repr(tool_name)
            tool_args = call.get("args", {}) or {}
            reasoning = call.get("reasoning", "")

            if not tool_name:
                batch_results.append("Missing \"tool\" key in one call — skipped.")
                continue

            if discovery_in_batch:
                batch_results.append(
                    f"Tool `{tool_name}` was not executed because search_tools "
                    "must complete before newly discovered tools can be called."
                )
                continue

            contract_error = validate_tool_call(
                tool_name,
                tool_args,
                active_tools=active_tools,
            )
            if contract_error:
                tool_result = contract_error
                tool_call_count += 1
                batch_results.append(
                    f"Tool result for `{tool_name}`:\n```json\n"
                    f"{compact_tool_result(tool_result)}\n```"
                )
                if emitter:
                    emitter.emit(
                        "tool.result",
                        milestone_id=ms_id,
                        tool=tool_name,
                        success=False,
                        call_index=tool_call_count,
                        **event_diagnostics(tool_name, tool_args, tool_result, 0.0),
                    )
                signature = tool_failure_signature(tool_name, tool_args, tool_result)
                failure_counts[signature] = failure_counts.get(signature, 0) + 1
                consecutive_failures += 1
                if failure_counts[signature] >= MAX_SAME_TOOL_FAILURES:
                    breaker_reason = (
                        f"Repeated identical tool failure ({signature}) for "
                        f"`{tool_name}`. Do not retry the same call; use "
                        "search_tools or change the arguments."
                    )
                    break
                if consecutive_failures >= MAX_CONSECUTIVE_TOOL_FAILURES:
                    breaker_reason = (
                        f"{MAX_CONSECUTIVE_TOOL_FAILURES} consecutive tool calls "
                        "failed. Stop guessing and request clarification."
                    )
                    break
                continue

            print(
                f"    [Worker] tool_call[{tool_call_count + 1}]: "
                f"{tool_name}({list(tool_args.keys())}) — {reasoning}"
            )

            if emitter:
                emitter.emit(
                    "tool.called",
                    milestone_id=ms_id,
                    tool=tool_name,
                    args_keys=list(tool_args.keys()),
                    reasoning=reasoning,
                    call_index=tool_call_count + 1,
                )

            started = time.perf_counter()
            if tool_name == "search_tools":
                limit = int(tool_args.get("limit", 3))
                if tool_searcher is None:
                    tool_result = {
                        "success": False,
                        "error_category": "tool_discovery_unavailable",
                        "error": "Tool discovery is unavailable for this run.",
                    }
                else:
                    with span_tool_call(tool_name, ms_id, session=session):
                        tool_result = tool_searcher(
                            str(tool_args["query"]),
                            max(1, min(limit, 5)),
                        )
                    discovery_in_batch = True
                    new_tools = set(tool_result.get("tools", []))
                    active_tools.update(new_tools)
                    discovered_tools.update(new_tools)
                    if emitter:
                        emitter.emit(
                            "tool.discovery",
                            milestone_id=ms_id,
                            query=tool_args["query"],
                            tools=sorted(new_tools),
                            count=len(new_tools),
                        )
            else:
                with span_tool_call(tool_name, ms_id, session=session):
                    tool_result = dispatch(tool_name, tool_args)

            if tool_name == "serve_app":
                action = str(tool_args.get("action", "start")).lower().strip()
                if tool_result.get("success"):
                    if action == "start":
                        server_id = str(tool_result.get("server_id", "")).strip()
                        if server_id:
                            started_server_ids.add(server_id)
                    elif action == "stop":
                        server_id = str(tool_args.get("server_id", "")).strip()
                        started_server_ids.discard(server_id)

            tool_call_count += 1
            duration_ms = (time.perf_counter() - started) * 1000.0

            if emitter:
                emitter.emit(
                    "tool.result",
                    milestone_id=ms_id,
                    tool=tool_name,
                    success=tool_result.get("success", False),
                    call_index=tool_call_count,
                    **event_diagnostics(
                        tool_name, tool_args, tool_result, duration_ms
                    ),
                )

            if tool_name in ("write_file", "patch_file") and tool_result.get("success"):
                wrote_files = True
                ui_calls_since_write = 0
                fpath = normalize_workspace_path(tool_args.get("file_path", ""))
                if fpath:
                    files_modified.append(fpath)

            if tool_name == "inspect_ui":
                total_ui_inspect_calls += 1
                ui_calls_since_write += 1

            result_text = compact_tool_result(tool_result)
            hint = tool_result.get("hint")
            if isinstance(hint, str) and hint.strip():
                result_text += f"\n\nHint: {hint.strip()}"
            batch_results.append(
                f"Tool result for `{tool_name}`:\n```json\n{result_text}\n```"
            )

            handoff_failure = (
                tool_name == "inspect_ui"
                and not tool_result.get("success")
                and tool_result.get("suggest_handoff")
            )
            if tool_result.get("success") or handoff_failure:
                consecutive_failures = 0
            elif not handoff_failure:
                signature = tool_failure_signature(tool_name, tool_args, tool_result)
                failure_counts[signature] = failure_counts.get(signature, 0) + 1
                consecutive_failures += 1
                if failure_counts[signature] >= MAX_SAME_TOOL_FAILURES:
                    if handoff_failure:
                        breaker_reason = None
                    else:
                        breaker_reason = (
                            f"Repeated identical tool failure ({signature}) for "
                            f"`{tool_name}`. Do not retry the same call; use "
                            "search_tools or change the arguments."
                        )
                elif consecutive_failures >= MAX_CONSECUTIVE_TOOL_FAILURES:
                    if handoff_failure:
                        breaker_reason = None
                    else:
                        breaker_reason = (
                            f"{MAX_CONSECUTIVE_TOOL_FAILURES} consecutive tool calls "
                            "failed. Stop guessing and request clarification."
                        )

            if handoff_failure and is_ui_milestone:
                batch_results.append(
                    "## UI handoff\n"
                    "This UI probe did not confirm the text/interaction. Prefer a "
                    "targeted code fix if validator feedback is clear, otherwise "
                    'signal `{"status": "complete", ...}` and let the validator run '
                    "official browser smoke checks."
                )

            if breaker_reason or tool_call_count >= MAX_WORKER_TOOL_CALLS:
                break

        next_msg = "\n\n".join(batch_results) if batch_results else "(no tool calls executed)"
        if call_note:
            next_msg += f"\n\n{call_note}"
        if breaker_reason:
            next_msg += (
                f"\n\n## TOOL FAILURE CIRCUIT BREAKER\n{breaker_reason}\n"
                "Do not repeat the failed operation. If another capability is "
                "needed, call search_tools; otherwise signal blocked with a "
                "specific clarification request."
            )

        # Harness-side contract auto-run: test feedback with zero LLM turns.
        if wrote_files and autorun_count < AUTORUN_MAX:
            auto = _autorun_contract(contract, ms_id, emitter)
            if auto:
                autorun_count += 1
                next_msg += f"\n\n{auto}"

        next_msg += "\n\nContinue toward the required deliverables."

        if target_files:
            ok, missing = target_files_exist(target_files)
            if missing:
                next_msg += f"\n\nStill missing: {missing}"
            else:
                next_msg += (
                    '\n\nAll required files exist. Signal {"status": "complete", ...} '
                    "if the work is done."
                )

        if is_ui_milestone:
            ui_nudge = _ui_handoff_nudge(ui_calls_since_write, total_ui_inspect_calls)
            if ui_nudge:
                next_msg += f"\n\n{ui_nudge}"

        conversation.append({"role": "assistant", "content": raw})
        conversation.append({"role": "user", "content": next_msg})

        if breaker_reason:
            if emitter:
                emitter.emit(
                    "worker.tool_failure_breaker",
                    milestone_id=ms_id,
                    reason=breaker_reason,
                    consecutive_failures=consecutive_failures,
                    distinct_failures=len(failure_counts),
                )
            return _finish({
                "status": "blocked",
                "reason": breaker_reason,
                "needs_clarification": (
                    "Choose a different available tool or provide the missing "
                    "workspace/contract information."
                ),
                "tool_calls": tool_call_count,
            })

    # Budget exhausted
    ok, missing = target_files_exist(target_files)
    if target_files and not ok:
        if emitter:
            emitter.emit(
                "worker.blocked",
                milestone_id=ms_id,
                reason=f"Tool budget exhausted. Missing deliverables: {missing}",
                tool_calls=tool_call_count,
            )
        return _finish({
            "status": "blocked",
            "reason": f"Tool budget exhausted. Missing deliverables: {missing}",
            "files_modified": list(set(files_modified)),
            "tool_calls": tool_call_count,
        })
    if emitter:
        emitter.emit(
            "worker.complete",
            milestone_id=ms_id,
            tool_calls=tool_call_count,
            files_modified=list(set(files_modified)),
        )
    return _finish({
        "status": "complete",
        "summary": f"Tool call budget exhausted after {MAX_WORKER_TOOL_CALLS} calls.",
        "files_modified": list(set(files_modified)),
        "tool_calls": tool_call_count,
    })


# ---------------------------------------------------------------------------
# Batch parsing + contract auto-run
# ---------------------------------------------------------------------------

def _extract_tool_calls(parsed: dict) -> tuple[list[dict], Optional[str], Optional[str]]:
    """
    Normalise a worker response into a list of tool-call dicts.

    Accepts either a single call object ({"tool": ..., "args": ...}) or a
    batch ({"calls": [...]}). Batches are capped at MAX_BATCH_CALLS.
    Returns (calls, error_message, note) — error_message is set when neither
    shape is present; note carries non-fatal warnings (e.g. truncation).
    """
    raw_calls = parsed.get("calls")
    if isinstance(raw_calls, list) and raw_calls:
        calls = [c for c in raw_calls if isinstance(c, dict)]
        if not calls:
            return [], 'The "calls" array must contain objects with "tool" and "args".', None
        if len(calls) > MAX_BATCH_CALLS:
            return calls[:MAX_BATCH_CALLS], None, (
                f"NOTE: batch truncated to {MAX_BATCH_CALLS} calls — "
                "emit the remaining calls next turn."
            )
        return calls, None, None

    if parsed.get("tool"):
        return [parsed], None, None

    return [], (
        'Missing "tool" key. Emit a tool call {"tool": "<name>", "args": {...}, '
        '"reasoning": "..."}, a batch {"calls": [...]}, or a status object '
        '("complete" / "blocked").'
    ), None


def _autorun_contract(
    contract: dict,
    ms_id: str,
    emitter: Optional[EventEmitter],
) -> str:
    """
    Run the milestone validation contract harness-side after a file write.

    Returns a compact feedback block for the worker's next turn, or "" when
    auto-run is not possible (no command / no sandbox).
    """
    command = str(contract.get("command", "")).strip()
    if (
        not command
        or str(contract.get("type", "")).lower() == "ui_smoke"
        or get_sandbox_context() is None
    ):
        return ""

    result = execute_contract(contract, timeout=60, profile="validation")
    returncode = result.get("returncode", -1)

    if emitter:
        emitter.emit(
            "worker.contract_autorun",
            milestone_id=ms_id,
            returncode=returncode,
            execution_mode=result.get("execution_mode", "unknown"),
            policy_denied=result.get("policy_denied", False),
        )

    stdout_tail = (result.get("stdout", "") or "")[-AUTORUN_STDOUT_CHARS:]
    stderr_tail = (result.get("stderr", "") or "")[-AUTORUN_STDERR_CHARS:]
    status_line = "PASS (exit 0)" if returncode == 0 else f"FAILING (exit {returncode})"
    return (
        f"## Auto-run of validation contract (harness, free feedback)\n"
        f"Command: `{command}`\n"
        f"Result: {status_line}\n"
        f"```\nstdout (tail):\n{stdout_tail}\nstderr (tail):\n{stderr_tail}\n```"
    )


def _retry_entry_message(
    error_feedback: Optional[str],
    memory: Any,
    milestone: dict,
    target_files: list[str],
) -> str:
    """Build the user turn that resumes the worker's thread after a FAIL."""
    parts = [
        "## RETRY — the Validator rejected your previous attempt.",
        "Your full conversation above shows everything you already did. "
        "Do NOT redo completed work; apply a MINIMAL fix with patch_file.",
    ]
    if error_feedback:
        parts.append(f"\n### Validator failure report\n{error_feedback}")
    constraints = _memory_constraints_block(memory, milestone)
    if constraints.strip():
        parts.append(constraints)
    _, missing = target_files_exist(target_files)
    if missing:
        parts.append(f"\nFiles still missing on disk: {missing}")
    parts.append("\nEmit your next JSON object now.")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Worker helper functions
# ---------------------------------------------------------------------------

def _normalize_milestone_for_worker(milestone: dict) -> dict:
    """Return a shallow copy of the milestone with workspace/-stripped paths."""
    ms = dict(milestone)
    ms["target_files"] = [
        normalize_workspace_path(p) for p in milestone.get("target_files", [])
    ]
    ms["acceptance_criteria"] = [
        str(item).strip()
        for item in milestone.get("acceptance_criteria", [])
        if str(item).strip()
    ]
    return ms


def _workspace_context_block(target_files: list[str] | None = None) -> str:
    """Build the workspace context header injected into every worker turn."""
    tree_result = dispatch("list_directory", {"target_dir": "."})
    tree = (
        tree_result.get("tree", "(empty)")
        if tree_result.get("success")
        else "(unavailable)"
    )
    targets_note = ""
    if target_files:
        targets_note = (
            "\n- Required files for this milestone: "
            + ", ".join(f"`{p}`" for p in target_files)
            + "\n- Other files/dirs in the tree are context only — the harness "
              "REJECTS writes to files not listed above.\n"
        )
    return (
        "## Workspace Context\n"
        f"- Project root (your cwd): `{get_workspace_root()}`\n"
        "- All tool paths are relative to this directory.\n"
        "- Do NOT prefix paths with `workspace/`.\n"
        f"{targets_note}\n"
        f"### Current directory tree\n```\n{tree}\n```\n"
    )


def _worker_milestone_brief(ms: dict) -> str:
    """Build an unambiguous milestone brief for the non-thinking worker."""
    target_files = ms.get("target_files", [])
    targets_md = (
        "\n".join(f"- `{p}`" for p in target_files)
        if target_files
        else "- (none listed)"
    )
    red_phase_note = ""
    is_scaffold = str(ms.get("validation_profile", "")).lower() == "test_scaffold" or (
        target_files
        and all(
            p.startswith("tests/") or p.startswith("test_") or "/test_" in p
            for p in target_files
        )
    )
    if is_scaffold:
        red_phase_note = (
            "- TDD RED PHASE: this is a test-scaffolding milestone. The module under "
            "test does NOT exist yet BY DESIGN. Collection errors / ModuleNotFoundError "
            "for it in the auto-run output are EXPECTED — do not try to fix them. "
            "Write the tests, then signal complete.\n"
        )

    return (
        "## REQUIRED deliverables (must all exist before complete)\n"
        f"{targets_md}\n\n"
        "## Acceptance criteria (the Validator will compile the checks)\n"
        + "\n".join(f"- {item}" for item in ms.get("acceptance_criteria", []))
        + "\n\n"
        f"Validation profile: {ms.get('validation_profile', 'auto')}\n\n"
        "## Rules for this milestone\n"
        "- Work ONLY on the required deliverables above — writes to other files are rejected.\n"
        "- Do NOT create `__init__.py` or other scaffolding unless listed above.\n"
        "- Ignore empty directories not listed above.\n"
        f"{red_phase_note}"
        f"{_ui_milestone_rules(ms)}"
        "- Every response must be a single JSON object (tool call, batch, complete, or blocked).\n"
    )


_UI_HINTS = ("ui", "html", "frontend", "react", "vite", "streamlit", "browser", "web")


def _is_ui_milestone(milestone: dict, contract: dict) -> bool:
    profile = str(milestone.get("validation_profile", "")).lower()
    if profile == "ui":
        return True
    if str(contract.get("type", "")).lower() == "ui_smoke":
        return True
    text = " ".join(
        [
            str(milestone.get("title", "")),
            str(milestone.get("description", "")),
            " ".join(str(item) for item in milestone.get("acceptance_criteria", [])),
        ]
    ).lower()
    targets = [str(path).lower() for path in milestone.get("target_files", [])]
    return any(hint in text for hint in _UI_HINTS) or any(
        path.endswith((".html", ".jsx", ".tsx", ".vue")) for path in targets
    )


def _ui_worker_guidance_block() -> str:
    return (
        "## UI milestone — division of labor\n"
        "- You implement the UI; the **validator** runs authoritative `ui_smoke` "
        "checks after you signal `complete`.\n"
        "- Optional preflight: `serve_app` → `inspect_ui` with `accessibility` or "
        "`audit` to catch obvious load/a11y issues.\n"
        "- For multi-step interactions (fill → click → assert), use ONE "
        '`inspect_ui` call with `action: "flow"` and a `steps` array.\n'
        "- Separate `fill`/`click` calls do **not** share browser state.\n"
        "- Do not loop on UI probes without a code change — fix the issue or "
        "signal `complete` for validator smoke.\n\n"
    )


def _ui_milestone_rules(ms: dict) -> str:
    if str(ms.get("validation_profile", "")).lower() != "ui":
        return ""
    return (
        "- UI PROFILE: keep manual UI checks lightweight. Validator smoke is the "
        "official test pass.\n"
    )


def _ui_handoff_nudge(calls_since_write: int, total_ui_calls: int) -> str:
    if total_ui_calls < UI_NUDGE_AFTER:
        return ""
    if calls_since_write >= UI_STRONG_NUDGE_AFTER:
        return (
            "## UI handoff reminder\n"
            "You have run many UI checks without changing code. Unless you are "
            "fixing a specific validator failure, signal `complete` now — the "
            "validator will run full browser smoke tests on its own server."
        )
    if calls_since_write >= UI_NUDGE_AFTER:
        return (
            "## UI handoff reminder\n"
            "Prefer targeted code fixes over repeated UI probes. When deliverables "
            "look ready (or after fixing validator feedback), signal `complete` "
            "and let the validator run official smoke checks."
        )
    return ""


def _memory_constraints_block(memory: Any, milestone: dict | None = None) -> str:
    """
    Return persisted negative constraints from memory_store.json for worker injection.

    Filters to constraints mentioning the current milestone's target files when
    a milestone is provided.
    """
    if memory is None:
        return ""
    try:
        constraints = memory.get_error_constraints()
        if not constraints:
            return ""
        if milestone:
            target_files = [
                normalize_workspace_path(p)
                for p in milestone.get("target_files", [])
            ]
            if target_files:
                lines = constraints.splitlines()
                header = lines[0] if lines else constraints
                body = lines[1:]
                relevant = [
                    line for line in body
                    if any(tf in line for tf in target_files)
                ]
                if relevant:
                    constraints = header + "\n" + "\n".join(relevant)
        return f"\n{constraints}\n"
    except Exception as exc:
        print(f"[Runtime] Memory constraints unavailable: {exc}")
        return ""


def _fetch_memory_grounding(memory: Any, milestone: dict) -> str:
    """Query structural memory for anti-hallucination grounding hints."""
    if memory is None:
        return ""
    try:
        class_hints = memory.query_structural_memory(milestone.get("title", ""))
        if class_hints:
            return f"\n\n## Memory Grounding\n{class_hints}"
    except Exception as exc:
        print(f"    [Worker] Memory grounding unavailable: {exc}")
    return ""


def _worker_invalid_json_message(raw: str, target_files: list[str]) -> str:
    """Build a corrective user turn after a failed JSON parse attempt."""
    _, missing = target_files_exist(target_files)
    if missing:
        missing_line = f"Files still missing on disk: {missing}"
    elif target_files:
        missing_line = (
            "All target files exist on disk — fix JSON format, then continue or signal complete."
        )
    else:
        missing_line = "Fix JSON format, then continue or signal complete."

    msg = (
        "INVALID OUTPUT. You must emit EXACTLY ONE valid JSON object.\n"
        "No markdown fences, no prose before or after the JSON.\n\n"
        "Tool call:\n"
        '{"tool": "<name>", "args": {...}, "reasoning": "<one sentence>"}\n\n'
        "Batch (up to 3 calls):\n"
        '{"calls": [{"tool": "<name>", "args": {...}, "reasoning": "..."}, ...]}\n\n'
        "Done:\n"
        '{"status": "complete", "summary": "...", "files_modified": ["..."]}\n\n'
        f"{missing_line}"
    )

    if _looks_like_tool_call_attempt(raw):
        msg += (
            "\n\nYour last output LOOKED like JSON but failed to parse.\n"
            "Common cause: unescaped double quotes inside `write_file` → `args` → `content`.\n"
            "Python code belongs INSIDE the JSON string — the outer JSON must stay valid.\n\n"
            "Fix ONE of:\n"
            '1. Escape inner double quotes: evaluate(\\"2 + 3\\")\n'
            "2. Use single quotes in Python inside the JSON string: evaluate('2 + 3')\n"
            "3. For small edits: read_file + patch_file instead of rewriting the whole file"
        )

    return msg
