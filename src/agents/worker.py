"""
Worker Agent — Phase 3.

Runs a multi-turn tool-call conversation loop until the worker signals
"complete" or "blocked", or the per-milestone tool call budget is exhausted.

The worker receives a curated tool set from the skill router (Phase 2),
executes tool calls, and accumulates file modifications for the Validator.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Optional

from src.llm_client import call_llm, ModelChoice
from src.telemetry import span_llm_call, span_tool_call
from src.tools import dispatch
from src.tools.paths import normalize_workspace_path, normalize_shell_command, get_workspace_root
from src.agents.utils import (
    parse_json_from_text,
    flatten_conversation,
    trim_conversation,
    target_files_exist,
    _looks_like_tool_call_attempt,
)

# ---------------------------------------------------------------------------
# Paths and configuration
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).parent.parent.parent
_CONFIG_DIR = _ROOT / "config"

_WORKER_MD = (_CONFIG_DIR / "worker.md").read_text()

MAX_TOKENS_WORKER    = int(os.getenv("MAX_TOKENS_WORKER", "5120"))
MAX_WORKER_TOOL_CALLS = 20
_NON_JSON_RETRIES     = 3


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
) -> dict[str, Any]:
    """
    Execute the worker agent loop for a single milestone.

    The worker is given the milestone brief, workspace context, curated tools,
    and (on retries) error feedback from the Validator. It emits JSON tool
    calls one at a time until it signals complete/blocked or exhausts its budget.

    Args:
        milestone:        The milestone dict from the plan.
        plan:             The full mission plan (for title context).
        curated_tools_md: Markdown of top-k skill blocks from the router.
        error_feedback:   Validator error message from the previous retry, or None.
        retry_count:      Current retry attempt (0-indexed).
        model:            LLM backend to use.
        memory:           MissionMemory instance, or None if disabled.

    Returns:
        dict with keys: status, summary, files_modified, tool_calls.
    """
    ms_id = milestone.get("id", "?")
    print(f"\n  [Phase 3] WORKER — milestone {ms_id} (attempt {retry_count + 1})")

    ms = _normalize_milestone_for_worker(milestone)
    target_files = ms.get("target_files", [])

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
        "Start now. Emit ONE JSON tool call for the first required step."
    )

    conversation: list[dict] = [{"role": "user", "content": user_turn}]
    tool_call_count = 0
    non_json_retries = 0
    files_modified: list[str] = []

    while tool_call_count < MAX_WORKER_TOOL_CALLS:
        trimmed = trim_conversation(conversation, max_turns=12)
        prompt = (
            trimmed[-1]["content"]
            if len(trimmed) == 1
            else flatten_conversation(trimmed)
        )

        with span_llm_call("worker", ms_id, model):
            llm_result = call_llm(
                prompt=prompt,
                model=model,
                max_tokens=MAX_TOKENS_WORKER,
                system_prompt=_WORKER_MD,
            )

        raw = llm_result.text.strip()
        parsed = parse_json_from_text(raw)

        if parsed is None:
            non_json_retries += 1
            print(f"    [Worker] Non-JSON response ({non_json_retries}/{_NON_JSON_RETRIES}).")
            if non_json_retries >= _NON_JSON_RETRIES:
                return {
                    "status": "blocked",
                    "reason": f"Worker failed to emit valid JSON after {_NON_JSON_RETRIES} attempts.",
                    "tool_calls": tool_call_count,
                }
            conversation.append({"role": "assistant", "content": raw})
            conversation.append({
                "role": "user",
                "content": _worker_invalid_json_message(raw, target_files),
            })
            continue

        non_json_retries = 0
        status = parsed.get("status")

        if status == "complete":
            files_modified.extend(parsed.get("files_modified", []))
            print(f"    [Worker] Signalled COMPLETE after {tool_call_count} tool call(s).")
            ok, missing = target_files_exist(target_files)
            if target_files and not ok:
                print(f"    [Worker] Rejected premature COMPLETE — missing: {missing}")
                conversation.append({"role": "assistant", "content": raw})
                conversation.append({
                    "role": "user",
                    "content": (
                        f"REJECTED. These required files are still missing: {missing}. "
                        "Use write_file to create them, then signal complete again."
                    ),
                })
                continue
            return {
                "status": "complete",
                "summary": parsed.get("summary", ""),
                "files_modified": list(set(files_modified)),
                "tool_calls": tool_call_count,
            }

        if status == "blocked":
            reason = parsed.get("reason", "Unknown block")
            clarification = parsed.get("needs_clarification", "")
            print(f"    [Worker] BLOCKED after {tool_call_count} tool call(s).")
            print(f"    [Worker] Reason: {reason}")
            if clarification:
                print(f"    [Worker] Needs clarification: {clarification}")
            return {"status": "blocked", "reason": reason, "tool_calls": tool_call_count}

        # Tool call
        tool_name = parsed.get("tool", "")
        tool_args = parsed.get("args", {}) or {}
        reasoning = parsed.get("reasoning", "")

        if not tool_name:
            conversation.append({"role": "assistant", "content": raw})
            conversation.append({
                "role": "user",
                "content": 'Missing "tool" key. Emit a valid tool call JSON.',
            })
            continue

        print(
            f"    [Worker] tool_call[{tool_call_count + 1}]: "
            f"{tool_name}({list(tool_args.keys())}) — {reasoning}"
        )

        with span_tool_call(tool_name, ms_id):
            tool_result = dispatch(tool_name, tool_args)

        tool_call_count += 1

        if tool_name in ("write_file", "patch_file") and tool_result.get("success"):
            fpath = normalize_workspace_path(tool_args.get("file_path", ""))
            if fpath:
                files_modified.append(fpath)

        result_text = json.dumps(tool_result, indent=2)[:4000]
        next_msg = (
            f"Tool result for `{tool_name}`:\n```json\n{result_text}\n```\n\n"
            "Continue toward the required deliverables."
        )

        if tool_name in ("write_file", "patch_file") and target_files:
            written = normalize_workspace_path(tool_args.get("file_path", ""))
            if written and written not in target_files:
                next_msg += (
                    f"\n\nWARNING: `{written}` is NOT in Required deliverables: {target_files}. "
                    "Do not add scaffolding. Create/edit only the required files next."
                )

        ok, missing = target_files_exist(target_files)
        if missing:
            next_msg += f"\n\nStill missing: {missing}"
        else:
            next_msg += (
                '\n\nAll required files exist. Signal {"status": "complete", ...} if the work is done.'
            )

        conversation.append({"role": "assistant", "content": raw})
        conversation.append({"role": "user", "content": next_msg})

    # Budget exhausted
    ok, missing = target_files_exist(target_files)
    if target_files and not ok:
        return {
            "status": "blocked",
            "reason": f"Tool budget exhausted. Missing deliverables: {missing}",
            "files_modified": list(set(files_modified)),
            "tool_calls": tool_call_count,
        }
    return {
        "status": "complete",
        "summary": f"Tool call budget exhausted after {MAX_WORKER_TOOL_CALLS} calls.",
        "files_modified": list(set(files_modified)),
        "tool_calls": tool_call_count,
    }


# ---------------------------------------------------------------------------
# Worker helper functions
# ---------------------------------------------------------------------------

def _normalize_milestone_for_worker(milestone: dict) -> dict:
    """Return a shallow copy of the milestone with workspace/-stripped paths."""
    ms = dict(milestone)
    ms["target_files"] = [
        normalize_workspace_path(p) for p in milestone.get("target_files", [])
    ]
    contract = dict(milestone.get("validation_contract", {}))
    if contract.get("command"):
        contract["command"] = normalize_shell_command(contract["command"])
    ms["validation_contract"] = contract
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
            + "\n- Other files/dirs in the tree are context only — do not edit them unless listed above.\n"
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
    contract = ms.get("validation_contract", {})
    return (
        "## REQUIRED deliverables (must all exist before complete)\n"
        f"{targets_md}\n\n"
        "## Validation command (your code must pass this)\n"
        f"```\n{contract.get('command', '(none)')}\n```\n"
        f"Pass criteria: {contract.get('pass_criteria', '(none)')}\n\n"
        "## Rules for this milestone\n"
        "- Work ONLY on the required deliverables above.\n"
        "- Do NOT create `__init__.py` or other scaffolding unless listed above.\n"
        "- Ignore empty directories not listed above.\n"
        "- Every response must be a single JSON object (tool call, complete, or blocked).\n"
    )


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
