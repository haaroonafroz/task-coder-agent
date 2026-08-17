"""Read-only workspace exploration for the Orchestrator before plan emission."""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any, Optional

from src.agents.tool_diagnostics import compact_tool_result, event_diagnostics
from src.agents.utils import parse_json_from_text, trim_conversation, validate_plan_payload
from src.events import EventEmitter
from src.llm_client import ModelChoice, call_llm, resolve_model_config
from src.telemetry import span_llm_call, span_tool_call, TelemetryContext
from src.tools import dispatch
from src.tools.tool_contracts import validate_tool_call
from src.agents.llm_stream_events import stream_context_for

_ORCHESTRATOR_READ_TOOLS = frozenset({
    "list_directory",
    "read_file",
    "search_grep",
    "project_info",
    "git_diff",
    "view_git_log",
})

_TEXT_SUFFIXES = {
    ".c", ".css", ".csv", ".go", ".html", ".ini", ".java", ".js", ".json",
    ".jsx", ".md", ".py", ".rs", ".sh", ".sql", ".toml", ".ts", ".tsx",
    ".txt", ".xml", ".yaml", ".yml",
}
_SKIP_DIRS = {".git", ".pytest_cache", ".venv", "__pycache__", "node_modules", "target"}

MAX_ORCHESTRATOR_EXPLORE_CALLS = int(os.getenv("MAX_ORCHESTRATOR_EXPLORE_CALLS", "10"))
MAX_ORCHESTRATOR_EXPLORE_LIGHT = int(os.getenv("MAX_ORCHESTRATOR_EXPLORE_LIGHT", "5"))
ORCHESTRATOR_EXPLORE_ENABLED = os.getenv(
    "ORCHESTRATOR_EXPLORE_ENABLED", "true"
).lower() not in ("0", "false", "no")
MAX_ORCHESTRATOR_BATCH_CALLS = 3
_NON_JSON_RETRIES = 2

_FILE_PATH_RE = re.compile(
    r"""['"`]?([\w./-]+\.(?:html|htm|py|js|jsx|ts|tsx|json|md|css|yaml|yml|toml|xml|vue))['"`]?""",
    re.IGNORECASE,
)
_QUOTED_RE = re.compile(r"""['"]([^'"]{3,80})['"]""")
_PORT_RE = re.compile(r"\b(?:localhost|127\.0\.0\.1):\d{2,5}\b")
_URL_RE = re.compile(r"https?://[^\s'\"<>]+")

_EXPLORE_TOOLS_MD = """\
## Read-only exploration tools

You may call these tools to orient yourself before emitting the plan.
You must NOT modify files — planning only.

| Tool | Purpose |
|------|---------|
| `search_grep` | Find symbols, strings, URLs — **use this first** |
| `read_file` | Read a slice (`offset`, `limit` lines) after grep narrows the target |
| `list_directory` | Directory tree |
| `project_info` | Ecosystem manifests and bounded file list |
| `git_diff` | Recent uncommitted changes |
| `view_git_log` | Recent commits (`limit` ≤ 5) |

Rules:
- **grep before read** — never read a whole large file without grep first.
- `read_file` must use `offset` and `limit` unless the file is tiny (<80 lines).
- `target_files` in the final plan must be files you found via these tools.
- Acceptance criteria describe observable behavior, not line numbers.

Emit ONE JSON object per turn — either a tool call or the final plan:

```json
{"tool": "search_grep", "args": {"query": "8080"}, "reasoning": "Locate endpoint config."}
```

When ready, emit the plan:

```json
{"action": "plan", "plan": { ... full plan object ... }}
```
"""


def should_explore(
    run_kind: str,
    workspace_root: Optional[Path],
    *,
    enabled: bool = ORCHESTRATOR_EXPLORE_ENABLED,
) -> tuple[bool, str]:
    """Return (explore, mode) where mode is ``full``, ``light``, or ``none``."""
    if not enabled or workspace_root is None or not workspace_root.exists():
        return False, "none"
    if run_kind == "repair":
        return True, "full"
    if run_kind == "new" and _workspace_has_meaningful_code(workspace_root):
        return True, "full"
    if run_kind == "resume" and _workspace_has_meaningful_code(workspace_root):
        return True, "light"
    return False, "none"


def explore_budget(mode: str) -> int:
    if mode == "light":
        return MAX_ORCHESTRATOR_EXPLORE_LIGHT
    if mode == "full":
        return MAX_ORCHESTRATOR_EXPLORE_CALLS
    return 0


def build_workspace_orientation(
    workspace_root: Path,
    user_request: str,
    previous_plan: Optional[dict[str, Any]] = None,
    *,
    max_grep_patterns: int = 6,
    max_matches_per_pattern: int = 8,
) -> str:
    """Deterministic preflight: tree, project info, and grep hits (no full files)."""
    sections: list[str] = ["## Session orientation (harness preflight)\n"]

    project = dispatch("project_info", {"max_entries": 80})
    if project.get("success"):
        sections.append(
            "### Project\n"
            f"- ecosystems: {', '.join(project.get('ecosystems') or []) or '(none)'}\n"
            f"- manifests: {', '.join(project.get('manifests') or []) or '(none)'}\n"
        )

    tree = dispatch("list_directory", {"target_dir": ".", "max_depth": 4})
    if tree.get("success"):
        tree_text = str(tree.get("tree", ""))[:4000]
        sections.append(f"### Directory tree\n```\n{tree_text}\n```\n")

    if previous_plan:
        completed = [
            m.get("id", "?")
            for m in previous_plan.get("milestones", [])
            if m.get("status") == "completed"
        ]
        if completed:
            sections.append(
                "### Previous plan milestones (completed)\n"
                + ", ".join(completed)
                + "\n"
            )

    patterns = _grep_patterns_from_request(user_request, previous_plan)[:max_grep_patterns]
    if patterns:
        sections.append("### Grep preflight\n")
        for pattern in patterns:
            result = dispatch("search_grep", {"query": pattern, "target_dir": "."})
            if not result.get("success"):
                continue
            matches = result.get("matches") or []
            if not matches:
                sections.append(f"- `{pattern}` → no matches\n")
                continue
            lines = []
            for match in matches[:max_matches_per_pattern]:
                lines.append(
                    f"  {match.get('file')}:{match.get('line_no')}: "
                    f"{str(match.get('text', '')).strip()[:120]}"
                )
            extra = len(matches) - max_matches_per_pattern
            suffix = f"  … +{extra} more\n" if extra > 0 else ""
            sections.append(f"- `{pattern}`\n" + "\n".join(lines) + "\n" + suffix)

    return "\n".join(sections)


def run_orchestration_explore(
    *,
    user_request: str,
    run_kind: str,
    parent_plan_id: Optional[str],
    previous_plan: Optional[dict[str, Any]],
    orientation_block: str,
    triage_report: Optional[dict[str, Any]],
    orchestrator_md: str,
    model: ModelChoice,
    explore_mode: str,
    session: Optional[TelemetryContext],
    emitter: Optional[EventEmitter],
) -> dict[str, Any]:
    """Exploration loop ending in a validated plan JSON object."""
    budget = explore_budget(explore_mode)
    print(f"  [Orchestrator] Exploration mode={explore_mode} (budget={budget} tool calls)")

    if emitter:
        emitter.emit("orchestrator.explore.started", run_kind=run_kind, mode=explore_mode, budget=budget)

    user_turn = (
        f"{orientation_block}\n\n"
        f"## Run Mode\n{run_kind}\n\n"
        f"## Parent Plan\n{parent_plan_id or '(none)'}\n\n"
        f"## User Request\n{user_request}\n\n"
    )
    if previous_plan:
        user_turn += (
            "## Previous Plan (summary)\n"
            f"```json\n{json.dumps(previous_plan, indent=2)[:8000]}\n```\n\n"
        )
    if triage_report:
        user_turn += (
            "## Read-only Triage Report\n"
            f"```json\n{json.dumps(triage_report, indent=2)[:8000]}\n```\n\n"
        )
    user_turn += (
        f"{_EXPLORE_TOOLS_MD}\n"
        "Explore the workspace with read-only tools, then emit "
        '`{"action": "plan", "plan": {...}}`.'
    )

    conversation: list[dict[str, Any]] = [{"role": "user", "content": user_turn}]
    tool_call_count = 0
    non_json_retries = 0

    while tool_call_count < budget:
        messages = trim_conversation(conversation, max_turns=16)
        span_model = (
            resolve_model_config(model, "orchestrator").model_name
            if model != "auto"
            else model
        )
        with span_llm_call("orchestrator", "explore", span_model, session=session):
            llm_result = call_llm(
                messages=messages,
                model=model,
                max_tokens=int(os.getenv("MAX_TOKENS_ORCHESTRATOR", "24576")),
                system_prompt=orchestrator_md,
                json_mode=True,
                role="orchestrator",
                stream_context=stream_context_for(
                    emitter,
                    "orchestrator",
                    phase="explore",
                    output_kind="json",
                ),
            )

        raw = llm_result.text.strip()
        parsed = parse_json_from_text(raw)
        if parsed is None:
            non_json_retries += 1
            if non_json_retries >= _NON_JSON_RETRIES:
                raise RuntimeError(
                    "Orchestrator exploration failed: invalid JSON after retries."
                )
            conversation.append({"role": "assistant", "content": raw})
            conversation.append({
                "role": "user",
                "content": (
                    "Invalid JSON. Emit a tool call "
                    '{"tool": "...", "args": {...}, "reasoning": "..."} '
                    'or final plan {"action": "plan", "plan": {...}}.'
                ),
            })
            continue
        non_json_retries = 0

        if parsed.get("action") == "plan":
            plan = parsed.get("plan")
            if not isinstance(plan, dict):
                conversation.append({"role": "assistant", "content": raw})
                conversation.append({
                    "role": "user",
                    "content": '"plan" must be a JSON object when action is "plan".',
                })
                continue
            error = validate_plan_payload(plan)
            if error:
                conversation.append({"role": "assistant", "content": raw})
                conversation.append({
                    "role": "user",
                    "content": f"Plan rejected: {error}. Fix and emit action=plan again.",
                })
                continue
            if emitter:
                emitter.emit(
                    "orchestrator.explore.finished",
                    tool_calls=tool_call_count,
                    mode=explore_mode,
                )
            return plan

        calls, call_error, call_note = _extract_explore_calls(parsed)
        if call_error:
            conversation.append({"role": "assistant", "content": raw})
            conversation.append({"role": "user", "content": call_error})
            continue

        batch_results: list[str] = []
        for call in calls:
            tool_name = str(call.get("tool", "") or "")
            tool_args = call.get("args", {}) or {}
            reasoning = str(call.get("reasoning", "") or "")

            if tool_name not in _ORCHESTRATOR_READ_TOOLS:
                batch_results.append(
                    f"Tool `{tool_name}` is not available during exploration. "
                    f"Allowed: {sorted(_ORCHESTRATOR_READ_TOOLS)}"
                )
                continue

            contract_error = validate_tool_call(tool_name, tool_args)
            if contract_error:
                tool_call_count += 1
                batch_results.append(
                    f"Tool result for `{tool_name}`:\n```json\n"
                    f"{compact_tool_result(contract_error)}\n```"
                )
                continue

            print(
                f"  [Orchestrator] explore[{tool_call_count + 1}]: "
                f"{tool_name}({list(tool_args.keys())}) — {reasoning[:80]}"
            )
            if emitter:
                emitter.emit(
                    "tool.called",
                    tool=tool_name,
                    args_keys=list(tool_args.keys()),
                    reasoning=reasoning,
                    call_index=tool_call_count + 1,
                    role="orchestrator",
                    phase="explore",
                )

            started = time.perf_counter()
            with span_tool_call(tool_name, "orchestrator_explore", session=session):
                tool_result = dispatch(tool_name, tool_args)
            tool_call_count += 1
            duration_ms = (time.perf_counter() - started) * 1000.0

            if emitter:
                emitter.emit(
                    "tool.result",
                    tool=tool_name,
                    success=tool_result.get("success", False),
                    call_index=tool_call_count,
                    role="orchestrator",
                    phase="explore",
                    **event_diagnostics(tool_name, tool_args, tool_result, duration_ms),
                )

            batch_results.append(
                f"Tool result for `{tool_name}`:\n```json\n"
                f"{compact_tool_result(tool_result)}\n```"
            )

        feedback = "\n\n".join(batch_results)
        if call_note:
            feedback = f"{call_note}\n\n{feedback}"
        conversation.append({"role": "assistant", "content": raw})
        conversation.append({"role": "user", "content": feedback})

    raise RuntimeError(
        f"Orchestrator exploration exhausted budget ({budget} tool calls) "
        "without emitting action=plan."
    )


def _workspace_has_meaningful_code(workspace_root: Path) -> bool:
    count = 0
    for path in workspace_root.rglob("*"):
        if not path.is_file() or any(part in _SKIP_DIRS for part in path.parts):
            continue
        if path.name in {".gitkeep", "pytest.ini"}:
            continue
        if path.suffix.lower() in _TEXT_SUFFIXES or path.suffix == "":
            count += 1
            if count >= 2:
                return True
    return False


def _grep_patterns_from_request(
    user_request: str,
    previous_plan: Optional[dict[str, Any]],
) -> list[str]:
    patterns: list[str] = []
    seen: set[str] = set()

    def _add(raw: str) -> None:
        token = raw.strip()
        if not token or token in seen:
            return
        seen.add(token)
        patterns.append(token)

    for match in _FILE_PATH_RE.finditer(user_request):
        _add(Path(match.group(1)).name)
        _add(match.group(1).replace("workspace/", ""))

    for match in _QUOTED_RE.finditer(user_request):
        _add(match.group(1))

    for match in _PORT_RE.finditer(user_request):
        _add(match.group(0))

    for match in _URL_RE.finditer(user_request):
        _add(match.group(0))

    plan = previous_plan or {}
    for milestone in plan.get("milestones", []):
        for raw in milestone.get("target_files", []):
            _add(str(raw))
            _add(Path(str(raw)).name)

    # Common defect tokens when the user did not quote them.
    for token in ("error", "fail", "bug", "fix", "localhost", "Exception"):
        if token.lower() in user_request.lower():
            _add(token)

    return patterns


def _extract_explore_calls(
    parsed: dict,
) -> tuple[list[dict], Optional[str], Optional[str]]:
    raw_calls = parsed.get("calls")
    if isinstance(raw_calls, list) and raw_calls:
        calls = [c for c in raw_calls if isinstance(c, dict)]
        if not calls:
            return [], 'The "calls" array must contain tool objects.', None
        if len(calls) > MAX_ORCHESTRATOR_BATCH_CALLS:
            return calls[:MAX_ORCHESTRATOR_BATCH_CALLS], None, (
                f"NOTE: batch truncated to {MAX_ORCHESTRATOR_BATCH_CALLS} calls."
            )
        return calls, None, None
    if parsed.get("tool"):
        return [parsed], None, None
    return [], (
        'Emit {"tool": "...", "args": {...}, "reasoning": "..."} '
        'or {"action": "plan", "plan": {...}}.'
    ), None
