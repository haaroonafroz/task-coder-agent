"""
Missions Runtime Engine — Phase 3.

Implements the four-phase serial execution loop described in the blueprint:

  1. ORCHESTRATION  — LLM reads orchestrator.md, decomposes request → plan.json
  2. SKILL ROUTING  — DynamicToolRouter queries Qdrant to inject top-3 tools
  3. WORKER         — LLM reads worker.md + curated tools, executes tool calls
  4. VALIDATION     — LLM reads validator.md, runs contract, PASS → commit / FAIL → retry

Hardware note: only ONE LLM inference runs at any moment (serial design)

Usage:
    python -m src.main "Build a Python REST API with FastAPI that exposes /health and /echo endpoints"

    Or programmatically:
        from src.main import MissionsRuntime
        runtime = MissionsRuntime()
        runtime.run("Add unit tests for the utils module")
"""

from __future__ import annotations

import json
import os
import sys
import time
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Internal imports
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from src.llm_client import call_llm, LLMResult, ModelChoice
from src.tool_registry import DynamicToolRouter
from src.telemetry import initialize_observability, span_llm_call, span_tool_call
from src.tools import dispatch, AVAILABLE_TOOLS

_CONFIG_DIR = _ROOT / "config"
_MISSION_DIR = _ROOT / "active_mission"
_PLAN_PATH = _MISSION_DIR / "plan.json"
_HANDOFFS_DIR = _MISSION_DIR / "handoffs"
_WORKSPACE_DIR = _ROOT / "workspace"

_ORCHESTRATOR_MD = (_CONFIG_DIR / "orchestrator.md").read_text()
_WORKER_MD = (_CONFIG_DIR / "worker.md").read_text()
_VALIDATOR_MD = (_CONFIG_DIR / "validator.md").read_text()
_SKILLS_PATH = _CONFIG_DIR / "skills.md"

MAX_WORKER_TOOL_CALLS = 20   # per milestone
MAX_RETRY_CYCLES = 3         # validator FAIL → worker retry limit
_NON_JSON_RETRIES = 3  # max format-recovery attempts per worker run

MAX_TOKENS_ORCHESTRATOR = int(os.getenv("MAX_TOKENS_ORCHESTRATOR", "8192"))
MAX_TOKENS_WORKER       = int(os.getenv("MAX_TOKENS_WORKER", "5120"))
MAX_TOKENS_VALIDATOR    = int(os.getenv("MAX_TOKENS_VALIDATOR", "3072"))

_PYTEST_INI = """\
[pytest]
pythonpath = .
testpaths = tests
"""

def _ensure_workspace() -> None:
    """Create workspace/ and pytest import bootstrap before any tool calls."""
    _WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)

    gitkeep = _WORKSPACE_DIR / ".gitkeep"
    if not gitkeep.exists():
        gitkeep.touch()

    pytest_ini = _WORKSPACE_DIR / "pytest.ini"
    if not pytest_ini.exists():
        pytest_ini.write_text(_PYTEST_INI, encoding="utf-8")

    (_WORKSPACE_DIR / "tests").mkdir(parents=True, exist_ok=True)

def _workspace_context_block(target_files: list[str] | None = None) -> str:
    tree_result = dispatch("list_directory", {"target_dir": "."})
    tree = tree_result.get("tree", "(empty)") if tree_result.get("success") else "(unavailable)"
    targets_note = ""
    if target_files:
        targets_note = (
            "\n- Required files for this milestone: "
            + ", ".join(f"`{p}`" for p in target_files)
            + "\n- Other files/dirs in the tree are context only — do not edit them unless listed above.\n"
        )
    return (
        "## Workspace Context\n"
        f"- Project root (your cwd): `{_WORKSPACE_DIR}`\n"
        "- All tool paths are relative to this directory.\n"
        "- Do NOT prefix paths with `workspace/`.\n"
        f"{targets_note}\n"
        f"### Current directory tree\n```\n{tree}\n```\n"
    )

def _normalize_milestone_for_worker(milestone: dict) -> dict:
    """Return a copy with workspace/-stripped paths for LLM consumption."""
    from src.tools.paths import normalize_workspace_path, normalize_shell_command
    ms = dict(milestone)
    ms["target_files"] = [
        normalize_workspace_path(p) for p in milestone.get("target_files", [])
    ]
    contract = dict(milestone.get("validation_contract", {}))
    if contract.get("command"):
        contract["command"] = normalize_shell_command(contract["command"])
    ms["validation_contract"] = contract
    return ms

def _target_files_exist(target_files: list[str]) -> tuple[bool, list[str]]:
    from src.tools.paths import resolve_workspace_path
    missing = []
    for rel in target_files:
        p = resolve_workspace_path(rel)
        if not p.exists():
            missing.append(rel)
    return (len(missing) == 0, missing)

def _worker_milestone_brief(ms: dict) -> str:
    """Build an unambiguous milestone brief for the non-thinking worker."""
    target_files = ms.get("target_files", [])
    targets_md = "\n".join(f"- `{p}`" for p in target_files) if target_files else "- (none listed)"
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


def _memory_constraints_block(memory: Any | None, milestone: dict | None = None) -> str:
    """
    Return persisted negative constraints from memory_store.json for worker injection.

    Optionally filters to errors mentioning target files from the current milestone.
    """
    if memory is None:
        return ""

    try:
        constraints = memory.get_error_constraints()
        if not constraints:
            return ""

        # Optional: filter to constraints relevant to this milestone's target files
        if milestone:
            from src.tools.paths import normalize_workspace_path
            target_files = [
                normalize_workspace_path(p) for p in milestone.get("target_files", [])
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

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class MilestoneHandoff:
    milestone_id: str
    title: str
    worker_summary: str
    files_modified: list[str]
    tool_calls_made: int
    retry_count: int
    verdict: str         # "PASS" | "FAIL" | "BLOCKED"
    commit_hash: str
    elapsed_ms: float
    error_log: list[str] = field(default_factory=list)


@dataclass
class MissionResult:
    mission_id: str
    title: str
    status: str           # "completed" | "partial" | "failed"
    milestones_passed: int
    milestones_total: int
    handoffs: list[MilestoneHandoff]
    total_elapsed_ms: float
    model_used: str


# ---------------------------------------------------------------------------
# Runtime engine
# ---------------------------------------------------------------------------

class MissionsRuntime:
    """
    Serial Missions execution engine.

    One agent role runs at a time; each phase completes before the next begins.
    """

    def __init__(
        self,
        model: ModelChoice = "auto",
        telemetry: bool = True,
        memory: bool = True,
    ) -> None:
        self.model = model
        self._router = DynamicToolRouter(_SKILLS_PATH)

        if telemetry:
            initialize_observability()

        self._memory = None
        if memory:
            try:
                from src.memory_layer import MissionMemory
                self._memory = MissionMemory()
            except Exception as exc:
                print(f"[Runtime] Memory layer unavailable: {exc}")

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self, user_request: str) -> MissionResult:
        """
        Execute a full mission from a user request.

        Args:
            user_request: Plain-English description of what to build or fix.

        Returns:
            MissionResult with per-milestone handoff telemetry.
        """
        t_mission_start = time.perf_counter()
        print(f"\n{'='*70}")
        print(f"  MISSIONS RUNTIME — starting")
        print(f"  Request: {user_request[:80]}{'...' if len(user_request) > 80 else ''}")
        print(f"{'='*70}\n")

        _ensure_workspace()
        print(f"[Runtime] Workspace ready: {_WORKSPACE_DIR}")

        # Phase 1 — Orchestration
        plan = self._orchestrate(user_request)
        milestones = plan.get("milestones", [])
        mission_id = plan.get("mission_id", str(uuid.uuid4())[:8])
        title = plan.get("title", "Untitled Mission")

        print(f"[Orchestrator] Mission '{title}' decomposed into {len(milestones)} milestones.")

        handoffs: list[MilestoneHandoff] = []
        passed = 0

        ms_index = 0
        while ms_index < len(milestones):
            ms = milestones[ms_index]
            ms_id = ms.get("id", "?")
            ms_title = ms.get("title", "")

            print(f"\n{'─'*70}")
            print(f"  MILESTONE {ms_id}: {ms_title}")
            print(f"{'─'*70}")

            # Check milestone-level crash recovery from Cognee
            if self._memory:
                resume = self._memory.check_resume_point(ms_id)
                if resume and resume.get("status") == "completed":
                    print(f"  [Memory] Milestone {ms_id} already completed — skipping.")
                    passed += 1
                    ms_index += 1
                    continue

            handoff = self._execute_milestone(ms, plan)
            handoffs.append(handoff)

            if handoff.verdict == "PASS":
                passed += 1
                # Save successful state to Cognee
                if self._memory:
                    self._memory.log_milestone_state(ms_id, asdict(handoff), "completed")
                ms_index += 1
            elif handoff.verdict == "REPLAN":
                # Trigger replan, update the plan in memory
                plan = self._replan_mission(plan, handoff.error_log[-1])
                milestones = plan.get("milestones", [])
                continue
            else:
                print(f"  [Runtime] Milestone {ms_id} FAILED after {handoff.retry_count} retries — halting mission.")
                break

        total_ms = (time.perf_counter() - t_mission_start) * 1000.0
        status = "completed" if passed == len(milestones) else ("partial" if passed > 0 else "failed")

        result = MissionResult(
            mission_id=mission_id,
            title=title,
            status=status,
            milestones_passed=passed,
            milestones_total=len(milestones),
            handoffs=handoffs,
            total_elapsed_ms=round(total_ms, 2),
            model_used=self.model,
        )

        self._print_summary(result)
        return result

    # ------------------------------------------------------------------
    # Phase 1 — Orchestration
    # ------------------------------------------------------------------

    def _orchestrate(self, user_request: str) -> dict:
        """
        Call the Orchestrator LLM to decompose the request into a milestone plan.
        Saves the result to active_mission/plan.json.
        """
        print("\n[Phase 1] ORCHESTRATION — decomposing request…")

        # Load prior plan if it exists and is valid (crash-recovery path)
        if _PLAN_PATH.exists():
            raw_plan = _PLAN_PATH.read_text(encoding="utf-8").strip()
            if raw_plan:
                try:
                    existing = json.loads(raw_plan)
                    pending = [
                        m for m in existing.get("milestones", [])
                        if m.get("status") != "completed"
                    ]
                    if pending:
                        print(
                            f"  [Orchestrator] Resuming existing plan "
                            f"'{existing.get('title')}' — {len(pending)} milestones pending."
                        )
                        return existing
                except json.JSONDecodeError as exc:
                    print(f"  [Orchestrator] Corrupt plan.json ignored: {exc}")
            else:
                print("  [Orchestrator] Empty plan.json ignored — generating new plan.")

        prompt = (
            f"{_ORCHESTRATOR_MD}\n\n"
            f"---\n\n"
            f"User Request:\n{user_request}\n\n"
            f"Output the JSON plan now:"
        )

        with span_llm_call("orchestrator", "init", self.model):
            result = call_llm(prompt, model=self.model, max_tokens=MAX_TOKENS_ORCHESTRATOR,
             json_mode=True, enable_thinking=True)

        try:
            plan = json.loads(result.text)
        except json.JSONDecodeError:
            # Attempt extraction from text block
            import re
            match = re.search(r"\{.*\}", result.text, re.DOTALL)
            if match:
                plan = json.loads(match.group())
            else:
                raise RuntimeError(f"Orchestrator returned non-JSON output:\n{result.text[:500]}")

        # Persist plan
        _MISSION_DIR.mkdir(parents=True, exist_ok=True)
        _PLAN_PATH.write_text(json.dumps(plan, indent=2), encoding="utf-8")
        print(f"  [Orchestrator] Plan saved → {_PLAN_PATH}")
        return plan
    
    # Define Replan Logic
    def _replan_mission(self, current_plan: dict, replan_guidance: str) -> dict:
        """
        Call the Orchestrator LLM to patch the plan based on Validator feedback.
        """
        print(f"\n[Phase 1.5] DYNAMIC RESCOPING — Orchestrator patching plan…")
        
        prompt = (
            f"{_ORCHESTRATOR_MD}\n\n"
            f"---\n\n"
            f"## Negotiation Boundary: Plan Flaw Detected\n"
            f"The Validator has rejected the current plan due to a structural flaw or command mismatch.\n\n"
            f"### Current Plan\n```json\n{json.dumps(current_plan, indent=2)}\n```\n\n"
            f"### Validator's Replan Guidance\n{replan_guidance}\n\n"
            f"Output an UPDATED `plan.json` fixing this issue. Keep completed milestones intact."
        )
        with span_llm_call("orchestrator_replan", "REPLAN", self.model):
            result = call_llm(prompt, model=self.model, max_tokens=MAX_TOKENS_ORCHESTRATOR, json_mode=True, enable_thinking=True)
        parsed = _parse_json_from_text(result.text)
        if parsed is None:
            raise RuntimeError(f"Orchestrator returned non-JSON during replan:\n{result.text[:500]}")
        _PLAN_PATH.write_text(json.dumps(parsed, indent=2), encoding="utf-8")
        print("  [Orchestrator] Plan successfully patched and saved.")
        return parsed

    # ------------------------------------------------------------------
    # Phase 2 + 3 + 4 — Skill routing + Worker + Validation
    # ------------------------------------------------------------------

    def _execute_milestone(self, milestone: dict, plan: dict) -> MilestoneHandoff:
        """
        Run the full Phase 2→3→4 loop for a single milestone.
        Retries the worker up to MAX_RETRY_CYCLES times on validator FAIL.
        """
        ms_id = milestone.get("id", "?")
        ms_title = milestone.get("title", "")
        contract = milestone.get("validation_contract", {})
        t_start = time.perf_counter()
        retry_count = 0
        error_log: list[str] = []
        
        # Programmatic test lock toggle: Check if this is a designated spec or test writing milestone
        ms_title_lower = ms_title.lower()
        ms_desc_lower = milestone.get("description", "").lower()
        is_test_milestone = "test" in ms_title_lower or "spec" in ms_title_lower or "test" in ms_desc_lower or "spec" in ms_desc_lower
        
        from src.tools.file_ops import set_allow_test_edits
        set_allow_test_edits(is_test_milestone)
        
        while retry_count < MAX_RETRY_CYCLES:
            # Phase 2 — Dynamic skill routing
            intent = f"{ms_title}: {milestone.get('description', '')}"
            curated_tools_md = self._router.fetch_curated_skills(intent, top_k=3)
            print(f"\n  [Phase 2] SKILL ROUTING — top tools selected for milestone {ms_id}.")

            # Phase 3 — Worker implementation
            worker_result = self._run_worker(
                milestone=milestone,
                plan=plan,
                curated_tools_md=curated_tools_md,
                error_feedback=error_log[-1] if error_log else None,
                retry_count=retry_count,
            )

            if worker_result.get("status") == "blocked":
                reason = worker_result.get("reason", "Unknown block")
                error_log.append(f"Worker blocked: {reason}")
                print(f"  [Runtime] Worker BLOCKED: {reason}")
                elapsed_ms = (time.perf_counter() - t_start) * 1000.0
                return MilestoneHandoff(
                    milestone_id=ms_id, title=ms_title,
                    worker_summary=reason, files_modified=[],
                    tool_calls_made=worker_result.get("tool_calls", 0),
                    retry_count=retry_count, verdict="BLOCKED",
                    commit_hash="", elapsed_ms=round(elapsed_ms, 2),
                    error_log=error_log,
                )

            # Phase 4 — Adversarial validation
            print(f"\n  [Phase 4] ADVERSARIAL VALIDATION — milestone {ms_id}…")
            verdict_data = self._run_validator(milestone, worker_result)
            verdict = verdict_data.get("verdict", "FAIL")

            if verdict == "PASS":
                # Commit the workspace changes
                commit_msg = f"feat({ms_id}): {ms_title}"
                commit_result = dispatch("git_commit", {"message": commit_msg})
                commit_hash = commit_result.get("commit_hash", "")
                elapsed_ms = (time.perf_counter() - t_start) * 1000.0
                self._save_handoff(ms_id, ms_title, worker_result, verdict_data, commit_hash, retry_count)
                print(f"\n  [✓] Milestone {ms_id} PASSED — commit {commit_hash}")
                return MilestoneHandoff(
                    milestone_id=ms_id, title=ms_title,
                    worker_summary=worker_result.get("summary", ""),
                    files_modified=worker_result.get("files_modified", []),
                    tool_calls_made=worker_result.get("tool_calls", 0),
                    retry_count=retry_count, verdict="PASS",
                    commit_hash=commit_hash, elapsed_ms=round(elapsed_ms, 2),
                    error_log=error_log,
                )
            elif verdict == "REPLAN":
                replan_guidance = verdict_data.get("replan_guidance")
                if not _valid_replan_guidance(replan_guidance):
                    print("\n  [Validator] REPLAN rejected — missing replan_guidance. Treating as FAIL.")
                    error_log.append("REPLAN rejected: empty replan_guidance.")
                    retry_count += 1
                    continue
                error_log.append(f"REPLAN requested: {replan_guidance}")
                print(f"\n  [!] Validator requested REPLAN: {replan_guidance}")
                elapsed_ms = (time.perf_counter() - t_start) * 1000.0
                return MilestoneHandoff(
                    milestone_id=ms_id, title=ms_title,
                    worker_summary="Validator requested plan negotiation.",
                    files_modified=worker_result.get("files_modified", []),
                    tool_calls_made=worker_result.get("tool_calls", 0),
                    retry_count=retry_count, verdict="REPLAN",
                    commit_hash="", elapsed_ms=round(elapsed_ms, 2),
                    error_log=error_log,
                )
            else:
                # FAIL — collect error telemetry, log to Cognee, retry
                errors = verdict_data.get("errors", [])
                fix_guidance = verdict_data.get("fix_guidance", "")
                error_summary = f"[Retry {retry_count+1}] Errors: {errors} | Guidance: {fix_guidance}"
                error_log.append(error_summary)
                print(f"\n  [✗] Milestone {ms_id} FAILED (retry {retry_count+1}/{MAX_RETRY_CYCLES})")
                print(f"      Errors: {'; '.join(errors[:3])}")

                # Log failure to Cognee for negative constraint injection
                if self._memory:
                    files = worker_result.get("files_modified", [])
                    for fpath in files:
                        self._memory.log_compilation_failure(fpath, "\n".join(errors))

                retry_count += 1

        # Exhausted retries
        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        return MilestoneHandoff(
            milestone_id=ms_id, title=ms_title,
            worker_summary="Exhausted retries",
            files_modified=[], tool_calls_made=0,
            retry_count=retry_count, verdict="FAIL",
            commit_hash="", elapsed_ms=round(elapsed_ms, 2),
            error_log=error_log,
        )

    # ------------------------------------------------------------------
    # Worker multi-turn tool-call loop
    # ------------------------------------------------------------------

    def _run_worker(
        self,
        milestone: dict,
        plan: dict,
        curated_tools_md: str,
        error_feedback: Optional[str],
        retry_count: int,
    ) -> dict[str, Any]:
        """
        Run the worker agent in a multi-turn conversation loop until it
        signals "complete" or "blocked", or the tool call budget is exhausted.

        Returns a dict with: status, summary, files_modified, tool_calls.
        """
        ms_id = milestone.get("id", "?")
        print(f"\n  [Phase 3] WORKER — milestone {ms_id} (attempt {retry_count + 1})")
        ms = _normalize_milestone_for_worker(milestone)
        target_files = ms.get("target_files", [])

        # Fetch memory grounding if available
        grounding = ""
        if self._memory:
            try:
                class_hints = self._memory.query_structural_memory(milestone.get("title", ""))
                if class_hints:
                    grounding = f"\n\n## Memory Grounding\n{class_hints}"
            except Exception as exc:
                print(f"    [Worker] Memory grounding unavailable: {exc}")

        memory_constraints = _memory_constraints_block(self._memory, milestone)
        if memory_constraints.strip():
            print("    [Worker] Injecting memory constraints from prior failures.")

        system_prompt = _WORKER_MD

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
            + (f"\n## Error Feedback from Validator (latest retry)\n{error_feedback}\n" if error_feedback else "")
            + f"\n## Available Tools\n{curated_tools_md}\n\n"
            "Start now. Emit ONE JSON tool call for the first required step."
        )

        conversation: list[dict] = [{"role": "user", "content": user_turn}]
        tool_call_count = 0
        non_json_retries = 0
        files_modified: list[str] = []

        while tool_call_count < MAX_WORKER_TOOL_CALLS:
            with span_llm_call("worker", ms_id, self.model):
                llm_result = call_llm(
                    prompt=conversation[-1]["content"] if len(conversation) == 1 else _flatten_conversation(conversation),
                    model=self.model,
                    max_tokens=MAX_TOKENS_WORKER,
                    system_prompt=system_prompt,
                )

            raw = llm_result.text.strip()
            parsed = _parse_json_from_text(raw)

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

            non_json_retries = 0  # reset on valid JSON
            status = parsed.get("status")

            if status == "complete":
                files_modified.extend(parsed.get("files_modified", []))
                print(f"    [Worker] Signalled COMPLETE after {tool_call_count} tool call(s).")
                ok, missing = _target_files_exist(target_files)
                if target_files and not ok:
                    print(f"    [Worker] Rejected premature COMPLETE — missing: {missing}")
                    conversation.append({"role": "assistant", "content": raw})
                    conversation.append({
                        "role": "user",
                        "content": (
                            f"REJECTED. These required files are still missing: {missing}. "
                            f"Use write_file to create them, then signal complete again."
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

            print(f"    [Worker] tool_call[{tool_call_count+1}]: {tool_name}({list(tool_args.keys())}) — {reasoning}")

            with span_tool_call(tool_name, ms_id):
                tool_result = dispatch(tool_name, tool_args)

            tool_call_count += 1

            if tool_name in ("write_file", "patch_file") and tool_result.get("success"):
                from src.tools.paths import normalize_workspace_path
                fpath = normalize_workspace_path(tool_args.get("file_path", ""))
                if fpath:
                    files_modified.append(fpath)

            result_text = json.dumps(tool_result, indent=2)[:4000]
            next_msg = (
                f"Tool result for `{tool_name}`:\n```json\n{result_text}\n```\n\n"
                "Continue toward the required deliverables."
            )

            # Warn if worker wrote something not in Target Files
            if tool_name in ("write_file", "patch_file") and target_files:
                from src.tools.paths import normalize_workspace_path
                written = normalize_workspace_path(tool_args.get("file_path", ""))
                if written and written not in target_files:
                    next_msg += (
                        f"\n\nWARNING: `{written}` is NOT in Required deliverables: {target_files}. "
                        "Do not add scaffolding. Create/edit only the required files next."
                    )

            ok, missing = _target_files_exist(target_files)
            if missing:
                next_msg += f"\n\nStill missing: {missing}"
            else:
                next_msg += "\n\nAll required files exist. Signal {\"status\": \"complete\", ...} if the work is done."

            conversation.append({"role": "assistant", "content": raw})
            conversation.append({"role": "user", "content": next_msg})

        # Budget exhausted — fail if deliverables missing
        ok, missing = _target_files_exist(target_files)
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

    # ------------------------------------------------------------------
    # Validator
    # ------------------------------------------------------------------

    def _run_validator(self, milestone: dict, worker_result: dict) -> dict[str, Any]:
        """
        Execute the validation contract and ask the validator LLM to render a verdict.
        """
        ms_id = milestone.get("id", "?")
        contract = milestone.get("validation_contract", {})
        command = contract.get("command", "")
        target_files = milestone.get("target_files", [])
        retry_count = worker_result.get("retry_count", 0)

        # 1. Structural Audit: Ensure the worker didn't edit any test file
        modified_files = worker_result.get("files_modified", [])
        unauthorized_edits = [
            f for f in modified_files 
            if "tests/" in f or "test_" in f
        ]
        
        # If this is not a designated spec/test-writing milestone, reject test edits
        is_test_milestone = "test" in milestone.get("title", "").lower() or "spec" in milestone.get("title", "").lower()
        if unauthorized_edits and not is_test_milestone:
            print(f"    [Validator] REJECTED: Worker illegally modified test files: {unauthorized_edits}")
            return {
                "verdict": "FAIL",
                "milestone_id": ms_id,
                "errors": [
                    f"Specification Gaming Detected: Worker altered test scripts {unauthorized_edits} to force a pass."
                ],
                "root_cause": "Worker altered the validation test files instead of correcting implementation code.",
                "fix_guidance": "Revert all modifications to test files. Correct the implementation code in your assigned source files so that it correctly passes the original, unmodified test suite."
            }

        # Execute the contract command if specified
        contract_output = ""
        if command:
            print(f"    [Validator] Running: {command}")
            tool_result = dispatch("run_shellscript", {"script": command, "timeout": 120})
            contract_output = (
                f"stdout:\n{tool_result.get('stdout', '')}\n"
                f"stderr:\n{tool_result.get('stderr', '')}\n"
                f"returncode: {tool_result.get('returncode', -1)}"
            )
            print(f"    [Validator] returncode={tool_result.get('returncode')}")

        prompt = (
            f"{_VALIDATOR_MD}\n\n"
            f"---\n\n"
            f"## Milestone Under Review\n"
            f"**ID**: {ms_id}\n"
            f"**Title**: {milestone.get('title', '')}\n"
            f"**Description**: {milestone.get('description', '')}\n"
            f"**Target files (worker may ONLY edit these)**: {target_files}\n\n"
            f"## Validation Contract\n```json\n{json.dumps(contract, indent=2)}\n```\n\n"
            f"## Contract Execution Output\n```\n{contract_output or '(no command run)'}\n```\n\n"
            f"## Worker Handoff\n"
            f"Files modified: {worker_result.get('files_modified', [])}\n"
            f"Summary: {worker_result.get('summary', '')}\n\n"
            f"**Retry attempt**: {retry_count + 1} of {MAX_RETRY_CYCLES}\n"
            f"Emit your PASS, FAIL, or REPLAN JSON verdict now:"
        )

        returncode = tool_result.get("returncode") if command else None
        replan = _detect_contract_replan(milestone, worker_result, contract_output, returncode)
        if replan:
            print(f"    [Validator] Deterministic REPLAN: {replan['replan_guidance']}")
            return replan
        
        if returncode == 0 and not (unauthorized_edits and not is_test_milestone):
            return {
                "verdict": "PASS",
                "milestone_id": ms_id,
                "validation_details": "Contract command exited 0.",
            }

        
        with span_llm_call("validator", ms_id, self.model):
            result = call_llm(prompt, model=self.model, max_tokens=MAX_TOKENS_VALIDATOR, json_mode=True, enable_thinking=True)

        parsed = _parse_json_from_text(result.text)
        if parsed is None:
            # If we can't parse the verdict and returncode == 0, auto-PASS
            if command and "returncode: 0" in contract_output:
                return {"verdict": "PASS", "milestone_id": ms_id, "validation_details": "Contract command exited 0."}
            return {
                "verdict": "FAIL",
                "milestone_id": ms_id,
                "errors": ["Could not parse validator response."],
                "fix_guidance": result.text[:500],
            }
        
        parsed = _normalize_validator_verdict(parsed, returncode=returncode)
        # Contract success is authoritative — don't allow LLM to REPLAN over a green run
        if returncode == 0 and parsed.get("verdict") == "REPLAN":
            print("    [Validator] Overriding REPLAN — contract command returned 0.")
            parsed["verdict"] = "PASS"
            parsed.setdefault(
                "validation_details",
                "Contract command exited 0.",
            )
        return parsed

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _save_handoff(
        self,
        ms_id: str,
        ms_title: str,
        worker_result: dict,
        verdict_data: dict,
        commit_hash: str,
        retry_count: int,
    ) -> None:
        """Persist milestone handoff metadata to active_mission/handoffs/."""
        _HANDOFFS_DIR.mkdir(parents=True, exist_ok=True)
        handoff = {
            "milestone_id": ms_id,
            "title": ms_title,
            "verdict": verdict_data.get("verdict"),
            "validation_details": verdict_data.get("validation_details"),
            "worker_summary": worker_result.get("summary"),
            "files_modified": worker_result.get("files_modified"),
            "tool_calls": worker_result.get("tool_calls"),
            "retry_count": retry_count,
            "commit_hash": commit_hash,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        path = _HANDOFFS_DIR / f"{ms_id}_{int(time.time())}.json"
        path.write_text(json.dumps(handoff, indent=2), encoding="utf-8")

        # Update plan.json status
        if _PLAN_PATH.exists():
            plan = json.loads(_PLAN_PATH.read_text())
            for m in plan.get("milestones", []):
                if m.get("id") == ms_id:
                    m["status"] = "completed"
                    break
            _PLAN_PATH.write_text(json.dumps(plan, indent=2), encoding="utf-8")

    @staticmethod
    def _print_summary(result: MissionResult) -> None:
        print(f"\n{'='*70}")
        print(f"  MISSION COMPLETE — {result.status.upper()}")
        print(f"  Title       : {result.title}")
        print(f"  Milestones  : {result.milestones_passed}/{result.milestones_total} passed")
        print(f"  Total time  : {result.total_elapsed_ms/1000:.1f}s")
        print(f"{'='*70}")
        for h in result.handoffs:
            icon = "✓" if h.verdict == "PASS" else "✗"
            print(f"  [{icon}] {h.milestone_id}: {h.title} — {h.verdict} ({h.elapsed_ms/1000:.1f}s, {h.retry_count} retries)")
        print()


# ---------------------------------------------------------------------------
# JSON parsing helpers
# ---------------------------------------------------------------------------

def _looks_like_tool_call_attempt(text: str) -> bool:
    """Heuristic: model tried a tool call but JSON did not parse."""
    lowered = text.lower()
    return (
        '"tool"' in lowered
        or '"write_file"' in lowered
        or '"patch_file"' in lowered
        or '"read_file"' in lowered
    )


def _worker_invalid_json_message(raw: str, target_files: list[str]) -> str:
    """Build a corrective user turn after a failed JSON parse."""
    _, missing = _target_files_exist(target_files)
    if missing:
        missing_line = f"Files still missing on disk: {missing}"
    elif target_files:
        missing_line = "All target files exist on disk — fix JSON format, then continue or signal complete."
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

def _load_json_candidate(text: str, *, repair: bool = False) -> Optional[Any]:
    """Parse one JSON candidate; optionally repair with json-repair."""
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

# Define Replan Logic
def _detect_contract_replan(
    milestone: dict,
    worker_result: dict,
    contract_output: str,
    returncode: int | None,
) -> dict | None:
    import re
    from src.tools.paths import resolve_workspace_path

    command = milestone.get("validation_contract", {}).get("command", "")
    target_files = milestone.get("target_files", [])
    modified = worker_result.get("files_modified", [])

    # Worker only touched allowed source files
    worker_only_source = modified and all(f not in modified for f in target_files if f.startswith("tests/"))
    worker_did_not_touch_tests = not any("tests/" in f or "test_" in f for f in modified)

    # pytest exit code 5 = no tests collected
    if returncode == 5 and "pytest" in command:
        # If tests exist on disk but none ran, likely command/filter mismatch
        test_files = list(resolve_workspace_path(".").glob("tests/test_*.py"))
        if test_files and worker_did_not_touch_tests:
            kw_match = re.search(r"-k\s+(\S+)", command)
            keyword = kw_match.group(1) if kw_match else None
            replan = (
                "Pytest collected zero tests (exit code 5) but test files exist. "
                "This is likely an Orchestrator validation command mismatch."
            )
            if keyword:
                replan += f" The -k filter '{keyword}' may not match any test names."
            return {
                "verdict": "REPLAN",
                "milestone_id": milestone.get("id"),
                "validation_details": replan,
                "errors": ["pytest exit code 5: no tests collected"],
                "root_cause": "Validation contract command does not match existing tests.",
                "fix_guidance": None,
                "replan_guidance": (
                    f"Fix validation_contract.command for {milestone.get('id')}. "
                    f"Current command: {command!r}. "
                    f"Inspect tests/test_*.py function names and use a matching -k expression "
                    f"(e.g. -k tokenize instead of -k tokenizer), or run the specific test functions directly."
                ),
            }

    return None

def _valid_replan_guidance(value: Any) -> bool:
    """True only for a non-empty, meaningful replan instruction."""
    if not isinstance(value, str):
        return False
    cleaned = value.strip().lower()
    return bool(cleaned) and cleaned not in {"null", "none", "n/a", "na"}


def _normalize_validator_verdict(
    parsed: dict,
    *,
    returncode: int | None,
) -> dict:
    """Coerce/normalize validator output so empty REPLAN cannot trigger replan."""
    verdict_raw = str(parsed.get("verdict", "FAIL")).strip().upper()

    if verdict_raw.startswith("PASS"):
        parsed["verdict"] = "PASS"
    elif verdict_raw.startswith("REPLAN"):
        parsed["verdict"] = "REPLAN"
    else:
        parsed["verdict"] = "FAIL"

    # Only upgrade to REPLAN when guidance is actually present
    if parsed["verdict"] != "REPLAN" and _valid_replan_guidance(parsed.get("replan_guidance")):
        parsed["verdict"] = "REPLAN"

    # REPLAN requires actionable guidance
    if parsed["verdict"] == "REPLAN" and not _valid_replan_guidance(parsed.get("replan_guidance")):
        if returncode == 0:
            print("    [Validator] Ignoring REPLAN with empty guidance — contract passed (returncode 0).")
            parsed["verdict"] = "PASS"
            parsed.setdefault(
                "validation_details",
                "Contract command exited 0; ignored invalid REPLAN verdict.",
            )
        else:
            print("    [Validator] Ignoring REPLAN with empty guidance — treating as FAIL.")
            parsed["verdict"] = "FAIL"
            parsed.setdefault(
                "errors",
                ["Validator emitted REPLAN without replan_guidance."],
            )

    return parsed


def _parse_json_from_text(text: str) -> Optional[dict]:
    """Extract and parse the first JSON object from a text string."""
    import re

    candidates: list[str] = []
    stripped = text.strip()
    if stripped:
        candidates.append(stripped)

    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        candidates.append(match.group(1))

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        candidates.append(match.group())

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


def _flatten_conversation(conversation: list[dict]) -> str:
    """Serialise a multi-turn conversation into a single prompt string."""
    parts = []
    for msg in conversation:
        role = msg["role"].upper()
        parts.append(f"[{role}]\n{msg['content']}")
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(
        description="Missions Runtime — serial multi-agent coding engine"
    )
    parser.add_argument("request", nargs="?", help="Coding request to execute")
    parser.add_argument(
        "--model",
        choices=["auto", "local", "gemini", "gpt4o"],
        default="auto",
        help="LLM backend to use (default: auto — tries local → gemini → gpt4o)",
    )
    parser.add_argument("--no-telemetry", action="store_true", help="Disable Arize Phoenix telemetry")
    parser.add_argument("--no-memory", action="store_true", help="Disable Cognee memory layer")
    args = parser.parse_args()

    request = args.request
    if not request:
        request = input("Enter your coding request: ").strip()
    if not request:
        print("No request provided. Exiting.")
        sys.exit(1)

    runtime = MissionsRuntime(
        model=args.model,
        telemetry=not args.no_telemetry,
        memory=not args.no_memory,
    )
    result = runtime.run(request)
    sys.exit(0 if result.status == "completed" else 1)


if __name__ == "__main__":
    main()
