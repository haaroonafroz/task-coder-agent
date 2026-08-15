"""
Missions Runtime Engine.

Implements the four-phase serial execution loop:

  1. ORCHESTRATION  — LLM reads orchestrator.md, decomposes request → plan.json
  2. SKILL ROUTING  — DynamicToolRouter queries Qdrant to inject top-3 tools
  3. WORKER         — LLM reads worker.md + curated tools, executes tool calls
  4. VALIDATION     — LLM reads validator.md, runs contract, PASS → commit / FAIL → retry

Hardware note: only ONE LLM inference runs at any moment (serial design).

Session scoping
---------------
Every run now executes inside an isolated session directory tree under
``sessions/<session_id>/``. The runtime points the global workspace root
(``src.tools.paths.set_workspace_root``) at the session's workspace before
any tool call, so all file/shell tools operate inside the session sandbox.
Legacy ``active_mission/`` + root ``workspace/`` remain untouched.

Usage:
    # Create a new session and run
    python -m src.main "Build a Python REST API with FastAPI"

    # Resume an existing session
    python -m src.main --session <session_id> "Continue the API"

    # Programmatic
        from src.main import MissionsRuntime
        runtime = MissionsRuntime()
        result = runtime.run("Add unit tests for the utils module")
"""

from __future__ import annotations

import json
import os
import sys
import time
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Optional

# ---------------------------------------------------------------------------
# Internal imports
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from src.llm_client import call_llm, ModelChoice
from src.tool_registry import DynamicToolRouter
from src.telemetry import (
    initialize_observability,
    span_mission_run,
    telemetry_context_from_session,
)
from src.tools import dispatch
from src.tools.file_ops import (
    set_allow_test_edits,
    begin_milestone_write_policy,
    clear_milestone_write_policy,
)
from src.tools.paths import set_workspace_root, get_workspace_root
from src.session import SessionContext, SessionManager
from src.sandbox import activate_sandbox, deactivate_sandbox
from src.sandbox.process_manager import format_allowed_ports_block
from src.sandbox.dependency_check import milestone_suggests_dependencies
from src.run_control import RunCancelledError, ensure_not_cancelled
from src.events import EventEmitter, emitter_for_session, register_emitter, unregister_emitter
from src.agents import (
    replan_mission,
    run_orchestration,
    run_triage,
    run_validator,
    run_worker,
)
from src.agents.orchestrator import repair_plan_issues
from src.agents.plan_lint import lint_plan
from src.agents.mission_summary import build_mission_summary

_CONFIG_DIR   = _ROOT / "config"
_SKILLS_PATH  = _CONFIG_DIR / "skills.md"

MAX_RETRY_CYCLES = 3
MAX_REPLANS_PER_MILESTONE = int(os.getenv("MAX_REPLANS_PER_MILESTONE", "2"))
_CORE_WORKER_TOOLS = (
    "read_file",
    "write_file",
    "patch_file",
    "list_directory",
    "search_grep",
    "run_pytest",
    "run_linter",
    "project_info",
    "run_checks",
)
_UI_HINTS = ("ui", "frontend", "react", "vite", "streamlit", "browser", "web app")

_PYTEST_INI = """\
[pytest]
pythonpath = .
testpaths = tests
"""


# ---------------------------------------------------------------------------
# Workspace bootstrap
# ---------------------------------------------------------------------------

def _ensure_session_workspace(ctx: SessionContext) -> None:
    """Create the session workspace, sandbox jail, and pytest bootstrap before any tool calls."""
    ctx.ensure_dirs()
    activate_sandbox(ctx)
    gitkeep = ctx.workspace_root / ".gitkeep"
    if not gitkeep.exists():
        gitkeep.touch()

    pytest_ini = ctx.workspace_root / "pytest.ini"
    if not pytest_ini.exists():
        pytest_ini.write_text(_PYTEST_INI, encoding="utf-8")

    (ctx.workspace_root / "tests").mkdir(parents=True, exist_ok=True)
    set_workspace_root(ctx.workspace_root)


def _is_test_milestone(milestone: dict) -> bool:
    """True when a milestone is explicitly a test/spec writing phase.

    Uses target-file patterns only.
    Free-text keyword matching against title/description is intentionally
    absent because implementation milestones routinely mention "tests"
    in their descriptions (e.g. "implement X to pass M1 tests"), which would
    cause false-positive classification and trigger incorrect replans.
    """
    target_files = milestone.get("target_files", [])
    if not target_files:
        return False
    return all(
        f.startswith("tests/") or f.startswith("test_") or "/test_" in f
        for f in target_files
    )


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
    verdict: str           # "PASS" | "FAIL" | "BLOCKED" | "REPLAN"
    commit_hash: str
    elapsed_ms: float
    error_log: list[str] = field(default_factory=list)
    failure_signature: str = ""  # deterministic fingerprint for replan dedup


@dataclass
class MissionResult:
    mission_id: str
    title: str
    status: str            # "completed" | "partial" | "failed" | "cancelled"
    milestones_passed: int
    milestones_total: int
    handoffs: list[MilestoneHandoff]
    total_elapsed_ms: float
    model_used: str
    session_id: str
    run_kind: str = "new"
    plan_id: Optional[str] = None
    summary_text: str = ""
    failure_reason: str = ""


# ---------------------------------------------------------------------------
# Runtime engine
# ---------------------------------------------------------------------------

class MissionsRuntime:
    """
    Serial Missions execution engine.

    One agent role runs at a time; each phase completes before the next begins.
    Every run is scoped to a :class:`SessionContext` whose workspace is
    activated globally before tool execution.
    """

    def __init__(
        self,
        model: ModelChoice = "auto",
        telemetry: bool = True,
        memory: bool = True,
    ) -> None:
        self.model = model
        self._router = DynamicToolRouter(_SKILLS_PATH)
        self._session_manager = SessionManager()

        if telemetry:
            initialize_observability()

        self._memory_enabled = memory
        self._memory = None
        self._session: Optional[SessionContext] = None
        self._emitter: Optional[EventEmitter] = None
        self._telemetry_ctx = None  # set per-run in run() (Phase 5)
        self._cancel_check: Optional[Callable[[], bool]] = None
        self._active_run_kind = "new"

    def _check_cancelled(self) -> None:
        """Raise if the active run received a cancel request."""
        ensure_not_cancelled(self._cancel_check)

    # ------------------------------------------------------------------
    # Public entry points
    # ------------------------------------------------------------------

    def run(
        self,
        user_request: str,
        session: Optional[SessionContext] = None,
        cancel_check: Optional[Callable[[], bool]] = None,
        run_kind: str = "auto",
    ) -> MissionResult:
        """
        Execute a full mission from a user request inside a session.

        If ``session`` is None a new session is created automatically. If a
        session is provided with an existing pending plan, the mission
        resumes from the first incomplete milestone.

        Args:
            user_request: Plain-English description of what to build or fix.
            session:      Optional pre-existing session to run inside.
            cancel_check: Optional callable returning True when the run should stop.
            run_kind: ``auto`` selects resume for pending plans and repair for
                completed plans; explicit values are ``new``, ``resume``, and
                ``repair``.

        Returns:
            MissionResult with per-milestone handoff telemetry.
        """
        t_mission_start = time.perf_counter()
        self._cancel_check = cancel_check

        # Resolve or create the session
        if session is None:
            title = user_request[:60] + ("..." if len(user_request) > 60 else "")
            session = self._session_manager.create_session(
                title=title, model=self.model
            )
        self._session = session
        effective_run_kind = self._resolve_run_kind(session, run_kind)
        previous_plan = self._load_plan(session)
        parent_plan_id = (
            str(previous_plan.get("plan_id") or previous_plan.get("mission_id") or "")
            if effective_run_kind == "repair"
            else None
        )

        _ensure_session_workspace(session)
        self._session_manager.update_status(session, "running")

        # Event stream for this session (singleton — same instance SSE subscribes to)
        self._emitter = emitter_for_session(session.session_id, session.events_path)
        register_emitter(self._emitter)
        self._emitter.emit(
            "session.started",
            request=user_request,
            model=self.model,
            run_kind=effective_run_kind,
            workspace=str(session.workspace_root),
        )

        print(f"\n{'='*70}")
        print(f"  MISSIONS RUNTIME — session {session.session_id}")
        print(f"  Request: {user_request[:80]}{'...' if len(user_request) > 80 else ''}")
        print(f"  Workspace: {session.workspace_root}")
        print(f"{'='*70}\n")

        # Initialise memory for this session
        self._init_memory(session)

        # Phase 5 — bind this run to a Phoenix session span. The root span
        # parents every child LLM/tool span for the session and carries
        # session.id so traces are filterable per session in the Phoenix UI.
        telemetry_ctx = telemetry_context_from_session(session)
        self._telemetry_ctx = telemetry_ctx

        with span_mission_run(
            session_id=session.phoenix_session_id or session.session_id,
            title=session.title,
            project=session.phoenix_project,
            model=self.model,
        ):
            try:
                return self._run_mission_body(
                    user_request,
                    session,
                    telemetry_ctx,
                    t_mission_start,
                    run_kind=effective_run_kind,
                    parent_plan_id=parent_plan_id,
                    previous_plan=previous_plan,
                )
            except RunCancelledError:
                return self._finish_cancelled(session, t_mission_start)
            finally:
                clear_milestone_write_policy()
                deactivate_sandbox()
                self._cancel_check = None

    def _finish_cancelled(
        self,
        session: SessionContext,
        t_mission_start: float,
    ) -> MissionResult:
        """Emit cancellation events and return a partial mission result."""
        total_ms = (time.perf_counter() - t_mission_start) * 1000.0
        self._session_manager.update_status(session, "paused")

        plan: dict = {}
        if session.plan_path.exists():
            try:
                plan = json.loads(session.plan_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                pass

        milestones = plan.get("milestones", [])
        passed = sum(1 for m in milestones if m.get("status") == "completed")

        if self._emitter:
            self._emitter.emit(
                "mission.cancelled",
                milestones_passed=passed,
                milestones_total=len(milestones),
                total_elapsed_ms=round(total_ms, 2),
            )
            unregister_emitter(session.session_id)

        self._telemetry_ctx = None
        print("\n  [Runtime] Run cancelled by user.")

        return MissionResult(
            mission_id=plan.get("mission_id", ""),
            title=plan.get("title", session.title),
            status="cancelled",
            milestones_passed=passed,
            milestones_total=len(milestones),
            handoffs=[],
            total_elapsed_ms=round(total_ms, 2),
            model_used=self.model,
            session_id=session.session_id,
            run_kind=getattr(self, "_active_run_kind", "new"),
            plan_id=plan.get("plan_id") or plan.get("mission_id"),
        )

    @staticmethod
    def _load_plan(session: SessionContext) -> dict[str, Any]:
        """Load the current plan without treating it as a resume decision."""
        if not session.plan_path.exists():
            return {}
        try:
            value = json.loads(session.plan_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return value if isinstance(value, dict) else {}

    @classmethod
    def _resolve_run_kind(cls, session: SessionContext, requested: str) -> str:
        """Distinguish crash recovery from a user-requested repair."""
        if requested in {"new", "resume", "repair"}:
            return requested
        plan = cls._load_plan(session)
        if any(m.get("status") != "completed" for m in plan.get("milestones", [])):
            return "resume"
        if plan.get("milestones"):
            return "repair"
        return "new"

    # ------------------------------------------------------------------
    # Mission body (orchestration → milestone loop → summary)
    # ------------------------------------------------------------------

    def _run_mission_body(
        self,
        user_request: str,
        session: SessionContext,
        telemetry_ctx,
        t_mission_start: float,
        *,
        run_kind: str,
        parent_plan_id: Optional[str],
        previous_plan: dict[str, Any],
    ) -> MissionResult:
        """Run orchestration + the milestone loop + summary inside a span."""
        self._check_cancelled()

        self._active_run_kind = run_kind
        triage_report: Optional[dict[str, Any]] = None

        # Repair runs get a read-only diagnosis before planning. A triage
        # failure is non-fatal: the Orchestrator can still use the request and
        # current workspace snapshot.
        if run_kind == "repair":
            try:
                triage_report = run_triage(
                    user_request,
                    workspace_root=session.workspace_root,
                    session_root=session.root,
                    model=self.model,
                    previous_plan=previous_plan,
                    session=telemetry_ctx,
                    emitter=self._emitter,
                )
            except Exception as exc:
                print(f"  [Triage] Triage failed (non-fatal): {exc}")
                self._emitter.emit("triage.failed", error=str(exc), fatal=False)

        # Phase 1 — Orchestration (graceful failure: no uncaught tracebacks)
        try:
            plan = run_orchestration(
                user_request, self.model,
                plan_path=session.plan_path,
                mission_dir=session.root,
                run_kind=run_kind,
                parent_plan_id=parent_plan_id,
                triage_report=triage_report,
                session=telemetry_ctx,
                emitter=self._emitter,
            )
        except Exception as exc:
            print(f"  [Orchestrator] Plan generation failed: {exc}")
            self._emitter.emit("mission.failed", phase="orchestration", error=str(exc))
            return MissionResult(
                mission_id="", title=session.title, status="failed",
                milestones_passed=0, milestones_total=0, handoffs=[],
                total_elapsed_ms=round(
                    (time.perf_counter() - t_mission_start) * 1000.0, 2
                ),
                model_used=self.model,
                session_id=session.session_id,
                run_kind=run_kind,
                plan_id=None,
            )

        # Phase 1.2 — deterministic plan lint: fix what's fixable in code,
        # repair the rest with one Orchestrator patch pass — before a single
        # worker cycle is burned on a structurally broken plan.
        plan = self._lint_and_repair_plan(plan, session, telemetry_ctx)

        milestones = plan.get("milestones", [])
        mission_id = plan.get("mission_id", str(uuid.uuid4())[:8])
        plan_id = str(plan.get("plan_id") or mission_id)
        title = plan.get("title", "Untitled Mission")

        self._emitter.emit(
            "plan.created",
            mission_id=mission_id,
            title=title,
            run_kind=run_kind,
            plan_id=plan_id,
            parent_plan_id=parent_plan_id,
            milestones_total=len(milestones),
            milestone_ids=[m.get("id", "?") for m in milestones],
        )

        print(f"[Orchestrator] Mission '{title}' decomposed into {len(milestones)} milestones.")

        handoffs: list[MilestoneHandoff] = []
        passed = 0
        ms_index = 0
        # Replan circuit breakers: consecutive-replan budget per milestone +
        # failure-fingerprint dedup (identical failure → identical replan is
        # always futile, regardless of paraphrased guidance).
        replan_counts: dict[str, int] = {}
        replan_signatures: dict[str, set] = {}

        while ms_index < len(milestones):
            self._check_cancelled()
            ms = milestones[ms_index]
            ms_id    = ms.get("id", "?")
            ms_title = ms.get("title", "")

            print(f"\n{'─'*70}")
            print(f"  MILESTONE {ms_id}: {ms_title}")
            print(f"{'─'*70}")

            # Crash-recovery: only a resume run may reuse completion memory,
            # and the memory must belong to this exact plan.
            if run_kind == "resume" and self._memory:
                resume = self._memory.check_resume_point(ms_id, plan_id=plan_id)
                if resume and resume.get("status") == "completed":
                    print(f"  [Memory] Milestone {ms_id} already completed — skipping.")
                    ms["status"] = "completed"
                    try:
                        session.plan_path.write_text(
                            json.dumps(plan, indent=2),
                            encoding="utf-8",
                        )
                    except OSError as exc:
                        print(f"  [Memory] Could not persist skipped milestone: {exc}")
                    self._emitter.emit("milestone.skipped", milestone_id=ms_id, title=ms_title)
                    passed += 1
                    ms_index += 1
                    continue

            self._emitter.emit("milestone.started", milestone_id=ms_id, title=ms_title)
            handoff = self._execute_milestone(ms, plan, session)
            handoffs.append(handoff)

            if handoff.verdict == "PASS":
                passed += 1
                ms["status"] = "completed"
                if self._memory:
                    self._memory.log_milestone_state(
                        ms_id,
                        asdict(handoff),
                        "completed",
                        plan_id=plan_id,
                    )
                ms_index += 1
            elif handoff.verdict == "REPLAN":
                self._emitter.emit(
                    "milestone.replan",
                    milestone_id=ms_id,
                    guidance=handoff.error_log[-1] if handoff.error_log else "",
                )

                # --- Replan circuit breakers ----------------------------------
                sig = handoff.failure_signature
                seen = replan_signatures.setdefault(ms_id, set())
                replan_count = replan_counts.get(ms_id, 0)

                if sig and sig in seen:
                    print(
                        f"  [Runtime] REPLAN LOOP DETECTED for {ms_id}: identical "
                        f"failure fingerprint ({sig}) seen before. Halting."
                    )
                    self._emitter.emit(
                        "milestone.failed",
                        milestone_id=ms_id,
                        reason="replan_loop_same_failure",
                        failure_signature=sig,
                    )
                    break

                if replan_count >= MAX_REPLANS_PER_MILESTONE:
                    print(
                        f"  [Runtime] Replan budget exhausted for {ms_id} "
                        f"({replan_count}/{MAX_REPLANS_PER_MILESTONE}). Halting."
                    )
                    self._emitter.emit(
                        "milestone.failed",
                        milestone_id=ms_id,
                        reason="replan_budget_exhausted",
                        replan_count=replan_count,
                    )
                    break

                if sig:
                    seen.add(sig)
                replan_counts[ms_id] = replan_count + 1

                try:
                    plan = replan_mission(
                        plan, handoff.error_log[-1], self.model,
                        plan_path=session.plan_path,
                        session=telemetry_ctx,
                        emitter=self._emitter,
                    )
                except Exception as exc:
                    print(f"  [Runtime] Replan failed: {exc} — halting mission.")
                    self._emitter.emit(
                        "milestone.failed",
                        milestone_id=ms_id,
                        reason="replan_failed",
                        error=str(exc),
                    )
                    break

                milestones = plan.get("milestones", [])
                self._emitter.emit(
                    "plan.updated",
                    milestones_total=len(milestones),
                    milestone_ids=[m.get("id", "?") for m in milestones],
                )

                # Re-anchor the loop BY MILESTONE ID — patch-based replans can
                # insert/remove milestones, so index arithmetic is unreliable.
                new_index = next(
                    (i for i, m in enumerate(milestones) if m.get("id") == ms_id),
                    None,
                )
                if new_index is None:
                    # The Orchestrator removed this milestone — its work is
                    # deemed unnecessary; continue with whatever followed it.
                    print(f"  [Runtime] Milestone {ms_id} removed by replan — continuing.")
                    self._emitter.emit("milestone.removed_by_replan", milestone_id=ms_id)
                    continue
                ms_index = new_index
            else:
                print(
                    f"  [Runtime] Milestone {ms_id} FAILED after "
                    f"{handoff.retry_count} retries — halting mission."
                )
                self._emitter.emit(
                    "milestone.failed",
                    milestone_id=ms_id,
                    retry_count=handoff.retry_count,
                    errors=handoff.error_log[-3:] if handoff.error_log else [],
                )
                break

        total_ms = (time.perf_counter() - t_mission_start) * 1000.0
        status = (
            "completed" if passed == len(milestones)
            else ("partial" if passed > 0 else "failed")
        )
        incomplete_ids = [
            str(m.get("id", "?"))
            for m in milestones
            if m.get("status") != "completed"
        ]
        audit_passed = not incomplete_ids and passed == len(milestones)
        if not audit_passed:
            status = "partial" if passed > 0 else "failed"
        self._emitter.emit(
            "mission.audit",
            passed=audit_passed,
            incomplete_milestones=incomplete_ids,
            run_kind=run_kind,
            plan_id=plan_id,
        )

        self._session_manager.update_status(session, status)

        result = MissionResult(
            mission_id=mission_id,
            title=title,
            status=status,
            milestones_passed=passed,
            milestones_total=len(milestones),
            handoffs=handoffs,
            total_elapsed_ms=round(total_ms, 2),
            model_used=self.model,
            session_id=session.session_id,
            run_kind=run_kind,
            plan_id=plan_id,
        )

        summary = build_mission_summary(
            title=title,
            status=status,
            milestones_passed=passed,
            milestones_total=len(milestones),
            total_elapsed_ms=result.total_elapsed_ms,
            handoffs=handoffs,
            incomplete_milestone_ids=incomplete_ids,
        )
        result.summary_text = summary["summary_text"]
        result.failure_reason = summary.get("failure_reason", "")

        if self._emitter:
            self._emitter.emit(
                "mission.complete",
                status=status,
                run_kind=run_kind,
                plan_id=plan_id,
                milestones_passed=passed,
                milestones_total=len(milestones),
                total_elapsed_ms=result.total_elapsed_ms,
                summary_text=result.summary_text,
                failure_reason=result.failure_reason,
                files_modified=summary.get("files_modified", []),
            )

            # Phase 6 — optional auto-eval (default off via MISSIONS_AUTO_EVAL).
            try:
                from src.evals.runner import auto_eval_enabled, run_session_evals
                if auto_eval_enabled():
                    run_session_evals(
                        session,
                        persist=True,
                        model=self.model,
                        emitter=self._emitter,
                    )
            except Exception as exc:
                print(f"[Evals] Auto-eval skipped or failed: {exc}")

            unregister_emitter(session.session_id)

        # Clear the per-run telemetry context now that the mission is done.
        self._telemetry_ctx = None

        self._print_summary(result)
        return result

    # ------------------------------------------------------------------
    # Plan lint (Phase 1.2)
    # ------------------------------------------------------------------

    def _lint_and_repair_plan(
        self,
        plan: dict,
        session: SessionContext,
        telemetry_ctx,
    ) -> dict:
        """
        Deterministically lint the plan; repair remaining issues with one
        Orchestrator patch pass. Never blocks the mission on lint failure —
        worst case the validator catches the flaw at runtime as before.
        """
        try:
            plan, fixes, issues = lint_plan(plan)
        except Exception as exc:
            print(f"  [PlanLint] Linting failed (non-fatal): {exc}")
            return plan

        if fixes:
            for fix in fixes:
                print(f"  [PlanLint] fixed: {fix}")
            session.plan_path.write_text(
                json.dumps(plan, indent=2), encoding="utf-8"
            )
            self._emitter.emit("plan.linted", fixes=fixes)

        if not issues:
            return plan

        for issue in issues:
            print(f"  [PlanLint] issue: {issue}")
        self._emitter.emit("plan.lint_issues", issues=issues)

        try:
            plan = repair_plan_issues(
                plan, issues, self.model,
                plan_path=session.plan_path,
                session=telemetry_ctx,
                emitter=self._emitter,
            )
            _, _, remaining = lint_plan(plan)
            if remaining:
                print(
                    f"  [PlanLint] {len(remaining)} issue(s) remain after repair — "
                    "continuing; the validator will guard at runtime."
                )
                self._emitter.emit("plan.lint_issues", issues=remaining, stage="post_repair")
        except Exception as exc:
            print(f"  [PlanLint] Repair pass failed (non-fatal): {exc}")
            self._emitter.emit("plan.lint_repair_failed", error=str(exc))

        return plan

    # ------------------------------------------------------------------
    # Milestone execution loop (glue: Phase 2 → 3 → 4)
    # ------------------------------------------------------------------

    def _execute_milestone(
        self, milestone: dict, plan: dict, session: SessionContext
    ) -> MilestoneHandoff:
        """
        Run the full Phase 2 → 3 → 4 loop for a single milestone.
        Retries the worker up to MAX_RETRY_CYCLES times on validator FAIL.
        """
        ms_id    = milestone.get("id", "?")
        ms_title = milestone.get("title", "")
        t_start  = time.perf_counter()
        retry_count = 0
        error_log: list[str] = []

        is_test = _is_test_milestone(milestone)
        set_allow_test_edits(is_test)

        # Write jail: scope write/patch calls to this milestone's target_files
        # so out-of-scope edits are rejected at the tool layer (one cheap turn)
        # instead of post-hoc by the validator (one full worker cycle).
        target_files = milestone.get("target_files", [])
        allow_new_test_files = (
            not is_test
            and any(
                str(path).startswith("tests/")
                or str(path).startswith("test_")
                or "/test_" in str(path)
                for path in target_files
            )
        )
        begin_milestone_write_policy(
            target_files,
            allow_new_test_files=allow_new_test_files,
        )

        # The worker's conversation survives validator FAILs within this
        # milestone: retries resume the same thread with new feedback instead
        # of re-deriving the workspace from scratch (cold-start cost).
        conversation: Optional[list] = None
        active_tools: set[str] = set()
        tool_failure_state: dict[str, Any] = {}

        # Route once per packet. Core inspection/edit/test tools are always
        # available so a retry cannot lose the ability to inspect or repair
        # the files it already touched.
        acceptance = milestone.get("acceptance_criteria", [])
        intent = (
            f"{ms_title}: {milestone.get('description', '')}\n"
            f"Acceptance criteria: {acceptance}\n"
            f"Validation profile: {milestone.get('validation_profile', 'auto')}"
        )
        initial_discovery = self._router.search_tools(intent, top_k=3)
        curated_tools_md = initial_discovery.get("documentation", "")
        active_tools.update(initial_discovery.get("tools", []))
        for tool_name in _CORE_WORKER_TOOLS:
            skill = self._router.get_skill_by_name(tool_name)
            if skill and skill not in curated_tools_md:
                curated_tools_md = (
                    f"{skill}\n\n---\n\n{curated_tools_md}"
                    if curated_tools_md
                    else skill
                )
            active_tools.add(tool_name)
        is_ui_packet = (
            str(milestone.get("validation_profile", "")).lower() == "ui"
            or any(str(path).lower().endswith((".html", ".jsx", ".tsx", ".vue"))
                   for path in target_files)
        )
        if is_ui_packet or any(
            hint in f"{ms_title} {milestone.get('description', '')}".lower()
            for hint in _UI_HINTS
        ):
            for tool_name in ("serve_app", "inspect_ui"):
                skill = self._router.get_skill_by_name(tool_name)
                if skill and skill not in curated_tools_md:
                    curated_tools_md = (
                        f"{skill}\n\n---\n\n{curated_tools_md}"
                        if curated_tools_md
                        else skill
                    )
                active_tools.add(tool_name)
            curated_tools_md = (
                f"{curated_tools_md}\n\n{format_allowed_ports_block()}"
                if curated_tools_md
                else format_allowed_ports_block()
            )
        self._emitter.emit(
            "tool.routing",
            milestone_id=ms_id,
            query=intent,
            tools=sorted(active_tools),
            count=len(active_tools),
            routed_tools=sorted(initial_discovery.get("tools", [])),
            core_tools=list(_CORE_WORKER_TOOLS),
        )
        if milestone_suggests_dependencies(milestone, plan):
            install_skill = self._router.get_skill_by_name("install_dependency")
            if install_skill and "install_dependency" not in curated_tools_md:
                curated_tools_md = (
                    f"{install_skill}\n\n---\n\n{curated_tools_md}"
                    if curated_tools_md
                    else install_skill
                )
                active_tools.add("install_dependency")
        print(f"\n  [Phase 2] SKILL ROUTING — tools selected for milestone {ms_id}.")

        while retry_count < MAX_RETRY_CYCLES:
            self._check_cancelled()

            # Phase 3 — Worker implementation
            worker_result = run_worker(
                milestone=milestone,
                plan=plan,
                curated_tools_md=curated_tools_md,
                error_feedback=error_log[-1] if error_log else None,
                retry_count=retry_count,
                model=self.model,
                memory=self._memory,
                emitter=self._emitter,
                session=self._telemetry_ctx,
                cancel_check=self._cancel_check,
                prior_conversation=conversation,
                initial_tool_names=active_tools,
                prior_active_tools=active_tools,
                tool_searcher=self._router.search_tools,
                prior_failure_state=tool_failure_state,
            )
            conversation = worker_result.get("conversation") or conversation
            active_tools.update(worker_result.get("active_tools", []))
            tool_failure_state = worker_result.get(
                "failure_state", tool_failure_state
            )

            if worker_result.get("status") == "cancelled":
                raise RunCancelledError("Run cancelled by user")

            worker_status = worker_result.get("status")
            if worker_status in {"blocked", "request_scope"}:
                reason = worker_result.get("reason", "Unknown block")
                clarification = str(worker_result.get("clarification", "") or "")
                requested_paths = worker_result.get("requested_paths", [])
                if worker_status == "request_scope" and requested_paths:
                    clarification = (
                        f"Requested workspace paths: {requested_paths}. "
                        "The current work packet must be expanded before implementation "
                        "can continue."
                    )
                error_log.append(f"Worker blocked: {reason}")
                print(f"  [Runtime] Worker BLOCKED: {reason}")
                elapsed_ms = (time.perf_counter() - t_start) * 1000.0
                if self._emitter:
                    self._emitter.emit(
                        "milestone.blocked",
                        milestone_id=ms_id,
                        reason=reason,
                        tool_calls=worker_result.get("tool_calls", 0),
                        elapsed_ms=round(elapsed_ms, 2),
                    )

                # A block WITH a clarification request usually means the plan
                # itself is incoherent (missing target file, contradictory
                # contract). Route it to the Orchestrator as a REPLAN instead
                # of dead-halting the mission — replan circuit breakers bound
                # the loop; a bare block still halts immediately.
                if clarification.strip():
                    error_log.append(
                        f"Worker blocked with clarification request: {reason} "
                        f"| Question: {clarification} | Fix the plan so this "
                        "milestone's writable scope, target_files, and validation contract match "
                        "what the implementation actually requires."
                    )
                    return MilestoneHandoff(
                        milestone_id=ms_id, title=ms_title,
                        worker_summary=reason, files_modified=[],
                        tool_calls_made=worker_result.get("tool_calls", 0),
                        retry_count=retry_count, verdict="REPLAN",
                        commit_hash="", elapsed_ms=round(elapsed_ms, 2),
                        error_log=error_log,
                    )

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
            verdict_data = run_validator(
                milestone=milestone,
                worker_result=worker_result,
                retry_count=retry_count,
                is_test_milestone=is_test,
                model=self.model,
                emitter=self._emitter,
                session=self._telemetry_ctx,
                plan=plan,
            )
            verdict = verdict_data.get("verdict", "FAIL")

            if verdict == "PASS":
                commit_msg    = f"feat({ms_id}): {ms_title}"
                commit_result = dispatch("git_commit", {"message": commit_msg})
                commit_hash   = commit_result.get("commit_hash", "")
                elapsed_ms    = (time.perf_counter() - t_start) * 1000.0
                self._save_handoff(
                    ms_id, ms_title, worker_result, verdict_data,
                    commit_hash, retry_count, session,
                )
                if self._emitter:
                    self._emitter.emit(
                        "milestone.passed",
                        milestone_id=ms_id,
                        commit_hash=commit_hash,
                        tool_calls=worker_result.get("tool_calls", 0),
                        files_modified=worker_result.get("files_modified", []),
                        elapsed_ms=round(elapsed_ms, 2),
                    )
                    self._emitter.emit(
                        "handoff.saved",
                        milestone_id=ms_id,
                        commit_hash=commit_hash,
                    )
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

            if verdict == "REPLAN":
                from src.agents.validator import _valid_replan_guidance
                replan_guidance = verdict_data.get("replan_guidance")
                if not _valid_replan_guidance(replan_guidance):
                    print("\n  [Validator] REPLAN rejected — missing replan_guidance. Treating as FAIL.")
                    error_log.append("REPLAN rejected: empty replan_guidance.")
                    retry_count += 1
                    continue
                worker_replan = verdict_data.get("worker_replan")
                if isinstance(worker_replan, dict):
                    error_log.append(
                        f"REPLAN requested: {replan_guidance}\n"
                        "Structured worker replan request:\n"
                        f"{json.dumps(worker_replan, sort_keys=True)}"
                    )
                else:
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
                    failure_signature=verdict_data.get("failure_signature", ""),
                )

            # FAIL — log, optionally store in memory, retry
            errors       = verdict_data.get("errors", [])
            fix_guidance = verdict_data.get("fix_guidance", "")
            error_summary = f"[Retry {retry_count + 1}] Errors: {errors} | Guidance: {fix_guidance}"
            # Hand the worker the UN-digested contract output — LLM summaries
            # of tracebacks routinely lose the exact assertion/line the fix needs.
            raw_output = str(verdict_data.get("contract_output", "") or "")
            if raw_output:
                if len(raw_output) > 1500:
                    raw_output = (
                        raw_output[:500]
                        + "\n...[middle omitted]...\n"
                        + raw_output[-1000:]
                    )
                error_summary += (
                    f"\nRaw validation output (bounded):\n{raw_output}"
                )
            error_log.append(error_summary)
            print(f"\n  [✗] Milestone {ms_id} FAILED (retry {retry_count + 1}/{MAX_RETRY_CYCLES})")
            print(f"      Errors: {'; '.join(errors[:3])}")

            if self._emitter:
                self._emitter.emit(
                    "milestone.retry",
                    milestone_id=ms_id,
                    retry=retry_count + 1,
                    max_retries=MAX_RETRY_CYCLES,
                    errors=errors[:3],
                    fix_guidance=fix_guidance,
                )

            if self._memory:
                for fpath in worker_result.get("files_modified", []):
                    self._memory.log_compilation_failure(fpath, "\n".join(errors))

            retry_count += 1

        # Retries exhausted
        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        if self._emitter:
            self._emitter.emit(
                "milestone.retries_exhausted",
                milestone_id=ms_id,
                retry_count=retry_count,
                elapsed_ms=round(elapsed_ms, 2),
            )
        return MilestoneHandoff(
            milestone_id=ms_id, title=ms_title,
            worker_summary="Exhausted retries",
            files_modified=[], tool_calls_made=0,
            retry_count=retry_count, verdict="FAIL",
            commit_hash="", elapsed_ms=round(elapsed_ms, 2),
            error_log=error_log,
        )

    # ------------------------------------------------------------------
    # Memory
    # ------------------------------------------------------------------

    def _init_memory(self, session: SessionContext) -> None:
        """Initialise the memory layer for this session (if enabled)."""
        self._memory = None
        if not self._memory_enabled:
            return
        try:
            from src.memory_layer import MissionMemory
            self._memory = MissionMemory(memory_file_path=session.memory_store_path)
        except Exception as exc:
            print(f"[Runtime] Memory layer unavailable: {exc}")

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def _save_handoff(
        self,
        ms_id: str,
        ms_title: str,
        worker_result: dict,
        verdict_data: dict,
        commit_hash: str,
        retry_count: int,
        session: SessionContext,
    ) -> None:
        """Persist milestone handoff metadata to the session's handoffs/ dir."""
        session.handoffs_dir.mkdir(parents=True, exist_ok=True)
        handoff = {
            "milestone_id":       ms_id,
            "title":              ms_title,
            "verdict":            verdict_data.get("verdict"),
            "validation_details": verdict_data.get("validation_details"),
            "worker_summary":     worker_result.get("summary"),
            "files_modified":     worker_result.get("files_modified"),
            "tool_calls":         worker_result.get("tool_calls"),
            "retry_count":        retry_count,
            "commit_hash":        commit_hash,
            "timestamp":          time.strftime("%Y-%m-%dT%H:%M:%S"),
            "session_id":         session.session_id,
        }
        path = session.handoffs_dir / f"{ms_id}_{int(time.time())}.json"
        path.write_text(json.dumps(handoff, indent=2), encoding="utf-8")

        # Mark milestone as completed in plan.json
        if session.plan_path.exists():
            try:
                plan = json.loads(session.plan_path.read_text(encoding="utf-8"))
                for m in plan.get("milestones", []):
                    if m.get("id") == ms_id:
                        m["status"] = "completed"
                        break
                session.plan_path.write_text(json.dumps(plan, indent=2), encoding="utf-8")
            except (json.JSONDecodeError, OSError) as exc:
                print(f"  [Runtime] Could not update plan.json: {exc}")

    @staticmethod
    def _print_summary(result: MissionResult) -> None:
        print(f"\n{'='*70}")
        print(f"  MISSION COMPLETE — {result.status.upper()}")
        print(f"  Session     : {result.session_id}")
        print(f"  Title       : {result.title}")
        print(f"  Milestones  : {result.milestones_passed}/{result.milestones_total} passed")
        print(f"  Total time  : {result.total_elapsed_ms / 1000:.1f}s")
        print(f"{'='*70}")
        for h in result.handoffs:
            icon = "✓" if h.verdict == "PASS" else "✗"
            print(
                f"  [{icon}] {h.milestone_id}: {h.title} — "
                f"{h.verdict} ({h.elapsed_ms / 1000:.1f}s, {h.retry_count} retries)"
            )
        print()

    # ------------------------------------------------------------------
    # Session access
    # ------------------------------------------------------------------

    @property
    def session(self) -> Optional[SessionContext]:
        """Return the session for the current (or most recent) run."""
        return self._session

    @property
    def session_manager(self) -> SessionManager:
        return self._session_manager

    @property
    def emitter(self) -> Optional[EventEmitter]:
        """Return the event emitter for the current (or most recent) run."""
        return self._emitter


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
        default="local",
        help="LLM backend to use (default: auto — tries local → gemini → gpt4o)",
    )
    parser.add_argument("--no-telemetry", action="store_true", help="Disable Arize Phoenix telemetry")
    parser.add_argument("--no-memory",    action="store_true", help="Disable Cognee memory layer")
    parser.add_argument(
        "--session",
        default=None,
        help="Resume an existing session by id (under sessions/<id>/)",
    )
    parser.add_argument(
        "--run-kind",
        choices=["auto", "new", "resume", "repair"],
        default="auto",
        help="Run lifecycle mode (default: auto; repair reuses the current workspace)",
    )
    parser.add_argument(
        "--list-sessions",
        action="store_true",
        help="List existing sessions and exit",
    )
    parser.add_argument(
        "--serve",
        action="store_true",
        help="Start the FastAPI Control API server instead of running a mission",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="API server bind host (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8088,
        help="API server bind port (default: 8088)",
    )
    args = parser.parse_args()

    # --serve: launch the Control API and return (does not run a mission).
    if args.serve:
        import uvicorn
        from src.api import create_app
        app = create_app()
        uvicorn.run(app, host=args.host, port=args.port)
        return

    runtime = MissionsRuntime(
        model=args.model,
        telemetry=not args.no_telemetry,
        memory=not args.no_memory,
    )

    if args.list_sessions:
        sessions = runtime.session_manager.list_sessions()
        if not sessions:
            print("No sessions found.")
            return
        print(f"{'SESSION ID':<14} {'STATUS':<10} {'MODEL':<8} {'TITLE'}")
        print("-" * 70)
        for s in sessions:
            print(
                f"{s.get('session_id', '?'):<14} "
                f"{s.get('status', '?'):<10} "
                f"{s.get('selected_model', '?'):<8} "
                f"{s.get('title', '?')}"
            )
        return

    request = args.request
    if not request:
        request = input("Enter your coding request: ").strip()
    if not request:
        print("No request provided. Exiting.")
        sys.exit(1)

    session = None
    if args.session:
        session = runtime.session_manager.load_session(args.session)
        if session is None:
            print(f"Session '{args.session}' not found under sessions/. Exiting.")
            sys.exit(1)
        print(f"[Runtime] Resuming session {session.session_id} ({session.title}).")

    result = runtime.run(request, session=session, run_kind=args.run_kind)
    sys.exit(0 if result.status == "completed" else 1)


if __name__ == "__main__":
    main()
