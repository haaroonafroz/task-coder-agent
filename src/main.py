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
Every run keeps state under ``sessions/<session_id>/`` and binds tools to
either a harness-managed workspace or an explicitly attached external project.
The runtime points ``src.tools.paths`` at that binding before tool execution.

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
import re
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
from src.tools.paths import set_workspace_root, get_workspace_root, reset_workspace_root
from src.session import SessionContext, SessionManager
from src.sandbox import activate_sandbox, deactivate_sandbox
from src.sandbox.process_manager import format_allowed_ports_block
from src.sandbox.dependency_check import milestone_suggests_dependencies
from src.run_control import RunCancelledError, ensure_not_cancelled
from src.events import EventEmitter, emitter_for_session, register_emitter, unregister_emitter
from src.workspace.git_state import snapshot_git_state
from src.agents import (
    replan_mission,
    run_code_review,
    run_hotfix,
    run_orchestration,
    run_triage,
    run_validator,
    run_worker,
)
from src.agents.contracts import (
    RouteDecision,
    build_mission_brief,
    hotfix_milestone_from_review,
    hotfix_milestone_from_route,
    normalize_route_decision,
)
from src.agents.escalation import deterministic_guard, run_escalation_triage
from src.agents.fix_verifier import run_fix_verification
from src.decisions import save_pending_decision
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
    "probe_dependency",
    "run_checks",
)
_UI_HINTS = ("frontend", "react", "vite", "streamlit", "browser", "web app")

_PYTEST_INI = """\
[pytest]
pythonpath = .
testpaths = tests
"""


# ---------------------------------------------------------------------------
# Workspace bootstrap
# ---------------------------------------------------------------------------

def _prepare_session_runtime(ctx: SessionContext) -> None:
    """Activate harness state and tools without modifying attached code."""
    ctx.ensure_dirs()
    activate_sandbox(ctx)
    set_workspace_root(ctx.workspace_root)


def _bootstrap_managed_workspace(ctx: SessionContext) -> None:
    """Create greenfield-only convenience files."""
    gitkeep = ctx.workspace_root / ".gitkeep"
    if not gitkeep.exists():
        gitkeep.touch()

    pytest_ini = ctx.workspace_root / "pytest.ini"
    if not pytest_ini.exists():
        pytest_ini.write_text(_PYTEST_INI, encoding="utf-8")

    (ctx.workspace_root / "tests").mkdir(parents=True, exist_ok=True)


def _inspect_external_workspace(
    ctx: SessionContext,
    manager: SessionManager,
) -> dict[str, Any]:
    """Inspect an external project read-only and persist bounded metadata."""
    profile = manager.workspace_service.inspect(ctx.workspace).to_dict()
    ctx.project_profile = profile
    ctx.git_preflight = profile.get("git")
    manager._save_meta(ctx)
    return profile


def _workspace_has_user_files(workspace_root: Path) -> bool:
    """Return True when the workspace contains code beyond bootstrap files."""
    ignored_names = {".gitkeep", "pytest.ini"}
    ignored_dirs = {
        ".git", ".pytest_cache", ".venv", "__pycache__", "node_modules", "target"
    }
    for path in workspace_root.rglob("*"):
        if (
            path.is_file()
            and path.name not in ignored_names
            and not any(part in ignored_dirs for part in path.parts)
        ):
            return True
    return False


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


def _normalize_failure_signature(errors: list, fix_guidance: str = "") -> str:
    """Reduce a FAIL verdict to a stable signature for repeat detection.

    Line numbers, memory addresses, and absolute paths change cosmetically
    between identical failures — normalize them away so only a genuinely
    different error produces a different signature. Empty errors yield ""
    (never trips the breaker).
    """
    if not errors:
        return ""
    text = " | ".join(str(e) for e in errors)
    if fix_guidance:
        text += " || " + str(fix_guidance)[:300]
    text = text.lower()
    text = re.sub(r"0x[0-9a-f]+", "0x#", text)
    text = re.sub(r"[\\/][\w.\-~]+(?:[\\/][\w.\-~]+)+", "<path>", text)
    text = re.sub(r"\d+", "#", text)
    return " ".join(text.split())


def _map_fix_verification(
    verification: dict[str, Any],
) -> tuple[str, list[str], str]:
    """Map a FixVerifier verdict onto the worker retry loop.

    Returns (loop_verdict, errors, fix_guidance):
    - VERIFIED → PASS. UNVERIFIED → PASS as well: the verifier could not
      prove failure (missing tests/toolchain), so burning retries is wrong;
      the post-PASS verify step owns the setup_env/run_smoke pause.
    - NEEDS_REWORK → FAIL with the verifier's guidance (retry).
    - ESCALATE_TO_MISSION → ESCALATE (caller returns a REPLAN handoff, which
      the hotfix routes already forward to escalation-triage + mission brief).
    """
    verification = verification or {}
    verdict = str(verification.get("verdict", "UNVERIFIED"))
    if verdict == "NEEDS_REWORK":
        summary = str(
            verification.get("summary", "") or "fix did not satisfy criteria"
        )
        guidance = str(
            verification.get("fix_guidance", "") or verification.get("summary", "")
        )
        return "FAIL", [summary], guidance
    if verdict == "ESCALATE_TO_MISSION":
        return "ESCALATE", [], ""
    return "PASS", [], ""


def _format_review_summary(
    report: dict[str, Any],
    *,
    handoff: Optional["MilestoneHandoff"] = None,
    verification: Optional[dict[str, Any]] = None,
) -> str:
    findings = report.get("findings", []) if isinstance(report, dict) else []
    lines = [
        "Code review",
        f"Verdict: {report.get('verdict', 'unknown')}",
        str(report.get("summary", "")).strip(),
    ]
    if findings:
        lines.extend(["", "Findings:"])
        for finding in findings:
            if not isinstance(finding, dict):
                continue
            lines.append(
                f"- [{str(finding.get('severity', 'risk')).upper()}] "
                f"{finding.get('title', 'Untitled finding')}: "
                f"{finding.get('issue', '')}"
            )
    if handoff is not None:
        lines.extend([
            "",
            "Fix outcome:",
            f"- {handoff.verdict}: {handoff.worker_summary}",
        ])
    if verification is not None:
        if verification.get("error"):
            lines.extend(["", f"Verification unavailable: {verification['error']}"])
        else:
            lines.extend([
                "",
                "Post-fix review:",
                f"- {verification.get('verdict', 'unknown')}: "
                f"{verification.get('summary', '')}",
            ])
    return "\n".join(lines)


_INFRA_GAP_MARKERS = (
    "no module named pytest",
    "pytest: command not found",
    "no module named ",
    "pip install",
    "connection refused",
    "network is unreachable",
    "temporary failure in name resolution",
    "environment prospect",
)


def _infra_gap_note(error_tail: list[str]) -> str:
    """Detect a toolchain/environment gap (vs a code defect) in error text.

    An infra REPLAN (e.g. validator demanding pytest in a bare venv) must
    feed FixVerifier/UNVERIFIED rather than mission escalation — the code
    change may be fine, only the harness to prove it is missing.
    """
    haystack = "\n".join(str(line) for line in error_tail).lower()
    for marker in _INFRA_GAP_MARKERS:
        if marker in haystack:
            return (
                "Validator failure looks like a missing toolchain/dependency "
                f"(matched {marker!r}), not a code defect. Prefer UNVERIFIED "
                "over mission escalation."
            )
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
    verdict: str           # "PASS" | "FAIL" | "BLOCKED" | "REPLAN"
    commit_hash: str
    elapsed_ms: float
    error_log: list[str] = field(default_factory=list)
    failure_signature: str = ""  # deterministic fingerprint for replan dedup
    # Hotfix route only: in-loop FixVerifier verdict (VERIFIED | NEEDS_REWORK |
    # UNVERIFIED | ESCALATE_TO_MISSION dict). Lets the post-PASS verify step
    # reuse the verdict instead of re-running the verifier LLM call.
    verification: Optional[dict[str, Any]] = None


@dataclass
class MissionResult:
    mission_id: str
    title: str
    status: str  # "completed" | "partial" | "failed" | "cancelled" | "awaiting_decision"
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
    execution_route: str = "mission"
    pending_decision: Optional[dict[str, Any]] = None


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
        self._active_execution_route = "mission"
        self._active_review_fix_mode = "ask"
        self._active_run_id: Optional[str] = None
        self._escalation_brief: Optional[dict[str, Any]] = None

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
        execution_route: str = "auto",
        review_fix_mode: str = "ask",
        review_fix_packet: Optional[dict[str, Any]] = None,
        run_id: Optional[str] = None,
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
        self._active_run_id = run_id
        self._active_review_fix_mode = (
            review_fix_mode if review_fix_mode in ("auto", "ask") else "ask"
        )
        self._escalation_brief = None

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

        _prepare_session_runtime(session)
        if session.workspace.kind == "managed":
            _bootstrap_managed_workspace(session)
        else:
            _inspect_external_workspace(session, self._session_manager)
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
                    requested_route=execution_route,
                    review_fix_packet=review_fix_packet,
                )
            except RunCancelledError:
                return self._finish_cancelled(session, t_mission_start)
            finally:
                if session.workspace.kind == "external":
                    self._persist_run_artifact(
                        session,
                        "git_postflight",
                        snapshot_git_state(session.workspace_root).to_dict(),
                    )
                clear_milestone_write_policy()
                deactivate_sandbox()
                reset_workspace_root()
                self._cancel_check = None
                self._active_run_id = None

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
            execution_route=getattr(self, "_active_execution_route", "mission"),
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
        requested_route: str,
        review_fix_packet: Optional[dict[str, Any]] = None,
    ) -> MissionResult:
        """Run orchestration + the milestone loop + summary inside a span."""
        self._check_cancelled()

        self._active_run_kind = run_kind
        triage_report: Optional[dict[str, Any]] = None

        # HITL follow-ups (apply_fix / run_smoke / setup_env) execute their
        # stored packet directly — no re-triage, no re-review. Escalation
        # from a packet falls straight through to the mission path below.
        if review_fix_packet is not None:
            packet_result = self._run_packet_run(
                user_request, session, review_fix_packet,
                t_mission_start, run_kind=run_kind,
            )
            if packet_result is not None:
                return packet_result
            triage_report = self._merge_escalation_brief(
                triage_report,
                route="packet",
                review_report=(
                    review_fix_packet.get("report")
                    or review_fix_packet.get("review_report")
                ),
            )

        # Lifecycle and execution intent are orthogonal. Crash recovery always
        # resumes the current mission; other requests are routed by a focused
        # triage profile (or an explicit API override).
        if run_kind == "resume":
            route_decision = RouteDecision(
                route="mission",
                confidence="high",
                rationale="Pending milestones require crash-safe resume.",
                source="lifecycle",
            )
        elif review_fix_packet is not None:
            route_decision = RouteDecision(
                route="mission",
                confidence="high",
                rationale="Follow-up packet escalated to a planned mission.",
                source="packet",
            )
        else:
            deterministic_route = requested_route
            request_lower = user_request.strip().lower()
            if requested_route == "auto" and (
                request_lower.startswith(("review ", "review:", "audit ", "audit:"))
                or "code review" in request_lower
            ):
                deterministic_route = "review"
            if (
                deterministic_route == "auto"
                and run_kind == "new"
                and not _workspace_has_user_files(session.workspace_root)
            ):
                deterministic_route = "mission"

            if deterministic_route in {"auto", "hotfix"}:
                try:
                    triage_report = run_triage(
                        user_request,
                        workspace_root=session.workspace_root,
                        session_root=session.root,
                        model=self.model,
                        previous_plan=previous_plan,
                        project_profile=session.project_profile,
                        session=telemetry_ctx,
                        emitter=self._emitter,
                    )
                except Exception as exc:
                    print(f"  [Triage] Routing failed — mission fallback: {exc}")
                    self._emitter.emit(
                        "triage.failed", error=str(exc), fatal=False
                    )

            route_decision = normalize_route_decision(
                triage_report,
                workspace_root=session.workspace_root,
                requested_route=deterministic_route,
            )

        self._active_execution_route = route_decision.route
        self._emitter.emit(
            "triage.route.selected",
            execution_route=route_decision.route,
            confidence=route_decision.confidence,
            rationale=route_decision.rationale,
            source=route_decision.source,
            candidate_files=route_decision.candidate_files,
        )
        self._persist_run_artifact(
            session,
            "route",
            {
                "decision": route_decision.to_dict(),
                "run_id": self._active_run_id,
                "session_id": session.session_id,
                "run_kind": run_kind,
                "parent_plan_id": parent_plan_id,
            },
        )

        if route_decision.route == "hotfix":
            focused = self._run_hotfix_route(
                user_request,
                route_decision,
                session,
                previous_plan,
                t_mission_start,
            )
            if focused is not None:
                return focused
            # The hotfix route stashed a MissionBrief (or escalated at guard
            # level); either way the mission path inherits focused context.
            triage_report = self._merge_escalation_brief(
                triage_report, route="hotfix"
            )
            self._active_execution_route = "mission"

        if route_decision.route == "review":
            focused, review_report = self._run_review_route(
                user_request,
                route_decision,
                session,
                previous_plan,
                t_mission_start,
            )
            if focused is not None:
                return focused
            triage_report = self._merge_escalation_brief(
                triage_report,
                route="review",
                review_report=review_report,
            )
            self._active_execution_route = "mission"

        # Phase 1 — Orchestration (graceful failure: no uncaught tracebacks)
        try:
            plan = run_orchestration(
                user_request, self.model,
                plan_path=session.plan_path,
                mission_dir=session.root,
                run_kind=run_kind,
                parent_plan_id=parent_plan_id,
                triage_report=triage_report,
                previous_plan=previous_plan,
                workspace_root=session.workspace_root,
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
            execution_route=self._active_execution_route,
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
                execution_route=self._active_execution_route,
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
    # Focused execution routes
    # ------------------------------------------------------------------

    def _merge_escalation_brief(
        self,
        triage_report: Optional[dict[str, Any]],
        *,
        route: str,
        review_report: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Fold a stashed MissionBrief into orchestration-bound triage context.

        The orchestrator already receives the full triage JSON, so the brief
        (structured + rendered text) rides along — the mission planner starts
        from verified review/hotfix evidence instead of the raw request.
        """
        brief = self._escalation_brief
        self._escalation_brief = None
        base = dict(triage_report or {})
        if brief is None:
            base.update({
                "route": "mission",
                "summary": (
                    f"The {route} route escalated to a planned mission "
                    "without a structured brief (guard-level escalation)."
                ),
            })
        else:
            base.update({
                "route": "mission",
                "summary": (
                    "Escalated to a planned mission. Read MISSION BRIEF first — "
                    "it carries verified review/hotfix context. Do not re-derive "
                    "what is already established there."
                ),
                "mission_brief": brief,
                "escalation_brief_text": brief.get("briefing", ""),
            })
        if review_report:
            base["review_report"] = review_report
        return base

    def _pause_for_decision(
        self,
        session: SessionContext,
        started_at: float,
        title: str,
        execution_route: str,
        decision_type: str,
        decision_title: str,
        summary: str,
        options: list[str],
        payload: dict[str, Any],
        handoffs: Optional[list[MilestoneHandoff]] = None,
    ) -> MissionResult:
        """Persist a HITL decision and finish the run as awaiting_decision."""
        pending = save_pending_decision(
            session,
            {
                "type": decision_type,
                "title": decision_title,
                "summary": summary,
                "options": options,
                "payload": payload,
            },
        )
        self._persist_run_artifact(session, "pending_decision", pending)
        self._emitter.emit(
            "run.awaiting_decision",
            decision_type=decision_type,
            title=decision_title,
            options=options,
        )
        return self._finish_focused_result(
            session=session,
            started_at=started_at,
            title=title,
            execution_route=execution_route,
            handoffs=handoffs or [],
            plan_id=None,
            status_override="awaiting_decision",
            summary_override=summary,
            pending_decision=pending,
        )

    def _escalate_hotfix_failure(
        self,
        user_request: str,
        session: SessionContext,
        *,
        origin: str,  # "hotfix" | "review_fix"
        target_files: list[str],
        handoff: MilestoneHandoff,
        review_report: Optional[dict[str, Any]] = None,
        started_at: Optional[float] = None,
        milestone: Optional[dict[str, Any]] = None,
    ) -> Optional[MissionResult]:
        """Route a failed focused packet via LLM escalation triage.

        Returns a finished MissionResult (HITL pause) when the triage says
        the mission path is not warranted, otherwise stashes a MissionBrief
        and returns None so the caller falls through to the full mission
        path. File count never decides — the LLM triage does, with a
        deterministic guard downgrading only sensitive high-risk stays.
        """
        started = started_at if started_at is not None else time.perf_counter()
        error_tail = handoff.error_log[-4:] if handoff.error_log else []
        packet = milestone or {"target_files": target_files}
        diff_summary = (
            f"origin={origin} verdict={handoff.verdict} "
            f"files_modified={handoff.files_modified}\n"
            "error_tail:\n" + "\n".join(str(line) for line in error_tail)
            + "\n" + _infra_gap_note(error_tail)
        )
        triage = run_escalation_triage(
            review_report=review_report or {},
            hotfix_packet=packet,
            hotfix_handoff=asdict(handoff),
            diff_summary=diff_summary,
            project_profile=session.project_profile,
            model=self.model,
            session=self._telemetry_ctx,
            emitter=self._emitter,
        )
        decision = deterministic_guard(triage, target_files=target_files)
        self._persist_run_artifact(session, "escalation_decision", decision)

        if decision.get("decision") == "escalate_mission":
            self._stash_mission_brief(
                user_request, session, origin=origin,
                target_files=target_files, handoff=handoff,
                review_report=review_report,
                rationale=str(decision.get("reason", "")),
                escalation=decision,
            )
            return None
        # stay_hotfix after retries means the bounded track is stuck but the
        # work is still local — pause for a human call instead of silently
        # escalating to a mission.
        return self._pause_for_decision(
            session, started, "Focused fix",
            origin if origin != "hotfix" else "hotfix",
            decision_type="hotfix_followup",
            decision_title="Hotfix stalled — how should we proceed?",
            summary=(
                f"The focused fix ended with verdict {handoff.verdict}. "
                f"Escalation triage says the work is still bounded: "
                f"{decision.get('reason', '')}"
            ),
            options=["apply_fix", "escalate_mission", "dismiss"],
            payload={
                "kind": "hotfix_retry",
                "milestone": milestone,
                "target_files": target_files,
                "verdict": handoff.verdict,
                "errors": error_tail,
                "review_report": review_report or {},
                "triage_rationale": decision.get("reason", ""),
            },
            handoffs=[handoff],
        )

    def _stash_mission_brief(
        self,
        user_request: str,
        session: SessionContext,
        *,
        origin: str,
        target_files: list[str],
        handoff: MilestoneHandoff,
        review_report: Optional[dict[str, Any]],
        rationale: str,
        verification: Optional[dict[str, Any]] = None,
        escalation: Optional[dict[str, Any]] = None,
    ) -> None:
        """Build the orchestrator-bound brief for a mission escalation."""
        escalation = escalation or {"reason": rationale}
        brief = build_mission_brief(
            original_request=user_request,
            review_report=review_report,
            hotfix_packet={
                "target_files": target_files,
                "origin": origin,
            },
            hotfix_handoff=asdict(handoff),
            verification=verification,
            escalation=escalation,
        )
        self._escalation_brief = brief
        self._persist_run_artifact(session, "mission_brief", brief)
        self._emitter.emit(
            "hotfix.escalated" if origin == "hotfix" else "review.escalated",
            reason="escalation_triage",
            affected_files=target_files,
            escalated_from=origin,
        )

    def _verify_hotfix_result(
        self,
        user_request: str,
        session: SessionContext,
        *,
        origin: str,
        milestone: dict[str, Any],
        handoff: MilestoneHandoff,
        review_report: Optional[dict[str, Any]],
        started_at: float,
        verification: Optional[dict[str, Any]] = None,
    ) -> Optional[MissionResult]:
        """Run FixVerifier over a PASS hotfix; map VERIFIED/* to outcomes.

        UNVERIFIED (missing tests/toolchain — including the 'user says do the
        tests' path) never auto-escalates to a mission; it pauses with
        setup_env / run_smoke options. Returns None only when the verdict
        routes to the mission path (brief stashed).

        Pass the in-loop verification to skip re-running the verifier when
        the milestone is the packet the loop already verified (callers pass
        handoff.verification; the setup_env re-verify path passes None so a
        fresh verdict is computed against the repaired toolchain).
        """
        started = started_at
        touched = sorted(set(handoff.files_modified) | set(milestone.get("target_files", [])))
        if verification is None:
            verification = run_fix_verification(
                review_report=review_report or {},
                hotfix_packet=milestone,
                hotfix_handoff=asdict(handoff),
                model=self.model,
                session=self._telemetry_ctx,
                emitter=self._emitter,
            )
        self._persist_run_artifact(session, "fix_verification", verification)
        verdict = str(verification.get("verdict", "UNVERIFIED"))

        title = "Focused hotfix" if origin == "hotfix" else "Code review"
        if verdict == "VERIFIED":
            summary = _format_review_summary(
                review_report or {"verdict": "n/a", "summary": user_request, "findings": []},
                handoff=handoff,
            )
            summary += (
                "\n\nFix verification: VERIFIED — "
                f"{verification.get('summary', '')}"
            )
            return self._finish_focused_result(
                session=session,
                started_at=started_at,
                title=title,
                execution_route=origin,
                handoffs=[handoff],
                plan_id=None,
                summary_override=summary,
            )
        if verdict == "ESCALATE_TO_MISSION":
            self._stash_mission_brief(
                user_request, session, origin=origin,
                target_files=touched, handoff=handoff,
                review_report=review_report,
                rationale=str(
                    verification.get("escalation_reason")
                    or verification.get("summary", "")
                ),
                verification=verification,
                escalation={"reason": verification.get("escalation_reason", "")},
            )
            return None
        if verdict == "NEEDS_REWORK":
            if self._active_review_fix_mode == "auto":
                # Headless mode: let escalation triage decide rework routing.
                escalated = self._escalate_hotfix_failure(
                    user_request, session, origin=origin,
                    target_files=touched, handoff=handoff,
                    review_report=review_report,
                    started_at=started,
                    milestone=milestone,
                )
                if escalated is not None:
                    return escalated
                return None
            return self._pause_for_decision(
                session, started_at, title, origin,
                decision_type="fix_followup",
                decision_title="Fix needs rework — how should we proceed?",
                summary=(
                    "The fix landed but verification says NEEDS_REWORK: "
                    f"{verification.get('summary', '')} "
                    f"Guidance: {verification.get('fix_guidance', '')}"
                ),
                options=["apply_fix", "escalate_mission", "dismiss"],
                payload={
                    "kind": "fix_rework",
                    "milestone": milestone,
                    "target_files": touched,
                    "verdict": handoff.verdict,
                    "verification": verification,
                    "review_report": review_report or {},
                },
                handoffs=[handoff],
            )
        # UNVERIFIED — infra gap, not a scope signal. Pause with the
        # "user says do the tests" path (setup_env / run_smoke); never
        # auto-escalate to a mission.
        missing = verification.get("missing_checks", [])
        return self._pause_for_decision(
            session, started_at, title, origin,
            decision_type="fix_unverified",
            decision_title="Fix landed but could not be verified — set up tests?",
            summary=(
                f"Files changed: {', '.join(touched) or '(none)'}. "
                f"Verification is UNVERIFIED: {verification.get('summary', '')} "
                f"Missing checks: {', '.join(missing) or 'none listed'}."
            ),
            options=["run_smoke", "setup_env", "escalate_mission", "dismiss"],
            payload={
                "kind": "fix_unverified",
                "milestone": milestone,
                "target_files": touched,
                "touched_files": touched,
                "verdict": handoff.verdict,
                "missing_checks": missing,
                "packages": missing,
                "verification": verification,
                "review_report": review_report or {},
            },
            handoffs=[handoff],
        )

    def _run_packet_run(
        self,
        user_request: str,
        session: SessionContext,
        packet: dict[str, Any],
        started_at: float,
        *,
        run_kind: str,
    ) -> Optional[MissionResult]:
        """Execute a stored HITL follow-up packet (no re-triage, no re-review).

        Kinds: ``review_fix`` (approved milestone), ``smoke`` (run the suite,
        fix only trivial breakage), ``setup_env`` (install missing test deps
        via the ``pip`` shell profile, then re-verify). Returns None when the
        packet escalates (brief stashed) so the caller falls to the mission.
        """
        kind = str(packet.get("kind") or "review_fix")
        report = packet.get("report") or packet.get("review_report") or {}
        self._emitter.emit(
            "packet.started", kind=kind,
            target_files=packet.get("target_files", []),
        )

        if kind == "setup_env":
            packages = packet.get("packages") or packet.get("missing_checks") or []
            milestone = {
                "id": "SETUP-ENV",
                "title": "Set up test environment for verification",
                "description": (
                    "Install the missing test dependencies into the session "
                    f"virtualenv: {', '.join(str(p) for p in packages) or '(see missing checks)'}. "
                    "Use run_shellscript with profile='pip' "
                    "(e.g. python -m pip install pytest). Do NOT modify any "
                    "project code — this packet only repairs the toolchain."
                ),
                "depends_on": [],
                "target_files": [],
                "acceptance_criteria": [
                    "The missing test dependencies import successfully.",
                    "No project source files were modified.",
                ],
                "validation_profile": "auto",
                "status": "pending",
                "route": "packet",
            }
        elif kind == "smoke":
            touched = packet.get("touched_files") or packet.get("target_files") or []
            milestone = {
                "id": "SMOKE",
                "title": "Run smoke checks for the recent fix",
                "description": (
                    "Run the project's test suite and linters over the recent "
                    "change. Fix only trivial breakage directly caused by the "
                    f"recent change. Touched files: {', '.join(touched)}."
                ),
                "depends_on": [],
                "target_files": list(touched),
                "acceptance_criteria": [
                    "Test suite and linters pass for the touched area.",
                ],
                "validation_profile": "auto",
                "status": "pending",
                "route": "packet",
            }
        else:  # review_fix — the exact milestone approved by the user
            milestone = dict(packet.get("milestone") or {})
            if not milestone:
                return self._pause_for_decision(
                    session, started_at, "Follow-up fix", "review",
                    decision_type="hotfix_followup",
                    decision_title="Approved fix packet is empty",
                    summary="The approved fix packet carried no milestone; nothing to execute.",
                    options=["escalate_mission", "dismiss"],
                    payload={"kind": kind, "review_report": report},
                )

        plan = {
            "mission_id": f"packet-{uuid.uuid4().hex[:10]}",
            "title": f"Follow-up: {kind}",
            "run_kind": run_kind,
            "execution_route": "review",
            "milestones": [milestone],
        }
        self._persist_run_artifact(session, "followup_packet", milestone)
        handoff = self._execute_milestone(
            milestone, plan, session, agent_profile="hotfix",
            review_report=report if isinstance(report, dict) else {},
        )
        self._persist_run_artifact(session, "followup_handoff", asdict(handoff))

        if handoff.verdict in {"REPLAN", "BLOCKED", "FAIL"}:
            return self._escalate_hotfix_failure(
                user_request, session, origin="review_fix",
                target_files=list(milestone.get("target_files", [])),
                handoff=handoff,
                review_report=report if isinstance(report, dict) else {},
                started_at=started_at,
                milestone=milestone,
            )

        verify_milestone = milestone
        if kind == "setup_env":
            # Re-verify the original touched files now the toolchain exists.
            verify_milestone = {
                "target_files": packet.get("touched_files")
                or packet.get("target_files") or [],
            }
        return self._verify_hotfix_result(
            user_request, session, origin="review_fix",
            milestone=verify_milestone, handoff=handoff,
            review_report=report if isinstance(report, dict) else {},
            started_at=started_at,
            # setup_env re-verifies different files with a repaired toolchain:
            # only reuse the in-loop verdict for the packet it verified.
            verification=(
                handoff.verification if verify_milestone is milestone else None
            ),
        )

    def _run_hotfix_route(
        self,
        user_request: str,
        decision: RouteDecision,
        session: SessionContext,
        previous_plan: dict[str, Any],
        started_at: float,
    ) -> Optional[MissionResult]:
        """Run one scoped Hotfix packet, or return None to escalate to mission.

        Entry requires only that triage named target files — packet width
        never decides escalation. Failures route through LLM escalation
        triage; PASS packets go through FixVerifier.
        """
        if not decision.candidate_files:
            self._emitter.emit(
                "hotfix.escalated",
                reason="no_targets",
                candidate_files=decision.candidate_files,
            )
            return None

        milestone = hotfix_milestone_from_route(decision, user_request)
        plan = {
            "mission_id": f"hotfix-{uuid.uuid4().hex[:10]}",
            "plan_id": previous_plan.get("plan_id"),
            "title": "Focused hotfix",
            "run_kind": self._active_run_kind,
            "execution_route": "hotfix",
            "milestones": [milestone],
        }
        self._persist_run_artifact(session, "hotfix_packet", milestone)
        handoff = self._execute_milestone(
            milestone, plan, session, agent_profile="hotfix",
            review_report=None,
        )
        self._persist_run_artifact(session, "hotfix_handoff", asdict(handoff))

        if handoff.verdict in {"REPLAN", "BLOCKED", "FAIL"}:
            return self._escalate_hotfix_failure(
                user_request, session, origin="hotfix",
                target_files=list(decision.candidate_files),
                handoff=handoff,
                started_at=started_at,
                milestone=milestone,
            )

        return self._verify_hotfix_result(
            user_request, session, origin="hotfix",
            milestone=milestone, handoff=handoff,
            review_report=None, started_at=started_at,
            verification=handoff.verification,
        )

    def _run_review_route(
        self,
        user_request: str,
        decision: RouteDecision,
        session: SessionContext,
        previous_plan: dict[str, Any],
        started_at: float,
    ) -> tuple[Optional[MissionResult], Optional[dict[str, Any]]]:
        """Run read-only review and conditionally dispatch actionable findings."""
        try:
            report = run_code_review(
                user_request=(
                    f"{user_request}\n\nRequested review scope: {decision.review_scope}"
                ),
                workspace_root=session.workspace_root,
                model=self.model,
                previous_plan=previous_plan,
                emitter=self._emitter,
                session=self._telemetry_ctx,
                cancel_check=self._cancel_check,
            )
        except RunCancelledError:
            raise
        except Exception as exc:
            summary = f"Code review failed before producing a report: {exc}"
            return (
                self._finish_focused_result(
                    session=session,
                    started_at=started_at,
                    title="Code review",
                    execution_route="review",
                    handoffs=[],
                    plan_id=previous_plan.get("plan_id"),
                    status_override="failed",
                    summary_override=summary,
                    failure_reason=summary,
                ),
                None,
            )

        report_dict = report.to_dict()
        self._persist_run_artifact(session, "review_report", report_dict)
        actionable = report.actionable_findings
        if not actionable:
            summary = _format_review_summary(report_dict)
            return (
                self._finish_focused_result(
                    session=session,
                    started_at=started_at,
                    title="Code review",
                    execution_route="review",
                    handoffs=[],
                    plan_id=previous_plan.get("plan_id"),
                    summary_override=summary,
                ),
                report_dict,
            )

        if session.workspace.access_mode == "read_only":
            summary = _format_review_summary(report_dict)
            return (
                self._finish_focused_result(
                    session=session,
                    started_at=started_at,
                    title="Code review",
                    execution_route="review",
                    handoffs=[],
                    plan_id=previous_plan.get("plan_id"),
                    summary_override=summary,
                ),
                report_dict,
            )

        milestone = hotfix_milestone_from_review(report)
        if not milestone.get("target_files"):
            self._emitter.emit(
                "review.escalated",
                reason="no_actionable_targets",
                affected_files=[],
            )
            return None, report_dict

        self._persist_run_artifact(session, "review_fix_packet", milestone)

        # HITL gate: in ask mode (default) a review with actionable findings
        # pauses so the user chooses fix vs mission vs dismiss. Auto mode
        # preserves the old fire-and-forget review→fix behaviour.
        if self._active_review_fix_mode != "auto":
            summary = _format_review_summary(report_dict)
            pending_result = self._pause_for_decision(
                session, started_at, "Code review", "review",
                decision_type="review_fix",
                decision_title="Review found actionable defects — apply the fix?",
                summary=summary,
                options=["apply_fix", "escalate_mission", "dismiss"],
                payload={
                    "kind": "review_fix",
                    "milestone": milestone,
                    "target_files": milestone.get("target_files", []),
                    "review_report": report_dict,
                },
            )
            return pending_result, report_dict

        plan = {
            "mission_id": f"review-fix-{uuid.uuid4().hex[:10]}",
            "plan_id": previous_plan.get("plan_id"),
            "title": "Review-driven fix",
            "run_kind": self._active_run_kind,
            "execution_route": "review",
            "milestones": [milestone],
        }
        handoff = self._execute_milestone(
            milestone, plan, session, agent_profile="hotfix",
            review_report=report_dict,
        )
        self._persist_run_artifact(session, "review_fix_handoff", asdict(handoff))
        if handoff.verdict in {"REPLAN", "BLOCKED", "FAIL"}:
            return (
                self._escalate_hotfix_failure(
                    user_request, session, origin="review_fix",
                    target_files=list(milestone.get("target_files", [])),
                    handoff=handoff,
                    review_report=report_dict,
                    started_at=started_at,
                    milestone=milestone,
                ),
                report_dict,
            )

        # PASS packets go through FixVerifier (deterministic checks + focused
        # LLM verdict) instead of a second full review pass.
        verified = self._verify_hotfix_result(
            user_request, session, origin="review_fix",
            milestone=milestone, handoff=handoff,
            review_report=report_dict, started_at=started_at,
            verification=handoff.verification,
        )
        return verified, report_dict

    def _finish_focused_result(
        self,
        *,
        session: SessionContext,
        started_at: float,
        title: str,
        execution_route: str,
        handoffs: list[MilestoneHandoff],
        plan_id: Optional[str],
        status_override: Optional[str] = None,
        summary_override: Optional[str] = None,
        failure_reason: str = "",
        pending_decision: Optional[dict[str, Any]] = None,
    ) -> MissionResult:
        elapsed_ms = round((time.perf_counter() - started_at) * 1000.0, 2)
        passed = sum(1 for handoff in handoffs if handoff.verdict == "PASS")
        total = len(handoffs)
        status = status_override or (
            "completed" if passed == total else ("partial" if passed else "failed")
        )
        if total == 0 and status_override is None:
            status = "completed"
        result = MissionResult(
            mission_id=f"{execution_route}-{uuid.uuid4().hex[:10]}",
            title=title,
            status=status,
            milestones_passed=passed,
            milestones_total=total,
            handoffs=handoffs,
            total_elapsed_ms=elapsed_ms,
            model_used=self.model,
            session_id=session.session_id,
            run_kind=self._active_run_kind,
            plan_id=plan_id,
            execution_route=execution_route,
            pending_decision=pending_decision,
        )
        if summary_override is not None:
            result.summary_text = summary_override
            result.failure_reason = failure_reason
        else:
            recap = build_mission_summary(
                title=title,
                status=status,
                milestones_passed=passed,
                milestones_total=total,
                total_elapsed_ms=elapsed_ms,
                handoffs=handoffs,
            )
            result.summary_text = recap["summary_text"]
            result.failure_reason = recap.get("failure_reason", "")

        self._session_manager.update_status(session, status)
        self._emitter.emit(
            "mission.audit",
            passed=status == "completed",
            incomplete_milestones=(
                [] if status == "completed" else [
                    handoff.milestone_id
                    for handoff in handoffs
                    if handoff.verdict != "PASS"
                ]
            ),
            run_kind=self._active_run_kind,
            execution_route=execution_route,
            plan_id=plan_id,
        )
        self._emitter.emit(
            "mission.complete",
            status=status,
            run_kind=self._active_run_kind,
            execution_route=execution_route,
            plan_id=plan_id,
            milestones_passed=passed,
            milestones_total=total,
            total_elapsed_ms=elapsed_ms,
            summary_text=result.summary_text,
            failure_reason=result.failure_reason,
            files_modified=sorted({
                path for handoff in handoffs for path in handoff.files_modified
            }),
        )
        unregister_emitter(session.session_id)
        self._telemetry_ctx = None
        self._print_summary(result)
        return result

    def _persist_run_artifact(
        self,
        session: SessionContext,
        name: str,
        payload: dict[str, Any],
    ) -> None:
        """Persist immutable routing/profile evidence beside the run record."""
        if not self._active_run_id:
            return
        directory = session.root / "runs" / f"{self._active_run_id}.artifacts"
        try:
            directory.mkdir(parents=True, exist_ok=True)
            (directory / f"{name}.json").write_text(
                json.dumps(payload, indent=2, default=str),
                encoding="utf-8",
            )
        except OSError as exc:
            print(f"  [Runtime] Could not persist {name} artifact: {exc}")

    # ------------------------------------------------------------------
    # Milestone execution loop (glue: Phase 2 → 3 → 4)
    # ------------------------------------------------------------------

    def _execute_milestone(
        self,
        milestone: dict,
        plan: dict,
        session: SessionContext,
        *,
        agent_profile: str = "worker",
        review_report: Optional[dict[str, Any]] = None,
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
        # Signature of the previous FAIL verdict — an identical repeat means
        # the worker is stuck, so stop burning retry cycles (see FAIL branch).
        prev_failure_sig: Optional[str] = None

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
            role="hotfix" if agent_profile == "hotfix" else "worker",
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

            # Phase 3 — focused implementation profile
            implementation_runner = (
                run_hotfix if agent_profile == "hotfix" else run_worker
            )
            worker_result = implementation_runner(
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
                print(f"  [Runtime] {agent_profile} BLOCKED: {reason}")
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

            # Phase 4 — validation. The hotfix route uses the FixVerifier
            # (diff + fix criteria, no test contracts, REPLAN impossible);
            # the mission route uses the adversarial contract validator.
            loop_verification: Optional[dict[str, Any]] = None
            if agent_profile == "hotfix":
                print(f"\n  [Phase 4] FIX VERIFICATION — milestone {ms_id}…")
                loop_verification = run_fix_verification(
                    review_report=review_report or {},
                    hotfix_packet=milestone,
                    hotfix_handoff={
                        "files_modified": worker_result.get("files_modified", []),
                        "summary": worker_result.get("summary", ""),
                    },
                    model=self.model,
                    session=self._telemetry_ctx,
                    emitter=self._emitter,
                )
                self._persist_run_artifact(session, "fix_verification", loop_verification)
                loop_verdict, v_errors, v_guidance = _map_fix_verification(
                    loop_verification
                )
                if loop_verdict == "ESCALATE":
                    reason = str(
                        loop_verification.get("escalation_reason", "")
                        or loop_verification.get("summary", "")
                    )
                    error_log.append(f"FixVerifier escalated to mission: {reason}")
                    print(f"\n  [!] FixVerifier ESCALATE_TO_MISSION: {reason}")
                    elapsed_ms = (time.perf_counter() - t_start) * 1000.0
                    return MilestoneHandoff(
                        milestone_id=ms_id, title=ms_title,
                        worker_summary=worker_result.get("summary", ""),
                        files_modified=worker_result.get("files_modified", []),
                        tool_calls_made=worker_result.get("tool_calls", 0),
                        retry_count=retry_count, verdict="REPLAN",
                        commit_hash="", elapsed_ms=round(elapsed_ms, 2),
                        error_log=error_log,
                        verification=loop_verification,
                    )
                if loop_verdict == "FAIL":
                    verdict_data = {
                        "verdict": "FAIL",
                        "errors": v_errors,
                        "fix_guidance": v_guidance,
                    }
                else:
                    verdict_data = {
                        "verdict": "PASS",
                        "validation_details": str(
                            loop_verification.get("summary", "")
                        ),
                    }
            else:
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
                commit_prefix = "fix" if agent_profile == "hotfix" else "feat"
                commit_msg    = f"{commit_prefix}({ms_id}): {ms_title}"
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
                    verification=loop_verification,
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

            # Repeat circuit-breaker: an identical normalized failure twice in
            # a row means the worker is stuck (same fix attempted, same error).
            # Stop early instead of burning the remaining retry budget.
            failure_sig = _normalize_failure_signature(errors, fix_guidance)
            if failure_sig and failure_sig == prev_failure_sig:
                error_log.append(
                    "Repeated identical failure — stopping retries early. "
                    f"Signature: {failure_sig[:200]}"
                )
                print(
                    f"  [!] Identical failure repeated — "
                    f"stopping retries early ({retry_count + 1}/{MAX_RETRY_CYCLES} used)."
                )
                break
            prev_failure_sig = failure_sig

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
                updated = False
                for m in plan.get("milestones", []):
                    if m.get("id") == ms_id:
                        m["status"] = "completed"
                        updated = True
                        break
                if updated:
                    session.plan_path.write_text(
                        json.dumps(plan, indent=2), encoding="utf-8"
                    )
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
        "--workspace",
        default=None,
        help="Attach a new session to an existing project directory",
    )
    parser.add_argument(
        "--read-only",
        action="store_true",
        help="Attach --workspace with read-only project access",
    )
    parser.add_argument(
        "--run-kind",
        choices=["auto", "new", "resume", "repair"],
        default="auto",
        help="Run lifecycle mode (default: auto; repair reuses the current workspace)",
    )
    parser.add_argument(
        "--execution-route",
        choices=["auto", "mission", "hotfix", "review"],
        default="auto",
        help="Agent profile route (default: auto, selected by triage)",
    )
    parser.add_argument(
        "--review-fix-mode",
        choices=["ask", "auto"],
        default="ask",
        help=(
            "Review HITL gate: 'ask' pauses when review finds actionable "
            "defects; 'auto' applies fixes without asking (default: ask)"
        ),
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
    if args.read_only and not args.workspace:
        parser.error("--read-only requires --workspace")

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
        if args.workspace:
            parser.error("--session and --workspace cannot be used together")
        session = runtime.session_manager.load_session(args.session)
        if session is None:
            print(f"Session '{args.session}' not found under sessions/. Exiting.")
            sys.exit(1)
        print(f"[Runtime] Resuming session {session.session_id} ({session.title}).")
    elif args.workspace:
        try:
            session = runtime.session_manager.create_session(
                title=request[:60] + ("..." if len(request) > 60 else ""),
                model=args.model,
                workspace_kind="external",
                workspace_path=args.workspace,
                workspace_access_mode="read_only" if args.read_only else "read_write",
            )
        except ValueError as exc:
            parser.error(str(exc))
        print(
            f"[Runtime] Attached {session.workspace_root} "
            f"({session.workspace.access_mode})."
        )

    result = runtime.run(
        request,
        session=session,
        run_kind=args.run_kind,
        execution_route=args.execution_route,
        review_fix_mode=args.review_fix_mode,
    )
    sys.exit(0 if result.status in ("completed", "awaiting_decision") else 1)


if __name__ == "__main__":
    main()
