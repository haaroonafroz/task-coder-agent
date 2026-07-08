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
from typing import Any, Optional

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
from src.tools.file_ops import set_allow_test_edits
from src.tools.paths import set_workspace_root, get_workspace_root
from src.session import SessionContext, SessionManager
from src.events import EventEmitter, register_emitter, unregister_emitter
from src.agents import run_orchestration, replan_mission, run_worker, run_validator

_CONFIG_DIR   = _ROOT / "config"
_SKILLS_PATH  = _CONFIG_DIR / "skills.md"

MAX_RETRY_CYCLES = 3

_PYTEST_INI = """\
[pytest]
pythonpath = .
testpaths = tests
"""


# ---------------------------------------------------------------------------
# Workspace bootstrap
# ---------------------------------------------------------------------------

def _ensure_session_workspace(ctx: SessionContext) -> None:
    """Create the session workspace and pytest import bootstrap before any tool calls."""
    ctx.ensure_dirs()
    gitkeep = ctx.workspace_root / ".gitkeep"
    if not gitkeep.exists():
        gitkeep.touch()

    pytest_ini = ctx.workspace_root / "pytest.ini"
    if not pytest_ini.exists():
        pytest_ini.write_text(_PYTEST_INI, encoding="utf-8")

    (ctx.workspace_root / "tests").mkdir(parents=True, exist_ok=True)
    set_workspace_root(ctx.workspace_root)


def _is_test_milestone(milestone: dict) -> bool:
    """True when a milestone is explicitly a test/spec writing phase."""
    title = milestone.get("title", "").lower()
    desc  = milestone.get("description", "").lower()
    return "test" in title or "spec" in title or "test" in desc or "spec" in desc


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


@dataclass
class MissionResult:
    mission_id: str
    title: str
    status: str            # "completed" | "partial" | "failed"
    milestones_passed: int
    milestones_total: int
    handoffs: list[MilestoneHandoff]
    total_elapsed_ms: float
    model_used: str
    session_id: str


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

    # ------------------------------------------------------------------
    # Public entry points
    # ------------------------------------------------------------------

    def run(self, user_request: str, session: Optional[SessionContext] = None) -> MissionResult:
        """
        Execute a full mission from a user request inside a session.

        If ``session`` is None a new session is created automatically. If a
        session is provided with an existing pending plan, the mission
        resumes from the first incomplete milestone.

        Args:
            user_request: Plain-English description of what to build or fix.
            session:      Optional pre-existing session to run inside.

        Returns:
            MissionResult with per-milestone handoff telemetry.
        """
        t_mission_start = time.perf_counter()

        # Resolve or create the session
        if session is None:
            title = user_request[:60] + ("..." if len(user_request) > 60 else "")
            session = self._session_manager.create_session(
                title=title, model=self.model
            )
        self._session = session

        _ensure_session_workspace(session)
        self._session_manager.update_status(session, "running")

        # Event stream for this session
        self._emitter = EventEmitter(session.events_path, session.session_id)
        register_emitter(self._emitter)
        self._emitter.emit(
            "session.started",
            request=user_request,
            model=self.model,
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
            return self._run_mission_body(
                user_request, session, telemetry_ctx, t_mission_start
            )

    # ------------------------------------------------------------------
    # Mission body (orchestration → milestone loop → summary)
    # ------------------------------------------------------------------

    def _run_mission_body(
        self,
        user_request: str,
        session: SessionContext,
        telemetry_ctx,
        t_mission_start: float,
    ) -> MissionResult:
        """Run orchestration + the milestone loop + summary inside a span."""
        # Phase 1 — Orchestration
        plan = run_orchestration(
            user_request, self.model,
            plan_path=session.plan_path,
            mission_dir=session.root,
            session=telemetry_ctx,
        )
        milestones = plan.get("milestones", [])
        mission_id = plan.get("mission_id", str(uuid.uuid4())[:8])
        title = plan.get("title", "Untitled Mission")

        self._emitter.emit(
            "plan.created",
            mission_id=mission_id,
            title=title,
            milestones_total=len(milestones),
            milestone_ids=[m.get("id", "?") for m in milestones],
        )

        print(f"[Orchestrator] Mission '{title}' decomposed into {len(milestones)} milestones.")

        handoffs: list[MilestoneHandoff] = []
        passed = 0
        ms_index = 0

        while ms_index < len(milestones):
            ms = milestones[ms_index]
            ms_id    = ms.get("id", "?")
            ms_title = ms.get("title", "")

            print(f"\n{'─'*70}")
            print(f"  MILESTONE {ms_id}: {ms_title}")
            print(f"{'─'*70}")

            # Crash-recovery: skip already-completed milestones
            if self._memory:
                resume = self._memory.check_resume_point(ms_id)
                if resume and resume.get("status") == "completed":
                    print(f"  [Memory] Milestone {ms_id} already completed — skipping.")
                    self._emitter.emit("milestone.skipped", milestone_id=ms_id, title=ms_title)
                    passed += 1
                    ms_index += 1
                    continue

            self._emitter.emit("milestone.started", milestone_id=ms_id, title=ms_title)
            handoff = self._execute_milestone(ms, plan, session)
            handoffs.append(handoff)

            if handoff.verdict == "PASS":
                passed += 1
                if self._memory:
                    self._memory.log_milestone_state(ms_id, asdict(handoff), "completed")
                ms_index += 1
            elif handoff.verdict == "REPLAN":
                self._emitter.emit(
                    "milestone.replan",
                    milestone_id=ms_id,
                    guidance=handoff.error_log[-1] if handoff.error_log else "",
                )
                plan = replan_mission(
                    plan, handoff.error_log[-1], self.model,
                    plan_path=session.plan_path,
                    session=telemetry_ctx,
                )
                milestones = plan.get("milestones", [])
                self._emitter.emit(
                    "plan.updated",
                    milestones_total=len(milestones),
                    milestone_ids=[m.get("id", "?") for m in milestones],
                )
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
        )

        if self._emitter:
            self._emitter.emit(
                "mission.complete",
                status=status,
                milestones_passed=passed,
                milestones_total=len(milestones),
                total_elapsed_ms=result.total_elapsed_ms,
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

        while retry_count < MAX_RETRY_CYCLES:
            # Phase 2 — Dynamic skill routing
            intent = f"{ms_title}: {milestone.get('description', '')}"
            curated_tools_md = self._router.fetch_curated_skills(intent, top_k=3)
            print(f"\n  [Phase 2] SKILL ROUTING — top tools selected for milestone {ms_id}.")

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
            )

            if worker_result.get("status") == "blocked":
                reason = worker_result.get("reason", "Unknown block")
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
            )
            verdict = verdict_data.get("verdict", "FAIL")

            if verdict == "PASS":
                commit_msg    = f"feat({ms_id}): {ms_title}"
                stage_paths = [str(session.root.relative_to(_ROOT)) + "/"]
                commit_result = dispatch("git_commit", {
                    "message": commit_msg,
                    "stage_paths": stage_paths,
                })
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

            # FAIL — log, optionally store in memory, retry
            errors       = verdict_data.get("errors", [])
            fix_guidance = verdict_data.get("fix_guidance", "")
            error_summary = f"[Retry {retry_count + 1}] Errors: {errors} | Guidance: {fix_guidance}"
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
        default="auto",
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

    result = runtime.run(request, session=session)
    sys.exit(0 if result.status == "completed" else 1)


if __name__ == "__main__":
    main()
