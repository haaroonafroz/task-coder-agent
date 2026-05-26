"""
Missions Runtime Engine.

Implements the four-phase serial execution loop:

  1. ORCHESTRATION  — LLM reads orchestrator.md, decomposes request → plan.json
  2. SKILL ROUTING  — DynamicToolRouter queries Qdrant to inject top-3 tools
  3. WORKER         — LLM reads worker.md + curated tools, executes tool calls
  4. VALIDATION     — LLM reads validator.md, runs contract, PASS → commit / FAIL → retry

Hardware note: only ONE LLM inference runs at any moment (serial design).

Usage:
    python -m src.main "Build a Python REST API with FastAPI"

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
from typing import Any

# ---------------------------------------------------------------------------
# Internal imports
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from src.llm_client import call_llm, ModelChoice
from src.tool_registry import DynamicToolRouter
from src.telemetry import initialize_observability
from src.tools import dispatch
from src.tools.file_ops import set_allow_test_edits
from src.agents import run_orchestration, replan_mission, run_worker, run_validator

_CONFIG_DIR   = _ROOT / "config"
_MISSION_DIR  = _ROOT / "active_mission"
_PLAN_PATH    = _MISSION_DIR / "plan.json"
_HANDOFFS_DIR = _MISSION_DIR / "handoffs"
_WORKSPACE_DIR = _ROOT / "workspace"
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
        plan = run_orchestration(user_request, self.model)
        milestones = plan.get("milestones", [])
        mission_id = plan.get("mission_id", str(uuid.uuid4())[:8])
        title = plan.get("title", "Untitled Mission")

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
                    passed += 1
                    ms_index += 1
                    continue

            handoff = self._execute_milestone(ms, plan)
            handoffs.append(handoff)

            if handoff.verdict == "PASS":
                passed += 1
                if self._memory:
                    self._memory.log_milestone_state(ms_id, asdict(handoff), "completed")
                ms_index += 1
            elif handoff.verdict == "REPLAN":
                plan = replan_mission(plan, handoff.error_log[-1], self.model)
                milestones = plan.get("milestones", [])
            else:
                print(
                    f"  [Runtime] Milestone {ms_id} FAILED after "
                    f"{handoff.retry_count} retries — halting mission."
                )
                break

        total_ms = (time.perf_counter() - t_mission_start) * 1000.0
        status = (
            "completed" if passed == len(milestones)
            else ("partial" if passed > 0 else "failed")
        )

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
    # Milestone execution loop (glue: Phase 2 → 3 → 4)
    # ------------------------------------------------------------------

    def _execute_milestone(self, milestone: dict, plan: dict) -> MilestoneHandoff:
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
            verdict_data = run_validator(
                milestone=milestone,
                worker_result=worker_result,
                retry_count=retry_count,
                is_test_milestone=is_test,
                model=self.model,
            )
            verdict = verdict_data.get("verdict", "FAIL")

            if verdict == "PASS":
                commit_msg    = f"feat({ms_id}): {ms_title}"
                commit_result = dispatch("git_commit", {"message": commit_msg})
                commit_hash   = commit_result.get("commit_hash", "")
                elapsed_ms    = (time.perf_counter() - t_start) * 1000.0
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

            if self._memory:
                for fpath in worker_result.get("files_modified", []):
                    self._memory.log_compilation_failure(fpath, "\n".join(errors))

            retry_count += 1

        # Retries exhausted
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
    ) -> None:
        """Persist milestone handoff metadata to active_mission/handoffs/."""
        _HANDOFFS_DIR.mkdir(parents=True, exist_ok=True)
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
        }
        path = _HANDOFFS_DIR / f"{ms_id}_{int(time.time())}.json"
        path.write_text(json.dumps(handoff, indent=2), encoding="utf-8")

        # Mark milestone as completed in plan.json
        if _PLAN_PATH.exists():
            try:
                plan = json.loads(_PLAN_PATH.read_text(encoding="utf-8"))
                for m in plan.get("milestones", []):
                    if m.get("id") == ms_id:
                        m["status"] = "completed"
                        break
                _PLAN_PATH.write_text(json.dumps(plan, indent=2), encoding="utf-8")
            except (json.JSONDecodeError, OSError) as exc:
                print(f"  [Runtime] Could not update plan.json: {exc}")

    @staticmethod
    def _print_summary(result: MissionResult) -> None:
        print(f"\n{'='*70}")
        print(f"  MISSION COMPLETE — {result.status.upper()}")
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
