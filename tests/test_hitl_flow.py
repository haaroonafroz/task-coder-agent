"""Coverage for the review/hotfix HITL loop.

Escalation is decided by LLM triage (never file count), MissionBriefs carry
focused-run context into missions, UNVERIFIED pauses instead of escalating,
and decisions resolve exactly once into follow-up runs.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.agents.contracts import (
    build_mission_brief,
    normalize_escalation_decision,
    normalize_fix_verification,
)
from src.agents.escalation import deterministic_guard
from src.api.messages import MessageStore
from src.api.routers.decisions import _escalation_request, _packet_for_action
from src.api.run_queue import RunQueue, RunRegistry
from src.decisions import (
    clear_pending_decision,
    load_pending_decision,
    save_pending_decision,
)
from src.main import _infra_gap_note
from src.sandbox.policy import SandboxMode, validate_shell_script
from src.session import SessionContext
from src.workspace.models import WorkspaceBinding


def _session(tmp_path: Path) -> SessionContext:
    sid = "hitl01"
    root = tmp_path / "sessions" / sid
    ws = root / "workspace"
    for d in (root, ws, root / "handoffs", root / "uploads", root / "parsed_requirements"):
        d.mkdir(parents=True)
    return SessionContext(
        session_id=sid,
        title="hitl test",
        state_root=root,
        workspace=WorkspaceBinding(kind="managed", root=ws),
        plan_path=root / "plan.json",
        handoffs_dir=root / "handoffs",
        memory_store_path=root / "memory_store.json",
        events_path=root / "events.jsonl",
        uploads_dir=root / "uploads",
        parsed_requirements_dir=root / "parsed_requirements",
        meta_path=root / "session.json",
        selected_model="auto",
        created_at="now",
        status="created",
    )


# ---------------------------------------------------------------------------
# pending_decision.json lifecycle
# ---------------------------------------------------------------------------

def test_pending_decision_round_trip(tmp_path: Path) -> None:
    ctx = _session(tmp_path)
    assert load_pending_decision(ctx) is None
    saved = save_pending_decision(ctx, {
        "type": "review_fix",
        "title": "Apply the fix?",
        "summary": "2 blockers",
        "options": ["apply_fix", "dismiss"],
        "payload": {"target_files": ["a.py"]},
    })
    assert saved["session_id"] == ctx.session_id
    loaded = load_pending_decision(ctx)
    assert loaded is not None
    assert loaded["type"] == "review_fix"
    assert loaded["payload"] == {"target_files": ["a.py"]}
    clear_pending_decision(ctx)
    assert load_pending_decision(ctx) is None
    clear_pending_decision(ctx)  # idempotent


# ---------------------------------------------------------------------------
# escalation contracts
# ---------------------------------------------------------------------------

def test_escalation_normalize_falls_back_to_stay() -> None:
    decision = normalize_escalation_decision({"decision": "nonsense"})
    assert decision.decision == "stay_hotfix"
    assert decision.source == "escalation_triage"
    fallback = normalize_escalation_decision(None)
    assert fallback.decision == "stay_hotfix"
    assert fallback.source == "deterministic"


def test_escalation_guard_upgrades_sensitive_high_risk() -> None:
    stay = {
        "decision": "stay_hotfix",
        "reason": "small patch",
        "risk": "high",
        "scope_type": "localized",
    }
    upgraded = deterministic_guard(stay, target_files=["migrations/0001.py"])
    assert upgraded["decision"] == "escalate_mission"
    assert upgraded["source"] == "deterministic"

    untouched = deterministic_guard(stay, target_files=["src/util.py"])
    assert untouched["decision"] == "stay_hotfix"

    low_risk = deterministic_guard(
        {**stay, "risk": "low"}, target_files=["migrations/0001.py"]
    )
    assert low_risk["decision"] == "stay_hotfix"


def test_escalation_file_count_never_decides() -> None:
    """Five files with the same one-line fix must survive the guard."""
    stay = {
        "decision": "stay_hotfix",
        "reason": "same import fix in five files",
        "risk": "low",
        "scope_type": "localized",
    }
    out = deterministic_guard(
        stay, target_files=[f"src/mod{i}.py" for i in range(5)]
    )
    assert out["decision"] == "stay_hotfix"


def test_fix_verification_normalize_unverified_fallback() -> None:
    assert normalize_fix_verification(None).verdict == "UNVERIFIED"
    assert normalize_fix_verification({"verdict": "bogus"}).verdict == "UNVERIFIED"
    assert normalize_fix_verification({"verdict": "verified"}).verdict == "VERIFIED"


def test_mission_brief_carries_review_evidence() -> None:
    report = {
        "summary": "two blockers",
        "findings": [
            {
                "severity": "blocker",
                "confidence": "high",
                "fix_criteria": ["close the file handle"],
            }
        ],
    }
    brief = build_mission_brief(
        original_request="review this repo",
        review_report=report,
        hotfix_packet={"target_files": ["a.py"]},
        hotfix_handoff={"files_modified": ["a.py"], "verdict": "FAIL"},
        escalation={"reason": "needs design", "scope_type": "architectural"},
    )
    assert "do not plan another review" in brief["briefing"].lower()
    assert "close the file handle" in brief["briefing"]
    assert brief["review_report"] is report


def test_infra_gap_note_detects_missing_pytest() -> None:
    note = _infra_gap_note(["FAILED - No module named pytest"])
    assert "UNVERIFIED" in note
    assert _infra_gap_note(["assertion error in test_foo"]) == ""


# ---------------------------------------------------------------------------
# safe shell profiles
# ---------------------------------------------------------------------------

def test_fixverify_profile_allows_compile_denies_install() -> None:
    ok = validate_shell_script(
        "python -m py_compile src/a.py",
        profile="fixverify",
        mode=SandboxMode.BALANCED,
    )
    assert ok.allowed
    denied = validate_shell_script(
        "python -m pip install pytest",
        profile="fixverify",
        mode=SandboxMode.BALANCED,
    )
    assert not denied.allowed


def test_review_profile_allows_probe_denies_install() -> None:
    probe = validate_shell_script(
        "python -m pip show pytest",
        profile="review",
        mode=SandboxMode.BALANCED,
    )
    assert probe.allowed
    install = validate_shell_script(
        "python -m pip install pytest",
        profile="review",
        mode=SandboxMode.BALANCED,
    )
    assert not install.allowed


# ---------------------------------------------------------------------------
# decision packet builders
# ---------------------------------------------------------------------------

def test_apply_fix_packet_reuses_stored_milestone() -> None:
    milestone = {"id": "REVIEW-FIX", "target_files": ["a.py"]}
    packet = _packet_for_action(
        "apply_fix",
        {"title": "t"},
        {"milestone": milestone, "target_files": ["a.py"],
         "review_report": {"summary": "s"}},
    )
    assert packet["kind"] == "review_fix"
    assert packet["milestone"] == milestone


def test_apply_fix_without_milestone_is_422() -> None:
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as exc:
        _packet_for_action("apply_fix", {"title": "t"}, {})
    assert exc.value.status_code == 422


def test_setup_env_packet_carries_missing_checks() -> None:
    packet = _packet_for_action(
        "setup_env", {"title": "t"},
        {"missing_checks": ["pytest not installed"],
         "touched_files": ["a.py"], "review_report": {}},
    )
    assert packet["kind"] == "setup_env"
    assert packet["packages"] == ["pytest not installed"]
    assert packet["touched_files"] == ["a.py"]


def test_escalation_request_embeds_prior_evidence() -> None:
    text = _escalation_request(
        {"title": "Hotfix stalled", "summary": "validator kept failing"},
        {"verdict": "REPLAN", "errors": ["No module named pytest"],
         "target_files": ["a.py"]},
    )
    assert "REPLAN" in text
    assert "No module named pytest" in text
    assert "a.py" in text


# ---------------------------------------------------------------------------
# run queue plumbing
# ---------------------------------------------------------------------------

def _verifier_runtime(tmp_path: Path):
    from src.main import MissionsRuntime

    runtime = object.__new__(MissionsRuntime)
    runtime.model = "auto"
    runtime._telemetry_ctx = None
    runtime._active_review_fix_mode = "ask"
    runtime._active_run_kind = "new"
    runtime._emitter = MagicMock()
    runtime._session_manager = MagicMock()
    runtime._persist_run_artifact = lambda *args, **kwargs: None
    return runtime


def _handoff(verdict: str = "PASS"):
    from src.main import MilestoneHandoff

    return MilestoneHandoff(
        milestone_id="HOTFIX",
        title="Focused hotfix",
        worker_summary="Patched app.py",
        files_modified=["app.py"],
        tool_calls_made=2,
        retry_count=0,
        verdict=verdict,
        commit_hash="abc",
        elapsed_ms=10,
    )


def test_verify_verified_completes(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import src.main as main_mod

    monkeypatch.setattr(
        main_mod,
        "run_fix_verification",
        lambda **kwargs: {
            "verdict": "VERIFIED",
            "summary": "criteria met",
            "evidence": [],
            "fix_guidance": "",
            "missing_checks": [],
            "escalation_reason": "",
        },
    )
    runtime = _verifier_runtime(tmp_path)
    session = MagicMock()
    session.session_id = "s1"
    result = runtime._verify_hotfix_result(
        "fix it", session, origin="hotfix",
        milestone={"target_files": ["app.py"]},
        handoff=_handoff("PASS"),
        review_report=None, started_at=0.0,
    )
    assert result is not None
    assert result.status == "completed"
    assert "VERIFIED" in result.summary_text


def test_verify_unverified_pauses_with_setup_options(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import src.main as main_mod

    monkeypatch.setattr(
        main_mod,
        "run_fix_verification",
        lambda **kwargs: {
            "verdict": "UNVERIFIED",
            "summary": "no pytest available",
            "evidence": [],
            "fix_guidance": "",
            "missing_checks": ["pytest not installed"],
            "escalation_reason": "",
        },
    )
    runtime = _verifier_runtime(tmp_path)
    state_root = tmp_path / "sessions" / "s1"
    state_root.mkdir(parents=True)
    session = MagicMock()
    session.session_id = "s1"
    session.state_root = state_root
    result = runtime._verify_hotfix_result(
        "fix it", session, origin="review_fix",
        milestone={"target_files": ["app.py"]},
        handoff=_handoff("PASS"),
        review_report={"summary": "s"}, started_at=0.0,
    )
    assert result is not None
    assert result.status == "awaiting_decision"
    assert result.pending_decision is not None
    assert result.pending_decision["type"] == "fix_unverified"
    assert "setup_env" in result.pending_decision["options"]
    assert "run_smoke" in result.pending_decision["options"]
    # No silent mission escalation for infra gaps.
    assert "escalate_mission" in result.pending_decision["options"]
    stored = load_pending_decision(session)
    assert stored is not None
    assert stored["payload"]["missing_checks"] == ["pytest not installed"]


def test_decisions_routes_registered() -> None:
    from src.api.app import create_app

    app = create_app()
    paths = {(getattr(r, "path", ""), tuple(sorted(getattr(r, "methods", []) or []))) for r in app.routes}
    assert ("/api/v1/sessions/{sid}/decisions", ("GET",)) in paths
    assert ("/api/v1/sessions/{sid}/decisions", ("POST",)) in paths


def test_enqueue_carries_review_fix_mode(tmp_path: Path) -> None:
    registry = RunRegistry(tmp_path / "sessions")
    queue = RunQueue(MagicMock(), registry, MagicMock(), MessageStore(tmp_path / "sessions"))
    ctx = _session(tmp_path)
    rec = queue.enqueue(ctx, "review this", review_fix_mode="auto")
    assert rec.review_fix_mode == "auto"
    queue.shutdown(wait=False)


def test_enqueue_packet_delivers_to_runtime(tmp_path: Path) -> None:
    sessions_root = tmp_path / "sessions"
    registry = RunRegistry(sessions_root)
    runtime = MagicMock()
    session_manager = MagicMock()
    queue = RunQueue(runtime, registry, session_manager, MessageStore(sessions_root))
    ctx = _session(tmp_path)
    session_manager.load_session.return_value = ctx

    packet = {"kind": "smoke", "touched_files": ["a.py"]}
    rec = queue.enqueue_packet(ctx, packet, "smoke follow-up")
    assert rec.run_kind == "new"  # never resume-hijacked

    from src.main import MissionResult
    runtime.run.return_value = MissionResult(
        mission_id="m", title="t", status="completed",
        milestones_passed=1, milestones_total=1, handoffs=[],
        total_elapsed_ms=1.0, model_used="auto", session_id=ctx.session_id,
    )
    try:
        queue._execute(rec, ctx, "auto", packet)
    finally:
        queue.shutdown(wait=False)
    _, kwargs = runtime.run.call_args
    assert kwargs["review_fix_packet"] == packet


# ---------------------------------------------------------------------------
# hotfix-route validation: FixVerifier in-loop (no REPLAN possible)
# ---------------------------------------------------------------------------

from src.agents.fix_verifier import _fix_criteria
from src.main import _map_fix_verification


def test_map_fix_verification() -> None:
    assert _map_fix_verification({"verdict": "VERIFIED"})[0] == "PASS"
    # UNVERIFIED cannot prove failure — PASS so the post-PASS step can pause
    # with setup_env/run_smoke instead of burning retries.
    assert _map_fix_verification({"verdict": "UNVERIFIED"})[0] == "PASS"
    assert _map_fix_verification({})[0] == "PASS"
    loop_verdict, errors, guidance = _map_fix_verification({
        "verdict": "NEEDS_REWORK",
        "summary": "criterion 2 unmet",
        "fix_guidance": "patch app.py line 9",
    })
    assert loop_verdict == "FAIL"
    assert errors == ["criterion 2 unmet"]
    assert guidance == "patch app.py line 9"
    assert _map_fix_verification({"verdict": "ESCALATE_TO_MISSION"})[0] == "ESCALATE"


def test_fix_criteria_prefers_review_then_packet() -> None:
    report = {"findings": [
        {"severity": "bug", "confidence": "high", "fix_criteria": ["no crash on empty"]},
        {"severity": "style", "confidence": "high", "fix_criteria": ["rename x"]},
    ]}
    packet = {"acceptance_criteria": ["issue is fixed"]}
    assert _fix_criteria(report, packet) == ["no crash on empty"]
    # Pure-hotfix route: no actionable findings → packet criteria.
    assert _fix_criteria({}, packet) == ["issue is fixed"]
    assert _fix_criteria(None, {}) == []


def test_verify_reuses_precomputed_verdict(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import src.main as main_mod

    def _boom(**kwargs):
        raise AssertionError("verifier must not re-run")

    monkeypatch.setattr(main_mod, "run_fix_verification", _boom)
    runtime = _verifier_runtime(tmp_path)
    session = MagicMock()
    session.session_id = "s1"
    result = runtime._verify_hotfix_result(
        "fix it", session, origin="hotfix",
        milestone={"target_files": ["app.py"]},
        handoff=_handoff("PASS"),
        review_report=None, started_at=0.0,
        verification={"verdict": "VERIFIED", "summary": "criteria met",
                      "evidence": [], "fix_guidance": "",
                      "missing_checks": [], "escalation_reason": ""},
    )
    assert result is not None
    assert result.status == "completed"
