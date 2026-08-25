"""Focused profile routing and contract regression tests."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from src.agents import code_review, hotfix
from src.agents.triage import _validate_report
from src.agents.contracts import (
    hotfix_milestone_from_review,
    normalize_review_report,
    normalize_route_decision,
)
from src.api.schemas import MessageCreate, RunCreate
from src.main import MilestoneHandoff, MissionsRuntime


def test_low_confidence_hotfix_falls_back_to_mission(tmp_path: Path):
    decision = normalize_route_decision(
        {
            "route": "hotfix",
            "confidence": "low",
            "rationale": "Possibly localized.",
            "candidate_files": ["app.py"],
        },
        workspace_root=tmp_path,
    )
    assert decision.route == "mission"


def test_high_confidence_hotfix_requires_candidate_files(tmp_path: Path):
    decision = normalize_route_decision(
        {
            "route": "hotfix",
            "confidence": "high",
            "rationale": "Localized defect.",
            "candidate_files": [],
        },
        workspace_root=tmp_path,
    )
    assert decision.route == "mission"


def test_explicit_review_override_is_preserved(tmp_path: Path):
    decision = normalize_route_decision(
        None,
        workspace_root=tmp_path,
        requested_route="review",
    )
    assert decision.route == "review"
    assert decision.source == "override"


def test_review_only_high_confidence_bugs_are_actionable(tmp_path: Path):
    report = normalize_review_report(
        {
            "verdict": "issues_found",
            "summary": "One bug and one style concern.",
            "scope": "current_diff",
            "findings": [
                {
                    "severity": "bug",
                    "confidence": "high",
                    "title": "Wrong fallback",
                    "issue": "The fallback returns the wrong value.",
                    "evidence": ["app.py:10"],
                    "affected_files": ["app.py"],
                    "fix_criteria": ["Fallback returns the configured value."],
                },
                {
                    "severity": "style",
                    "confidence": "high",
                    "title": "Naming",
                    "issue": "Variable name is terse.",
                    "evidence": ["app.py:3"],
                    "affected_files": ["app.py"],
                    "fix_criteria": ["Rename it."],
                },
            ],
        },
        workspace_root=tmp_path,
    )
    assert len(report.actionable_findings) == 1
    milestone = hotfix_milestone_from_review(report)
    assert milestone["target_files"] == ["app.py"]
    assert milestone["acceptance_criteria"] == [
        "Fallback returns the configured value."
    ]


def test_hotfix_wrapper_uses_distinct_role_and_budget(monkeypatch: pytest.MonkeyPatch):
    captured = {}

    def fake_run_worker(**kwargs):
        captured.update(kwargs)
        return {"status": "complete", "files_modified": ["app.py"]}

    monkeypatch.setattr(hotfix, "run_worker", fake_run_worker)
    result = hotfix.run_hotfix(
        milestone={"id": "H1"},
        plan={"milestones": []},
        curated_tools_md="",
        error_feedback=None,
        retry_count=0,
        model="auto",
        memory=None,
    )
    assert result["status"] == "complete"
    assert result["hotfix_result"]["status"] == "complete"
    assert captured["agent_role"] == "hotfix"
    assert captured["system_prompt"] == hotfix._HOTFIX_MD
    assert captured["max_tool_calls"] == hotfix.MAX_HOTFIX_TOOL_CALLS


def test_reviewer_accepts_xml_tool_calls(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    responses = iter([
        SimpleNamespace(
            text=(
                "Review the diff first.\n"
                "<tool_call><function=git_diff></function></tool_call>"
            )
        ),
        SimpleNamespace(
            text=(
                '{"action":"review","report":{"verdict":"clean",'
                '"summary":"No defects found.","scope":"git diff","findings":[]}}'
            )
        ),
    ])
    monkeypatch.setattr(
        code_review, "build_workspace_orientation", lambda *args, **kwargs: "tree"
    )
    monkeypatch.setattr(code_review, "call_llm", lambda **kwargs: next(responses))
    monkeypatch.setattr(
        code_review,
        "dispatch",
        lambda name, args: {"success": True, "diff": ""},
    )

    report = code_review.run_code_review(
        user_request="Review app.py",
        workspace_root=tmp_path,
        model="auto",
    )
    assert report.verdict == "clean"


def test_reviewer_denies_write_tools(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    responses = iter([
        SimpleNamespace(
            text='{"tool":"write_file","args":{"file_path":"app.py","content":"x"},"reasoning":"fix"}'
        ),
        SimpleNamespace(
            text=(
                '{"action":"review","report":{"verdict":"clean",'
                '"summary":"No defects found.","scope":"app.py","findings":[]}}'
            )
        ),
    ])
    dispatched = []
    monkeypatch.setattr(
        code_review, "build_workspace_orientation", lambda *args, **kwargs: "tree"
    )
    monkeypatch.setattr(code_review, "call_llm", lambda **kwargs: next(responses))
    monkeypatch.setattr(
        code_review,
        "dispatch",
        lambda name, args: dispatched.append((name, args)) or {"success": True},
    )

    report = code_review.run_code_review(
        user_request="Review app.py",
        workspace_root=tmp_path,
        model="auto",
    )
    assert report.verdict == "clean"
    assert dispatched == []


def test_api_schemas_accept_execution_route_override():
    assert MessageCreate(content="review this", execution_route="review").execution_route == "review"
    assert RunCreate(request="fix this", execution_route="hotfix").execution_route == "hotfix"


def test_triage_contract_requires_route_fields():
    assert _validate_report({
        "route": "hotfix",
        "rationale": "One known file.",
        "summary": "Localized change.",
        "candidate_files": ["app.py"],
        "evidence": ["app.py:1"],
        "constraints": [],
        "validation_intent": ["Behavior is corrected."],
        "confidence": "high",
    }) is None
    assert "route" in (_validate_report({
        "summary": "Missing route.",
        "evidence": [],
        "candidate_files": [],
        "constraints": [],
        "validation_intent": [],
        "confidence": "low",
    }) or "")


def test_hotfix_route_preserves_active_plan(tmp_path: Path):
    plan_path = tmp_path / "plan.json"
    original = '{"plan_id":"plan-1","milestones":[{"id":"M1","status":"completed"}]}'
    plan_path.write_text(original, encoding="utf-8")
    runtime = object.__new__(MissionsRuntime)
    runtime._active_run_id = "run-1"
    runtime._active_run_kind = "repair"
    runtime._emitter = SimpleNamespace(emit=lambda *args, **kwargs: None)
    runtime._execute_milestone = lambda *args, **kwargs: MilestoneHandoff(
        milestone_id="HOTFIX",
        title="Focused hotfix",
        worker_summary="Patched app.py",
        files_modified=["app.py"],
        tool_calls_made=2,
        retry_count=0,
        verdict="PASS",
        commit_hash="abc",
        elapsed_ms=10,
    )
    sentinel = object()
    runtime._finish_focused_result = lambda **kwargs: sentinel
    session = SimpleNamespace(
        root=tmp_path,
        plan_path=plan_path,
        handoffs_dir=tmp_path / "handoffs",
        session_id="session-1",
    )
    decision = normalize_route_decision(
        {
            "route": "hotfix",
            "confidence": "high",
            "rationale": "Localized.",
            "candidate_files": ["app.py"],
            "validation_intent": ["Bug is fixed."],
        },
        workspace_root=tmp_path,
    )

    result = runtime._run_hotfix_route(
        "Fix app.py", decision, session, {"plan_id": "plan-1"}, 0.0
    )
    assert result is sentinel
    runtime._save_handoff(
        "HOTFIX",
        "Focused hotfix",
        {"summary": "Patched app.py", "files_modified": ["app.py"], "tool_calls": 2},
        {"verdict": "PASS", "validation_details": "Validated."},
        "abc",
        0,
        session,
    )
    assert plan_path.read_text(encoding="utf-8") == original
    assert (tmp_path / "runs" / "run-1.artifacts" / "hotfix_packet.json").exists()
