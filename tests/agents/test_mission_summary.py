"""Tests for deterministic mission summaries."""

from __future__ import annotations

from dataclasses import dataclass, field

from src.agents.mission_summary import build_mission_summary


@dataclass
class _Handoff:
    milestone_id: str
    title: str
    worker_summary: str
    files_modified: list[str]
    tool_calls_made: int = 0
    retry_count: int = 0
    verdict: str = "PASS"
    commit_hash: str = ""
    elapsed_ms: float = 0.0
    error_log: list[str] = field(default_factory=list)


def test_completed_mission_summary_lists_files_and_milestones():
    summary = build_mission_summary(
        title="Task scheduler",
        status="completed",
        milestones_passed=2,
        milestones_total=2,
        total_elapsed_ms=120_000,
        handoffs=[
            _Handoff(
                "M1",
                "Parser",
                "Implemented schedule parser",
                ["schedule_parser.py", "tests/test_parser.py"],
            ),
            _Handoff(
                "M2",
                "Scheduler",
                "Added scheduler loop",
                ["scheduler.py"],
            ),
        ],
    )

    text = summary["summary_text"]
    assert "Task scheduler" in text
    assert "completed" in text
    assert "schedule_parser.py" in text
    assert "scheduler.py" in text
    assert "All milestones passed validation." in text
    assert summary["failure_reason"] == ""


def test_failed_mission_summary_explains_stop_reason():
    summary = build_mission_summary(
        title="Habit tracker",
        status="failed",
        milestones_passed=0,
        milestones_total=2,
        total_elapsed_ms=45_000,
        handoffs=[
            _Handoff(
                "M1",
                "UI shell",
                "Built React shell",
                ["src/App.tsx"],
                verdict="FAIL",
                error_log=[
                    "[Retry 3] Errors: ['assert_text failed'] | Guidance: Fix button label"
                ],
            ),
        ],
        incomplete_milestone_ids=["M1", "M2"],
    )

    assert "failed" in summary["summary_text"]
    assert "Fix button label" in summary["summary_text"]
    assert "retries were exhausted" in summary["failure_reason"]
