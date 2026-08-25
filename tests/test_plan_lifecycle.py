"""Regression tests for plan identity and repair-run lifecycle handling."""

import json
from pathlib import Path
from types import SimpleNamespace

from src.main import MissionsRuntime
from src.memory_layer import MissionMemory


def test_milestone_completion_is_scoped_to_plan(tmp_path: Path):
    memory = MissionMemory(tmp_path / "memory.json")
    memory.log_milestone_state(
        "M1",
        {"plan_id": "plan-a", "title": "Original"},
        "completed",
        plan_id="plan-a",
    )

    assert memory.check_resume_point("M1", plan_id="plan-a")["status"] == "completed"
    assert memory.check_resume_point("M1", plan_id="plan-b") is None


def test_legacy_milestone_entries_do_not_complete_new_plan(tmp_path: Path):
    memory_path = tmp_path / "memory.json"
    memory_path.write_text(
        json.dumps(
            {
                "milestones": {"M1": {"status": "completed"}},
                "error_log": [],
            }
        ),
        encoding="utf-8",
    )
    memory = MissionMemory(memory_path)

    assert memory.check_resume_point("M1", plan_id="new-repair-plan") is None


def test_run_kind_resolution_distinguishes_resume_and_repair(tmp_path: Path):
    plan_path = tmp_path / "plan.json"
    session = SimpleNamespace(plan_path=plan_path)

    assert MissionsRuntime._resolve_run_kind(session, "auto") == "new"

    plan_path.write_text(
        json.dumps({"plan_id": "plan-a", "milestones": [{"id": "M1", "status": "pending"}]}),
        encoding="utf-8",
    )
    assert MissionsRuntime._resolve_run_kind(session, "auto") == "resume"

    plan_path.write_text(
        json.dumps({"plan_id": "plan-a", "milestones": [{"id": "M1", "status": "completed"}]}),
        encoding="utf-8",
    )
    assert MissionsRuntime._resolve_run_kind(session, "auto") == "repair"
    assert MissionsRuntime._resolve_run_kind(session, "new") == "new"
