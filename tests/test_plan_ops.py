"""Tests for safe high-level work-packet patches."""

from __future__ import annotations

import pytest

from src.agents.plan_ops import PlanPatchError, apply_plan_patch


def _milestone(ms_id: str, status: str = "pending") -> dict:
    return {
        "id": ms_id,
        "title": f"Milestone {ms_id}",
        "description": "Implement the feature.",
        "depends_on": [],
        "target_files": [f"{ms_id.lower()}.py"],
        "acceptance_criteria": ["The feature behaves correctly."],
        "validation_profile": "python",
        "status": status,
    }


def test_updates_acceptance_intent_without_executable_contract() -> None:
    plan = {"milestones": [_milestone("M1")]}
    patched = apply_plan_patch(plan, [{
        "op": "update_milestone",
        "milestone_id": "M1",
        "fields": {
            "acceptance_criteria": ["The result is visible to the user."],
            "validation_profile": "structural",
        },
    }])

    assert patched["milestones"][0]["acceptance_criteria"] == [
        "The result is visible to the user."
    ]
    assert "validation_contract" not in patched["milestones"][0]
    assert plan["milestones"][0]["validation_profile"] == "python"


def test_insert_requires_acceptance_criteria() -> None:
    with pytest.raises(PlanPatchError, match="acceptance_criteria"):
        apply_plan_patch({"milestones": [_milestone("M1")]}, [{
            "op": "insert_milestone_after",
            "after_id": "M1",
            "milestone": {
                "id": "M2",
                "title": "Missing intent",
                "target_files": ["m2.py"],
            },
        }])


def test_completed_milestone_is_immutable() -> None:
    with pytest.raises(PlanPatchError, match="completed"):
        apply_plan_patch({"milestones": [_milestone("M1", "completed")]}, [{
            "op": "update_milestone",
            "milestone_id": "M1",
            "fields": {"acceptance_criteria": ["Changed"]},
        }])


def test_contract_operations_are_rejected() -> None:
    with pytest.raises(PlanPatchError, match="unsupported op"):
        apply_plan_patch({"milestones": [_milestone("M1")]}, [{
            "op": "set_contract",
            "milestone_id": "M1",
            "validation_contract": {"type": "pytest"},
        }])
