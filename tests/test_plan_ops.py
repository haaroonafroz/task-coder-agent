"""Tests for deterministic plan patch operations (src/agents/plan_ops.py)."""

from __future__ import annotations

import pytest

from src.agents.plan_ops import apply_plan_patch, PlanPatchError


def _plan() -> dict:
    return {
        "mission_id": "abc123",
        "title": "Test Mission",
        "milestones": [
            {
                "id": "M1",
                "title": "Scaffold tests",
                "description": "Write tests.",
                "depends_on": [],
                "target_files": ["tests/test_core.py"],
                "validation_contract": {
                    "type": "pytest",
                    "command": "python -m pytest tests/test_core.py --collect-only -q",
                },
                "status": "completed",
            },
            {
                "id": "M2",
                "title": "Implement core",
                "description": "Implement core logic.",
                "depends_on": ["M1"],
                "target_files": ["core.py"],
                "validation_contract": {
                    "type": "pytest",
                    "command": "python -m pytest tests/test_core.py -v",
                },
                "status": "pending",
            },
            {
                "id": "M3",
                "title": "CLI",
                "description": "Wire CLI.",
                "depends_on": ["M2"],
                "target_files": ["main.py"],
                "validation_contract": {
                    "type": "pytest",
                    "command": "python -m pytest tests/test_core.py -v",
                },
                "status": "pending",
            },
        ],
    }


def test_set_contract_replaces_only_target_milestone_contract() -> None:
    plan = _plan()
    new_contract = {"type": "pytest", "command": "python -m pytest tests/test_core.py -v -k move"}
    patched = apply_plan_patch(plan, [
        {"op": "set_contract", "milestone_id": "M2", "validation_contract": new_contract},
    ])
    assert patched["milestones"][1]["validation_contract"] == new_contract
    # Original plan untouched (no mutation)
    assert plan["milestones"][1]["validation_contract"]["command"].endswith("-v")
    # Other milestones untouched
    assert patched["milestones"][2]["validation_contract"] == plan["milestones"][2]["validation_contract"]


def test_set_contract_on_completed_milestone_is_rejected() -> None:
    with pytest.raises(PlanPatchError, match="immutable"):
        apply_plan_patch(_plan(), [
            {"op": "set_contract", "milestone_id": "M1",
             "validation_contract": {"type": "pytest", "command": "x"}},
        ])


def test_update_milestone_fields() -> None:
    patched = apply_plan_patch(_plan(), [
        {"op": "update_milestone", "milestone_id": "M2",
         "fields": {"description": "Implement core logic with wrap-around.",
                    "target_files": ["core.py", "core_utils.py"]}},
    ])
    m2 = patched["milestones"][1]
    assert m2["description"].endswith("wrap-around.")
    assert m2["target_files"] == ["core.py", "core_utils.py"]
    assert m2["title"] == "Implement core"  # untouched


def test_update_milestone_rejects_unknown_fields() -> None:
    with pytest.raises(PlanPatchError, match="cannot update fields"):
        apply_plan_patch(_plan(), [
            {"op": "update_milestone", "milestone_id": "M2", "fields": {"id": "M9"}},
        ])


def test_insert_milestone_after() -> None:
    patched = apply_plan_patch(_plan(), [
        {"op": "insert_milestone_after", "after_id": "M1", "milestone": {
            "id": "M1a", "title": "Extra", "description": "Extra work.",
            "target_files": ["extra.py"],
            "validation_contract": {"type": "pytest", "command": "python -m pytest tests/test_core.py -q"},
        }},
    ])
    ids = [m["id"] for m in patched["milestones"]]
    assert ids == ["M1", "M1a", "M2", "M3"]
    assert patched["milestones"][1]["status"] == "pending"


def test_insert_duplicate_id_is_rejected() -> None:
    with pytest.raises(PlanPatchError, match="already exists"):
        apply_plan_patch(_plan(), [
            {"op": "insert_milestone_after", "after_id": "M1", "milestone": {
                "id": "M2", "title": "Dup", "target_files": ["x.py"],
                "validation_contract": {"type": "pytest", "command": "y"},
            }},
        ])


def test_insert_missing_anchor_is_rejected() -> None:
    with pytest.raises(PlanPatchError, match="anchor"):
        apply_plan_patch(_plan(), [
            {"op": "insert_milestone_after", "after_id": "M99", "milestone": {
                "id": "M1a", "title": "Extra", "target_files": ["x.py"],
                "validation_contract": {"type": "pytest", "command": "y"},
            }},
        ])


def test_remove_milestone_cleans_depends_on() -> None:
    patched = apply_plan_patch(_plan(), [
        {"op": "remove_milestone", "milestone_id": "M3"},
    ])
    ids = [m["id"] for m in patched["milestones"]]
    assert ids == ["M1", "M2"]


def test_remove_completed_milestone_is_rejected() -> None:
    with pytest.raises(PlanPatchError, match="immutable"):
        apply_plan_patch(_plan(), [
            {"op": "remove_milestone", "milestone_id": "M1"},
        ])


def test_empty_operations_rejected() -> None:
    with pytest.raises(PlanPatchError, match="non-empty"):
        apply_plan_patch(_plan(), [])


def test_unsupported_op_rejected() -> None:
    with pytest.raises(PlanPatchError, match="unsupported op"):
        apply_plan_patch(_plan(), [{"op": "regenerate_everything"}])
