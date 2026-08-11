"""Tests for deterministic high-level work-packet linting."""

from __future__ import annotations

from src.agents.plan_lint import lint_plan


def _milestone(ms_id: str, **overrides) -> dict:
    base = {
        "id": ms_id,
        "title": f"Milestone {ms_id}",
        "description": "Do the work.",
        "depends_on": [],
        "target_files": ["core.py"],
        "acceptance_criteria": ["The feature behaves correctly."],
        "validation_profile": "python",
    }
    base.update(overrides)
    return base


def test_valid_packet_passes_cleanly() -> None:
    _, fixes, issues = lint_plan({"milestones": [_milestone("M1")]})
    assert fixes == []
    assert issues == []


def test_acceptance_criteria_are_required() -> None:
    _, _, issues = lint_plan({
        "milestones": [_milestone("M1", acceptance_criteria=[])]
    })
    assert any("acceptance_criteria" in issue for issue in issues)


def test_validation_profile_is_bounded() -> None:
    _, _, issues = lint_plan({
        "milestones": [_milestone("M1", validation_profile="shell")]
    })
    assert any("unsupported validation_profile" in issue for issue in issues)


def test_contracts_are_rejected_from_packets() -> None:
    _, _, issues = lint_plan({
        "milestones": [_milestone(
            "M1",
            validation_contract={"type": "pytest", "command": "pytest"},
        )]
    })
    assert any("validation_contract is not allowed" in issue for issue in issues)


def test_workspace_prefixes_are_stripped() -> None:
    plan = {"milestones": [_milestone("M1", target_files=["workspace/core.py"])]}
    _, fixes, issues = lint_plan(plan)
    assert plan["milestones"][0]["target_files"] == ["core.py"]
    assert any("workspace/" in fix for fix in fixes)
    assert issues == []


def test_invalid_depends_on_dropped() -> None:
    plan = {"milestones": [
        _milestone("M1"),
        _milestone("M2", depends_on=["M1", "M99", "M2"]),
    ]}
    _, fixes, issues = lint_plan(plan)
    assert plan["milestones"][1]["depends_on"] == ["M1"]
    assert any("depends_on" in fix for fix in fixes)
    assert issues == []


def test_forward_dependency_is_rejected() -> None:
    _, _, issues = lint_plan({"milestones": [
        _milestone("M1", depends_on=["M2"]),
        _milestone("M2"),
    ]})
    assert any("earlier milestones" in issue for issue in issues)


def test_environment_setup_milestone_flagged() -> None:
    _, _, issues = lint_plan({"milestones": [_milestone(
        "M1",
        title="Environment Setup",
        description="Install the required dependencies with pip install.",
    )]})
    assert any("environment-setup" in issue for issue in issues)


def test_duplicate_ids_flagged() -> None:
    _, _, issues = lint_plan({"milestones": [_milestone("M1"), _milestone("M1")]})
    assert any("duplicate milestone id" in issue for issue in issues)
