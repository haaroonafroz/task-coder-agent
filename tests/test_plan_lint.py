"""Tests for deterministic plan lint (src/agents/plan_lint.py)."""

from __future__ import annotations

from src.agents.plan_lint import lint_plan


def _milestone(ms_id: str, **overrides) -> dict:
    base = {
        "id": ms_id,
        "title": f"Milestone {ms_id}",
        "description": "Do the work.",
        "depends_on": [],
        "target_files": ["core.py"],
        "validation_contract": {
            "type": "pytest",
            "command": "python -m pytest tests/test_core.py -v",
        },
    }
    base.update(overrides)
    return base


def test_shell_contract_retyped_to_pytest() -> None:
    plan = {"milestones": [_milestone("M1", validation_contract={
        "type": "shell",
        "command": "python -m pytest tests/test_core.py -v",
    })]}
    _, fixes, issues = lint_plan(plan)
    contract = plan["milestones"][0]["validation_contract"]
    assert contract["type"] == "pytest"
    assert any("retyped" in f for f in fixes)
    assert issues == []


def test_shell_py_compile_retyped() -> None:
    plan = {"milestones": [_milestone("M1", validation_contract={
        "type": "shell",
        "command": "python -m py_compile main.py",
    })]}
    _, fixes, _ = lint_plan(plan)
    assert plan["milestones"][0]["validation_contract"]["type"] == "py_compile"
    assert any("retyped" in f for f in fixes)


def test_workspace_prefixes_stripped() -> None:
    plan = {"milestones": [_milestone(
        "M1",
        target_files=["workspace/core.py"],
        validation_contract={
            "type": "pytest",
            "command": "python -m pytest workspace/tests/test_core.py -v",
        },
    )]}
    _, fixes, _ = lint_plan(plan)
    ms = plan["milestones"][0]
    assert ms["target_files"] == ["core.py"]
    assert "workspace/" not in ms["validation_contract"]["command"]
    assert len(fixes) >= 2


def test_invalid_depends_on_dropped() -> None:
    plan = {"milestones": [
        _milestone("M1"),
        _milestone("M2", depends_on=["M1", "M99", "M2"]),
    ]}
    _, fixes, issues = lint_plan(plan)
    assert plan["milestones"][1]["depends_on"] == ["M1"]
    assert any("depends_on" in f for f in fixes)
    assert issues == []


def test_environment_setup_milestone_flagged() -> None:
    plan = {"milestones": [_milestone(
        "M1",
        title="Environment Setup",
        description="Install the required dependencies with pip install.",
    )]}
    _, _, issues = lint_plan(plan)
    assert any("environment-setup" in i for i in issues)


def test_test_scaffold_missing_fields_flagged() -> None:
    plan = {"milestones": [_milestone(
        "M1",
        target_files=["tests/test_core.py"],
        validation_contract={
            "type": "test_scaffold",
            "command": "python -m pytest tests/test_core.py --collect-only -q",
        },
    )]}
    _, _, issues = lint_plan(plan)
    assert any("test_scaffold contract is missing fields" in i for i in issues)


def test_policy_denied_command_flagged() -> None:
    plan = {"milestones": [_milestone("M1", validation_contract={
        "type": "shell",
        "command": "curl http://localhost:8000/health",
    })]}
    _, _, issues = lint_plan(plan)
    assert any("not executable" in i for i in issues)


def test_missing_target_files_flagged() -> None:
    plan = {"milestones": [_milestone("M1", target_files=[])]}
    _, _, issues = lint_plan(plan)
    assert any("target_files" in i for i in issues)


def test_duplicate_ids_flagged() -> None:
    plan = {"milestones": [_milestone("M1"), _milestone("M1")]}
    _, _, issues = lint_plan(plan)
    assert any("duplicate milestone id" in i for i in issues)


def test_valid_plan_passes_cleanly() -> None:
    plan = {"milestones": [
        _milestone("M1", target_files=["tests/test_core.py"], validation_contract={
            "type": "pytest",
            "command": "python -m pytest tests/test_core.py --collect-only -q",
        }),
        _milestone("M2", depends_on=["M1"]),
    ]}
    _, fixes, issues = lint_plan(plan)
    assert fixes == []
    assert issues == []
