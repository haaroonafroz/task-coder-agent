"""Unit tests for validator deterministic gates."""

from __future__ import annotations

from src.agents.validator import (
    _block_infrastructure_replan,
    _detect_all_skipped_spec_gaming,
    _detect_collect_only_pass,
    _detect_policy_denial_replan,
    _pytest_ran_in_output,
    _target_file_boundary_fail,
    _unauthorized_test_edits,
)


def test_pytest_ran_detects_collected_tests() -> None:
    output = "stdout:\n22 tests collected in 0.05s\nreturncode: 0"
    assert _pytest_ran_in_output(output, 0) is True


def test_pytest_ran_detects_argv_execution_mode() -> None:
    output = "stdout:\n\nreturncode: 1\nexecution_mode: argv"
    assert _pytest_ran_in_output(output, 1) is True


def test_collect_only_pass_on_zero_exit() -> None:
    milestone = {"id": "M1", "target_files": ["tests/test_x.py"]}
    contract = {
        "command": "python -m pytest tests/test_x.py --collect-only -q",
        "pass_criteria": "discovers at least 8 test functions",
    }
    output = "stdout:\n22 tests collected in 0.01s\nreturncode: 0\nexecution_mode: argv"

    result = _detect_collect_only_pass(milestone, contract, output, 0, True)
    assert result is not None
    assert result["verdict"] == "PASS"


def test_block_infrastructure_replan_broad_markers() -> None:
    parsed = {
        "verdict": "REPLAN",
        "replan_guidance": (
            "Verify that pytest is correctly installed and virtual environments "
            "are activated before re-executing the contract."
        ),
    }
    output = "stdout:\n22 tests collected\nreturncode: 1\nexecution_mode: argv"

    blocked = _block_infrastructure_replan(parsed, output, 1)
    assert blocked["verdict"] == "FAIL"
    assert blocked.get("replan_guidance") is None


def test_block_infrastructure_replan_shell_127_without_pytest() -> None:
    parsed = {
        "verdict": "REPLAN",
        "replan_guidance": (
            "The validation environment is misconfigured. Please verify the "
            "integrity of the virtual environment or reset the environment path."
        ),
    }
    output = (
        "stdout:\n\nstderr:\n/bin/bash: python: command not found\n"
        "returncode: 127\nexecution_mode: shell\npolicy_denied: False"
    )

    blocked = _block_infrastructure_replan(parsed, output, 127)
    assert blocked["verdict"] == "FAIL"
    assert blocked.get("replan_guidance") is None
    assert any("shell mode" in err.lower() for err in blocked["errors"])


def test_all_skipped_spec_gaming_fails(tmp_path) -> None:
    test_file = tmp_path / "tests" / "test_x.py"
    test_file.parent.mkdir(parents=True)
    test_file.write_text(
        "import pytest\n"
        "@pytest.mark.skip(reason='later')\n"
        "def test_a(): pass\n"
        "@pytest.mark.skip(reason='later')\n"
        "def test_b(): pass\n",
        encoding="utf-8",
    )

    # Patch resolve_workspace_path to point at tmp_path file
    from unittest.mock import patch

    milestone = {"id": "M1", "target_files": ["tests/test_x.py"]}
    worker = {"files_modified": ["tests/test_x.py"]}

    with patch("src.agents.validator.resolve_workspace_path", return_value=test_file):
        result = _detect_all_skipped_spec_gaming(milestone, worker, "", True)

    assert result is not None
    assert result["verdict"] == "FAIL"


def test_policy_denial_replan_includes_allowlist() -> None:
    milestone = {"id": "M1"}
    contract = {
        "command": "grep -q eval tests/test_math_eval.py",
        "type": "shell",
    }
    tool_result = {
        "policy_denied": True,
        "stderr": "Policy denied: Command 'grep' not in validation allowlist",
        "returncode": -1,
    }
    result = _detect_policy_denial_replan(milestone, contract, tool_result, "")
    assert result is not None
    assert result["verdict"] == "REPLAN"
    assert "pytest" in result["replan_guidance"]
    assert "collect-only" in result["replan_guidance"]
    assert result["policy_reference"]["profile"] == "validation"


def test_policy_denial_not_triggered_without_flag() -> None:
    tool_result = {"returncode": 1, "stderr": "some error"}
    assert _detect_policy_denial_replan({}, {}, tool_result, "") is None



def test_target_file_boundary_rejects_out_of_scope_edits() -> None:
    milestone = {"id": "M1", "target_files": ["tests/test_x.py"]}
    worker = {"files_modified": ["tests/test_x.py", "snake_logic.py"]}

    result = _target_file_boundary_fail(milestone, worker)

    assert result is not None
    assert result["verdict"] == "FAIL"
    assert result["out_of_scope_files"] == ["snake_logic.py"]


def test_target_file_boundary_allows_workspace_prefix_equivalence() -> None:
    milestone = {"id": "M1", "target_files": ["tests/test_x.py"]}
    worker = {"files_modified": ["workspace/tests/test_x.py"]}

    assert _target_file_boundary_fail(milestone, worker) is None


def test_new_agent_owned_test_is_not_spec_gaming() -> None:
    worker = {
        "files_modified": ["tests/test_roman.py"],
        "created_files": ["tests/test_roman.py"],
    }

    assert _unauthorized_test_edits(worker) == []


def test_existing_acceptance_test_edit_is_spec_gaming() -> None:
    worker = {
        "files_modified": ["tests/test_existing.py"],
        "created_files": [],
    }

    assert _unauthorized_test_edits(worker) == ["tests/test_existing.py"]
