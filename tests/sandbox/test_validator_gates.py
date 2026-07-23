"""Unit tests for validator deterministic gates."""

from __future__ import annotations

from src.agents.validator import (
    _block_infrastructure_replan,
    _detect_all_skipped_spec_gaming,
    _detect_collect_only_pass,
    _detect_policy_denial_replan,
    _pytest_ran_in_output,
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
