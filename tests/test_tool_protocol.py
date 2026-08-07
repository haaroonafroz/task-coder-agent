"""Tests for semantic tool validation and bounded tool diagnostics."""

from __future__ import annotations

from src.agents.tool_diagnostics import (
    classify_tool_result,
    event_diagnostics,
    tool_failure_signature,
)
from src.tools.tool_contracts import validate_tool_call


def test_unknown_tool_is_rejected_with_discovery_guidance() -> None:
    result = validate_tool_call(
        "run_command",
        {"command": "pytest"},
        active_tools={"read_file", "search_tools"},
    )
    assert result is not None
    assert result["error_category"] == "unknown_tool"
    assert "search_tools" in result["error"]


def test_known_but_undisclosed_tool_is_rejected() -> None:
    result = validate_tool_call(
        "run_shellscript",
        {"script": "pwd"},
        active_tools={"read_file", "search_tools"},
    )
    assert result is not None
    assert result["error_category"] == "tool_not_available"


def test_run_pytest_rejects_wrong_argument_name() -> None:
    result = validate_tool_call(
        "run_pytest",
        {"command": "pytest tests/test_app.py"},
        active_tools={"run_pytest", "search_tools"},
    )
    assert result is not None
    assert result["error_category"] == "invalid_arguments"
    assert "test_path" in result["error"]


def test_valid_shell_call_is_accepted() -> None:
    assert validate_tool_call(
        "run_shellscript",
        {"script": "python -c 'import app'", "timeout": 10},
        active_tools={"run_shellscript", "search_tools"},
    ) is None


def test_search_tools_limit_is_bounded() -> None:
    result = validate_tool_call(
        "search_tools",
        {"query": "run tests", "limit": 9},
        active_tools={"search_tools"},
    )
    assert result is not None
    assert result["error_category"] == "invalid_arguments"


def test_diagnostics_capture_shell_failure_without_unbounded_output() -> None:
    result = {
        "success": False,
        "returncode": 1,
        "stdout": "normal output",
        "stderr": "API_KEY=sk-super-secret\n" + ("x" * 3000),
        "timed_out": False,
        "execution_mode": "shell",
        "cwd": "/tmp/sessions/abc/workspace",
    }
    diagnostics = event_diagnostics(
        "run_shellscript",
        {"script": "python -c 'print(1)'"},
        result,
        12.3456,
    )
    assert diagnostics["error_category"] == "nonzero_exit"
    assert diagnostics["duration_ms"] == 12.35
    assert len(diagnostics["stderr_tail"]) <= 1600
    assert "sk-super-secret" not in diagnostics["stderr_tail"]
    assert diagnostics["failure_signature"]


def test_failure_signature_is_stable_across_session_paths() -> None:
    args = {"script": "python -c 'import app'"}
    result_a = {
        "success": False,
        "returncode": 1,
        "stderr": "/tmp/sessions/aaa/workspace/app.py:12: error",
    }
    result_b = {
        "success": False,
        "returncode": 1,
        "stderr": "/tmp/sessions/bbb/workspace/app.py:99: error",
    }
    assert tool_failure_signature("run_shellscript", args, result_a) == (
        tool_failure_signature("run_shellscript", args, result_b)
    )


def test_failure_classifier_prioritizes_policy_and_timeout() -> None:
    assert classify_tool_result({
        "success": False,
        "policy_denied": True,
        "returncode": -1,
    }) == "policy_denied"
    assert classify_tool_result({
        "success": False,
        "timed_out": True,
        "returncode": -1,
    }) == "timeout"
