"""Tests for semantic tool validation and bounded tool diagnostics."""

from __future__ import annotations

from src.agents.tool_diagnostics import (
    classify_tool_result,
    event_diagnostics,
    tool_failure_signature,
)
from src.tools.git_ops import _filter_sensitive_diff
from src.tools.paths import reset_workspace_root, set_workspace_root
from src.tools.system_ops import search_grep
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


def test_serve_app_accepts_contract_entry_point() -> None:
    assert validate_tool_call(
        "serve_app",
        {
            "action": "start",
            "kind": "generic",
            "port": 9000,
            "entry_point": "index.html",
            "command": ["python", "-m", "http.server", "9000"],
        },
        active_tools={"serve_app"},
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


def test_search_grep_filters_sensitive_files_from_rg_results(tmp_path, monkeypatch) -> None:
    import src.tools.system_ops as system_ops

    (tmp_path / "src").mkdir()
    monkeypatch.setattr(
        system_ops,
        "_run_argv",
        lambda *args, **kwargs: {
            "returncode": 0,
            "stdout": (
                "src/app.py:1:API endpoint\n"
                "secrets.json:2:API_KEY=should-not-leak\n"
                "credentials.json:3:password=should-not-leak"
            ),
            "stderr": "",
        },
    )
    set_workspace_root(tmp_path)
    try:
        result = search_grep("API|password", ".", max_results=20)
    finally:
        reset_workspace_root()

    assert result["success"] is True
    assert [match["file"] for match in result["matches"]] == ["src/app.py"]
    assert "should-not-leak" not in str(result)


def test_git_diff_drops_sensitive_file_sections() -> None:
    raw = (
        "diff --git a/src/app.py b/src/app.py\n"
        "--- a/src/app.py\n"
        "+++ b/src/app.py\n"
        "@@ -1 +1 @@\n"
        "-old\n"
        "+new\n"
        "diff --git a/secrets.json b/secrets.json\n"
        "--- a/secrets.json\n"
        "+++ b/secrets.json\n"
        "@@ -1 +1 @@\n"
        "-old-secret\n"
        "+new-secret\n"
    )

    filtered = _filter_sensitive_diff(raw)

    assert "src/app.py" in filtered
    assert "secrets.json" not in filtered
    assert "new-secret" not in filtered


def test_compact_event_args_truncates_file_bodies() -> None:
    from src.agents.tool_diagnostics import compact_event_args

    compact = compact_event_args({"file_path": "app.py", "content": "x" * 2000})
    assert compact["file_path"] == "app.py"
    assert len(compact["content"]) < 500
    assert str(compact["content"]).endswith("…")
