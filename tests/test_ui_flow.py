"""Tests for single-session UI flow and browser-backed assertions."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from src.tools.ui import _looks_dynamic_html, _run_flow_step, _static_snapshot, inspect_ui


def test_looks_dynamic_html_detects_script_tags() -> None:
    assert _looks_dynamic_html("<html><script></script></html>") is True
    assert _looks_dynamic_html("<html><body>Hello</body></html>") is False


def test_static_snapshot_marks_dynamic_pages() -> None:
    body = "<html><head><title>T</title></head><body><script></script></body></html>"
    with patch("src.tools.ui._fetch", return_value=(200, body)):
        result = _static_snapshot("http://127.0.0.1:1/", action="assert_text", text="Hello")

    assert result["render_mode"] == "static_fetch"
    assert result["suggest_handoff"] is True
    assert "JavaScript-rendered" in result["hint"]


def test_flow_step_assert_text_reports_handoff_hint() -> None:
    page = MagicMock()
    page.locator.return_value.inner_text.return_value = "Weekly Habits"
    result = _run_flow_step(
        page,
        {"action": "assert_text", "text": "Exercise"},
        url="http://127.0.0.1:9000/",
    )

    assert result["success"] is False
    assert result["suggest_handoff"] is True
    assert "signal" in result["hint"]


def test_inspect_ui_flow_delegates_to_browser_layer() -> None:
    expected = {
        "success": True,
        "render_mode": "browser",
        "flow_results": [{"success": True}],
    }
    with patch(
        "src.tools.ui.server_status",
        return_value={"success": True, "running": True, "port": 9000},
    ), patch("src.tools.ui._with_browser", return_value=expected) as browser:
        result = inspect_ui(
            "server-1",
            action="flow",
            steps=[{"action": "snapshot"}],
        )

    assert result == expected
    browser.assert_called_once()


def test_inspect_ui_assert_text_delegates_to_browser_layer() -> None:
    expected = {
        "success": False,
        "render_mode": "browser",
        "suggest_handoff": True,
        "hint": "handoff",
    }
    with patch(
        "src.tools.ui.server_status",
        return_value={"success": True, "running": True, "port": 9000},
    ), patch("src.tools.ui._with_browser", return_value=expected):
        result = inspect_ui("server-1", action="assert_text", text="Exercise")

    assert result["render_mode"] == "browser"
    assert result["suggest_handoff"] is True
