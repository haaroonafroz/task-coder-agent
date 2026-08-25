"""Tests for worker-owned server cleanup on milestone finish."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from src.agents.worker import run_worker


def _llm_response(text: str) -> MagicMock:
    response = MagicMock()
    response.text = text
    response.model_used = "test"
    response.tokens_prompt = 0
    response.tokens_generated = 0
    response.prefill_ms = 0
    response.decode_ms = 0
    response.total_ms = 0
    response.thinking_level = None
    response.fallback_used = False
    return response


def test_worker_stops_started_servers_on_complete() -> None:
    milestone = {
        "id": "M1",
        "title": "Build page",
        "description": "Create index.html",
        "target_files": ["index.html"],
        "acceptance_criteria": ["Page loads."],
        "validation_profile": "ui",
    }
    plan = {"title": "Demo"}

    with patch("src.agents.worker.call_llm") as llm:
        llm.side_effect = [
            _llm_response(
                '{"calls": ['
                '{"tool": "serve_app", "args": {"action": "start", "kind": "generic", '
                '"port": 9000, "command": ["python", "-m", "http.server", "9000"]}, '
                '"reasoning": "Start static server."}, '
                '{"tool": "inspect_ui", "args": {"server_id": "server-abc123", '
                '"action": "navigate"}, "reasoning": "Smoke check."}'
                ']}'
            ),
            _llm_response(
                '{"status": "complete", "summary": "done", "files_modified": ["index.html"]}'
            ),
        ]
        with patch("src.agents.worker.dispatch") as dispatch:
            def _dispatch(tool_name: str, tool_args: dict) -> dict:
                if tool_name == "list_directory":
                    return {"success": True, "tree": "(empty)"}
                if tool_name == "serve_app":
                    return {"success": True, "server_id": "server-abc123", "port": 9000}
                if tool_name == "inspect_ui":
                    return {"success": True, "url": "http://127.0.0.1:9000/"}
                raise AssertionError(f"unexpected tool: {tool_name}")

            dispatch.side_effect = _dispatch
            with patch("src.agents.worker.target_files_exist", return_value=(True, [])):
                with patch("src.agents.worker.check_target_file_dependencies") as dep:
                    dep.return_value.ok = True
                    with patch("src.agents.worker.stop_server") as stop_server:
                        result = run_worker(
                            milestone=milestone,
                            plan=plan,
                            curated_tools_md="",
                            error_feedback=None,
                            retry_count=0,
                            model="auto",
                            memory=None,
                            initial_tool_names={"serve_app", "inspect_ui", "search_tools"},
                        )

    assert result["status"] == "complete"
    stop_server.assert_called_once_with("server-abc123")
