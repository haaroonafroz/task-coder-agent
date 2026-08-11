"""Tests for deterministic UI smoke contract handling."""

from __future__ import annotations

from unittest.mock import patch

from src.agents.ui_validation import run_ui_smoke_contract


def test_ui_smoke_contract_stops_server_after_success() -> None:
    contract = {
        "type": "ui_smoke",
        "serve": {"kind": "vite", "port": 5173},
        "checks": [{"type": "visible_text", "text": "To Do"}],
    }

    with patch(
        "src.agents.ui_validation.serve_app",
        side_effect=[
            {"success": True, "server_id": "server-1"},
            {"success": True},
            {"success": True},
        ],
    ) as serve:
        with patch(
            "src.agents.ui_validation.inspect_ui",
            return_value={"success": True, "passed": True},
        ) as inspect:
            result = run_ui_smoke_contract(contract)

    assert result["verdict"] == "PASS"
    assert inspect.call_args.kwargs["action"] == "assert_text"
    assert serve.call_args_list[-1].kwargs == {"action": "stop", "server_id": "server-1"}


def test_ui_smoke_contract_replans_without_serve_object() -> None:
    result = run_ui_smoke_contract({"type": "ui_smoke", "checks": []})

    assert result["verdict"] == "REPLAN"


def test_ui_smoke_contract_replans_when_chromium_is_missing() -> None:
    contract = {
        "type": "ui_smoke",
        "command": "python -m http.server 9000 --bind 127.0.0.1",
        "serve": {"kind": "generic", "port": 9000, "entry_point": "index.html"},
        "checks": [{"action": "accessibility"}],
    }
    with patch(
        "src.agents.ui_validation.serve_app",
        side_effect=[
            {"success": True, "server_id": "server-1"},
            {"success": True, "logs": ""},
            {"success": True},
        ],
    ) as serve:
        with patch(
            "src.agents.ui_validation.inspect_ui",
            return_value={
                "success": False,
                "capability_missing": "chromium",
                "error": "Chromium is not installed",
            },
        ):
            result = run_ui_smoke_contract(contract)

    assert result["verdict"] == "REPLAN"
    assert result["ui_results"][0]["capability_missing"] == "chromium"
    assert serve.call_args_list[0].kwargs["command"] == [
        "python",
        "-m",
        "http.server",
        "9000",
        "--bind",
        "127.0.0.1",
    ]
    assert serve.call_args_list[-1].kwargs == {"action": "stop", "server_id": "server-1"}
