"""Deterministic validation for harness-managed local UI smoke contracts."""

from __future__ import annotations

import shlex
from typing import Any

from src.tools.ui import inspect_ui, serve_app


def run_ui_smoke_contract(contract: dict[str, Any]) -> dict[str, Any]:
    """Start a local app, run text-oriented checks, and always stop it."""
    serve = contract.get("serve", {})
    if (
        not isinstance(serve, dict)
        or not serve
        or ("kind" not in serve and "command" not in serve)
    ):
        return {
            "verdict": "REPLAN",
            "errors": ["ui_smoke contract must contain a 'serve' object."],
            "replan_guidance": "Add serve.kind, serve.port, and the app entry point.",
        }

    start_args = dict(serve)
    start_args["action"] = "start"
    if (
        str(start_args.get("kind", "")).lower() == "generic"
        and not start_args.get("command")
        and isinstance(contract.get("command"), str)
    ):
        try:
            start_args["command"] = shlex.split(contract["command"])
        except ValueError as exc:
            return {
                "verdict": "REPLAN",
                "errors": [f"Invalid generic server command: {exc}"],
                "replan_guidance": "Provide a shell-parseable command in the UI contract.",
            }
    started = serve_app(**start_args)
    if not started.get("success"):
        error = started.get("error", "UI server failed to start.")
        fix_guidance = "Fix the application startup command and inspect server logs."
        if started.get("servers"):
            fix_guidance = (
                "Use serve_app(action=\"list\") to inspect harness-managed servers, "
                "stop stale ones with serve_app(action=\"stop\", server_id=...), "
                "then retry validation."
            )
        return {
            "verdict": "FAIL" if not started.get("policy_denied") else "REPLAN",
            "errors": [error],
            "root_cause": "The managed UI server was not ready for validation.",
            "fix_guidance": fix_guidance,
            "active_servers": started.get("servers", []),
            "allowed_ports": started.get("allowed_ports", []),
        }

    server_id = str(started.get("server_id", ""))
    results: list[dict[str, Any]] = []
    failure: dict[str, Any] | None = None
    try:
        for check in contract.get("checks", []) or []:
            if not isinstance(check, dict):
                continue
            raw_action = str(check.get("action") or check.get("type") or "snapshot")
            action = {
                "visible_text": "assert_text",
                "browser_audit": "audit",
            }.get(raw_action.lower(), raw_action.lower())
            result = inspect_ui(
                server_id=server_id,
                action=action,
                path=str(check.get("path", "/")),
                text=str(check.get("text", "")),
                selector=str(check.get("selector", "")),
                value=str(check.get("value", "")),
            )
            results.append(result)
            if not result.get("success"):
                failure = {
                    "action": action,
                    "error": result.get("error")
                    or result.get("asserted_text", "")
                    or "the check returned success=false",
                    "capability_missing": result.get("capability_missing"),
                }
                break
        logs = serve_app(action="logs", server_id=server_id)
        if failure:
            missing = failure.get("capability_missing")
            return {
                "verdict": "REPLAN" if missing else "FAIL",
                "errors": [f"UI check '{failure['action']}' failed: {failure['error']}"],
                "root_cause": (
                    "The UI browser capability is unavailable."
                    if missing
                    else "The rendered UI did not satisfy the smoke contract."
                ),
                "fix_guidance": (
                    "Install the harness browser dependency and Chromium."
                    if missing
                    else "Use the URL, browser diagnostics, and server logs to fix the UI."
                ),
                "ui_results": results,
                "server_logs": logs.get("logs", ""),
            }
        return {
            "verdict": "PASS",
            "validation_details": "All UI smoke checks passed.",
            "ui_results": results,
            "server_logs": logs.get("logs", ""),
        }
    finally:
        serve_app(action="stop", server_id=server_id)
