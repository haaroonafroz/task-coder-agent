"""Browser-backed checks for the harness UI inspection tool."""

from __future__ import annotations

import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from src.tools.ui import inspect_ui

pytest.importorskip("playwright", reason="Playwright is not installed")


class _UiHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        body = (
            "<!doctype html><html><head><title>Demo board</title></head>"
            "<body><h1>To Do</h1><button>Save</button>"
            '<img alt="Board illustration" src="/image.svg"></body></html>'
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args: object) -> None:
        pass


def test_browser_audit_captures_clean_local_page(tmp_path: Path) -> None:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _UiHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    context = SimpleNamespace(jail_root=tmp_path)
    try:
        with patch(
            "src.tools.ui.server_status",
            return_value={
                "success": True,
                "running": True,
                "port": server.server_address[1],
            },
        ), patch("src.tools.ui.get_sandbox_context", return_value=context):
            result = inspect_ui("server-1", action="audit")
            accessibility = inspect_ui("server-1", action="accessibility")
    finally:
        server.shutdown()
        thread.join(timeout=2)

    assert result["success"] is True
    assert result["console_errors"] == []
    assert result["failed_requests"] == []
    assert accessibility["success"] is True
    assert accessibility["accessibility_issues"] == []
