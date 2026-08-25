"""Tests for serve_app list action wiring."""

from __future__ import annotations

from src.tools.ui import serve_app


def test_serve_app_list_returns_registry() -> None:
    result = serve_app(action="list")

    assert result["success"] is True
    assert isinstance(result["servers"], list)
    assert 9000 in result["allowed_ports"]
    assert result["max_servers"] == 4
