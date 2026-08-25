"""Tests for harness-managed local server lifecycle."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.sandbox import process_manager as pm


@pytest.fixture(autouse=True)
def _clear_process_registry() -> None:
    pm._PROCESSES.clear()
    yield
    pm._PROCESSES.clear()


def _managed(server_id: str, *, port: int = 9000, running: bool = True) -> pm.ManagedProcess:
    process = MagicMock()
    process.poll.return_value = None if running else 0
    return pm.ManagedProcess(
        server_id=server_id,
        process=process,
        port=port,
        log_path=Path("/tmp/server.log"),
        started_at=1.0,
    )


def test_list_servers_prunes_dead_entries() -> None:
    pm._PROCESSES["alive"] = _managed("alive", port=9001, running=True)
    pm._PROCESSES["dead"] = _managed("dead", port=9002, running=False)

    result = pm.list_servers()

    assert result["success"] is True
    assert [entry["server_id"] for entry in result["servers"]] == ["alive"]
    assert result["allowed_ports"] == pm.get_allowed_ports()
    assert "dead" not in pm._PROCESSES


def test_start_server_error_includes_active_servers() -> None:
    pm._PROCESSES["one"] = _managed("one", port=9000)
    pm._PROCESSES["two"] = _managed("two", port=9001)
    pm._PROCESSES["three"] = _managed("three", port=9002)
    pm._PROCESSES["four"] = _managed("four", port=9003)
    sandbox = MagicMock()
    sandbox.jail_root = Path("/tmp/jail")
    sandbox.workspace_root = Path("/tmp/workspace")

    with patch.object(pm, "_ctx", return_value=sandbox):
        result = pm.start_server(
            ["python", "-m", "http.server", "9004"],
            port=9004,
            ready_url="http://127.0.0.1:9004/",
        )

    assert result["success"] is False
    assert "At most 4 harness-managed servers" in result["error"]
    assert len(result["servers"]) == 4
    assert result["allowed_ports"] == pm.get_allowed_ports()


def test_start_server_rejects_disallowed_port_with_allowlist() -> None:
    sandbox = MagicMock()
    sandbox.jail_root = Path("/tmp/jail")
    sandbox.workspace_root = Path("/tmp/workspace")

    with patch.object(pm, "_ctx", return_value=sandbox):
        result = pm.start_server(
            ["python", "-m", "http.server", "8088"],
            port=8088,
            ready_url="http://127.0.0.1:8088/",
        )

    assert result["success"] is False
    assert "not in the session port allowlist" in result["error"]
    assert 8088 not in result["allowed_ports"]
    assert 9000 in result["allowed_ports"]
