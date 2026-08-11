"""Lifecycle manager for harness-owned local development servers."""

from __future__ import annotations

import os
import signal
import subprocess
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.sandbox.context import SandboxContext, get_sandbox_context
from src.sandbox.env import build_sandbox_env
from src.sandbox.policy import validate_argv


_ALLOWED_PORTS = set(range(9000, 9050))
_MAX_SERVERS = 4


def get_allowed_ports() -> list[int]:
    """Return sorted session ports the harness may bind for UI tests."""
    return sorted(_ALLOWED_PORTS)


def format_allowed_ports_block() -> str:
    """Markdown block injected when UI server tools are available."""
    ports = get_allowed_ports()
    if not ports:
        return "## Session server ports\nNo ports are configured for harness-managed servers."
    if len(ports) <= 12:
        listing = ", ".join(str(port) for port in ports)
    else:
        listing = f"{ports[0]}–{ports[-1]} ({len(ports)} ports)"
    return (
        "## Session server ports\n"
        f"The harness may start test servers only on: **{listing}**.\n"
        "Admin and infrastructure services on other ports are not reachable "
        "or stoppable by `serve_app`.\n"
        "Use `serve_app` with `action: \"list\"` to inspect harness-managed "
        "servers before stopping stale ones."
    )


@dataclass
class ManagedProcess:
    server_id: str
    process: subprocess.Popen
    port: int
    log_path: Path
    started_at: float


_PROCESSES: dict[str, ManagedProcess] = {}


def _prune_dead_servers() -> None:
    """Drop finished processes so they no longer consume the server slot cap."""
    dead = [
        server_id
        for server_id, managed in _PROCESSES.items()
        if managed.process.poll() is not None
    ]
    for server_id in dead:
        _PROCESSES.pop(server_id, None)


def _active_servers(include_logs: bool = False) -> list[dict[str, Any]]:
    _prune_dead_servers()
    servers: list[dict[str, Any]] = []
    for managed in _PROCESSES.values():
        entry = {
            "server_id": managed.server_id,
            "port": managed.port,
            "running": managed.process.poll() is None,
            "started_at": managed.started_at,
        }
        if include_logs:
            entry["logs_tail"] = _read_log(managed.log_path)
        servers.append(entry)
    servers.sort(key=lambda item: item["started_at"])
    return servers


def _ctx(ctx: SandboxContext | None = None) -> SandboxContext | None:
    return ctx or get_sandbox_context()


def _kill(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        proc.wait(timeout=3)
    except (OSError, subprocess.TimeoutExpired):
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except OSError:
            pass


def _read_log(path: Path, limit: int = 3000) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")[-limit:]
    except OSError:
        return ""


def _ready(url: str, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=1) as response:
                if 200 <= response.status < 500:
                    return True
        except (OSError, urllib.error.URLError):
            time.sleep(0.2)
    return False


def start_server(
    argv: list[str],
    *,
    port: int,
    ready_url: str,
    timeout: int = 45,
    ctx: SandboxContext | None = None,
) -> dict[str, Any]:
    """Start an allowlisted server inside the active session jail."""
    sandbox = _ctx(ctx)
    if sandbox is None:
        return {"success": False, "error": "No active sandbox context"}
    _prune_dead_servers()
    if len(_PROCESSES) >= _MAX_SERVERS:
        return {
            "success": False,
            "error": (
                f"At most {_MAX_SERVERS} harness-managed servers may run per session."
            ),
            "servers": _active_servers(),
            "allowed_ports": get_allowed_ports(),
        }
    if port not in _ALLOWED_PORTS:
        return {
            "success": False,
            "error": f"Port {port} is not in the session port allowlist.",
            "allowed_ports": get_allowed_ports(),
        }
    if not ready_url.startswith("http://127.0.0.1:"):
        return {"success": False, "error": "Servers must expose an http://127.0.0.1 URL."}
    verdict = validate_argv(argv, profile="devserver")
    if not verdict.allowed:
        return {"success": False, "error": f"Policy denied: {verdict.reason}", "policy_denied": True}

    artifact_dir = sandbox.jail_root / "artifacts" / "ui"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    server_id = f"server-{uuid.uuid4().hex[:8]}"
    log_path = artifact_dir / f"{server_id}.log"
    log_file = log_path.open("w", encoding="utf-8")
    env = build_sandbox_env(sandbox)
    try:
        process = subprocess.Popen(
            argv,
            cwd=str(sandbox.workspace_root),
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    except OSError as exc:
        log_file.close()
        return {"success": False, "error": f"Could not start server: {exc}"}
    finally:
        log_file.close()

    managed = ManagedProcess(server_id, process, port, log_path, time.time())
    _PROCESSES[server_id] = managed
    if not _ready(ready_url, max(1, min(timeout, 120))):
        _kill(process)
        _PROCESSES.pop(server_id, None)
        return {
            "success": False,
            "error": f"Server did not become ready at {ready_url}.",
            "logs_tail": _read_log(log_path),
        }
    return {
        "success": True,
        "server_id": server_id,
        "port": port,
        "ready": True,
        "logs_tail": _read_log(log_path),
        "sandbox": "session_jail",
        "allowed_ports": get_allowed_ports(),
    }


def list_servers(*, include_logs: bool = False) -> dict[str, Any]:
    """Return harness-managed servers registered for the active session."""
    return {
        "success": True,
        "servers": _active_servers(include_logs=include_logs),
        "allowed_ports": get_allowed_ports(),
        "max_servers": _MAX_SERVERS,
    }


def server_status(server_id: str) -> dict[str, Any]:
    managed = _PROCESSES.get(server_id)
    if managed is None:
        return {"success": False, "error": f"Unknown server '{server_id}'."}
    return {
        "success": True,
        "server_id": server_id,
        "running": managed.process.poll() is None,
        "port": managed.port,
        "logs_tail": _read_log(managed.log_path),
        "allowed_ports": get_allowed_ports(),
    }


def stop_server(server_id: str) -> dict[str, Any]:
    managed = _PROCESSES.pop(server_id, None)
    if managed is None:
        return {"success": False, "error": f"Unknown server '{server_id}'."}
    _kill(managed.process)
    return {"success": True, "server_id": server_id, "stopped": True}


def server_logs(server_id: str, limit: int = 3000) -> dict[str, Any]:
    managed = _PROCESSES.get(server_id)
    if managed is None:
        return {"success": False, "error": f"Unknown server '{server_id}'."}
    return {"success": True, "server_id": server_id, "logs": _read_log(managed.log_path, limit)}


def stop_all_servers() -> None:
    for server_id in list(_PROCESSES):
        stop_server(server_id)
