"""Minimal scrubbed environment for sandboxed subprocesses."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from src.sandbox.context import SandboxContext

# Vars that must never leak into child processes.
_SECRET_PREFIXES = (
    "OPENAI_", "GEMINI_", "HF_", "API_KEY", "SECRET", "TOKEN", "PASSWORD",
    "PHOENIX_", "QDRANT_", "AWS_", "AZURE_", "GOOGLE_",
)


def _is_secret_key(key: str) -> bool:
    upper = key.upper()
    return any(upper.startswith(p) or p in upper for p in _SECRET_PREFIXES)


def _system_path() -> str:
    return os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin")


def build_sandbox_env(
    ctx: SandboxContext,
    *,
    use_venv: bool = True,
) -> dict[str, str]:
    """
    Build a minimal environment for subprocess execution inside the jail.

    Strips host secrets and points HOME/TMPDIR/VIRTUAL_ENV at session paths.
    """
    python_bin = str(ctx.venv_bin) if use_venv and ctx.venv_python.exists() else ""
    path_parts = [p for p in (python_bin, _system_path()) if p]
    env: dict[str, str] = {
        "PATH": os.pathsep.join(path_parts),
        "HOME": str(ctx.home_dir),
        "TMPDIR": str(ctx.tmp_dir),
        "PYTHONPATH": str(ctx.workspace_root),
        "PIP_CACHE_DIR": str(ctx.pip_cache_dir),
        "NO_COLOR": "1",
        "TERM": "dumb",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
    }
    if use_venv and ctx.venv_python.exists():
        env["VIRTUAL_ENV"] = str(ctx.venv_path)
        env["MISSIONS_PYTHON"] = str(ctx.venv_python.resolve())

    # Preserve only safe, non-secret host vars needed for toolchains.
    safe_passthrough = ("SSL_CERT_FILE", "SSL_CERT_DIR", "HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY")
    for key in safe_passthrough:
        if key in os.environ and not _is_secret_key(key):
            env[key] = os.environ[key]

    return env


def resolve_python(ctx: SandboxContext) -> str:
    """Return python executable — prefer session venv when it exists."""
    if ctx.venv_python.exists():
        return str(ctx.venv_python)
    return sys.executable
