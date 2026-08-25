"""One-time session venv bootstrap with harness dev tools."""

from __future__ import annotations

import shutil
from pathlib import Path

from src.sandbox.context import SandboxContext
from src.sandbox.executor import get_executor
from src.sandbox.policy import NetworkMode

_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_REQUIREMENTS = _ROOT / "config" / "requirements-sandbox.txt"
_MARKER_NAME = ".harness_bootstrap"


def _bootstrap_marker(ctx: SandboxContext) -> Path:
    return ctx.venv_path / _MARKER_NAME


def _requirements_dest(ctx: SandboxContext) -> Path:
    """Copy requirements into the jail so bwrap can read them."""
    return ctx.jail_root / ".bootstrap-requirements.txt"


def bootstrap_sandbox_venv(
    ctx: SandboxContext,
    *,
    requirements_path: Path | None = None,
    force: bool = False,
) -> bool:
    """
    Install harness dev tools into the session venv (idempotent).

    Returns True if pip install ran, False if already bootstrapped.
    Raises RuntimeError on pip failure.
    """
    req_src = (requirements_path or _DEFAULT_REQUIREMENTS).resolve()
    if not req_src.exists():
        raise FileNotFoundError(f"Sandbox requirements file not found: {req_src}")

    marker = _bootstrap_marker(ctx)
    if marker.exists() and not force:
        return False

    python = str(ctx.ensure_venv())

    dest = _requirements_dest(ctx)
    shutil.copy2(req_src, dest)

    cmd = [python, "-m", "pip", "install", "-r", str(dest), "--quiet"]
    result = get_executor().run_argv(
        cmd,
        ctx=ctx,
        timeout=300,
        network=NetworkMode.PIP_EGRESS,
        profile="pip",
        cwd=ctx.workspace_root,
        use_venv=True,
    )

    if not result.get("success"):
        stderr = result.get("stderr", "")
        stdout = result.get("stdout", "")
        raise RuntimeError(
            f"Sandbox venv bootstrap failed (exit {result.get('returncode')}): "
            f"{stderr or stdout or 'unknown error'}"
        )

    marker.write_text(f"requirements={req_src.name}\n", encoding="utf-8")
    return True
