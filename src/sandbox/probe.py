"""Activation-time verification of the session toolchain."""

from __future__ import annotations

import re

from src.sandbox.context import SandboxContext
from src.sandbox.env import resolve_python
from src.sandbox.executor import get_executor


class SandboxToolchainError(RuntimeError):
    """Raised when the session venv does not satisfy harness requirements."""


def verify_sandbox_toolchain(ctx: SandboxContext) -> dict[str, str]:
    """
    Verify pytest and flake8 are importable from the session venv.

    Called once after bootstrap during sandbox activation. Raises
    :class:`SandboxToolchainError` on failure so missions fail fast
    instead of mid-validation.
    """
    python = resolve_python(ctx)
    executor = get_executor()

    result = executor.run_argv(
        [
            python,
            "-c",
            "import pytest, flake8; print(pytest.__version__)",
        ],
        ctx=ctx,
        timeout=60,
        profile="validation",
    )

    if not result.get("success"):
        stderr = result.get("stderr", "")
        stdout = result.get("stdout", "")
        raise SandboxToolchainError(
            f"Session toolchain probe failed (exit {result.get('returncode')}): "
            f"{stderr or stdout or 'unknown error'}. "
            f"python={python}"
        )

    version = _extract_version(result.get("stdout", ""))
    return {
        "python": python,
        "pytest_version": version or "unknown",
    }


def _extract_version(stdout: str) -> str:
    for line in stdout.splitlines():
        line = line.strip()
        if re.match(r"^\d+\.\d+", line):
            return line
    return stdout.strip().split()[-1] if stdout.strip() else ""
