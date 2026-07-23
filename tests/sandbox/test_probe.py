"""Unit tests for sandbox activation toolchain probe."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.sandbox.context import SandboxContext
from src.sandbox.probe import SandboxToolchainError, verify_sandbox_toolchain


def _ctx(tmp_path: Path) -> SandboxContext:
    root = tmp_path / "sess"
    ws = root / "workspace"
    venv = root / ".venv"
    for d in (root, ws, venv / "bin", root / ".tmp", root / ".home", root / ".cache" / "pip"):
        d.mkdir(parents=True)
    (venv / "bin" / "python").write_bytes(b"")
    return SandboxContext(
        session_id="test",
        jail_root=root,
        workspace_root=ws,
        venv_path=venv,
        tmp_dir=root / ".tmp",
        home_dir=root / ".home",
        pip_cache_dir=root / ".cache" / "pip",
    )


def test_verify_toolchain_returns_versions(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    mock_executor = MagicMock()
    mock_executor.run_argv.return_value = {
        "success": True,
        "returncode": 0,
        "stdout": "8.4.2",
        "stderr": "",
    }

    with patch("src.sandbox.probe.get_executor", return_value=mock_executor):
        info = verify_sandbox_toolchain(ctx)

    assert info["python"] == str(ctx.venv_python)
    assert info["pytest_version"] == "8.4.2"


def test_verify_toolchain_raises_on_failure(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    mock_executor = MagicMock()
    mock_executor.run_argv.return_value = {
        "success": False,
        "returncode": 1,
        "stdout": "",
        "stderr": "No module named pytest",
    }

    with patch("src.sandbox.probe.get_executor", return_value=mock_executor):
        with pytest.raises(SandboxToolchainError, match="No module named pytest"):
            verify_sandbox_toolchain(ctx)
