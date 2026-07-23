"""Unit tests for sandbox venv bootstrap."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.sandbox.bootstrap import bootstrap_sandbox_venv
from src.sandbox.context import SandboxContext


def _ctx(tmp_path: Path) -> SandboxContext:
    root = tmp_path / "sess"
    ws = root / "workspace"
    for d in (root, ws, root / ".tmp", root / ".home", root / ".cache" / "pip"):
        d.mkdir(parents=True)
    return SandboxContext(
        session_id="test",
        jail_root=root,
        workspace_root=ws,
        venv_path=root / ".venv",
        tmp_dir=root / ".tmp",
        home_dir=root / ".home",
        pip_cache_dir=root / ".cache" / "pip",
    )


def test_bootstrap_skips_when_marker_present(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    ctx.venv_path.mkdir(parents=True)
    marker = ctx.venv_path / ".harness_bootstrap"
    marker.write_text("ok\n", encoding="utf-8")

    with patch("src.sandbox.bootstrap.get_executor") as mock_exec:
        ran = bootstrap_sandbox_venv(ctx)

    assert ran is False
    mock_exec.assert_not_called()


def test_bootstrap_raises_when_requirements_missing(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    missing = tmp_path / "missing.txt"

    with pytest.raises(FileNotFoundError):
        bootstrap_sandbox_venv(ctx, requirements_path=missing)


def test_bootstrap_raises_on_pip_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ctx = _ctx(tmp_path)
    req = tmp_path / "req.txt"
    req.write_text("pytest\n", encoding="utf-8")

    monkeypatch.setattr(ctx, "ensure_venv", lambda: ctx.venv_path / "bin" / "python")

    mock_executor = MagicMock()
    mock_executor.run_argv.return_value = {
        "success": False,
        "returncode": 1,
        "stderr": "pip exploded",
        "stdout": "",
    }

    with patch("src.sandbox.bootstrap.get_executor", return_value=mock_executor):
        with pytest.raises(RuntimeError, match="pip exploded"):
            bootstrap_sandbox_venv(ctx, requirements_path=req)
