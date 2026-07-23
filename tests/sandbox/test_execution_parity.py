"""Integration tests for consistent session-venv execution across tool paths."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.sandbox.commands import canonicalize_shell_script, compile_contract_to_argv
from src.sandbox.context import SandboxContext
from src.sandbox.env import resolve_python
from src.tools.system_ops import run_pytest


def _ctx(tmp_path: Path) -> SandboxContext:
    root = tmp_path / "sess"
    ws = root / "workspace"
    venv = root / ".venv"
    for d in (root, ws, venv / "bin", root / ".tmp", root / ".home", root / ".cache" / "pip"):
        d.mkdir(parents=True)
    (venv / "bin" / "python").write_bytes(b"")
    (ws / "tests").mkdir(parents=True)
    (ws / "tests" / "test_x.py").write_text("def test_ok(): assert True\n", encoding="utf-8")
    return SandboxContext(
        session_id="test",
        jail_root=root,
        workspace_root=ws,
        venv_path=venv,
        tmp_dir=root / ".tmp",
        home_dir=root / ".home",
        pip_cache_dir=root / ".cache" / "pip",
    )


def test_run_pytest_and_validator_compile_use_same_python(tmp_path: Path) -> None:
    """Worker run_pytest argv[0] must match compiled validation contract python."""
    ctx = _ctx(tmp_path)
    contract = {
        "type": "shell",
        "command": "python -m pytest tests/test_x.py -v",
    }

    worker_python = resolve_python(ctx)
    compiled = compile_contract_to_argv(contract, ctx=ctx)

    assert compiled is not None
    assert compiled[0] == worker_python

    canonical = canonicalize_shell_script(contract["command"], ctx=ctx)
    assert worker_python in canonical


def test_run_pytest_invokes_resolve_python(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)

    mock_executor = MagicMock()
    mock_executor.run_argv.return_value = {
        "success": True,
        "returncode": 0,
        "stdout": "",
        "stderr": "",
        "python": str(ctx.venv_python),
        "venv": str(ctx.venv_path),
    }

    with patch("src.tools.system_ops.get_sandbox_context", return_value=ctx):
        with patch("src.tools.system_ops._executor", return_value=mock_executor):
            run_pytest("tests/test_x.py")

    argv = mock_executor.run_argv.call_args[0][0]
    assert argv[0] == str(ctx.venv_python)
    assert argv[1:4] == ["-m", "pytest", "tests/test_x.py"]


def test_run_shellscript_canonicalizes_python(tmp_path: Path) -> None:
    from src.tools.system_ops import run_shellscript

    ctx = _ctx(tmp_path)
    mock_executor = MagicMock()
    mock_executor.run_shell.return_value = {
        "success": True,
        "returncode": 0,
        "stdout": "",
        "stderr": "",
    }

    with patch("src.tools.system_ops.get_sandbox_context", return_value=ctx):
        with patch("src.tools.system_ops._executor", return_value=mock_executor):
            run_shellscript("python -m pytest tests/test_x.py -q", profile="validation")

    script = mock_executor.run_shell.call_args[0][0]
    assert str(ctx.venv_python) in script
    assert not script.startswith("python ")


def test_canonicalized_path_passes_validation_policy(tmp_path: Path) -> None:
    """Regression: canonicalized venv python must not be policy-denied."""
    from src.sandbox.policy import validate_shell_script

    ctx = _ctx(tmp_path)
    script = f"{ctx.venv_python} -m pytest tests/test_math_eval.py --collect-only -q"
    verdict = validate_shell_script(script, profile="validation")
    assert verdict.allowed, verdict.reason
