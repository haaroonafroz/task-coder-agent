"""Unit tests for harness command normalization and contract compilation."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.sandbox.commands import (
    canonicalize_shell_script,
    compile_contract_to_argv,
    execute_contract,
)
from src.sandbox.context import SandboxContext


def _ctx(tmp_path: Path) -> SandboxContext:
    root = tmp_path / "sess"
    ws = root / "workspace"
    venv = root / ".venv"
    for d in (root, ws, venv / "bin", root / ".tmp", root / ".home", root / ".cache" / "pip"):
        d.mkdir(parents=True)
    py = venv / "bin" / "python"
    py.write_bytes(b"")  # placeholder — resolve_python checks exists()
    return SandboxContext(
        session_id="test",
        jail_root=root,
        workspace_root=ws,
        venv_path=venv,
        tmp_dir=root / ".tmp",
        home_dir=root / ".home",
        pip_cache_dir=root / ".cache" / "pip",
    )


def test_canonicalize_rewrites_python_and_python3(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    from src.sandbox.env import resolve_python

    py = resolve_python(ctx)

    script = "python -m pytest tests/test_x.py && python3 -c 'import pytest'"
    result = canonicalize_shell_script(script, ctx=ctx)

    assert result.count(py) == 2
    assert not result.startswith("python ")
    assert " python " not in result
    assert "python3" not in result


def test_canonicalize_preserves_absolute_python_paths(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    script = "/usr/bin/python3 -m pytest tests/test_x.py"
    result = canonicalize_shell_script(script, ctx=ctx)
    assert result == script


def test_compile_py_compile_shell_contract_to_argv(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    contract = {
        "type": "shell",
        "command": "python -m py_compile main.py",
    }
    argv = compile_contract_to_argv(contract, ctx=ctx)

    assert argv is not None
    assert argv[0] == str(ctx.venv_python.absolute())
    assert argv == [str(ctx.venv_python.absolute()), "-m", "py_compile", "main.py"]


def test_compile_flake8_shell_contract_to_argv(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    contract = {
        "type": "shell",
        "command": "python -m flake8 main.py --max-line-length=120",
    }
    argv = compile_contract_to_argv(contract, ctx=ctx)

    assert argv is not None
    assert argv[0] == str(ctx.venv_python.absolute())
    assert argv[1:4] == ["-m", "flake8", "main.py"]
    assert "--max-line-length=120" in argv


def test_lint_type_contract_respects_command_target(tmp_path: Path) -> None:
    """Regression: type="lint" must NOT discard the command's explicit target.

    Previously a contract {"type": "lint", "command": "python -m flake8
    string_utils.py ..."} compiled to `flake8 .` — sweeping tests/ into the
    lint run and failing on scaffolded test files the worker cannot edit,
    which drove multi-replan loops.
    """
    ctx = _ctx(tmp_path)
    contract = {
        "type": "lint",
        "command": "python -m flake8 string_utils.py --max-line-length=120",
    }
    argv = compile_contract_to_argv(contract, ctx=ctx)

    assert argv is not None
    assert argv[1:4] == ["-m", "flake8", "string_utils.py"]
    assert "." not in argv[3:]


def test_lint_type_contract_without_command_falls_back_to_dot(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    contract = {"type": "lint", "target": "."}
    argv = compile_contract_to_argv(contract, ctx=ctx)

    assert argv is not None
    assert argv[1:4] == ["-m", "flake8", "."]


def test_execute_contract_py_compile_prefers_argv(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    contract = {"type": "shell", "command": "python -m py_compile main.py"}

    mock_executor = MagicMock()
    mock_executor.run_argv.return_value = {
        "success": True,
        "returncode": 0,
        "stdout": "",
        "stderr": "",
        "python": str(ctx.venv_python),
    }

    with patch("src.sandbox.commands.get_executor", return_value=mock_executor):
        with patch("src.sandbox.commands.get_sandbox_context", return_value=ctx):
            result = execute_contract(contract)

    mock_executor.run_argv.assert_called_once()
    mock_executor.run_shell.assert_not_called()
    assert result["execution_mode"] == "argv"


def test_compile_pytest_shell_contract_to_argv(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    contract = {
        "type": "shell",
        "command": "python -m pytest tests/test_email.py --collect-only -q",
    }
    argv = compile_contract_to_argv(contract, ctx=ctx)

    assert argv is not None
    assert argv[0] == str(ctx.venv_python.absolute())
    assert argv[1:4] == ["-m", "pytest", "tests/test_email.py"]
    assert "--collect-only" in argv
    assert "-q" in argv


def test_compile_structured_pytest_contract(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    contract = {
        "type": "pytest",
        "target": "tests/test_x.py",
        "args": "-v -k oracle",
    }
    argv = compile_contract_to_argv(contract, ctx=ctx)

    assert argv == [
        str(ctx.venv_python.absolute()),
        "-m",
        "pytest",
        "tests/test_x.py",
        "-v",
        "-k",
        "oracle",
    ]


def test_compile_unknown_contract_returns_none(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    contract = {"type": "shell", "command": "make build && ./run.sh"}
    assert compile_contract_to_argv(contract, ctx=ctx) is None


def test_execute_contract_prefers_argv(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    contract = {"type": "pytest", "target": "tests/test_x.py", "args": "-q"}

    mock_executor = MagicMock()
    mock_executor.run_argv.return_value = {
        "success": True,
        "returncode": 0,
        "stdout": "1 test collected",
        "stderr": "",
        "python": str(ctx.venv_python),
    }

    with patch("src.sandbox.commands.get_executor", return_value=mock_executor):
        with patch("src.sandbox.commands.get_sandbox_context", return_value=ctx):
            result = execute_contract(contract)

    mock_executor.run_argv.assert_called_once()
    mock_executor.run_shell.assert_not_called()
    assert result["execution_mode"] == "argv"


def test_execute_contract_passes_env_overlay_to_argv(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    contract = {"type": "pytest", "target": "tests/test_x.py", "args": "-q"}

    mock_executor = MagicMock()
    mock_executor.run_argv.return_value = {
        "success": True,
        "returncode": 0,
        "stdout": "1 passed",
        "stderr": "",
        "python": str(ctx.venv_python),
    }

    with patch("src.sandbox.commands.get_executor", return_value=mock_executor):
        with patch("src.sandbox.commands.get_sandbox_context", return_value=ctx):
            execute_contract(contract, env_overlay={"PYTHONPATH": "/tmp/stubs"})

    assert mock_executor.run_argv.call_args.kwargs["env_overlay"] == {
        "PYTHONPATH": "/tmp/stubs"
    }


def test_execute_contract_falls_back_to_shell(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    contract = {"type": "shell", "command": "echo hello"}

    mock_executor = MagicMock()
    mock_executor.run_shell.return_value = {
        "success": True,
        "returncode": 0,
        "stdout": "hello",
        "stderr": "",
    }

    with patch("src.sandbox.commands.get_executor", return_value=mock_executor):
        with patch("src.sandbox.commands.get_sandbox_context", return_value=ctx):
            result = execute_contract(contract)

    mock_executor.run_shell.assert_called_once()
    shell_arg = mock_executor.run_shell.call_args[0][0]
    assert str(ctx.venv_python) not in shell_arg  # echo doesn't use python
    assert result["execution_mode"] == "shell"
