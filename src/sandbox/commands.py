"""
Harness-owned command normalization and contract compilation.

Agents may emit bare ``python`` / ``python3`` in shell strings. The harness
rewrites those to the session venv interpreter before any subprocess runs.
"""

from __future__ import annotations

import re
import shlex
from typing import Any, Optional

from src.sandbox.context import SandboxContext, get_sandbox_context
from src.sandbox.env import resolve_python
from src.sandbox.executor import get_executor
from src.sandbox.policy import ShellProfile

# Match bare python/python3 tokens (not path segments like foo/python/bar).
_PYTHON3_RE = re.compile(r"(?<![\w/.-])python3(?![\w])")
_PYTHON_RE = re.compile(r"(?<![\w/.-])python(?![\w])")

# Shell command patterns the harness can compile to argv (no shell).
_PYTEST_MODULE_RE = re.compile(
    r"^\s*(?:python3?)\s+-m\s+pytest\s+(\S+)(?:\s+(.*))?$",
    re.IGNORECASE,
)
_BARE_PYTEST_RE = re.compile(
    r"^\s*pytest\s+(\S+)(?:\s+(.*))?$",
    re.IGNORECASE,
)
_PY_COMPILE_MODULE_RE = re.compile(
    r"^\s*(?:python3?)\s+-m\s+py_compile\s+(\S+)(?:\s+(.*))?$",
    re.IGNORECASE,
)
_FLAKE8_MODULE_RE = re.compile(
    r"^\s*(?:python3?)\s+-m\s+flake8\s+(\S+)(?:\s+(.*))?$",
    re.IGNORECASE,
)


def canonicalize_shell_script(script: str, ctx: Optional[SandboxContext] = None) -> str:
    """
    Rewrite bare ``python`` / ``python3`` tokens to the session venv interpreter.

    Called for every shell invocation (worker + validator) before execution.
    """
    sandbox = ctx or get_sandbox_context()
    if sandbox is None:
        return script

    py = resolve_python(sandbox)
    # Replace python3 before python to avoid leaving a trailing "3".
    rewritten = _PYTHON3_RE.sub(py, script)
    rewritten = _PYTHON_RE.sub(py, rewritten)
    return rewritten


def _split_args(arg_string: str) -> list[str]:
    """Split trailing CLI args safely."""
    arg_string = (arg_string or "").strip()
    if not arg_string:
        return []
    try:
        return shlex.split(arg_string, posix=True)
    except ValueError:
        return arg_string.split()


def compile_contract_to_argv(
    contract: dict[str, Any],
    ctx: Optional[SandboxContext] = None,
) -> Optional[list[str]]:
    """
    Compile a validation contract to an argv list when the harness recognizes it.

    Returns None when the contract must run through a shell string.
    """
    sandbox = ctx or get_sandbox_context()
    if sandbox is None:
        return None

    ctype = str(contract.get("type", "shell")).lower()
    command = str(contract.get("command", "")).strip()
    python = resolve_python(sandbox)

    # The command string is the source of truth when present: its target and
    # flags are more specific than contract-level defaults. Parsing it FIRST
    # prevents e.g. type="lint" + "flake8 string_utils.py" from degenerating
    # into linting "." (which sweeps in tests/ and fails on scaffolded test
    # files the worker is forbidden to touch).
    normalized = command.replace("workspace/", "").strip()
    if normalized:
        match = _PYTEST_MODULE_RE.match(normalized)
        if match:
            target, rest = match.group(1), match.group(2) or ""
            return [python, "-m", "pytest", target, *_split_args(rest)]

        match = _BARE_PYTEST_RE.match(normalized)
        if match:
            target, rest = match.group(1), match.group(2) or ""
            return [python, "-m", "pytest", target, *_split_args(rest)]

        match = _PY_COMPILE_MODULE_RE.match(normalized)
        if match:
            target, rest = match.group(1), match.group(2) or ""
            return [python, "-m", "py_compile", target, *_split_args(rest)]

        match = _FLAKE8_MODULE_RE.match(normalized)
        if match:
            target, rest = match.group(1), match.group(2) or ""
            return [python, "-m", "flake8", target, *_split_args(rest)]

    if ctype == "pytest":
        target = contract.get("target") or contract.get("test_path")
        if target:
            extra = _split_args(str(contract.get("args", "")))
            return [python, "-m", "pytest", str(target), *extra]

    if ctype in ("lint", "flake8"):
        target = contract.get("target", ".")
        return [python, "-m", "flake8", str(target), "--max-line-length=120"]

    return None


def execute_contract(
    contract: dict[str, Any],
    *,
    ctx: Optional[SandboxContext] = None,
    timeout: int = 120,
    profile: ShellProfile = "validation",
    env_overlay: Optional[dict[str, str]] = None,
) -> dict[str, Any]:
    """
    Execute a validation contract using argv when possible, else canonical shell.

    Always routes through the session sandbox executor.
    """
    sandbox = ctx or get_sandbox_context()
    if sandbox is None:
        return {
            "success": False,
            "returncode": -1,
            "stdout": "",
            "stderr": "No active sandbox context",
            "timed_out": False,
        }

    executor = get_executor()
    argv = compile_contract_to_argv(contract, ctx=sandbox)

    if argv is not None:
        result = executor.run_argv(
            argv,
            ctx=sandbox,
            timeout=timeout,
            profile=profile,
            env_overlay=env_overlay,
        )
        result["execution_mode"] = "argv"
        result["argv"] = argv
        return result

    command = str(contract.get("command", "")).strip()
    script = canonicalize_shell_script(command, ctx=sandbox)
    result = executor.run_shell(
        script,
        ctx=sandbox,
        timeout=timeout,
        profile=profile,
        env_overlay=env_overlay,
    )
    result["execution_mode"] = "shell"
    result["script"] = script
    return result
