"""
System operation tools — run_pytest, run_linter, install_dependency,
search_grep, run_shellscript.

All subprocess invocations route through the sandbox executor (Phase 9):
command policy, session jail, scrubbed env, process groups, optional bwrap.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any

from src.sandbox.commands import canonicalize_shell_script
from src.sandbox.context import get_sandbox_context
from src.sandbox.env import resolve_python
from src.sandbox.executor import get_executor
from src.sandbox.policy import NetworkMode, ShellProfile
from src.tools.paths import (
    get_workspace_root,
    normalize_shell_command,
    normalize_workspace_path,
    resolve_workspace_path,
)


def _executor():
    return get_executor()


def _run_argv(
    cmd: list[str],
    cwd: Path,
    timeout: int,
    *,
    profile: ShellProfile = "worker",
    network: NetworkMode = NetworkMode.NONE,
    use_venv: bool = True,
) -> dict[str, Any]:
    """Execute argv via sandbox executor."""
    result = _executor().run_argv(
        cmd,
        timeout=timeout,
        network=network,
        profile=profile,
        cwd=cwd,
        use_venv=use_venv,
    )
    return result


# ---------------------------------------------------------------------------
# run_pytest
# ---------------------------------------------------------------------------

def run_pytest(test_path: str, extra_args: str = "-v --tb=short") -> dict[str, Any]:
    """
    Run pytest on a target file or directory.

    Args:
        test_path:  Path relative to workspace/ (e.g. tests/test_email.py).
        extra_args: Additional pytest CLI flags as a space-separated string.

    Returns:
        {"passed": bool, "returncode": int, "stdout": str, "stderr": str, "cwd": str}
    """
    try:
        target = resolve_workspace_path(test_path)
    except ValueError as exc:
        return {"passed": False, "returncode": -1, "stdout": "", "stderr": str(exc)}

    rel = target.relative_to(get_workspace_root().resolve())
    extra = extra_args.split() if extra_args else []
    ctx = get_sandbox_context()
    python = resolve_python(ctx) if ctx else sys.executable
    cmd = [python, "-m", "pytest", str(rel)] + extra
    result = _run_argv(cmd, cwd=get_workspace_root(), timeout=120)
    result["passed"] = result["returncode"] == 0
    result["test_path"] = str(rel)
    return result


# ---------------------------------------------------------------------------
# run_linter
# ---------------------------------------------------------------------------

def run_linter(
    target_path: str,
    tool: str = "flake8",
    extra_args: str = "--max-line-length=120",
) -> dict[str, Any]:
    """
    Run flake8 or black --check on a target file or directory.

    Args:
        target_path: Path relative to workspace/.
        tool:        "flake8" or "black".
        extra_args:  Additional linter flags.

    Returns:
        {"clean": bool, "returncode": int, "stdout": str, "stderr": str, "cwd": str}
    """
    try:
        target = resolve_workspace_path(target_path)
    except ValueError as exc:
        return {"clean": False, "returncode": -1, "stdout": "", "stderr": str(exc)}

    rel = target.relative_to(get_workspace_root().resolve())
    extra = extra_args.split() if extra_args else []
    ctx = get_sandbox_context()
    python = resolve_python(ctx) if ctx else sys.executable

    if tool == "black":
        cmd = [python, "-m", "black", "--check", str(rel)] + extra
    else:
        cmd = [python, "-m", "flake8", str(rel)] + extra

    result = _run_argv(cmd, cwd=get_workspace_root(), timeout=60)
    result["clean"] = result["returncode"] == 0
    result["target_path"] = str(rel)
    return result


# ---------------------------------------------------------------------------
# install_dependency
# ---------------------------------------------------------------------------

def install_dependency(package_name: str) -> dict[str, Any]:
    """
    Install a Python package into the session-local .venv via pip.

    Args:
        package_name: Package name with optional version specifier (e.g. "httpx>=0.27.0").

    Returns:
        {"success": bool, "stdout": str, "stderr": str}
    """
    ctx = get_sandbox_context()
    if ctx is None:
        return {"success": False, "stdout": "", "stderr": "No active sandbox context"}

    try:
        python = str(ctx.ensure_venv())
    except Exception as exc:
        return {"success": False, "stdout": "", "stderr": f"venv creation failed: {exc}"}

    cmd = [python, "-m", "pip", "install", package_name, "--quiet"]
    result = _executor().run_argv(
        cmd,
        ctx=ctx,
        timeout=180,
        network=NetworkMode.PIP_EGRESS,
        profile="pip",
        cwd=ctx.workspace_root,
        use_venv=True,
    )

    req_file = get_workspace_root() / "requirements.txt"
    if result["success"] and req_file.exists():
        existing = req_file.read_text(encoding="utf-8")
        base_name = re.split(r"[><=!]", package_name)[0].strip()
        if base_name not in existing:
            req_file.write_text(existing.rstrip() + f"\n{package_name}\n", encoding="utf-8")

    return result


# ---------------------------------------------------------------------------
# uninstall_dependency
# ---------------------------------------------------------------------------

def uninstall_dependency(package_name: str) -> dict[str, Any]:
    """
    Uninstall a Python package from the session-local .venv.

    Args:
        package_name: Package name with optional version specifier.

    Returns:
        {"success": bool, "stdout": str, "stderr": str}
    """
    ctx = get_sandbox_context()
    if ctx is None:
        return {"success": False, "stdout": "", "stderr": "No active sandbox context"}

    if not ctx.venv_python.exists():
        return {"success": False, "stdout": "", "stderr": "Session venv not initialized"}

    python = str(ctx.venv_python)
    cmd = [python, "-m", "pip", "uninstall", package_name, "-y", "--quiet"]
    result = _executor().run_argv(
        cmd,
        ctx=ctx,
        timeout=180,
        network=NetworkMode.PIP_EGRESS,
        profile="pip",
        cwd=ctx.workspace_root,
    )

    req_file = get_workspace_root() / "requirements.txt"
    if result["success"] and req_file.exists():
        existing = req_file.read_text(encoding="utf-8")
        base_name = re.split(r"[><=!]", package_name)[0].strip()
        lines = existing.splitlines()
        new_lines = []
        for line in lines:
            clean_line = line.strip()
            if not clean_line or clean_line.startswith("#"):
                new_lines.append(line)
                continue
            if re.split(r"[><=!]", clean_line)[0].strip() == base_name:
                continue
            new_lines.append(line)
        req_file.write_text("\n".join(new_lines).rstrip() + "\n", encoding="utf-8")

    return result


# ---------------------------------------------------------------------------
# search_grep
# ---------------------------------------------------------------------------

def search_grep(query: str, target_dir: str = ".") -> dict[str, Any]:
    """
    Regex search across all files in target_dir using Python's re module.

    Falls back to ripgrep (rg) via sandbox executor if available.

    Args:
        query:      Python regex pattern.
        target_dir: Directory relative to workspace/ (default: workspace root).

    Returns:
        {"success": bool, "matches": [{"file", "line_no", "text"}], "match_count": int, "cwd": str}
    """
    try:
        target = resolve_workspace_path(target_dir or ".")
    except ValueError as exc:
        return {"success": False, "matches": [], "match_count": 0, "error": str(exc)}

    if not target.exists():
        return {
            "success": False,
            "matches": [],
            "match_count": 0,
            "error": f"Directory not found: {normalize_workspace_path(target_dir)}",
        }

    ws_root = get_workspace_root().resolve()

    rg_result = _run_argv(
        ["rg", "--line-number", "--no-heading", query, str(target.relative_to(ws_root))],
        cwd=ws_root,
        timeout=30,
    )
    if rg_result["returncode"] in (0, 1):
        matches = []
        for line in rg_result["stdout"].splitlines():
            parts = line.split(":", 2)
            if len(parts) >= 3:
                abs_file = ws_root / parts[0]
                try:
                    rel_file = str(abs_file.resolve().relative_to(ws_root))
                except ValueError:
                    rel_file = parts[0]
                matches.append({"file": rel_file, "line_no": parts[1], "text": parts[2]})
            elif len(parts) == 2:
                matches.append({"file": parts[0], "line_no": parts[1], "text": ""})
        return {
            "success": True,
            "matches": matches,
            "match_count": len(matches),
            "cwd": str(ws_root),
        }

    try:
        pattern = re.compile(query)
    except re.error as exc:
        return {"success": False, "matches": [], "match_count": 0, "error": f"Invalid regex: {exc}"}

    matches = []
    for filepath in target.rglob("*"):
        if not filepath.is_file():
            continue
        try:
            for line_no, line in enumerate(
                filepath.read_text(encoding="utf-8", errors="replace").splitlines(), 1
            ):
                if pattern.search(line):
                    matches.append({
                        "file": str(filepath.resolve().relative_to(ws_root)),
                        "line_no": str(line_no),
                        "text": line,
                    })
        except Exception:
            continue

    return {
        "success": True,
        "matches": matches,
        "match_count": len(matches),
        "cwd": str(ws_root),
    }


# ---------------------------------------------------------------------------
# run_shellscript
# ---------------------------------------------------------------------------

def run_shellscript(
    script: str,
    timeout: int = 30,
    profile: ShellProfile = "worker",
) -> dict[str, Any]:
    """
    Execute a shell script inside the session sandbox.

    The script is validated against the command policy for the given profile
    before execution.  ``validation`` profile is stricter (validator contracts).

    Args:
        script:   Shell command or multi-line bash script.
        timeout:  Wall-clock timeout in seconds.
        profile:  Policy profile — ``worker`` | ``validation`` | ``pip``.

    Returns:
        {"success": bool, "stdout": str, "stderr": str, "returncode": int, "timed_out": bool}
    """
    script = normalize_shell_command(script)
    ctx = get_sandbox_context()
    if ctx is not None:
        script = canonicalize_shell_script(script, ctx=ctx)
    return _executor().run_shell(script, timeout=timeout, profile=profile)
