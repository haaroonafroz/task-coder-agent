"""
System operation tools — run_pytest, run_linter, install_dependency,
search_grep, run_shellscript.

All subprocess invocations use a configurable timeout and capture both stdout
and stderr for clean result reporting.
"""

from __future__ import annotations

import re
import subprocess
import sys
import os
from pathlib import Path
from typing import Any

from src.tools.paths import (
    WORKSPACE_ROOT,
    normalize_shell_command,
    normalize_workspace_path,
    resolve_workspace_path,
)
_REPO_ROOT = WORKSPACE_ROOT.parent


def _workspace_env() -> dict[str, str]:
    env = os.environ.copy()
    root = str(WORKSPACE_ROOT.resolve())
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = root if not existing else f"{root}{os.pathsep}{existing}"
    return env

def _run(cmd: list[str], cwd: Path, timeout: int, env: dict[str, str] | None = None) -> dict[str, Any]:
    """Execute a subprocess and return structured result."""
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=_workspace_env(),
        )
        return {
            "returncode": result.returncode,
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
            "success": result.returncode == 0,
            "timed_out": False,
        }
    except subprocess.TimeoutExpired:
        return {
            "returncode": -1,
            "stdout": "",
            "stderr": f"Process timed out after {timeout}s",
            "success": False,
            "timed_out": True,
        }
    except FileNotFoundError as exc:
        return {
            "returncode": -1,
            "stdout": "",
            "stderr": str(exc),
            "success": False,
            "timed_out": False,
        }
    except Exception as exc:
        return {
            "returncode": -1,
            "stdout": "",
            "stderr": str(exc),
            "success": False,
            "timed_out": False,
        }


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

    rel = target.relative_to(WORKSPACE_ROOT.resolve())
    extra = extra_args.split() if extra_args else []
    cmd = [sys.executable, "-m", "pytest", str(rel)] + extra
    result = _run(cmd, cwd=WORKSPACE_ROOT, timeout=120, env=_workspace_env())
    result["passed"] = result["returncode"] == 0
    result["cwd"] = str(WORKSPACE_ROOT)
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

    rel = target.relative_to(WORKSPACE_ROOT.resolve())
    extra = extra_args.split() if extra_args else []

    if tool == "black":
        cmd = [sys.executable, "-m", "black", "--check", str(rel)] + extra
    else:
        cmd = [sys.executable, "-m", "flake8", str(rel)] + extra

    result = _run(cmd, cwd=WORKSPACE_ROOT, timeout=60, env=_workspace_env())
    result["clean"] = result["returncode"] == 0
    result["cwd"] = str(WORKSPACE_ROOT)
    result["target_path"] = str(rel)
    return result


# ---------------------------------------------------------------------------
# install_dependency
# ---------------------------------------------------------------------------

def install_dependency(package_name: str) -> dict[str, Any]:
    """
    Install a Python package via pip and optionally record it in workspace/requirements.txt.

    Args:
        package_name: Package name with optional version specifier (e.g. "httpx>=0.27.0").

    Returns:
        {"success": bool, "stdout": str, "stderr": str}
    """
    cmd = [sys.executable, "-m", "pip", "install", package_name, "--quiet"]
    result = _run(cmd, cwd=_REPO_ROOT, timeout=180)

    # Append to workspace/requirements.txt if it exists
    req_file = WORKSPACE_ROOT / "requirements.txt"
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
    Uninstall a Python package via pip and optionally remove it from workspace/requirements.txt.

    Args:
        package_name: Package name with optional version specifier (e.g. "httpx>=0.27.0").

    Returns:
        {"success": bool, "stdout": str, "stderr": str}
    """
    cmd = [sys.executable, "-m", "pip", "uninstall", package_name, "--quiet", "-y"]
    result = _run(cmd, cwd=_REPO_ROOT, timeout=180)

    # Remove from workspace/requirements.txt if it exists
    req_file = WORKSPACE_ROOT / "requirements.txt"
    if result["success"] and req_file.exists():
        existing = req_file.read_text(encoding="utf-8")
        # Extract base package name (e.g. httpx from httpx>=0.27.0)
        base_name = re.split(r"[><=!]", package_name)[0].strip()

        # Remove all lines matching the (potentially version-pinned) package name or base name
        lines = existing.splitlines()
        new_lines = []
        for line in lines:
            clean_line = line.strip()
            # skip empty/comment lines
            if not clean_line or clean_line.startswith("#"):
                new_lines.append(line)
                continue
            # Remove line if it matches the package (via base name match)
            # This matches e.g. "httpx", "httpx==...", "httpx>=...", etc.
            if re.split(r"[><=!]", clean_line)[0].strip() == base_name:
                continue
            new_lines.append(line)
        new_content = "\n".join(new_lines).rstrip() + "\n"
        req_file.write_text(new_content, encoding="utf-8")

    return result
        
# ---------------------------------------------------------------------------
# search_grep
# ---------------------------------------------------------------------------

def search_grep(query: str, target_dir: str = ".") -> dict[str, Any]:
    """
    Regex search across all files in target_dir using Python's re module.

    Falls back to ripgrep (rg) if available for speed.

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

    ws_root = WORKSPACE_ROOT.resolve()

    # Try ripgrep first for speed
    rg_result = _run(
        ["rg", "--line-number", "--no-heading", query, str(target)],
        cwd=WORKSPACE_ROOT,
        timeout=30,
    )
    if rg_result["returncode"] in (0, 1):  # 0=matches, 1=no matches
        matches = []
        for line in rg_result["stdout"].splitlines():
            parts = line.split(":", 2)
            if len(parts) >= 3:
                abs_file = Path(parts[0])
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
            "cwd": str(WORKSPACE_ROOT),
        }

    # Pure-Python fallback
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
        "cwd": str(WORKSPACE_ROOT),
    }

# ---------------------------------------------------------------------------
# run_shellscript
# ---------------------------------------------------------------------------

def run_shellscript(script: str, timeout: int = 30) -> dict[str, Any]:
    """
    Execute an arbitrary shell script inside the workspace directory.

    The script runs with cwd=workspace/ for convenience; use absolute paths
    or `cd` for operations outside workspace/.

    Args:
        script:  Shell command or multi-line bash script.
        timeout: Wall-clock timeout in seconds.

    Returns:
        {"success": bool, "stdout": str, "stderr": str, "returncode": int, "timed_out": bool}
    """
    script = normalize_shell_command(script)
    try:
        result = subprocess.run(
            script,
            shell=True,
            cwd=WORKSPACE_ROOT,
            env=_workspace_env(),
            capture_output=True,
            text=True,
            timeout=timeout,
            executable="/bin/bash",
        )
        return {
            "returncode": result.returncode,
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
            "success": result.returncode == 0,
            "timed_out": False,
            "cwd": str(WORKSPACE_ROOT),
        }
    except subprocess.TimeoutExpired:
        return {
            "returncode": -1,
            "stdout": "",
            "stderr": f"Script timed out after {timeout}s",
            "success": False,
            "timed_out": True,
        }
    except Exception as exc:
        return {
            "returncode": -1,
            "stdout": "",
            "stderr": str(exc),
            "success": False,
            "timed_out": False,
        }
