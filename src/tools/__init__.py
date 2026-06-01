"""
Tool dispatcher — routes tool_name → implementation function.

All tool functions accept a single `args` dict and return a plain dict
with at minimum a "success" key.
"""

from __future__ import annotations

from typing import Any, Callable

from src.tools.file_ops import read_file, write_file, patch_file, list_directory
from src.tools.git_ops import git_commit, git_diff, view_git_log
from src.tools.system_ops import run_pytest, run_linter, install_dependency, search_grep, run_shellscript
from src.tools.paths import normalize_shell_command, normalize_workspace_path

# ---------------------------------------------------------------------------
# Dispatch table  name → callable
# ---------------------------------------------------------------------------
_TOOLS: dict[str, Callable[..., dict[str, Any]]] = {
    "read_file":          lambda args: read_file(**args),
    "write_file":         lambda args: write_file(**args),
    "patch_file":         lambda args: patch_file(**args),
    "list_directory":     lambda args: list_directory(**args),
    "search_grep":        lambda args: search_grep(**args),
    "run_pytest":         lambda args: run_pytest(**args),
    "run_linter":         lambda args: run_linter(**args),
    "git_commit":         lambda args: git_commit(**args),
    "git_diff":           lambda args: git_diff(**args),
    "view_git_log":       lambda args: view_git_log(**args),
    "install_dependency": lambda args: install_dependency(**args),
    "run_shellscript":    lambda args: run_shellscript(**args),
}

AVAILABLE_TOOLS = list(_TOOLS.keys())

_PATH_KEYS = frozenset({"file_path", "target_dir", "target_path", "test_path"})


def _normalize_tool_args(tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
    """Normalize workspace paths and shell commands before tool execution."""
    normalized = dict(args)
    for key in _PATH_KEYS:
        value = normalized.get(key)
        if isinstance(value, str):
            cleaned = normalize_workspace_path(value)
            normalized[key] = cleaned if cleaned else "."

    if tool_name == "run_shellscript":
        script = normalized.get("script")
        if isinstance(script, str):
            normalized["script"] = normalize_shell_command(script)

    return normalized


def dispatch(tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
    """
    Execute a named tool with the given arguments.

    Args:
        tool_name: One of the keys in AVAILABLE_TOOLS.
        args:      Keyword arguments forwarded to the tool function.

    Returns:
        Tool result dict (always contains "success" key).
    """
    fn = _TOOLS.get(tool_name)
    if fn is None:
        return {
            "success": False,
            "error": f"Unknown tool '{tool_name}'. Available: {AVAILABLE_TOOLS}",
        }
    try:
        return fn(_normalize_tool_args(tool_name, args))
    except TypeError as exc:
        return {"success": False, "error": f"Bad arguments for '{tool_name}': {exc}"}
    except Exception as exc:
        return {"success": False, "error": f"Tool '{tool_name}' raised: {exc}"}