"""
File operation tools — read_file, write_file, patch_file, list_directory.

All paths are relative to workspace/ (the project root).
Examples: validator/__init__.py, tests/test_email.py, .

The tools also accept legacy paths prefixed with workspace/ — that prefix
is stripped automatically.

Each function returns a plain dict with a "success" key and either result
fields or an "error" message.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.tools.paths import WORKSPACE_ROOT, resolve_workspace_path


_ALLOW_TEST_EDITS = False
def set_allow_test_edits(allowed: bool) -> None:
    """Toggle whether tools are permitted to create or modify test files."""
    global _ALLOW_TEST_EDITS
    _ALLOW_TEST_EDITS = allowed

def _path_info(abs_path: Path) -> dict[str, str]:
    """Return consistent path metadata for LLM-facing tool results."""
    ws_root = WORKSPACE_ROOT.resolve()
    abs_resolved = abs_path.resolve()
    if abs_resolved == ws_root:
        rel = "."
    else:
        rel = str(abs_resolved.relative_to(ws_root))
    return {
        "path": str(abs_resolved),
        "workspace_relative_path": rel,
        "cwd": str(ws_root),
    }


# ---------------------------------------------------------------------------
# read_file
# ---------------------------------------------------------------------------

def read_file(file_path: str) -> dict[str, Any]:
    """
    Read and return the raw string content of a file.

    Args:
        file_path: Path relative to workspace/ (e.g. validator/email.py).

    Returns:
        {"success": True, "content": "<file text>", ...path info...}
        {"success": False, "error": "<message>"}
    """
    try:
        target = resolve_workspace_path(file_path)
    except ValueError as exc:
        return {"success": False, "error": str(exc)}

    if not target.exists():
        return {
            "success": False,
            "error": f"File not found: {_path_info(target)['workspace_relative_path']}",
            **_path_info(WORKSPACE_ROOT),
        }
    if not target.is_file():
        return {"success": False, "error": f"Path is not a file: {file_path}"}

    try:
        content = target.read_text(encoding="utf-8")
        return {"success": True, "content": content, **_path_info(target)}
    except Exception as exc:
        return {"success": False, "error": str(exc)}


# ---------------------------------------------------------------------------
# write_file
# ---------------------------------------------------------------------------

def write_file(file_path: str, content: str) -> dict[str, Any]:
    """
    Write content to a file, creating parent directories as needed.

    Args:
        file_path: Destination path relative to workspace/.
        content:   Full text content to write.

    Returns:
        {"success": True, "message": "<confirmation>", "bytes_written": <int>, ...}
        {"success": False, "error": "<message>"}
    """
    try:
        target = resolve_workspace_path(file_path)
    except ValueError as exc:
        return {"success": False, "error": str(exc)}

    target_rel = _path_info(target)["workspace_relative_path"]
    
    # Programmatic Test Freeze Guardrail
    if not _ALLOW_TEST_EDITS and ("tests/" in target_rel or "test_" in target_rel):
        return {
            "success": False,
            "error": (
                f"PROTECTED FILE BREACH: Modifying test files ('{target_rel}') is strictly forbidden "
                "during implementation milestones. You must correct your source code"
                " to make existing tests pass, rather than modifying the test cases."
            )
        }

    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        byte_count = len(content.encode("utf-8"))
        return {
            "success": True,
            "message": f"Written {byte_count} bytes to {_path_info(target)['workspace_relative_path']}",
            "bytes_written": byte_count,
            **_path_info(target),
        }
    except Exception as exc:
        return {"success": False, "error": str(exc)}


# ---------------------------------------------------------------------------
# patch_file
# ---------------------------------------------------------------------------

def patch_file(file_path: str, search_string: str, replace_string: str) -> dict[str, Any]:
    """
    Replace the first exact occurrence of search_string with replace_string.

    Args:
        file_path:      Path relative to workspace/.
        search_string:  Exact text to locate (must appear exactly once).
        replace_string: Replacement text.

    Returns:
        {"success": True, "message": "<confirmation>", ...}
        {"success": False, "error": "<message>"}
    """
    try:
        target = resolve_workspace_path(file_path)
    except ValueError as exc:
        return {"success": False, "error": str(exc)}

    target_rel = _path_info(target)["workspace_relative_path"]
    # Programmatic Test Freeze Guardrail (Patch-level)
    if not _ALLOW_TEST_EDITS and ("tests/" in target_rel or "test_" in target_rel):
        return {
            "success": False,
            "error": (
                f"PROTECTED FILE BREACH: Modifying test files ('{target_rel}') is strictly forbidden "
                "during implementation milestones. You must correct your source code "
                " to make existing tests pass, rather than modifying the test cases."
            )
        }

    if not target.exists():
        return {"success": False, "error": f"File not found: {file_path}"}

    try:
        original = target.read_text(encoding="utf-8")
        count = original.count(search_string)
        if count == 0:
            return {
                "success": False,
                "error": (
                    f"search_string not found in {_path_info(target)['workspace_relative_path']}. "
                    "Verify the exact whitespace."
                ),
            }
        if count > 1:
            return {
                "success": False,
                "error": (
                    f"search_string appears {count} times in "
                    f"{_path_info(target)['workspace_relative_path']}. "
                    "Provide more surrounding context to make it unique."
                ),
            }
        patched = original.replace(search_string, replace_string, 1)
        target.write_text(patched, encoding="utf-8")
        return {
            "success": True,
            "message": (
                f"Patched {_path_info(target)['workspace_relative_path']}: replaced 1 occurrence."
            ),
            **_path_info(target),
        }
    except Exception as exc:
        return {"success": False, "error": str(exc)}


# ---------------------------------------------------------------------------
# list_directory
# ---------------------------------------------------------------------------

def list_directory(target_dir: str = ".", max_depth: int = 6) -> dict[str, Any]:
    """
    Recursively list all files and subdirectories inside target_dir.

    Args:
        target_dir: Directory relative to workspace/ (. = workspace root).
        max_depth:  Maximum recursion depth (default 6).

    Returns:
        {"success": True, "tree": "<indented text tree>", "count": <int>, "cwd": ...}
        {"success": False, "error": "<message>"}
    """
    try:
        target = resolve_workspace_path(target_dir or ".")
    except ValueError as exc:
        return {"success": False, "error": str(exc)}

    if not target.exists():
        target.mkdir(parents=True, exist_ok=True)

    if not target.is_dir():
        return {"success": False, "error": f"Path is not a directory: {target_dir}"}

    lines: list[str] = []
    count = 0
    rel_root = _path_info(target)["workspace_relative_path"]

    def _walk(path: Path, depth: int, prefix: str) -> None:
        nonlocal count
        if depth > max_depth:
            lines.append(f"{prefix}... (max depth reached)")
            return
        try:
            entries = sorted(path.iterdir(), key=lambda e: (e.is_file(), e.name))
        except PermissionError:
            lines.append(f"{prefix}[permission denied]")
            return
        for i, entry in enumerate(entries):
            connector = "└── " if i == len(entries) - 1 else "├── "
            lines.append(f"{prefix}{connector}{entry.name}{'/' if entry.is_dir() else ''}")
            count += 1
            if entry.is_dir():
                extension = "    " if i == len(entries) - 1 else "│   "
                _walk(entry, depth + 1, prefix + extension)

    display_root = rel_root if rel_root != "." else "."
    lines.append(f"{display_root}/")
    _walk(target, 0, "")

    return {
        "success": True,
        "tree": "\n".join(lines) if count else f"{display_root}/\n(empty)",
        "count": count,
        "target_dir": rel_root,
        **_path_info(WORKSPACE_ROOT),
    }