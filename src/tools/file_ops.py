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
from typing import Any, Optional

from src.tools.paths import (
    get_workspace_root,
    normalize_workspace_path,
    resolve_workspace_path,
)


_ALLOW_TEST_EDITS = False
_NEW_TEST_PATHS: set[str] = set()
_CREATED_THIS_MILESTONE: set[str] = set()


def set_allow_test_edits(allowed: bool) -> None:
    """Toggle whether tools are permitted to create or modify test files."""
    global _ALLOW_TEST_EDITS
    _ALLOW_TEST_EDITS = allowed


# ---------------------------------------------------------------------------
# Per-milestone write policy
#
# The runtime scopes each milestone to its declared target_files. Writes
# outside that set are rejected AT THE TOOL LAYER (one cheap turn) instead of
# post-hoc by the validator (one full worker+validation cycle). Read tracking
# supports diff-first enforcement: full rewrites of large existing files are
# only accepted after the file was read this milestone, or with rewrite=True.
# ---------------------------------------------------------------------------
_ALLOWED_WRITE_PATHS: Optional[set[str]] = None
_READ_THIS_MILESTONE: set[str] = set()

REWRITE_MIN_LINES = 60  # files larger than this require read-before-rewrite


def _is_test_path(path: str) -> bool:
    return path.startswith("tests/") or path.startswith("test_") or "/test_" in path


def begin_milestone_write_policy(
    allowed_paths: Optional[list[str]],
    *,
    allow_new_test_files: bool = False,
) -> None:
    """Set the milestone write jail and reset read-before-rewrite tracking."""
    global _ALLOWED_WRITE_PATHS, _READ_THIS_MILESTONE
    global _NEW_TEST_PATHS, _CREATED_THIS_MILESTONE
    if allowed_paths:
        _ALLOWED_WRITE_PATHS = {
            normalize_workspace_path(p) for p in allowed_paths if str(p).strip()
        }
    else:
        _ALLOWED_WRITE_PATHS = None
    _READ_THIS_MILESTONE = set()
    _CREATED_THIS_MILESTONE = set()
    if allow_new_test_files and _ALLOWED_WRITE_PATHS:
        workspace = get_workspace_root().resolve()
        _NEW_TEST_PATHS = {
            path for path in _ALLOWED_WRITE_PATHS
            if _is_test_path(path) and not (workspace / path).exists()
        }
    else:
        _NEW_TEST_PATHS = set()


def clear_milestone_write_policy() -> None:
    """Remove the write jail (e.g. after the mission loop finishes)."""
    global _ALLOWED_WRITE_PATHS, _READ_THIS_MILESTONE
    global _NEW_TEST_PATHS, _CREATED_THIS_MILESTONE
    _ALLOWED_WRITE_PATHS = None
    _READ_THIS_MILESTONE = set()
    _NEW_TEST_PATHS = set()
    _CREATED_THIS_MILESTONE = set()


def get_created_files() -> list[str]:
    """Return files created during the active milestone."""
    return sorted(_CREATED_THIS_MILESTONE)


def _test_write_error(target_rel: str) -> Optional[dict[str, Any]]:
    """Protect existing tests while allowing explicitly planned new tests."""
    if _ALLOW_TEST_EDITS or not _is_test_path(target_rel):
        return None
    if target_rel in _NEW_TEST_PATHS:
        return None
    return {
        "success": False,
        "error": (
            f"PROTECTED FILE BREACH: Modifying existing acceptance test "
            f"('{target_rel}') is forbidden in this milestone. Create or modify "
            "an explicitly planned agent-owned test, or fix the implementation."
        ),
    }


def _write_jail_error(target_rel: str) -> Optional[dict[str, Any]]:
    """Return an error dict when the path is outside the milestone jail."""
    if _ALLOWED_WRITE_PATHS is None:
        return None
    if target_rel in _ALLOWED_WRITE_PATHS:
        return None
    allowed = sorted(_ALLOWED_WRITE_PATHS)
    return {
        "success": False,
        "error": (
            f"MILESTONE BOUNDARY BREACH: '{target_rel}' is not in this milestone's "
            f"target_files. You may ONLY create/edit: {allowed}. "
            "Do not add scaffolding or helper files. If the plan itself is wrong, "
            "signal blocked and explain why."
        ),
    }


def _path_info(abs_path: Path) -> dict[str, str]:
    """Return consistent path metadata for LLM-facing tool results."""
    ws_root = get_workspace_root().resolve()
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

def read_file(
    file_path: str,
    offset: int = 1,
    limit: Optional[int] = None,
) -> dict[str, Any]:
    """
    Read and return the raw string content of a file.

    Args:
        file_path: Path relative to workspace/ (e.g. validator/email.py).
        offset: One-based line number to start reading from.
        limit: Maximum number of lines to return.

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
            **_path_info(get_workspace_root()),
        }
    if not target.is_file():
        return {"success": False, "error": f"Path is not a file: {file_path}"}

    try:
        content = target.read_text(encoding="utf-8")
        _READ_THIS_MILESTONE.add(_path_info(target)["workspace_relative_path"])
        lines = content.splitlines(keepends=True)
        start = max(1, int(offset))
        if limit is None:
            selected = lines[start - 1:]
        else:
            selected = lines[start - 1:start - 1 + max(0, int(limit))]
        return {
            "success": True,
            "content": "".join(selected),
            "start_line": start,
            "end_line": start + len(selected) - 1 if selected else start - 1,
            "total_lines": len(lines),
            **_path_info(target),
        }
    except Exception as exc:
        return {"success": False, "error": str(exc)}


# ---------------------------------------------------------------------------
# write_file
# ---------------------------------------------------------------------------

def write_file(file_path: str, content: str, rewrite: bool = False) -> dict[str, Any]:
    """
    Write content to a file, creating parent directories as needed.

    Args:
        file_path: Destination path relative to workspace/.
        content:   Full text content to write.
        rewrite:   Escape hatch allowing a full rewrite of a large existing
                   file without a prior read_file this milestone. Prefer
                   read_file + patch_file for targeted edits — full rewrites
                   are the slowest possible edit on local models.

    Returns:
        {"success": True, "message": "<confirmation>", "bytes_written": <int>, ...}
        {"success": False, "error": "<message>"}
    """
    try:
        target = resolve_workspace_path(file_path)
    except ValueError as exc:
        return {"success": False, "error": str(exc)}

    target_rel = _path_info(target)["workspace_relative_path"]

    jail_error = _write_jail_error(target_rel)
    if jail_error:
        return jail_error

    test_error = _test_write_error(target_rel)
    if test_error:
        return test_error

    # Diff-first enforcement: rewriting a large existing file without reading
    # it first almost always means the model is guessing at current content.
    if target.exists() and not rewrite and target_rel not in _READ_THIS_MILESTONE:
        try:
            existing_lines = len(target.read_text(encoding="utf-8").splitlines())
        except OSError:
            existing_lines = 0
        if existing_lines > REWRITE_MIN_LINES:
            return {
                "success": False,
                "error": (
                    f"REWRITE REJECTED: '{target_rel}' already exists with "
                    f"{existing_lines} lines. Full-file rewrites are expensive and "
                    "error-prone. Either:\n"
                    "1. read_file it first, then patch_file the specific section, or\n"
                    "2. read_file it first, then write_file the full content, or\n"
                    '3. pass "rewrite": true if a complete rewrite is truly intended.'
                ),
            }

    was_new = not target.exists()
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        if was_new:
            _CREATED_THIS_MILESTONE.add(target_rel)
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

    jail_error = _write_jail_error(target_rel)
    if jail_error:
        return jail_error

    test_error = _test_write_error(target_rel)
    if test_error:
        return test_error

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
        **_path_info(get_workspace_root()),
    }
