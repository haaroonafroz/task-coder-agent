"""Git tools scoped to the active managed or external workspace."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.sandbox.context import get_sandbox_context
from src.sandbox.executor import get_executor
from src.sandbox.policy import NetworkMode
from src.tools.paths import is_sensitive_workspace_path

_REPO_ROOT = Path(__file__).parent.parent.parent  # legacy fallback


def _filter_sensitive_diff(diff_text: str) -> str:
    """Remove complete unified-diff sections for known secret-bearing paths."""
    sections: list[str] = []
    current: list[str] = []
    current_sensitive = False

    def _flush() -> None:
        if current and not current_sensitive:
            sections.extend(current)

    for line in diff_text.splitlines(keepends=True):
        if line.startswith("diff --git "):
            _flush()
            current = [line]
            header = line[len("diff --git "):].strip()
            left, separator, right = header.partition(" b/")
            left_path = left.removeprefix("a/").strip('"')
            right_path = right.strip('"') if separator else ""
            current_sensitive = (
                is_sensitive_workspace_path(left_path)
                or is_sensitive_workspace_path(right_path)
            )
        else:
            current.append(line)
            if line.startswith(("--- ", "+++ ")):
                marker_path = line[4:].strip().strip('"')
                marker_path = marker_path.removeprefix("a/").removeprefix("b/")
                if marker_path != "/dev/null" and is_sensitive_workspace_path(marker_path):
                    current_sensitive = True
    _flush()
    return "".join(sections)


def _git_root() -> Path:
    """Return the repository root for the active workspace."""
    ctx = get_sandbox_context()
    if ctx is not None:
        # Preserve history for sessions created before workspace separation.
        if ctx.workspace_kind == "managed" and (ctx.jail_root / ".git").exists():
            return ctx.jail_root
        return ctx.workspace_root
    return _REPO_ROOT


def _git(*args: str, cwd: Path | None = None, timeout: int = 30) -> dict[str, Any]:
    """Run a git sub-command via sandbox executor."""
    root = cwd or _git_root()
    cmd = ["git", *args]
    return get_executor().run_argv(
        cmd,
        cwd=root,
        timeout=timeout,
        network=NetworkMode.NONE,
        profile="worker",
        use_venv=False,
    )


def _ensure_git_repo(root: Path) -> dict[str, Any] | None:
    """Initialise Git for a managed workspace if needed."""
    if (root / ".git").exists():
        return None
    init = _git("init", cwd=root)
    if not init["success"]:
        return {"success": False, "error": f"git init failed: {init['stderr']}"}
    _git("config", "user.email", "agent@missions.local", cwd=root)
    _git("config", "user.name", "Missions Agent", cwd=root)
    return None


# ---------------------------------------------------------------------------
# git_commit
# ---------------------------------------------------------------------------

def git_commit(message: str, stage_paths: list[str] | None = None) -> dict[str, Any]:
    """
    Stage and commit a managed workspace.

    External workspaces are deliberately never auto-committed; their changes
    remain in the user's working tree for review.

    Args:
        message:     Conventional-commit-style message.
        stage_paths: Deprecated — ignored for session-scoped git.

    Returns:
        {"success": True, "commit_hash": "<sha>", "message": "<msg>"}
        {"success": False, "error": "<message>"}
    """
    ctx = get_sandbox_context()
    if ctx is not None and ctx.workspace_kind == "external":
        return {
            "success": True,
            "commit_hash": "not-committed",
            "message": "External workspace changes were not auto-committed.",
            "skipped": True,
        }
    if ctx is not None and ctx.workspace_mode == "read_only":
        return {
            "success": False,
            "commit_hash": "",
            "error": "Cannot commit a read-only workspace.",
        }

    root = _git_root()
    err = _ensure_git_repo(root)
    if err:
        return err

    stage_target = "workspace/" if ctx is not None and root == ctx.jail_root else "."
    stage = _git("add", stage_target, cwd=root)
    if not stage["success"] and "pathspec" not in stage.get("stderr", ""):
        return {"success": False, "error": f"git add failed: {stage['stderr']}"}

    status = _git("status", "--porcelain", cwd=root)
    if not status["stdout"]:
        return {
            "success": True,
            "commit_hash": "none",
            "message": "Nothing to commit — workspace already clean.",
        }

    commit = _git("commit", "-m", message, cwd=root)
    if not commit["success"]:
        return {"success": False, "error": f"git commit failed: {commit['stderr']}"}

    rev = _git("rev-parse", "--short", "HEAD", cwd=root)
    commit_hash = rev["stdout"] if rev["success"] else "unknown"
    return {
        "success": True,
        "commit_hash": commit_hash,
        "message": f"Committed [{commit_hash}]: {message}",
    }


# ---------------------------------------------------------------------------
# git_diff
# ---------------------------------------------------------------------------

def git_diff() -> dict[str, Any]:
    """
    Return the unified diff of all uncommitted changes against HEAD.

    Returns:
        {"success": True, "diff": "<unified diff text>", "has_changes": <bool>}
        {"success": False, "error": "<message>"}
    """
    root = _git_root()
    if not (root / ".git").exists():
        return {"success": True, "diff": "(no git repository — workspace is untracked)", "has_changes": False}

    result = _git("diff", "HEAD", cwd=root)
    if not result["success"] and result["stderr"]:
        result = _git("diff", "--cached", cwd=root)

    diff_text = _filter_sensitive_diff(result["stdout"]) or "(no changes)"
    ctx = get_sandbox_context()
    preflight = ctx.git_preflight if ctx is not None else None
    return {
        "success": True,
        "diff": diff_text,
        "has_changes": bool(diff_text.strip() and diff_text != "(no changes)"),
        "preexisting_changes": preflight or {},
        "note": (
            "Diff may include user changes that existed before this run; compare "
            "against preexisting_changes and worker-reported files."
            if preflight
            else ""
        ),
    }


# ---------------------------------------------------------------------------
# view_git_log
# ---------------------------------------------------------------------------

def view_git_log(limit: int = 10) -> dict[str, Any]:
    """
    Show recent Git history for the bound workspace.

    Args:
        limit: Maximum number of commits to show (default 10).

    Returns:
        {"success": True, "log": "<formatted log>", "count": <int>}
        {"success": False, "error": "<message>"}
    """
    root = _git_root()
    if not (root / ".git").exists():
        return {"success": True, "log": "(no git repository)", "count": 0}

    result = _git(
        "log",
        f"--max-count={limit}",
        "--pretty=format:%h  %ad  %s",
        "--date=short",
        cwd=root,
    )
    if not result["success"]:
        return {"success": False, "error": result["stderr"]}

    lines = [line for line in result["stdout"].splitlines() if line.strip()]
    return {
        "success": True,
        "log": result["stdout"] or "(no commits yet)",
        "count": len(lines),
    }
