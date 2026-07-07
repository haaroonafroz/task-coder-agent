"""
Git operation tools — git_commit, git_diff, view_git_log.

All git commands run in the repository root (task-coder-agent-v2/).
git_commit stages all changes under workspace/ to keep the commit scope clean.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).parent.parent.parent  # task-coder-agent-v2/


def _git(*args: str, cwd: Path = _REPO_ROOT, timeout: int = 30) -> dict[str, Any]:
    """Run a git sub-command and return structured output."""
    cmd = ["git", *args]
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return {
            "returncode": result.returncode,
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
            "success": result.returncode == 0,
        }
    except subprocess.TimeoutExpired:
        return {"returncode": -1, "stdout": "", "stderr": "git command timed out", "success": False}
    except FileNotFoundError:
        return {"returncode": -1, "stdout": "", "stderr": "git not found in PATH", "success": False}
    except Exception as exc:
        return {"returncode": -1, "stdout": "", "stderr": str(exc), "success": False}


# ---------------------------------------------------------------------------
# git_commit
# ---------------------------------------------------------------------------

def git_commit(message: str, stage_paths: list[str] | None = None) -> dict[str, Any]:
    """
    Stage modified files and create a git commit.

    Uses ``git add workspace/ active_mission/`` by default (legacy behaviour).
    When ``stage_paths`` is provided (session-scoped runs), only those
    repo-relative paths are staged — e.g. ``["sessions/<id>/"]``.

    Args:
        message:     Conventional-commit-style message.
        stage_paths: Optional list of repo-relative paths to stage. When
                     omitted, the legacy ``workspace/ active_mission/`` set
                     is staged.

    Returns:
        {"success": True, "commit_hash": "<sha>", "message": "<msg>"}
        {"success": False, "error": "<message>"}
    """
    # 1. Initialise repo if needed
    if not (_REPO_ROOT / ".git").exists():
        init = _git("init")
        if not init["success"]:
            return {"success": False, "error": f"git init failed: {init['stderr']}"}
        _git("config", "user.email", "agent@missions.local")
        _git("config", "user.name", "Missions Agent")

    # 2. Stage changes
    paths_to_stage = stage_paths if stage_paths is not None else ["workspace/", "active_mission/"]
    stage = _git("add", *paths_to_stage)
    if not stage["success"] and "pathspec" not in stage["stderr"]:
        return {"success": False, "error": f"git add failed: {stage['stderr']}"}

    # 3. Check if there's anything to commit
    status = _git("status", "--porcelain")
    if not status["stdout"]:
        return {"success": True, "commit_hash": "none", "message": "Nothing to commit — workspace already clean."}

    # 4. Commit
    commit = _git("commit", "-m", message)
    if not commit["success"]:
        return {"success": False, "error": f"git commit failed: {commit['stderr']}"}

    # 5. Extract commit hash
    rev = _git("rev-parse", "--short", "HEAD")
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
    # If no git repo yet, diff makes no sense
    if not (_REPO_ROOT / ".git").exists():
        return {"success": True, "diff": "(no git repository — workspace is untracked)", "has_changes": False}

    result = _git("diff", "HEAD")
    if not result["success"] and result["stderr"]:
        # HEAD may not exist on a brand-new repo; diff against empty tree
        result = _git("diff", "--cached")

    diff_text = result["stdout"] or "(no changes)"
    return {
        "success": True,
        "diff": diff_text,
        "has_changes": bool(result["stdout"].strip()),
    }


# ---------------------------------------------------------------------------
# view_git_log
# ---------------------------------------------------------------------------

def view_git_log(limit: int = 10) -> dict[str, Any]:
    """
    Show recent git commit history.

    Args:
        limit: Maximum number of commits to show (default 10).

    Returns:
        {"success": True, "log": "<formatted log>", "count": <int>}
        {"success": False, "error": "<message>"}
    """
    if not (_REPO_ROOT / ".git").exists():
        return {"success": True, "log": "(no git repository)", "count": 0}

    result = _git(
        "log",
        f"--max-count={limit}",
        "--pretty=format:%h  %ad  %s",
        "--date=short",
    )
    if not result["success"]:
        return {"success": False, "error": result["stderr"]}

    lines = [l for l in result["stdout"].splitlines() if l.strip()]
    return {
        "success": True,
        "log": result["stdout"] or "(no commits yet)",
        "count": len(lines),
    }
