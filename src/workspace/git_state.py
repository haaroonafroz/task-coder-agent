"""Read-only Git discovery and preflight snapshots."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Optional

from src.workspace.models import GitSnapshot


def _git(path: Path, *args: str) -> Optional[str]:
    try:
        result = subprocess.run(
            [
                "git",
                "-c", "core.fsmonitor=false",
                "-c", "core.hooksPath=/dev/null",
                "-C", str(path),
                *args,
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
            env={
                **os.environ,
                "GIT_OPTIONAL_LOCKS": "0",
                "GIT_TERMINAL_PROMPT": "0",
            },
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.rstrip("\r\n") if result.returncode == 0 else None


def detect_git_root(path: Path) -> Optional[Path]:
    raw = _git(path, "rev-parse", "--show-toplevel")
    if not raw:
        return None
    try:
        root = Path(raw).resolve()
        workspace = path.resolve()
    except OSError:
        return None
    # Do not silently bind a parent repository: Git could then expose or mutate
    # siblings outside the explicitly attached workspace.
    if root != workspace:
        return None
    return root


def snapshot_git_state(path: Path) -> GitSnapshot:
    root = detect_git_root(path)
    if root is None:
        return GitSnapshot()

    branch = _git(root, "branch", "--show-current") or None
    head_sha = _git(root, "rev-parse", "HEAD") or None
    porcelain = _git(root, "status", "--porcelain=v1", "--untracked-files=all") or ""
    modified: list[str] = []
    staged: list[str] = []
    untracked: list[str] = []

    for line in porcelain.splitlines():
        if len(line) < 4:
            continue
        status = line[:2]
        file_path = line[3:]
        if " -> " in file_path:
            file_path = file_path.split(" -> ", 1)[1]
        if status == "??":
            untracked.append(file_path)
            continue
        if status[0] not in {" ", "?"}:
            staged.append(file_path)
        if status[1] not in {" ", "?"}:
            modified.append(file_path)

    return GitSnapshot(
        root=root,
        branch=branch,
        head_sha=head_sha,
        modified=sorted(set(modified)),
        staged=sorted(set(staged)),
        untracked=sorted(set(untracked)),
    )
