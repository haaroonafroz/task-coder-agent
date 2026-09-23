"""Validation and construction boundary for arbitrary workspace paths."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from src.workspace.git_state import detect_git_root
from src.workspace.inspector import inspect_workspace
from src.workspace.models import (
    EnvironmentStrategy,
    ProjectProfile,
    WorkspaceAccessMode,
    WorkspaceBinding,
)

_PROTECTED_ROOTS = tuple(
    Path(path)
    for path in (
        "/", "/bin", "/boot", "/dev", "/etc", "/lib", "/lib64", "/proc",
        "/root", "/run", "/sbin", "/sys", "/usr", "/var",
    )
)


class WorkspaceValidationError(ValueError):
    """Raised when an external workspace is unsafe or unusable."""


class WorkspaceService:
    def __init__(self, managed_root: Path) -> None:
        self.managed_root = managed_root

    def create_managed(
        self,
        session_id: str,
        *,
        access_mode: WorkspaceAccessMode = "read_write",
        environment_strategy: EnvironmentStrategy = "harness",
    ) -> WorkspaceBinding:
        root = (self.managed_root / session_id).resolve()
        root.mkdir(parents=True, exist_ok=False)
        return WorkspaceBinding(
            kind="managed",
            root=root,
            access_mode=access_mode,
            environment_strategy=environment_strategy,
            sandbox_required=False,
        )

    def attach(
        self,
        path: str | Path,
        *,
        access_mode: WorkspaceAccessMode = "read_write",
        environment_strategy: EnvironmentStrategy = "auto",
    ) -> WorkspaceBinding:
        root = self.validate(path, require_write=access_mode == "read_write")
        return WorkspaceBinding(
            kind="external",
            root=root,
            access_mode=access_mode,
            git_root=detect_git_root(root),
            environment_strategy=environment_strategy,
            sandbox_required=True,
        )

    def validate(self, path: str | Path, *, require_write: bool = False) -> Path:
        raw = Path(path).expanduser()
        try:
            root = raw.resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise WorkspaceValidationError(f"Workspace path is unavailable: {raw}") from exc

        if not root.is_dir():
            raise WorkspaceValidationError(f"Workspace path is not a directory: {root}")
        if self._is_protected(root):
            raise WorkspaceValidationError(
                f"Attaching protected system path is forbidden: {root}"
            )
        if not os.access(root, os.R_OK | os.X_OK):
            raise WorkspaceValidationError(f"Workspace is not readable: {root}")
        if require_write and not os.access(root, os.W_OK | os.X_OK):
            raise WorkspaceValidationError(f"Workspace is not writable: {root}")
        return root

    def inspect(self, binding: WorkspaceBinding) -> ProjectProfile:
        return inspect_workspace(binding.root, binding.kind)

    @staticmethod
    def _is_protected(path: Path) -> bool:
        return any(
            path == protected
            or (protected != Path("/") and protected in path.parents)
            for protected in _PROTECTED_ROOTS
        )


def default_managed_root(sessions_root: Path) -> Path:
    return sessions_root.resolve().parent / "managed-workspaces"
