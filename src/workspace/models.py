"""Workspace and project metadata shared by sessions, APIs, and sandboxes."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal, Optional

WorkspaceKind = Literal["managed", "external"]
WorkspaceAccessMode = Literal["read_write", "read_only"]
EnvironmentStrategy = Literal["auto", "project", "harness"]


@dataclass
class WorkspaceBinding:
    """A session's reference to code, which may be harness- or user-owned."""

    kind: WorkspaceKind
    root: Path
    access_mode: WorkspaceAccessMode = "read_write"
    git_root: Optional[Path] = None
    environment_strategy: EnvironmentStrategy = "auto"
    sandbox_required: bool = False

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["root"] = str(self.root)
        data["git_root"] = str(self.git_root) if self.git_root else None
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "WorkspaceBinding":
        root = Path(data["root"])
        return cls(
            kind=data.get("kind", "managed"),
            root=root,
            access_mode=data.get("access_mode", "read_write"),
            git_root=Path(data["git_root"]) if data.get("git_root") else None,
            environment_strategy=data.get("environment_strategy", "auto"),
            sandbox_required=bool(
                data.get("sandbox_required", data.get("kind") == "external")
            ),
        )


@dataclass
class ProjectEnvironment:
    """Detected project environment without activating or modifying it."""

    kind: str
    root: Optional[Path] = None
    python: Optional[Path] = None
    package_manager: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "root": str(self.root) if self.root else None,
            "python": str(self.python) if self.python else None,
            "package_manager": self.package_manager,
        }


@dataclass
class GitSnapshot:
    """Git state captured without changing the repository."""

    root: Optional[Path] = None
    branch: Optional[str] = None
    head_sha: Optional[str] = None
    modified: list[str] = field(default_factory=list)
    staged: list[str] = field(default_factory=list)
    untracked: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "root": str(self.root) if self.root else None,
            "branch": self.branch,
            "head_sha": self.head_sha,
            "modified": self.modified,
            "staged": self.staged,
            "untracked": self.untracked,
        }


@dataclass
class ProjectProfile:
    """Bounded deterministic inspection result for an attached project."""

    root: Path
    workspace_kind: WorkspaceKind
    git: GitSnapshot = field(default_factory=GitSnapshot)
    languages: list[str] = field(default_factory=list)
    manifests: list[str] = field(default_factory=list)
    environment: Optional[ProjectEnvironment] = None
    detected_commands: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "root": str(self.root),
            "workspace_kind": self.workspace_kind,
            "git": self.git.to_dict(),
            "languages": self.languages,
            "manifests": self.manifests,
            "environment": self.environment.to_dict() if self.environment else None,
            "detected_commands": self.detected_commands,
        }
