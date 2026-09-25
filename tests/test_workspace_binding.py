"""Integration coverage for managed and externally attached workspaces."""

from __future__ import annotations

import asyncio
import subprocess
from pathlib import Path

import pytest

from src.api.routers.sessions import delete_session
from src.api.routers.workspace import _is_sensitive
from src.sandbox.context import activate_sandbox, deactivate_sandbox, sandbox_from_session
from src.sandbox.executor import ExecutorBackend, SubprocessExecutor
from src.session import SessionManager
from src.tools.file_ops import list_directory, read_file, write_file
from src.tools.git_ops import git_commit
from src.tools.paths import resolve_workspace_path, reset_workspace_root, set_workspace_root
from src.workspace.environment import detect_environment
from src.workspace.git_state import snapshot_git_state
from src.workspace.service import WorkspaceService, WorkspaceValidationError


def test_external_session_persists_binding_without_writing_project(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    marker = project / "app.py"
    marker.write_text("print('hello')\n", encoding="utf-8")
    manager = SessionManager(tmp_path / "sessions")

    ctx = manager.create_session(
        "existing project",
        workspace_kind="external",
        workspace_path=str(project),
    )
    loaded = manager.load_session(ctx.session_id)

    assert loaded is not None
    assert loaded.workspace.kind == "external"
    assert loaded.workspace_root == project.resolve()
    assert loaded.state_root != loaded.workspace_root
    assert list(project.iterdir()) == [marker]


def test_managed_workspace_is_separate_from_session_state(tmp_path: Path) -> None:
    manager = SessionManager(tmp_path / "sessions")
    ctx = manager.create_session("greenfield")

    assert ctx.workspace.kind == "managed"
    assert ctx.workspace_root.parent == tmp_path / "managed-workspaces"
    assert ctx.state_root.parent == tmp_path / "sessions"


def test_session_deletion_never_deletes_external_workspace(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    source = project / "main.py"
    source.write_text("value = 1\n", encoding="utf-8")
    manager = SessionManager(tmp_path / "sessions")
    ctx = manager.create_session(
        "external",
        workspace_kind="external",
        workspace_path=str(project),
    )

    asyncio.run(delete_session(ctx.session_id, manager))

    assert project.is_dir()
    assert source.exists()
    assert not ctx.state_root.exists()


def test_two_sessions_can_reference_same_canonical_workspace(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    manager = SessionManager(tmp_path / "sessions")

    first = manager.create_session(
        "first", workspace_kind="external", workspace_path=str(project)
    )
    second = manager.create_session(
        "second", workspace_kind="external", workspace_path=str(project / ".")
    )

    assert first.workspace_root == second.workspace_root
    assert first.state_root != second.state_root


def test_protected_system_directories_cannot_be_attached(tmp_path: Path) -> None:
    service = WorkspaceService(tmp_path / "managed")
    with pytest.raises(WorkspaceValidationError):
        service.attach("/etc")


def test_workspace_path_resolution_blocks_symlink_escape(tmp_path: Path) -> None:
    project = tmp_path / "project"
    outside = tmp_path / "outside"
    project.mkdir()
    outside.mkdir()
    (project / "escape").symlink_to(outside, target_is_directory=True)
    set_workspace_root(project)
    try:
        with pytest.raises(ValueError, match="escapes workspace"):
            resolve_workspace_path("escape/secret.txt")
    finally:
        reset_workspace_root()


def test_read_only_external_binding_denies_direct_file_writes(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    manager = SessionManager(tmp_path / "sessions")
    ctx = manager.create_session(
        "review",
        workspace_kind="external",
        workspace_path=str(project),
        workspace_access_mode="read_only",
    )
    activate_sandbox(ctx)
    set_workspace_root(project)
    try:
        result = write_file("new.py", "x = 1\n")
    finally:
        deactivate_sandbox()
        reset_workspace_root()

    assert result["success"] is False
    assert "READ ONLY" in result["error"]
    assert not (project / "new.py").exists()


def test_external_commands_fail_closed_without_bubblewrap(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("SANDBOX_REQUIRE_BWRAP", "true")
    from src.settings.store import reset_settings
    reset_settings()
    project = tmp_path / "project"
    project.mkdir()
    manager = SessionManager(tmp_path / "sessions")
    session = manager.create_session(
        "external",
        workspace_kind="external",
        workspace_path=str(project),
    )
    ctx = sandbox_from_session(session)
    ctx.ensure_dirs()

    result = SubprocessExecutor(ExecutorBackend.NATIVE).run_argv(
        ["python", "--version"],
        ctx=ctx,
    )

    assert result["success"] is False
    assert result["sandbox_denied"] is True


def test_external_commands_run_natively_when_bwrap_not_required(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("SANDBOX_REQUIRE_BWRAP", "false")
    from src.settings.store import reset_settings
    reset_settings()
    project = tmp_path / "project"
    project.mkdir()
    manager = SessionManager(tmp_path / "sessions")
    session = manager.create_session(
        "external",
        workspace_kind="external",
        workspace_path=str(project),
    )
    ctx = sandbox_from_session(session)
    ctx.ensure_dirs()

    result = SubprocessExecutor(ExecutorBackend.NATIVE).run_argv(
        ["python", "--version"],
        ctx=ctx,
    )

    assert result.get("sandbox_denied") is not True
    assert result["success"] is True or result["returncode"] == 0


def test_external_workspace_is_never_auto_committed(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    subprocess.run(["git", "init", "-q", str(project)], check=True)
    manager = SessionManager(tmp_path / "sessions")
    session = manager.create_session(
        "external",
        workspace_kind="external",
        workspace_path=str(project),
    )
    activate_sandbox(session)
    set_workspace_root(project)
    try:
        result = git_commit("feat: should not commit")
    finally:
        deactivate_sandbox()
        reset_workspace_root()

    assert result["success"] is True
    assert result["skipped"] is True
    assert result["commit_hash"] == "not-committed"


def test_project_environment_and_dirty_git_are_detected(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    python = project / ".venv" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.write_text("", encoding="utf-8")
    subprocess.run(["git", "init", "-q", str(project)], check=True)
    subprocess.run(["git", "-C", str(project), "config", "user.email", "test@example.com"], check=True)
    subprocess.run(["git", "-C", str(project), "config", "user.name", "Test"], check=True)
    tracked = project / "app.py"
    tracked.write_text("value = 1\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(project), "add", "app.py"], check=True)
    subprocess.run(["git", "-C", str(project), "commit", "-qm", "initial"], check=True)
    tracked.write_text("value = 2\n", encoding="utf-8")
    (project / "new.txt").write_text("new\n", encoding="utf-8")

    environment = detect_environment(project)
    snapshot = snapshot_git_state(project)

    assert environment is not None
    assert environment.kind == "python-venv"
    assert snapshot.head_sha
    assert snapshot.modified == ["app.py"]
    assert "new.txt" in snapshot.untracked


@pytest.mark.parametrize(
    "name",
    [".env", ".env.local", "private.key", "certificate.pem", "credentials.json"],
)
def test_sensitive_workspace_files_are_hidden(name: str) -> None:
    assert _is_sensitive(Path(name))


def test_sensitive_file_contents_are_not_exposed_to_agent_tools(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    (project / ".env").write_text("API_TOKEN=secret\n", encoding="utf-8")
    (project / ".env.example").write_text("API_TOKEN=\n", encoding="utf-8")
    set_workspace_root(project)
    try:
        denied = read_file(".env")
        example = read_file(".env.example")
        tree = list_directory(".")
    finally:
        reset_workspace_root()

    assert denied["success"] is False
    assert "SENSITIVE FILE" in denied["error"]
    assert example["success"] is True
    assert ".env\n" not in tree["tree"]
