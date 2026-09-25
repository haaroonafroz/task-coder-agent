"""Per-session sandbox context (jail root, venv, tmp)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from src.session import SessionContext
from src.workspace.environment import detect_environment

_active: Optional["SandboxContext"] = None


@dataclass
class SandboxContext:
    """
    Filesystem jail for one Missions session.

    Harness state and project code are independent roots. External projects
    require process isolation for command execution.
    """

    session_id: str
    jail_root: Path
    workspace_root: Path
    venv_path: Path
    tmp_dir: Path
    home_dir: Path
    pip_cache_dir: Path
    workspace_kind: str = "managed"
    workspace_mode: str = "read_write"
    environment_strategy: str = "harness"
    sandbox_required: bool = False
    uses_project_environment: bool = False
    git_preflight: Optional[dict[str, Any]] = None

    @property
    def state_root(self) -> Path:
        """Harness-owned runtime state root."""
        return self.jail_root

    @property
    def venv_bin(self) -> Path:
        return self.venv_path / "bin"

    @property
    def venv_python(self) -> Path:
        return self.venv_bin / "python"

    def ensure_dirs(self) -> None:
        """Create sandbox directories (idempotent)."""
        for d in (
            self.jail_root,
            self.tmp_dir,
            self.home_dir,
            self.pip_cache_dir,
        ):
            d.mkdir(parents=True, exist_ok=True)
        if self.workspace_kind == "managed":
            self.workspace_root.mkdir(parents=True, exist_ok=True)

    def ensure_project_venv(self, *, with_pip: bool = True) -> Path:
        """Ensure a venv *inside* the attached external project; bind to it.

        Reuse order: already-bound project venv → ``<root>/.venv`` →
        ``<root>/venv`` → create ``<root>/.venv``. The context is updated in
        place so dependency checks and installs in the same run immediately
        see the project interpreter. Managed sessions never reach here.
        """
        import subprocess

        if self.workspace_kind != "external":
            return self.ensure_venv()
        if self.workspace_mode == "read_only":
            raise RuntimeError(
                "Cannot create or use a project venv for a read-only workspace"
            )
        if self.uses_project_environment and self.venv_python.exists():
            return self.venv_python
        for name in (".venv", "venv"):
            candidate = self.workspace_root / name / "bin" / "python"
            if candidate.is_file():
                self.venv_path = candidate.parent.parent
                self.uses_project_environment = True
                return candidate
        venv_path = self.workspace_root / ".venv"
        argv = [str(self._system_python()), "-m", "venv", str(venv_path)]
        if not with_pip:
            argv.append("--without-pip")
        subprocess.run(argv, check=True, capture_output=True, text=True)
        self.venv_path = venv_path
        self.uses_project_environment = True
        return self.venv_python

    def ensure_venv(self) -> Path:
        """Create session-local venv if missing; return python path."""
        import subprocess

        if self.venv_python.exists():
            if not self.uses_project_environment:
                self._harden_venv_symlinks()
            return self.venv_python
        if self.uses_project_environment:
            raise RuntimeError(f"Project environment is incomplete: {self.venv_path}")

        self.venv_path.parent.mkdir(parents=True, exist_ok=True)
        # Use the system base interpreter so the venv does not inherit symlinks
        # from the host/project venv that live outside the session jail.
        subprocess.run(
            [str(self._system_python()), "-m", "venv", str(self.venv_path)],
            check=True,
            capture_output=True,
            text=True,
        )
        self._harden_venv_symlinks()
        return self.venv_python

    def _system_python(self) -> Path:
        """Return the host system interpreter used to anchor in-jail venv symlinks."""
        import sys

        base = getattr(sys, "base_executable", None) or sys.executable
        return Path(base).resolve()

    def _harden_venv_symlinks(self) -> None:
        """Re-point venv interpreter symlinks that escape the session jail."""
        if not self.venv_bin.is_dir():
            return

        jail = self.jail_root.resolve()
        base_python = self._system_python()
        interpreter_names = {
            "python", "python3",
            *(f"python{n}" for n in range(7, 14)),
        }

        for entry in self.venv_bin.iterdir():
            if not entry.is_symlink():
                continue
            try:
                target = entry.resolve()
            except (OSError, RuntimeError):
                continue
            if jail in target.parents or target == jail:
                continue
            if entry.name in interpreter_names:
                entry.unlink()
                entry.symlink_to(base_python)


def sandbox_from_session(session: SessionContext) -> SandboxContext:
    """Build a :class:`SandboxContext` from a :class:`SessionContext`."""
    root = session.state_root
    detected = (
        detect_environment(session.workspace_root)
        if session.workspace.kind == "external"
        and session.workspace.environment_strategy in {"auto", "project"}
        else None
    )
    project_venv = (
        detected.root
        if detected is not None
        and detected.kind == "python-venv"
        and detected.root is not None
        else None
    )
    return SandboxContext(
        session_id=session.session_id,
        jail_root=root,
        workspace_root=session.workspace_root,
        venv_path=project_venv or root / ".venv",
        tmp_dir=root / ".tmp",
        home_dir=root / ".home",
        pip_cache_dir=root / ".cache" / "pip",
        workspace_kind=session.workspace.kind,
        workspace_mode=session.workspace.access_mode,
        environment_strategy=session.workspace.environment_strategy,
        sandbox_required=session.workspace.sandbox_required,
        uses_project_environment=project_venv is not None,
        git_preflight=session.git_preflight,
    )


def activate_sandbox(session: SessionContext) -> SandboxContext:
    """Activate sandbox for the current run (serial — one active at a time)."""
    global _active
    from src.sandbox.bootstrap import bootstrap_sandbox_venv
    from src.sandbox.probe import verify_sandbox_toolchain

    ctx = sandbox_from_session(session)
    ctx.ensure_dirs()
    _active = ctx
    if ctx.workspace_kind == "managed":
        bootstrap_sandbox_venv(ctx)
        verify_sandbox_toolchain(ctx)
    return ctx


def get_sandbox_context() -> Optional[SandboxContext]:
    """Return the active sandbox, or None outside a session run."""
    return _active


def deactivate_sandbox() -> None:
    """Stop harness-owned processes and clear the active sandbox."""
    global _active
    from src.sandbox.process_manager import stop_all_servers

    stop_all_servers()
    _active = None
