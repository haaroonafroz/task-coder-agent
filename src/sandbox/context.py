"""Per-session sandbox context (jail root, venv, tmp)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from src.session import SessionContext

_active: Optional["SandboxContext"] = None


@dataclass
class SandboxContext:
    """
    Filesystem jail for one Missions session.

    ``jail_root`` is ``sessions/<id>/`` — the only writable tree for subprocesses.
    ``workspace_root`` is ``sessions/<id>/workspace/`` (code generation cwd).
    Git, venv, tmp, and home all live inside the jail.
    """

    session_id: str
    jail_root: Path
    workspace_root: Path
    venv_path: Path
    tmp_dir: Path
    home_dir: Path
    pip_cache_dir: Path

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
            self.workspace_root,
            self.tmp_dir,
            self.home_dir,
            self.pip_cache_dir,
            self.workspace_root / "tests",
        ):
            d.mkdir(parents=True, exist_ok=True)

    def ensure_venv(self) -> Path:
        """Create session-local venv if missing; return python path."""
        import subprocess

        if self.venv_python.exists():
            self._harden_venv_symlinks()
            return self.venv_python

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
    root = session.root
    return SandboxContext(
        session_id=session.session_id,
        jail_root=root,
        workspace_root=session.workspace_root,
        venv_path=root / ".venv",
        tmp_dir=root / ".tmp",
        home_dir=root / ".home",
        pip_cache_dir=root / ".cache" / "pip",
    )


def activate_sandbox(session: SessionContext) -> SandboxContext:
    """Activate sandbox for the current run (serial — one active at a time)."""
    global _active
    from src.sandbox.bootstrap import bootstrap_sandbox_venv
    from src.sandbox.probe import verify_sandbox_toolchain

    ctx = sandbox_from_session(session)
    ctx.ensure_dirs()
    _active = ctx
    bootstrap_sandbox_venv(ctx)
    verify_sandbox_toolchain(ctx)
    return ctx


def get_sandbox_context() -> Optional[SandboxContext]:
    """Return the active sandbox, or None outside a session run."""
    return _active


def deactivate_sandbox() -> None:
    """Clear the active sandbox after a run completes."""
    global _active
    _active = None
