"""Unit tests for sandbox context and session venv hardening."""

from __future__ import annotations

from pathlib import Path

from src.sandbox.context import SandboxContext


def _ctx(tmp_path: Path) -> SandboxContext:
    root = tmp_path / "sess"
    ws = root / "workspace"
    venv = root / ".venv"
    for d in (root, ws, venv / "bin", root / ".tmp", root / ".home", root / ".cache" / "pip"):
        d.mkdir(parents=True)
    return SandboxContext(
        session_id="test",
        jail_root=root,
        workspace_root=ws,
        venv_path=venv,
        tmp_dir=root / ".tmp",
        home_dir=root / ".home",
        pip_cache_dir=root / ".cache" / "pip",
    )


def test_harden_venv_symlinks_repoints_outside_jail(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    outside_python = outside / "python3"
    outside_python.write_bytes(b"")
    if ctx.venv_python.exists():
        ctx.venv_python.unlink()
    ctx.venv_python.symlink_to(outside_python)

    ctx._harden_venv_symlinks()

    assert ctx.venv_python.is_symlink()
    assert ctx.venv_python.resolve() == ctx._system_python()


def test_harden_venv_symlinks_keeps_in_jail_links(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    inside = ctx.venv_bin / "python-real"
    inside.write_bytes(b"")
    if ctx.venv_python.exists():
        ctx.venv_python.unlink()
    ctx.venv_python.symlink_to(inside)

    ctx._harden_venv_symlinks()

    assert ctx.venv_python.resolve() == inside.resolve()
