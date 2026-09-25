"""Structural validation-profile classification + project-venv installs.

Regression: a training-loop hotfix mentioning `build_trainer` compiled to a
ui_smoke contract because the substring "ui" matched. UI-ness now comes from
file suffixes, Streamlit imports, and frontend manifests — never prose.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.agents.validation_compiler import (
    classify_validation_target,
    compile_validation_contract,
)
from src.sandbox.context import SandboxContext


def _milestone(**overrides):
    base = {
        "id": "M1",
        "title": "Focused hotfix",
        "description": "Fix the training loop.",
        "target_files": ["src/training/trainer.py"],
        "acceptance_criteria": [
            "Verify that running `build_trainer` with a config that omits "
            "`run_name` no longer raises NameError."
        ],
        "validation_profile": "auto",
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# classifier
# ---------------------------------------------------------------------------

def test_build_trainer_prose_is_not_ui() -> None:
    """Exact regression: 'ui' inside 'build_trainer' must not trigger UI."""
    profile, reason = classify_validation_target(_milestone())
    assert profile == "python"
    contract, error = compile_validation_contract(_milestone())
    assert error is None
    assert contract is not None
    assert contract["type"] == "pytest"


@pytest.mark.parametrize("target", ["app.tsx", "index.html", "App.vue", "App.svelte"])
def test_ui_suffixes_classify_ui(target: str) -> None:
    profile, _ = classify_validation_target(_milestone(target_files=[target]))
    assert profile == "ui"


def test_lone_css_change_is_not_ui() -> None:
    profile, _ = classify_validation_target(_milestone(target_files=["style.css"]))
    assert profile == "python"


def test_streamlit_import_classifies_ui(tmp_path: Path) -> None:
    app = tmp_path / "app.py"
    app.write_text("import streamlit as st\nst.write('hi')\n", encoding="utf-8")
    profile, reason = classify_validation_target(
        _milestone(target_files=["app.py"]), workspace=tmp_path
    )
    assert profile == "ui"
    assert "Streamlit" in reason


def test_plain_python_imports_are_not_ui(tmp_path: Path) -> None:
    mod = tmp_path / "trainer.py"
    mod.write_text("import torch\nfrom ui_helpers import build_guidance\n", encoding="utf-8")
    profile, _ = classify_validation_target(
        _milestone(target_files=["trainer.py"]), workspace=tmp_path
    )
    assert profile == "python"


def test_package_json_plus_js_targets_is_ui(tmp_path: Path) -> None:
    (tmp_path / "package.json").write_text('{"dependencies": {"react": "19"}}', encoding="utf-8")
    profile, _ = classify_validation_target(
        _milestone(target_files=["src/app.js"]), workspace=tmp_path
    )
    assert profile == "ui"


def test_js_target_without_manifest_is_python(tmp_path: Path) -> None:
    profile, _ = classify_validation_target(
        _milestone(target_files=["scripts/run.js"]), workspace=tmp_path
    )
    assert profile == "python"


# ---------------------------------------------------------------------------
# project venv handling
# ---------------------------------------------------------------------------

def _ctx(tmp_path: Path, kind: str = "external", mode: str = "read_write") -> SandboxContext:
    root = tmp_path / "proj"
    root.mkdir(exist_ok=True)
    state = tmp_path / "state"
    return SandboxContext(
        session_id="s1",
        jail_root=state,
        workspace_root=root,
        venv_path=state / ".venv",
        tmp_dir=state / ".tmp",
        home_dir=state / ".home",
        pip_cache_dir=state / ".cache" / "pip",
        workspace_kind=kind,
        workspace_mode=mode,
    )


def test_ensure_project_venv_reuses_existing(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    venv_python = ctx.workspace_root / ".venv" / "bin" / "python"
    venv_python.parent.mkdir(parents=True)
    venv_python.write_text("#!/bin/sh\n", encoding="utf-8")
    assert ctx.ensure_project_venv() == venv_python
    assert ctx.uses_project_environment is True


def test_ensure_project_venv_prefers_dot_venv(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    for name in ("venv", ".venv"):
        p = ctx.workspace_root / name / "bin" / "python"
        p.parent.mkdir(parents=True)
        p.write_text("#!/bin/sh\n", encoding="utf-8")
    assert ctx.ensure_project_venv() == ctx.workspace_root / ".venv" / "bin" / "python"


def test_ensure_project_venv_creates_when_missing(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    python = ctx.ensure_project_venv(with_pip=False)
    assert python.is_file()
    assert ctx.venv_path == ctx.workspace_root / ".venv"
    assert ctx.uses_project_environment is True


def test_ensure_project_venv_refuses_read_only(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path, mode="read_only")
    with pytest.raises(RuntimeError):
        ctx.ensure_project_venv()


def test_ensure_project_venv_delegates_for_managed(tmp_path: Path, monkeypatch) -> None:
    ctx = _ctx(tmp_path, kind="managed")
    monkeypatch.setattr(
        SandboxContext, "ensure_venv", lambda self: Path("/session/python")
    )
    assert ctx.ensure_project_venv() == Path("/session/python")


# ---------------------------------------------------------------------------
# install_dependency routing
# ---------------------------------------------------------------------------

class _FakeExecutor:
    def __init__(self) -> None:
        self.commands: list[list[str]] = []
        self.show_ok = False

    def run_argv(self, argv, **kwargs):
        self.commands.append(list(argv))
        if "show" in argv:
            return {"success": self.show_ok, "stdout": "", "stderr": ""}
        return {"success": True, "stdout": "installed", "stderr": ""}


def _install(monkeypatch, ctx, **kwargs):
    import src.tools.system_ops as ops

    fake = _FakeExecutor()
    fake.show_ok = kwargs.get("show_ok", False)
    monkeypatch.setattr(ops, "get_sandbox_context", lambda: ctx)
    monkeypatch.setattr(ops, "_executor", lambda: fake)
    return ops.install_dependency("torch"), fake


def test_install_uses_project_venv_when_present(tmp_path: Path, monkeypatch) -> None:
    ctx = _ctx(tmp_path)
    venv_python = ctx.workspace_root / ".venv" / "bin" / "python"
    venv_python.parent.mkdir(parents=True)
    venv_python.write_text("#!/bin/sh\n", encoding="utf-8")
    result, fake = _install(monkeypatch, ctx)
    assert result["success"] is True
    assert result["environment"]["kind"] == "project"
    assert result["environment"]["python"] == str(venv_python)
    assert fake.commands[-1][0] == str(venv_python)


def test_install_skips_when_already_present(tmp_path: Path, monkeypatch) -> None:
    ctx = _ctx(tmp_path)
    venv_python = ctx.workspace_root / ".venv" / "bin" / "python"
    venv_python.parent.mkdir(parents=True)
    venv_python.write_text("#!/bin/sh\n", encoding="utf-8")
    result, fake = _install(monkeypatch, ctx, show_ok=True)
    assert result["success"] is True
    assert result["already_installed"] is True
    assert all("install" not in cmd for cmd in fake.commands)


def test_install_does_not_touch_external_requirements(tmp_path: Path, monkeypatch) -> None:
    ctx = _ctx(tmp_path)
    venv_python = ctx.workspace_root / ".venv" / "bin" / "python"
    venv_python.parent.mkdir(parents=True)
    venv_python.write_text("#!/bin/sh\n", encoding="utf-8")
    req = ctx.workspace_root / "requirements.txt"
    req.write_text("torch\n", encoding="utf-8")
    import src.tools.system_ops as ops

    fake = _FakeExecutor()
    monkeypatch.setattr(ops, "get_sandbox_context", lambda: ctx)
    monkeypatch.setattr(ops, "_executor", lambda: fake)
    monkeypatch.setattr(
        "src.tools.system_ops.get_workspace_root", lambda: ctx.workspace_root
    )
    result = ops.install_dependency("datasets")
    assert result["success"] is True
    assert req.read_text(encoding="utf-8") == "torch\n"


def test_install_managed_keeps_session_behavior(tmp_path: Path, monkeypatch) -> None:
    ctx = _ctx(tmp_path, kind="managed")
    session_python = ctx.venv_path / "bin" / "python"
    session_python.parent.mkdir(parents=True)
    session_python.write_text("#!/bin/sh\n", encoding="utf-8")
    result, fake = _install(monkeypatch, ctx)
    assert result["success"] is True
    assert result["environment"]["kind"] == "session"
    assert fake.commands[-1][0] == str(session_python)
