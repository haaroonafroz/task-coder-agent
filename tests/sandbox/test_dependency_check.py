"""Tests for third-party dependency detection and verification."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.agents.validator import _missing_dependency_fail
from src.sandbox.context import SandboxContext
from src.sandbox.dependency_check import (
    check_target_file_dependencies,
    collect_third_party_imports,
    milestone_suggests_dependencies,
    third_party_hints_in_text,
)


def _ctx(tmp_path: Path) -> SandboxContext:
    root = tmp_path / "sess"
    ws = root / "workspace"
    venv = root / ".venv"
    for d in (root, ws, venv / "bin", root / ".tmp", root / ".home", root / ".cache" / "pip"):
        d.mkdir(parents=True)
    (venv / "bin" / "python").write_bytes(b"")
    return SandboxContext(
        session_id="test",
        jail_root=root,
        workspace_root=ws,
        venv_path=venv,
        tmp_dir=root / ".tmp",
        home_dir=root / ".home",
        pip_cache_dir=root / ".cache" / "pip",
    )


def test_collect_third_party_imports_filters_local_and_stdlib(tmp_path: Path) -> None:
    ws = tmp_path / "workspace"
    ws.mkdir()
    (ws / "snake_logic.py").write_text("class SnakeGame: pass\n", encoding="utf-8")
    (ws / "main.py").write_text(
        "import os\nimport pygame\nfrom snake_logic import SnakeGame\n",
        encoding="utf-8",
    )

    required, checked, errors = collect_third_party_imports(
        ["main.py"],
        workspace_root=ws,
    )

    assert checked == ["main.py"]
    assert errors == []
    assert required == ["pygame"]


def test_third_party_hints_in_text_finds_pygame() -> None:
    hints = third_party_hints_in_text("Build a Snake game using pygame")
    assert "pygame" in hints


def test_milestone_suggests_dependencies() -> None:
    milestone = {
        "title": "Implement Pygame UI",
        "description": "Render the game using Pygame",
    }
    assert milestone_suggests_dependencies(milestone) is True
    assert milestone_suggests_dependencies({"title": "Add math helper", "description": "stdlib only"}) is False


def test_check_target_file_dependencies_reports_missing(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    ws = ctx.workspace_root
    (ws / "main.py").write_text("import pygame\n", encoding="utf-8")

    mock_executor = MagicMock()
    mock_executor.run_argv.return_value = {
        "returncode": 1,
        "stdout": "",
        "stderr": "",
    }

    with patch("src.sandbox.dependency_check.get_executor", return_value=mock_executor):
        with patch("src.sandbox.dependency_check.get_sandbox_context", return_value=ctx):
            report = check_target_file_dependencies(["main.py"])

    assert report.missing_imports == ["pygame"]
    assert report.missing_packages == ["pygame"]
    assert report.ok is False


def test_missing_dependency_fail_returns_fail_verdict(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    ws = ctx.workspace_root
    (ws / "main.py").write_text("import pygame\n", encoding="utf-8")

    mock_executor = MagicMock()
    mock_executor.run_argv.return_value = {"returncode": 1, "stdout": "", "stderr": ""}

    milestone = {"id": "M3", "target_files": ["main.py"]}
    with patch("src.sandbox.dependency_check.get_executor", return_value=mock_executor):
        with patch("src.sandbox.dependency_check.get_sandbox_context", return_value=ctx):
            result = _missing_dependency_fail(milestone, ["main.py"])

    assert result is not None
    assert result["verdict"] == "FAIL"
    assert "pygame" in result["fix_guidance"]


def test_missing_dependency_fail_ok_for_stdlib_only(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    ws = ctx.workspace_root
    (ws / "main.py").write_text("import os\nimport snake_logic\n", encoding="utf-8")
    (ws / "snake_logic.py").write_text("pass\n", encoding="utf-8")

    mock_executor = MagicMock()
    mock_executor.run_argv.return_value = {"returncode": 0, "stdout": "", "stderr": ""}

    milestone = {"id": "M3", "target_files": ["main.py"]}
    with patch("src.sandbox.dependency_check.get_executor", return_value=mock_executor):
        with patch("src.sandbox.dependency_check.get_sandbox_context", return_value=ctx):
            result = _missing_dependency_fail(milestone, ["main.py"])

    assert result is None
