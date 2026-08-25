"""Tests for protected acceptance tests and planned agent-owned tests."""

from __future__ import annotations

from pathlib import Path

from src.tools.file_ops import (
    begin_milestone_write_policy,
    clear_milestone_write_policy,
    patch_file,
    set_allow_test_edits,
    write_file,
)
from src.tools.paths import get_workspace_root, set_workspace_root


def test_new_planned_test_file_can_be_created_and_refined(tmp_path: Path) -> None:
    original_root = get_workspace_root()
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    set_workspace_root(workspace)
    set_allow_test_edits(False)
    begin_milestone_write_policy(
        ["roman.py", "tests/test_roman.py"],
        allow_new_test_files=True,
    )
    try:
        created = write_file("tests/test_roman.py", "def test_one():\n    assert True\n")
        refined = patch_file(
            "tests/test_roman.py",
            "assert True",
            "assert 1 == 1",
        )
        assert created["success"] is True
        assert refined["success"] is True
    finally:
        clear_milestone_write_policy()
        set_workspace_root(original_root)


def test_existing_acceptance_test_remains_protected(tmp_path: Path) -> None:
    original_root = get_workspace_root()
    workspace = tmp_path / "workspace"
    (workspace / "tests").mkdir(parents=True)
    (workspace / "tests" / "test_existing.py").write_text(
        "def test_existing():\n    assert True\n",
        encoding="utf-8",
    )
    set_workspace_root(workspace)
    set_allow_test_edits(False)
    begin_milestone_write_policy(
        ["source.py", "tests/test_existing.py"],
        allow_new_test_files=True,
    )
    try:
        result = patch_file(
            "tests/test_existing.py",
            "assert True",
            "assert False",
        )
        assert result["success"] is False
        assert "PROTECTED FILE BREACH" in result["error"]
    finally:
        clear_milestone_write_policy()
        set_workspace_root(original_root)
