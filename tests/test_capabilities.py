"""Tests for language-neutral project capabilities."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.tools.capabilities import project_info, run_checks


def test_project_info_detects_node_and_python_manifests(tmp_path: Path) -> None:
    (tmp_path / "package.json").write_text(
        json.dumps({"scripts": {"test": "vitest"}}),
        encoding="utf-8",
    )
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'demo'\n", encoding="utf-8")

    with patch("src.tools.capabilities.get_workspace_root", return_value=tmp_path):
        result = project_info()

    assert result["success"] is True
    assert result["ecosystems"] == ["node", "python"]
    assert result["manifests"] == ["package.json", "pyproject.toml"]


def test_run_checks_compiles_python_test_to_argv(tmp_path: Path) -> None:
    ctx = MagicMock()
    ctx.workspace_root = tmp_path
    executor = MagicMock()
    executor.run_argv.return_value = {
        "returncode": 0,
        "stdout": "2 passed",
        "stderr": "",
        "timed_out": False,
    }

    with patch("src.tools.capabilities.get_workspace_root", return_value=tmp_path):
        with patch("src.tools.capabilities.get_sandbox_context", return_value=ctx):
            with patch("src.tools.capabilities.get_executor", return_value=executor):
                result = run_checks(
                    ecosystem="python",
                    checks=["test"],
                    target="tests",
                )

    assert result["success"] is True
    assert result["checks"][0]["passed"] is True
    argv = executor.run_argv.call_args.args[0]
    assert argv[:3] == ["python", "-m", "pytest"]
    assert argv[3:] == ["tests", "-q"]
