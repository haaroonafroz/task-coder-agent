from __future__ import annotations

import importlib.util
from pathlib import Path

from src.agents.test_scaffold_validator import (
    build_python_stub_overlay,
    python_stub_env_overlay,
    validate_test_scaffold_structure,
)


def _milestone() -> dict:
    return {
        "id": "M1",
        "target_files": ["tests/test_snake_logic.py"],
        "validation_contract": {
            "type": "test_scaffold",
            "language": "python",
            "public_api": [
                {
                    "module": "snake_logic",
                    "name": "Direction",
                    "kind": "enum",
                    "members": ["UP", "DOWN", "LEFT", "RIGHT"],
                },
                {
                    "module": "snake_logic",
                    "name": "SnakeGame",
                    "kind": "class",
                    "methods": ["move", "spawn_food"],
                },
            ],
            "required_imports": ["snake_logic.Direction", "snake_logic.SnakeGame"],
            "forbidden_definitions": ["Direction", "SnakeGame", "move", "spawn_food"],
            "min_assertions": 1,
            "min_tests": 1,
        },
    }


def test_scaffold_rejects_embedded_production_class(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    test_file = workspace / "tests" / "test_snake_logic.py"
    test_file.parent.mkdir(parents=True)
    test_file.write_text(
        "class SnakeGame:\n"
        "    def move(self):\n"
        "        return True\n\n"
        "def test_move():\n"
        "    assert SnakeGame().move() is True\n",
        encoding="utf-8",
    )

    result = validate_test_scaffold_structure(_milestone(), workspace)

    assert result.ok is False
    assert any("SnakeGame" in error for error in result.errors)
    assert any("move" in error for error in result.errors)


def test_scaffold_accepts_external_import_contract(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    test_file = workspace / "tests" / "test_snake_logic.py"
    test_file.parent.mkdir(parents=True)
    test_file.write_text(
        "from snake_logic import Direction, SnakeGame\n\n"
        "def test_move_changes_head():\n"
        "    game = SnakeGame(10, 10)\n"
        "    assert game.move() is True\n",
        encoding="utf-8",
    )

    result = validate_test_scaffold_structure(_milestone(), workspace)

    assert result.ok is True


def test_scaffold_accepts_prompt_style_import_and_forbidden_formats(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    test_file = workspace / "tests" / "test_snake_logic.py"
    test_file.parent.mkdir(parents=True)
    test_file.write_text(
        "from snake_logic import Direction, SnakeGame\n\n"
        "def test_move_changes_head():\n"
        "    game = SnakeGame(10, 10)\n"
        "    assert game.move() is True\n",
        encoding="utf-8",
    )
    milestone = _milestone()
    milestone["validation_contract"]["required_imports"] = [
        "from snake_logic import SnakeGame",
        "from snake_logic import Direction",
    ]
    milestone["validation_contract"]["forbidden_definitions"] = [
        "class SnakeGame",
        "def move",
    ]

    result = validate_test_scaffold_structure(milestone, workspace)

    assert result.ok is True


def test_build_python_stub_overlay(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    stub_root = tmp_path / "stubs"

    build_python_stub_overlay(_milestone(), workspace, stub_root)
    stub_file = stub_root / "snake_logic.py"

    assert stub_file.exists()
    overlay = python_stub_env_overlay(stub_root, workspace)
    assert str(stub_root) in overlay["PYTHONPATH"]

    spec = importlib.util.spec_from_file_location("snake_logic", stub_file)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    assert hasattr(module, "Direction")
    assert hasattr(module, "SnakeGame")
