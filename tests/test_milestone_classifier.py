"""Tests for _is_test_milestone structured classification."""

from __future__ import annotations

from src.main import _is_test_milestone


def test_scaffold_contract_type_is_test_milestone() -> None:
    milestone = {
        "title": "Test Scaffold for Core Logic",
        "description": "Write pytest tests for movement and collision.",
        "target_files": ["tests/test_snake_logic.py"],
        "validation_contract": {"type": "test_scaffold"},
    }
    assert _is_test_milestone(milestone) is True


def test_test_only_target_files_is_test_milestone() -> None:
    milestone = {
        "title": "Spec",
        "description": "Write tests.",
        "target_files": ["tests/test_foo.py", "tests/test_bar.py"],
        "validation_contract": {"type": "pytest"},
    }
    assert _is_test_milestone(milestone) is True


def test_implementation_milestone_is_not_test_milestone() -> None:
    milestone = {
        "title": "Implement Core Game Logic",
        "description": "Implement the SnakeGame class to pass all M1 tests.",
        "target_files": ["snake_logic.py"],
        "validation_contract": {"type": "pytest"},
    }
    assert _is_test_milestone(milestone) is False


def test_ui_milestone_is_not_test_milestone() -> None:
    milestone = {
        "title": "Implement Pygame UI",
        "description": "Create game.py with pygame rendering.",
        "target_files": ["game.py"],
        "validation_contract": {"type": "lint"},
    }
    assert _is_test_milestone(milestone) is False


def test_mixed_target_files_is_not_test_milestone() -> None:
    milestone = {
        "title": "Implement and Test",
        "description": "Write code and tests together.",
        "target_files": ["snake_logic.py", "tests/test_snake_logic.py"],
        "validation_contract": {"type": "pytest"},
    }
    assert _is_test_milestone(milestone) is False
