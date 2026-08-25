"""Tests for _is_test_milestone structured classification."""

from __future__ import annotations

from src.main import _is_test_milestone


def test_test_only_target_files_are_test_milestones() -> None:
    milestone = {
        "title": "Test Scaffold for Core Logic",
        "description": "Write pytest tests for movement and collision.",
        "target_files": ["tests/test_snake_logic.py"],
        "acceptance_criteria": ["The specification is executable."],
        "validation_profile": "python",
    }
    assert _is_test_milestone(milestone) is True


def test_test_only_target_files_is_test_milestone() -> None:
    milestone = {
        "title": "Spec",
        "description": "Write tests.",
        "target_files": ["tests/test_foo.py", "tests/test_bar.py"],
        "acceptance_criteria": ["The tests describe the behavior."],
        "validation_profile": "python",
    }
    assert _is_test_milestone(milestone) is True


def test_implementation_milestone_is_not_test_milestone() -> None:
    milestone = {
        "title": "Implement Core Game Logic",
        "description": "Implement the SnakeGame class to pass all M1 tests.",
        "target_files": ["snake_logic.py"],
        "acceptance_criteria": ["The implementation behaves correctly."],
        "validation_profile": "python",
    }
    assert _is_test_milestone(milestone) is False


def test_ui_milestone_is_not_test_milestone() -> None:
    milestone = {
        "title": "Implement Pygame UI",
        "description": "Create game.py with pygame rendering.",
        "target_files": ["game.py"],
        "acceptance_criteria": ["The interface renders correctly."],
        "validation_profile": "ui",
    }
    assert _is_test_milestone(milestone) is False


def test_mixed_target_files_is_not_test_milestone() -> None:
    milestone = {
        "title": "Implement and Test",
        "description": "Write code and tests together.",
        "target_files": ["snake_logic.py", "tests/test_snake_logic.py"],
        "acceptance_criteria": ["Code and tests work together."],
        "validation_profile": "python",
    }
    assert _is_test_milestone(milestone) is False
