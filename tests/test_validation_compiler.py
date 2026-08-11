"""Tests for validator-owned compilation of high-level packets."""

from __future__ import annotations

from src.agents.validation_compiler import compile_validation_contract
from src.sandbox.process_manager import get_allowed_ports


def test_ui_packet_compiles_server_and_rendered_checks() -> None:
    contract, error = compile_validation_contract({
        "id": "M1",
        "target_files": ["index.html"],
        "acceptance_criteria": [
            'The page title is "Tiny Board".',
            '"To Do" is visible.',
        ],
        "validation_profile": "ui",
    })

    assert error is None
    assert contract is not None
    assert contract["type"] == "ui_smoke"
    assert contract["serve"]["kind"] == "generic"
    assert contract["serve"]["port"] in get_allowed_ports()
    assert {"action": "assert_text", "text": "Tiny Board"} in contract["checks"]


def test_python_packet_compiles_pytest_from_test_targets() -> None:
    contract, error = compile_validation_contract({
        "id": "M1",
        "target_files": ["tests/test_core.py", "core.py"],
        "acceptance_criteria": ["Core behavior passes its tests."],
        "validation_profile": "python",
    })

    assert error is None
    assert contract is not None
    assert contract["type"] == "pytest"
    assert contract["command"].endswith("tests/test_core.py -q")


def test_low_level_contract_is_not_a_compilation_input() -> None:
    contract, error = compile_validation_contract({
        "id": "M1",
        "target_files": ["core.py"],
        "validation_contract": {"type": "pytest", "command": "pytest"},
    })

    assert contract is None
    assert error == "work packet has no acceptance_criteria"
