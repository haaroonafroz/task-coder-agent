"""Tests for the worker wire protocol: batched calls + failure fingerprints."""

from __future__ import annotations

from src.agents.worker import _extract_tool_calls, MAX_BATCH_CALLS
from src.agents.utils import failure_signature


# ---------------------------------------------------------------------------
# Batch extraction
# ---------------------------------------------------------------------------

def test_single_tool_call_normalized_to_list() -> None:
    calls, error, note = _extract_tool_calls(
        {"tool": "read_file", "args": {"file_path": "a.py"}, "reasoning": "r"}
    )
    assert error is None
    assert len(calls) == 1
    assert calls[0]["tool"] == "read_file"


def test_batch_calls_accepted() -> None:
    calls, error, note = _extract_tool_calls({"calls": [
        {"tool": "read_file", "args": {"file_path": "a.py"}},
        {"tool": "read_file", "args": {"file_path": "b.py"}},
    ]})
    assert error is None
    assert len(calls) == 2
    assert note is None


def test_batch_truncated_at_max_with_note() -> None:
    calls, error, note = _extract_tool_calls({"calls": [
        {"tool": "read_file", "args": {}}
        for _ in range(MAX_BATCH_CALLS + 2)
    ]})
    assert error is None
    assert len(calls) == MAX_BATCH_CALLS
    assert note is not None and "truncated" in note


def test_non_dict_batch_entries_filtered() -> None:
    calls, error, _ = _extract_tool_calls({"calls": ["garbage", {"tool": "read_file", "args": {}}]})
    assert error is None
    assert len(calls) == 1


def test_empty_batch_is_error() -> None:
    calls, error, _ = _extract_tool_calls({"calls": ["garbage"]})
    assert calls == []
    assert error is not None


def test_missing_tool_key_is_error() -> None:
    calls, error, _ = _extract_tool_calls({"something": "else"})
    assert calls == []
    assert error is not None and "tool" in error


# ---------------------------------------------------------------------------
# Failure signatures
# ---------------------------------------------------------------------------

def test_same_failure_maps_to_same_signature() -> None:
    sig_a = failure_signature(
        "M3", "python -m pytest tests/test_snake.py -v",
        "FAILED tests/test_snake.py::test_move - assert (11, 7) == (9, 7)",
        1,
    )
    sig_b = failure_signature(
        "M3", "python -m pytest tests/test_snake.py -v",
        "FAILED tests/test_snake.py::test_move - assert (12, 8) == (10, 8)",
        1,
    )
    # Same error line modulo digits → same fingerprint
    assert sig_a == sig_b


def test_different_command_maps_to_different_signature() -> None:
    sig_a = failure_signature("M3", "python -m pytest tests/a.py -v", "error: boom", 1)
    sig_b = failure_signature("M3", "python -m pytest tests/b.py -v", "error: boom", 1)
    assert sig_a != sig_b


def test_different_returncode_maps_to_different_signature() -> None:
    sig_a = failure_signature("M3", "cmd", "error: boom", 1)
    sig_b = failure_signature("M3", "cmd", "error: boom", 127)
    assert sig_a != sig_b
