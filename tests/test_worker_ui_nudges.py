"""Tests for worker UI handoff nudges and failure handling."""

from __future__ import annotations

from src.agents.worker import _ui_handoff_nudge


def test_ui_nudge_escalates_without_hard_cap() -> None:
    assert _ui_handoff_nudge(calls_since_write=0, total_ui_calls=2) == ""
    assert "Prefer targeted code fixes" in _ui_handoff_nudge(4, 4)
    assert "signal `complete` now" in _ui_handoff_nudge(8, 8)
