"""Cooperative run cancellation helpers."""

from __future__ import annotations

from typing import Callable, Optional


class RunCancelledError(Exception):
    """Raised when a run is cancelled via the Control API."""


def ensure_not_cancelled(cancel_check: Optional[Callable[[], bool]]) -> None:
    """Raise :class:`RunCancelledError` if the cancel check returns True."""
    if cancel_check and cancel_check():
        raise RunCancelledError("Run cancelled by user")
