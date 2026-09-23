"""Canonical-path workspace locks for safe multi-session coordination."""

from __future__ import annotations

import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


class WorkspaceLockManager:
    """Process-local locks; the global queue currently makes these uncontended."""

    def __init__(self) -> None:
        self._guard = threading.Lock()
        self._locks: dict[str, threading.RLock] = {}

    def _lock_for(self, root: Path) -> threading.RLock:
        key = str(root.resolve())
        with self._guard:
            return self._locks.setdefault(key, threading.RLock())

    @contextmanager
    def write_lock(self, root: Path) -> Iterator[None]:
        lock = self._lock_for(root)
        with lock:
            yield

    @contextmanager
    def read_lock(self, root: Path) -> Iterator[None]:
        # A single exclusive lock is intentional for the MVP. The API remains
        # globally serial, while this interface can later become a RW lock.
        lock = self._lock_for(root)
        with lock:
            yield


workspace_locks = WorkspaceLockManager()
