"""
Cognee Knowledge Graph Memory Layer.

Provides three operational capabilities:

  A. Session Persistence & Crash Recovery
     — Logs milestone state into Cognee's entity graph so that a rebooting
       engine can locate and resume the last incomplete milestone.

  B. Anti-Hallucination Guardrails
     — Queries historical code graphs to inject context-grounded structural
       facts into the worker's context window, preventing API path guessing.

  C. Self-Improving Error Mitigation Loops
     — Commits validator failure logs into Cognee so that on the next retry
       the worker receives "negative constraints" — concrete anti-patterns
       extracted from its own previous mistakes.

If Cognee is not installed the class gracefully degrades to a lightweight
JSON-file-backed store (active_mission/memory_store.json) so the runtime
continues to function without the dependency.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Optional

_LEGACY_MEMORY_FILE = Path(__file__).parent.parent / "active_mission" / "memory_store.json"

# ---------------------------------------------------------------------------
# Optional Cognee import
#
# Cognee is OPT-IN: it performs cloud-LLM graph extraction on every write,
# which blocks the serial pipeline for tens of seconds per milestone and adds
# token cost. The JSON store serves every read path in the runtime, so the
# default backend is "json". Set MISSIONS_MEMORY_BACKEND=cognee to re-enable.
# ---------------------------------------------------------------------------
_MEMORY_BACKEND = os.getenv("MISSIONS_MEMORY_BACKEND", "json").strip().lower()

try:
    import cognee
    _COGNEE_INSTALLED = True
except ImportError:
    cognee = None  # type: ignore
    _COGNEE_INSTALLED = False

_COGNEE_AVAILABLE = _COGNEE_INSTALLED and _MEMORY_BACKEND == "cognee"


# ---------------------------------------------------------------------------
# Persistent async runner
#
# A dedicated background thread owns a single asyncio event loop for the
# lifetime of the process. Cognee coroutines are submitted to it via
# run_coroutine_threadsafe, avoiding the per-call ThreadPoolExecutor overhead
# and the deprecated asyncio.get_event_loop() API.
# ---------------------------------------------------------------------------

class _AsyncRunner:
    """
    Singleton background thread that runs a persistent asyncio event loop.

    Coroutines submitted via run() execute on the background thread, keeping
    the main (serial) runtime thread free and avoiding repeated loop creation.
    """

    def __init__(self) -> None:
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._loop.run_forever,
            daemon=True,
            name="cognee-async-runner",
        )
        self._thread.start()

    def run(self, coro, timeout: float = 30) -> Any:
        """Submit a coroutine and block until it completes or times out."""
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result(timeout=timeout)

    def submit(self, coro) -> None:
        """Schedule a coroutine fire-and-forget; never blocks the caller.

        Memory writes are not on the critical path of the serial pipeline —
        the JSON store is written synchronously for durability, so a dropped
        or slow Cognee write must never stall milestone execution.
        """
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)

        def _swallow(fut: concurrent.futures.Future) -> None:
            try:
                fut.result()
            except Exception as exc:
                print(f"[Memory] Background Cognee write failed: {exc}")

        future.add_done_callback(_swallow)


_async_runner: Optional[_AsyncRunner] = None
_runner_lock = threading.Lock()


def _get_runner() -> _AsyncRunner:
    """Return the process-wide async runner, creating it on first call."""
    global _async_runner
    if _async_runner is None:
        with _runner_lock:
            if _async_runner is None:
                _async_runner = _AsyncRunner()
    return _async_runner


def _run_async(coro, timeout: float = 30) -> Any:
    """Run a coroutine on the persistent background event loop."""
    return _get_runner().run(coro, timeout=timeout)


def _submit_async(coro) -> None:
    """Schedule a coroutine without blocking the serial runtime thread."""
    _get_runner().submit(coro)


# ---------------------------------------------------------------------------
# MissionMemory
# ---------------------------------------------------------------------------

class MissionMemory:
    """
    Unified memory interface — uses Cognee when available, falls back to
    a local JSON file store otherwise.
    """

    def __init__(self, memory_file_path: Optional[Path] = None) -> None:
        self._memory_file = memory_file_path if memory_file_path is not None else _LEGACY_MEMORY_FILE
        if _COGNEE_AVAILABLE:
            self._backend = "cognee"
            print("[Memory] Cognee knowledge graph backend active.")
        else:
            self._backend = "json"
            print("[Memory] Cognee not installed — using JSON file store.")
        self._ensure_store()

    # ------------------------------------------------------------------
    # A. Session Persistence & Crash Recovery
    # ------------------------------------------------------------------

    def log_milestone_state(
        self, milestone_id: str, plan_meta: dict, current_status: str
    ) -> None:
        """
        Persist the current state of a milestone into the memory backend.

        Args:
            milestone_id:   Milestone identifier (e.g. "M2").
            plan_meta:      Full handoff dict for this milestone.
            current_status: "pending" | "in_progress" | "completed" | "failed".
        """
        payload = (
            f"Milestone: {milestone_id}. "
            f"Status: {current_status}. "
            f"Metadata: {json.dumps(plan_meta)[:800]}."
        )
        if _COGNEE_AVAILABLE:
            try:
                _submit_async(self._cognee_add_and_cognify(payload))
            except Exception as exc:
                print(f"[Memory] Cognee write scheduling failed: {exc}")

        # Always write to the JSON store for durability
        self._json_write(milestone_id, current_status, plan_meta)

    def check_resume_point(self, milestone_id: str) -> Optional[dict]:
        """
        Check whether a milestone was previously completed.

        Returns the stored state dict if found, or None.
        """
        store = self._load_store()
        return store.get("milestones", {}).get(milestone_id)

    # ------------------------------------------------------------------
    # B. Anti-Hallucination Guardrails
    # ------------------------------------------------------------------

    def query_structural_memory(self, target_class_or_feature: str) -> str:
        """
        Query the knowledge graph for structural facts about a code entity.

        Injected into the worker's context to prevent hallucinated API paths.

        Args:
            target_class_or_feature: Natural-language description of what to look up.

        Returns:
            A string with known facts, or empty string if nothing found.
        """
        if _COGNEE_AVAILABLE:
            try:
                result = _run_async(
                    self._cognee_search(
                        f"What are the parameters, file locations, and dependencies "
                        f"related to {target_class_or_feature}?"
                    ),
                    timeout=10,
                )
                if result:
                    return str(result)[:2000]
            except Exception as exc:
                print(f"[Memory] Cognee structural query failed: {exc}")

        # JSON fallback — return any error log mentioning the target
        store = self._load_store()
        errors = store.get("error_log", [])
        relevant = [
            e for e in errors
            if target_class_or_feature.lower() in e.get("payload", "").lower()
        ]
        if relevant:
            facts = "\n".join(e["payload"] for e in relevant[-3:])
            return f"Known constraints from prior runs:\n{facts}"
        return ""

    # ------------------------------------------------------------------
    # C. Self-Improving Error Mitigation Loops
    # ------------------------------------------------------------------

    def log_compilation_failure(self, file_path: str, compile_error: str) -> None:
        """
        Commit a validator failure to memory so future worker iterations avoid
        replicating the same mistake.

        Args:
            file_path:     Path of the file that failed.
            compile_error: The exact error text from the validator.
        """
        payload = (
            f"File {file_path} failed validation with error: {compile_error[:600]}. "
            f"Do not replicate this syntax or structure."
        )
        if _COGNEE_AVAILABLE:
            try:
                _submit_async(self._cognee_add_and_cognify(payload))
            except Exception as exc:
                print(f"[Memory] Cognee error-log scheduling failed: {exc}")

        store = self._load_store()
        store.setdefault("error_log", []).append({
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "file_path": file_path,
            "payload": payload,
        })
        # Keep last 50 error entries to bound file size
        store["error_log"] = store["error_log"][-50:]
        self._save_store(store)
        print(f"[Memory] Error logged for {file_path}.")

    def get_error_constraints(self, top_n: int = 5) -> str:
        """
        Return the most recent error logs as negative constraints for injection
        into the worker's next retry context.

        Args:
            top_n: Maximum number of error entries to return.

        Returns:
            Formatted negative constraint block, or empty string.
        """
        store = self._load_store()
        errors = store.get("error_log", [])
        if not errors:
            return ""
        recent = errors[-top_n:]
        lines = [f"- {e['payload']}" for e in recent]
        return "## Negative Constraints (do NOT repeat these patterns)\n" + "\n".join(lines)

    def clear_error_log(self) -> None:
        """
        Clear all persisted error logs.

        Call at the start of a new mission to prevent stale constraints from a
        previous run from contaminating the worker's context.
        """
        store = self._load_store()
        store["error_log"] = []
        self._save_store(store)

    # ------------------------------------------------------------------
    # Cognee async helpers
    # ------------------------------------------------------------------

    @staticmethod
    async def _cognee_add_and_cognify(payload: str) -> None:
        await cognee.remember(payload)

    @staticmethod
    async def _cognee_search(query: str) -> Any:
        results = await cognee.recall(query_text=query)
        if not results:
            return ""
        return " ".join(r.text for r in results if hasattr(r, "text"))

    # ------------------------------------------------------------------
    # JSON file-store helpers
    # ------------------------------------------------------------------

    def _ensure_store(self) -> None:
        self._memory_file.parent.mkdir(parents=True, exist_ok=True)
        if not self._memory_file.exists():
            self._save_store({"milestones": {}, "error_log": []})

    def _load_store(self) -> dict:
        try:
            return json.loads(self._memory_file.read_text(encoding="utf-8"))
        except Exception:
            return {"milestones": {}, "error_log": []}

    def _save_store(self, data: dict) -> None:
        self._memory_file.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def _json_write(self, milestone_id: str, status: str, meta: dict) -> None:
        store = self._load_store()
        store.setdefault("milestones", {})[milestone_id] = {
            "status": status,
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            **{k: v for k, v in meta.items() if k != "error_log"},
        }
        self._save_store(store)
