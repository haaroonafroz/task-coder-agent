"""
Cognee Knowledge Graph Memory Layer

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
import json
import time
from pathlib import Path
from typing import Any, Optional

_MEMORY_FILE = Path(__file__).parent.parent / "active_mission" / "memory_store.json"

# ---------------------------------------------------------------------------
# Optional Cognee import
# ---------------------------------------------------------------------------
try:
    import cognee
    _COGNEE_AVAILABLE = True
except ImportError:
    cognee = None  # type: ignore
    _COGNEE_AVAILABLE = False


def _run_async(coro) -> Any:
    """Run a coroutine in the current or a new event loop."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(asyncio.run, coro)
                return future.result(timeout=30)
        return loop.run_until_complete(coro)
    except Exception:
        return asyncio.run(coro)


class MissionMemory:
    """
    Unified memory interface — uses Cognee when available, falls back to
    a local JSON file store otherwise.
    """

    def __init__(self) -> None:
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

        On crash/reboot, call `locate_resume_point()` to find where to restart.

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
                _run_async(self._cognee_add_and_cognify(payload))
            except Exception as exc:
                print(f"[Memory] Cognee write failed: {exc}; falling back to JSON.")

        # Always write to the JSON store for durability
        self._json_write(milestone_id, current_status, plan_meta)

    def check_resume_point(self, milestone_id: str) -> Optional[dict]:
        """
        Check whether a milestone was previously completed.

        Returns the stored state dict if found, or None.
        """
        store = self._load_store()
        return store.get("milestones", {}).get(milestone_id)

    def locate_resume_point(self) -> Optional[str]:
        """
        Find the last milestone that was marked incomplete.

        If Cognee is available, performs a semantic graph search.
        Falls back to scanning the JSON store.

        Returns:
            A string description of the resume context, or None.
        """
        if _COGNEE_AVAILABLE:
            try:
                result = _run_async(self._cognee_search(
                    "Find the last incomplete milestone and its architectural dependencies"
                ))
                if result:
                    return str(result)
            except Exception as exc:
                print(f"[Memory] Cognee resume search failed: {exc}")

        # JSON fallback
        store = self._load_store()
        milestones = store.get("milestones", {})
        for ms_id, data in reversed(list(milestones.items())):
            if data.get("status") != "completed":
                return f"Resume at milestone {ms_id}: status={data.get('status')}"
        return None

    # ------------------------------------------------------------------
    # B. Anti-Hallucination Guardrails
    # ------------------------------------------------------------------

    def query_structural_memory(self, target_class_or_feature: str) -> str:
        """
        Query the knowledge graph for structural facts about a code entity.

        Injected into the worker's context to prevent hallucinated API paths.

        Args:
            target_class_or_feature: Natural language description of what to look up.

        Returns:
            A string with known facts, or empty string if nothing found.
        """
        if _COGNEE_AVAILABLE:
            try:
                result = _run_async(self._cognee_search(
                    f"What are the parameters, file locations, and dependencies related to {target_class_or_feature}?"
                ))
                if result:
                    return str(result)[:2000]
            except Exception as exc:
                print(f"[Memory] Cognee structural query failed: {exc}")

        # JSON fallback — return any error log that mentions the target
        store = self._load_store()
        errors = store.get("error_log", [])
        relevant = [e for e in errors if target_class_or_feature.lower() in e.get("payload", "").lower()]
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
                _run_async(self._cognee_add_and_cognify(payload))
            except Exception as exc:
                print(f"[Memory] Cognee error-log write failed: {exc}")

        # JSON fallback
        store = self._load_store()
        store.setdefault("error_log", []).append({
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "file_path": file_path,
            "payload": payload,
        })
        # Keep last 50 error entries
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
        """Clear all persisted error logs (useful between missions)."""
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
        _MEMORY_FILE.parent.mkdir(parents=True, exist_ok=True)
        if not _MEMORY_FILE.exists():
            self._save_store({"milestones": {}, "error_log": []})

    def _load_store(self) -> dict:
        try:
            return json.loads(_MEMORY_FILE.read_text(encoding="utf-8"))
        except Exception:
            return {"milestones": {}, "error_log": []}

    def _save_store(self, data: dict) -> None:
        _MEMORY_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def _json_write(self, milestone_id: str, status: str, meta: dict) -> None:
        store = self._load_store()
        store.setdefault("milestones", {})[milestone_id] = {
            "status": status,
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            **{k: v for k, v in meta.items() if k not in ("error_log",)},
        }
        self._save_store(store)
