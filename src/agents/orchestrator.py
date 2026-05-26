"""
Orchestrator Agent — Phase 1 and Phase 1.5.

Phase 1  : Reads orchestrator.md and decomposes the user request into a
           structured milestone plan persisted to active_mission/plan.json.
           Resumes an existing plan automatically on crash-recovery.

Phase 1.5: Dynamic Rescoping — replans the mission when the Validator
           detects a structural flaw (e.g. a command/test name mismatch)
           that cannot be fixed by the worker alone.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from src.llm_client import call_llm, ModelChoice
from src.telemetry import span_llm_call
from src.agents.utils import parse_json_from_text

# ---------------------------------------------------------------------------
# Paths and configuration
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).parent.parent.parent
_CONFIG_DIR = _ROOT / "config"
_MISSION_DIR = _ROOT / "active_mission"
_PLAN_PATH = _MISSION_DIR / "plan.json"

_ORCHESTRATOR_MD = (_CONFIG_DIR / "orchestrator.md").read_text()

MAX_TOKENS_ORCHESTRATOR = int(os.getenv("MAX_TOKENS_ORCHESTRATOR", "8192"))


# ---------------------------------------------------------------------------
# Phase 1 — Orchestration
# ---------------------------------------------------------------------------

def run_orchestration(user_request: str, model: ModelChoice) -> dict:
    """
    Decompose a user request into a structured milestone plan.

    Behaviour:
      - If active_mission/plan.json exists with pending milestones, resumes it
        without calling the LLM (crash-recovery path).
      - Otherwise calls the Orchestrator LLM and persists the new plan.

    Args:
        user_request: Plain-English description of what to build or fix.
        model:        LLM backend to use.

    Returns:
        The plan dict with a "milestones" list.
    """
    print("\n[Phase 1] ORCHESTRATION — decomposing request…")

    if _PLAN_PATH.exists():
        raw_plan = _PLAN_PATH.read_text(encoding="utf-8").strip()
        if raw_plan:
            try:
                existing = json.loads(raw_plan)
                pending = [
                    m for m in existing.get("milestones", [])
                    if m.get("status") != "completed"
                ]
                if pending:
                    print(
                        f"  [Orchestrator] Resuming existing plan "
                        f"'{existing.get('title')}' — {len(pending)} milestones pending."
                    )
                    return existing
            except json.JSONDecodeError as exc:
                print(f"  [Orchestrator] Corrupt plan.json ignored: {exc}")
        else:
            print("  [Orchestrator] Empty plan.json ignored — generating new plan.")

    prompt = (
        f"{_ORCHESTRATOR_MD}\n\n"
        f"---\n\n"
        f"User Request:\n{user_request}\n\n"
        f"Output the JSON plan now:"
    )

    with span_llm_call("orchestrator", "init", model):
        result = call_llm(
            prompt, model=model, max_tokens=MAX_TOKENS_ORCHESTRATOR,
            json_mode=True, enable_thinking=True,
        )

    plan = parse_json_from_text(result.text)
    if plan is None:
        raise RuntimeError(f"Orchestrator returned non-JSON output:\n{result.text[:500]}")

    _MISSION_DIR.mkdir(parents=True, exist_ok=True)
    _PLAN_PATH.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    print(f"  [Orchestrator] Plan saved → {_PLAN_PATH}")
    return plan


# ---------------------------------------------------------------------------
# Phase 1.5 — Dynamic Rescoping / Replan
# ---------------------------------------------------------------------------

def replan_mission(current_plan: dict, replan_guidance: str, model: ModelChoice) -> dict:
    """
    Patch the current plan based on Validator feedback.

    Called when the Validator emits a REPLAN verdict — typically when the
    validation contract command doesn't match the generated test names, or
    when the plan has a structural flaw the worker cannot resolve alone.

    Args:
        current_plan:    The plan dict to patch.
        replan_guidance: Actionable guidance string from the Validator.
        model:           LLM backend to use.

    Returns:
        The updated plan dict (also persisted to plan.json).
    """
    print("\n[Phase 1.5] DYNAMIC RESCOPING — Orchestrator patching plan…")

    prompt = (
        f"{_ORCHESTRATOR_MD}\n\n"
        f"---\n\n"
        f"## Negotiation Boundary: Plan Flaw Detected\n"
        f"The Validator has rejected the current plan due to a structural flaw "
        f"or command mismatch.\n\n"
        f"### Current Plan\n```json\n{json.dumps(current_plan, indent=2)}\n```\n\n"
        f"### Validator's Replan Guidance\n{replan_guidance}\n\n"
        f"Output an UPDATED `plan.json` fixing this issue. Keep completed milestones intact."
    )

    with span_llm_call("orchestrator_replan", "REPLAN", model):
        result = call_llm(
            prompt, model=model, max_tokens=MAX_TOKENS_ORCHESTRATOR,
            json_mode=True, enable_thinking=True,
        )

    parsed = parse_json_from_text(result.text)
    if parsed is None:
        raise RuntimeError(
            f"Orchestrator returned non-JSON during replan:\n{result.text[:500]}"
        )

    _PLAN_PATH.write_text(json.dumps(parsed, indent=2), encoding="utf-8")
    print("  [Orchestrator] Plan successfully patched and saved.")
    return parsed
