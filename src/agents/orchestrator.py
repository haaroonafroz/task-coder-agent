"""
Orchestrator Agent — Phase 1 and Phase 1.5.

Phase 1  : Reads orchestrator.md and decomposes the user request into a
           structured milestone plan persisted to the session's plan.json.
           Resumes an existing plan automatically on crash-recovery.

Phase 1.5: Dynamic Rescoping — patches the plan when the Validator detects
           a structural flaw. The Orchestrator emits a SMALL PATCH OP-LIST
           (see src/agents/plan_ops.py) which the harness validates and
           applies deterministically — the whole plan is never regenerated,
           so completed milestones cannot be silently dropped or reordered.

Both phases use corrective JSON retries: a malformed LLM response triggers
one corrective re-prompt instead of crashing the mission.
"""

from __future__ import annotations

import json
import hashlib
import os
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Optional

from src.llm_client import call_llm, ModelChoice, resolve_model_config
from src.telemetry import span_llm_call, TelemetryContext
from src.agents.utils import parse_json_from_text
from src.agents.plan_ops import apply_plan_patch, PlanPatchError
from src.agents.llm_stream_events import stream_context_for
from src.events import EventEmitter

# ---------------------------------------------------------------------------
# Paths and configuration
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).parent.parent.parent
_CONFIG_DIR = _ROOT / "config"
_LEGACY_MISSION_DIR = _ROOT / "active_mission"
_LEGACY_PLAN_PATH = _LEGACY_MISSION_DIR / "plan.json"

_ORCHESTRATOR_MD = (_CONFIG_DIR / "orchestrator.md").read_text()

MAX_TOKENS_ORCHESTRATOR = int(os.getenv("MAX_TOKENS_ORCHESTRATOR", "24576"))
_JSON_CORRECTION_ATTEMPTS = 2


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _call_json_with_correction(
    initial_prompt: str,
    *,
    model: ModelChoice,
    role: str,
    span_name: str,
    span_label: str,
    session: Optional[TelemetryContext],
    emitter: Optional[EventEmitter],
    validate: Callable[[dict], Optional[str]],
) -> dict:
    """
    Call the LLM expecting a JSON object; on parse/validation failure, send
    ONE corrective re-prompt with the error before giving up.

    Args:
        validate: Function receiving the parsed dict, returning None when the
                  payload is acceptable or an error string describing the
                  problem (fed back to the model verbatim).

    Raises:
        RuntimeError: when all attempts fail (caller converts to a graceful
                      mission failure — never an uncaught traceback).
    """
    span_model = (
        resolve_model_config(model, role).model_name  # type: ignore[arg-type]
        if model != "auto" else model
    )
    prompt = initial_prompt
    last_error = "unknown"

    for attempt in range(1 + _JSON_CORRECTION_ATTEMPTS):
        with span_llm_call(span_name, span_label, span_model, session=session):
            result = call_llm(
                prompt, model=model, max_tokens=MAX_TOKENS_ORCHESTRATOR,
                json_mode=True, role=role,  # type: ignore[arg-type]
                stream_context=stream_context_for(
                    emitter,
                    "orchestrator",
                    phase=span_name,
                    output_kind="json",
                ),
            )

        parsed = parse_json_from_text(result.text)
        if parsed is None:
            last_error = "output was not a parseable JSON object"
        else:
            validation_error = validate(parsed)
            if validation_error is None:
                return parsed
            last_error = validation_error

        print(
            f"  [Orchestrator] Invalid {span_name} output "
            f"(attempt {attempt + 1}/{1 + _JSON_CORRECTION_ATTEMPTS}): {last_error}"
        )
        prompt = (
            f"{initial_prompt}\n\n"
            f"## CORRECTION REQUIRED\n"
            f"Your previous response was rejected: {last_error}.\n"
            f"Previous response (truncated):\n{result.text[:1500]}\n\n"
            f"Output ONLY a single valid JSON object that fixes this."
        )

    raise RuntimeError(
        f"Orchestrator produced invalid JSON after "
        f"{1 + _JSON_CORRECTION_ATTEMPTS} attempts ({span_name}): {last_error}"
    )


# ---------------------------------------------------------------------------
# Phase 1 — Orchestration
# ---------------------------------------------------------------------------

def run_orchestration(
    user_request: str,
    model: ModelChoice,
    plan_path: Path = _LEGACY_PLAN_PATH,
    mission_dir: Path = _LEGACY_MISSION_DIR,
    run_kind: str = "auto",
    parent_plan_id: Optional[str] = None,
    triage_report: Optional[dict[str, Any]] = None,
    session: Optional[TelemetryContext] = None,
    emitter: Optional[EventEmitter] = None,
) -> dict:
    """
    Decompose a user request into a structured milestone plan.

    Behaviour:
      - If ``run_kind`` is ``resume`` and ``plan_path`` has pending
        milestones, resumes it without calling the LLM.
      - ``repair`` and ``new`` always create a fresh plan while preserving
        the previous plan in the session's ``plans/`` history directory.
      - Otherwise calls the Orchestrator LLM and persists the new plan.

    Returns:
        The plan dict with a "milestones" list.
    """
    print("\n[Phase 1] ORCHESTRATION — decomposing request…")

    if plan_path.exists():
        raw_plan = plan_path.read_text(encoding="utf-8").strip()
        if raw_plan:
            try:
                existing = json.loads(raw_plan)
                pending = [
                    m for m in existing.get("milestones", [])
                    if m.get("status") != "completed"
                ]
                if pending and run_kind in ("auto", "resume"):
                    existing.setdefault(
                        "plan_id",
                        str(existing.get("mission_id") or f"legacy-{uuid.uuid4().hex[:12]}"),
                    )
                    existing.setdefault("run_kind", "resume")
                    plan_path.write_text(
                        json.dumps(existing, indent=2),
                        encoding="utf-8",
                    )
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
        f"## Run Mode\n{run_kind}\n\n"
        f"## Parent Plan\n{parent_plan_id or '(none)'}\n\n"
        f"User Request:\n{user_request}\n\n"
    )
    if triage_report:
        prompt += (
            "## Read-only Triage Report\n"
            f"```json\n{json.dumps(triage_report, indent=2)[:12000]}\n```\n\n"
        )
    prompt += "Output the JSON plan now:"

    def _validate_plan(parsed: dict) -> Optional[str]:
        milestones = parsed.get("milestones")
        if not isinstance(milestones, list) or not milestones:
            return "plan must contain a non-empty 'milestones' list"
        for i, ms in enumerate(milestones):
            if not isinstance(ms, dict) or not ms.get("id"):
                return f"milestone #{i + 1} is missing an 'id'"
            if not ms.get("target_files"):
                return f"milestone '{ms.get('id')}' must list 'target_files'"
            criteria = ms.get("acceptance_criteria")
            if (
                not isinstance(criteria, list)
                or not criteria
                or any(not isinstance(item, str) or not item.strip() for item in criteria)
            ):
                return (
                    f"milestone '{ms.get('id')}' must include a non-empty "
                    "'acceptance_criteria' list of non-empty strings"
                )
            profile = str(ms.get("validation_profile", "auto")).lower()
            if profile not in {"auto", "ui", "python", "lint", "structural"}:
                return (
                    f"milestone '{ms.get('id')}' has unsupported "
                    f"validation_profile '{profile}'"
                )
            if "validation_contract" in ms:
                return (
                    f"milestone '{ms.get('id')}' must not include "
                    "'validation_contract'; provide high-level intent only"
                )
        return None

    plan = _call_json_with_correction(
        prompt,
        model=model,
        role="orchestrator",
        span_name="orchestrator",
        span_label="init",
        session=session,
        emitter=emitter,
        validate=_validate_plan,
    )

    old_plan: Optional[dict[str, Any]] = None
    if plan_path.exists() and run_kind not in ("auto", "resume"):
        try:
            old_plan = json.loads(plan_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            old_plan = None
    if old_plan:
        _archive_plan(old_plan, mission_dir)

    model_mission_id = str(plan.get("mission_id") or "").strip()
    if run_kind == "repair" or not model_mission_id:
        model_mission_id = f"{run_kind}-{uuid.uuid4().hex[:10]}"
    plan["mission_id"] = model_mission_id
    plan["plan_id"] = f"{run_kind}-{uuid.uuid4().hex[:12]}"
    plan["run_kind"] = run_kind
    plan["parent_plan_id"] = parent_plan_id
    plan["request_hash"] = hashlib.sha256(
        user_request.encode("utf-8")
    ).hexdigest()[:16]

    mission_dir.mkdir(parents=True, exist_ok=True)
    plan_path.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    print(f"  [Orchestrator] Plan saved → {plan_path}")
    return plan


def _archive_plan(plan: dict[str, Any], mission_dir: Path) -> None:
    """Persist a completed or superseded plan before replacing plan.json."""
    plan_id = str(
        plan.get("plan_id")
        or plan.get("mission_id")
        or f"legacy-{int(time.time())}"
    )
    history_dir = mission_dir / "plans"
    history_dir.mkdir(parents=True, exist_ok=True)
    destination = history_dir / f"{plan_id}.json"
    if not destination.exists():
        destination.write_text(json.dumps(plan, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Phase 1.5 — Dynamic Rescoping / Replan (patch-based)
# ---------------------------------------------------------------------------

_PATCH_FORMAT_GUIDE = """\
## Output Format — PLAN PATCH (never a full plan)

Output a single JSON object with an "operations" list. Supported operations:

1. Update milestone intent:
   {"op": "update_milestone", "milestone_id": "M3", "fields": {
     "acceptance_criteria": ["..."], "validation_profile": "python"
   }}

3. Insert a new milestone after an existing one:
   {"op": "insert_milestone_after", "after_id": "M2", "milestone": {
     "id": "M2a", "target_files": ["..."],
     "acceptance_criteria": ["..."], "validation_profile": "auto"
   }}

4. Remove a pending milestone:
   {"op": "remove_milestone", "milestone_id": "M4"}

Rules:
- Use the SMALLEST patch that fixes the flaw. Update intent, not executable
  validation details.
- Milestones with status "completed" are immutable — the harness rejects patches touching them.
- Milestone ids must stay unique; never reorder the plan yourself.
"""


def _replan_prompt(current_plan: dict, replan_guidance: str) -> str:
    return (
        f"{_ORCHESTRATOR_MD}\n\n"
        f"---\n\n"
        f"## Negotiation Boundary: Plan Flaw Detected\n"
        f"The Validator has rejected the current plan due to a structural flaw or command mismatch.\n\n"
        f"### Current Plan\n```json\n{json.dumps(current_plan, indent=2)}\n```\n\n"
        f"### Validator's Replan Guidance\n{replan_guidance}\n\n"
        f"### Harness Constraints (non-negotiable)\n"
        f"- pytest, flake8, and black are pre-installed in the session venv at activation.\n"
        f"- NEVER add Environment Setup, pip install, or tooling verification milestones.\n"
        f"- Mission-specific packages are installed by the worker via "
        f"install_dependency, not via validation commands.\n"
        f"- Keep acceptance criteria observable and let the Validator choose commands.\n"
        f"- Repair packet fields only; do not add executable validation contracts.\n\n"
        f"{_PATCH_FORMAT_GUIDE}\n"
        f"Output the plan patch JSON now:"
    )


def _apply_ops_prompt(
    current_plan: dict,
    *,
    guidance_header: str,
    guidance_body: str,
) -> str:
    return (
        f"{_ORCHESTRATOR_MD}\n\n"
        f"---\n\n"
        f"## {guidance_header}\n\n"
        f"### Current Plan\n```json\n{json.dumps(current_plan, indent=2)}\n```\n\n"
        f"{guidance_body}\n\n"
        f"{_PATCH_FORMAT_GUIDE}\n"
        f"Output the plan patch JSON now:"
    )


def _request_plan_patch(
    prompt: str,
    current_plan: dict,
    model: ModelChoice,
    *,
    span_label: str,
    session: Optional[TelemetryContext],
    emitter: Optional[EventEmitter],
) -> dict:
    """Run one patch-ops LLM exchange and apply the result deterministically."""

    holder: dict[str, Any] = {}

    def _validate_ops(parsed: dict) -> Optional[str]:
        ops = parsed.get("operations")
        if not isinstance(ops, list) or not ops:
            return "response must contain a non-empty 'operations' list"
        try:
            holder["plan"] = apply_plan_patch(current_plan, ops)
        except PlanPatchError as exc:
            return f"invalid patch: {exc}"
        return None

    _call_json_with_correction(
        prompt,
        model=model,
        role="orchestrator",
        span_name="orchestrator_replan",
        span_label=span_label,
        session=session,
        emitter=emitter,
        validate=_validate_ops,
    )
    return holder["plan"]


def replan_mission(
    current_plan: dict,
    replan_guidance: str,
    model: ModelChoice,
    plan_path: Path = _LEGACY_PLAN_PATH,
    session: Optional[TelemetryContext] = None,
    emitter: Optional[EventEmitter] = None,
) -> dict:
    """
    Patch the current plan based on Validator feedback.

    Called when the Validator emits a REPLAN verdict. The Orchestrator emits
    a patch op-list which is validated and applied by code — completed
    milestones are immutable and plan order is preserved unless an explicit
    insert/remove op says otherwise.

    Returns:
        The updated plan dict (also persisted to plan_path).
    """
    print("\n[Phase 1.5] DYNAMIC RESCOPING — Orchestrator patching plan…")

    patched = _request_plan_patch(
        _replan_prompt(current_plan, replan_guidance),
        current_plan,
        model,
        span_label="REPLAN",
        session=session,
        emitter=emitter,
    )

    plan_path.write_text(json.dumps(patched, indent=2), encoding="utf-8")
    print("  [Orchestrator] Plan successfully patched and saved.")
    return patched


def repair_plan_issues(
    current_plan: dict,
    issues: list[str],
    model: ModelChoice,
    plan_path: Path,
    session: Optional[TelemetryContext] = None,
    emitter: Optional[EventEmitter] = None,
) -> dict:
    """
    Repair plan-lint findings the harness could not fix deterministically.

    Same patch-ops protocol as replan_mission, invoked once at plan time
    (before any worker cycle is burned) instead of at validation time.
    """
    print("\n[Phase 1.2] PLAN REPAIR — fixing lint issues before execution…")

    body = (
        "### Deterministic plan lint found these structural issues\n"
        + "\n".join(f"- {issue}" for issue in issues)
        + "\n\nFix them with the smallest possible patch."
    )
    patched = _request_plan_patch(
        _apply_ops_prompt(
            current_plan,
            guidance_header="Plan Lint: Structural Issues Detected Before Execution",
            guidance_body=body,
        ),
        current_plan,
        model,
        span_label="LINT_REPAIR",
        session=session,
        emitter=emitter,
    )

    plan_path.write_text(json.dumps(patched, indent=2), encoding="utf-8")
    print("  [Orchestrator] Plan repaired and saved.")
    return patched
