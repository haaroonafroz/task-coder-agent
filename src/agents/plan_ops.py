"""
Deterministic plan patch operations — negotiated replanning without
full-plan regeneration.

The old replan path asked the Orchestrator to regenerate the ENTIRE plan
from scratch: expensive (tens of thousands of tokens), risky (milestones
silently reordered/renamed/dropped), and unauditable. Instead the
Orchestrator emits a small op-list which this module validates and applies
in code:

  {"op": "update_milestone",       "milestone_id": "M3", "fields": {"title"?, "description"?, "target_files"?}}
  {"op": "insert_milestone_after", "after_id": "M2",     "milestone": {...}}
  {"op": "remove_milestone",       "milestone_id": "M4"}

Invariants enforced here (not by the LLM):
  - Milestones with status "completed" are immutable.
  - Milestone ids stay unique.
  - Inserted milestones must carry id/title/target_files/acceptance_criteria.
  - depends_on references are cleaned up after removals.
"""

from __future__ import annotations

import copy
from typing import Any

SUPPORTED_OPS = (
    "update_milestone",
    "insert_milestone_after",
    "remove_milestone",
)

_UPDATABLE_FIELDS = (
    "title", "description", "target_files", "depends_on",
    "acceptance_criteria", "validation_profile",
)


class PlanPatchError(ValueError):
    """Raised when a plan patch operation is invalid."""


def _is_completed(milestone: dict) -> bool:
    return str(milestone.get("status", "")).lower() == "completed"


def _index_by_id(milestones: list[dict]) -> dict[str, int]:
    index: dict[str, int] = {}
    for i, ms in enumerate(milestones):
        ms_id = ms.get("id")
        if ms_id:
            index[str(ms_id)] = i
    return index


def _validate_new_milestone(ms: Any, existing_ids: set[str]) -> dict:
    if not isinstance(ms, dict):
        raise PlanPatchError("inserted milestone must be a JSON object")
    ms_id = str(ms.get("id", "")).strip()
    if not ms_id:
        raise PlanPatchError("inserted milestone is missing 'id'")
    if ms_id in existing_ids:
        raise PlanPatchError(f"inserted milestone id '{ms_id}' already exists")
    if not ms.get("title"):
        raise PlanPatchError(f"inserted milestone '{ms_id}' is missing 'title'")
    if not isinstance(ms.get("target_files"), list) or not ms["target_files"]:
        raise PlanPatchError(
            f"inserted milestone '{ms_id}' must list non-empty 'target_files'"
        )
    if not isinstance(ms.get("acceptance_criteria"), list) or not ms["acceptance_criteria"]:
        raise PlanPatchError(
            f"inserted milestone '{ms_id}' must include non-empty 'acceptance_criteria'"
        )
    out = copy.deepcopy(ms)
    out.setdefault("depends_on", [])
    out["status"] = "pending"
    return out


def apply_plan_patch(plan: dict, operations: Any) -> dict:
    """
    Validate and apply a list of patch operations to a plan.

    Args:
        plan:       The current plan dict (not mutated).
        operations: List of operation dicts (see module docstring).

    Returns:
        A NEW plan dict with all operations applied.

    Raises:
        PlanPatchError: on the first invalid operation (message is worded so
                        it can be fed back to the Orchestrator for correction).
    """
    if not isinstance(operations, list) or not operations:
        raise PlanPatchError("'operations' must be a non-empty list of patch ops")

    new_plan = copy.deepcopy(plan)
    milestones = new_plan.setdefault("milestones", [])
    if not isinstance(milestones, list):
        raise PlanPatchError("plan 'milestones' must be a list")

    for i, op in enumerate(operations):
        if not isinstance(op, dict):
            raise PlanPatchError(f"operation #{i + 1} is not a JSON object")
        kind = op.get("op")
        if kind not in SUPPORTED_OPS:
            raise PlanPatchError(
                f"operation #{i + 1} has unsupported op '{kind}'. "
                f"Supported: {list(SUPPORTED_OPS)}"
            )

        index = _index_by_id(milestones)

        if kind == "update_milestone":
            ms_id = str(op.get("milestone_id", "")).strip()
            if ms_id not in index:
                raise PlanPatchError(f"update_milestone: milestone '{ms_id}' not found")
            target = milestones[index[ms_id]]
            if _is_completed(target):
                raise PlanPatchError(
                    f"update_milestone: milestone '{ms_id}' is completed and immutable"
                )
            fields = op.get("fields")
            if not isinstance(fields, dict) or not fields:
                raise PlanPatchError(
                    "update_milestone: 'fields' must be a non-empty object"
                )
            unknown = set(fields) - set(_UPDATABLE_FIELDS)
            if unknown:
                raise PlanPatchError(
                    f"update_milestone: cannot update fields {sorted(unknown)}. "
                    f"Allowed: {list(_UPDATABLE_FIELDS)}"
                )
            if "target_files" in fields and not isinstance(fields["target_files"], list):
                raise PlanPatchError("update_milestone: 'target_files' must be a list")
            if (
                "acceptance_criteria" in fields
                and (
                    not isinstance(fields["acceptance_criteria"], list)
                    or not fields["acceptance_criteria"]
                )
            ):
                raise PlanPatchError(
                    "update_milestone: 'acceptance_criteria' must be a non-empty list"
                )
            target.update(copy.deepcopy(fields))

        elif kind == "insert_milestone_after":
            after_id = str(op.get("after_id", "")).strip()
            if after_id not in index:
                raise PlanPatchError(
                    f"insert_milestone_after: anchor milestone '{after_id}' not found"
                )
            new_ms = _validate_new_milestone(op.get("milestone"), set(index))
            milestones.insert(index[after_id] + 1, new_ms)

        elif kind == "remove_milestone":
            ms_id = str(op.get("milestone_id", "")).strip()
            if ms_id not in index:
                raise PlanPatchError(f"remove_milestone: milestone '{ms_id}' not found")
            if _is_completed(milestones[index[ms_id]]):
                raise PlanPatchError(
                    f"remove_milestone: milestone '{ms_id}' is completed and immutable"
                )
            milestones.pop(index[ms_id])
            for ms in milestones:
                deps = ms.get("depends_on")
                if isinstance(deps, list) and ms_id in deps:
                    ms["depends_on"] = [d for d in deps if d != ms_id]

    return new_plan
