"""Deterministic linting for high-level mission work packets."""

from __future__ import annotations

_ENV_SETUP_TERMS = (
    "environment setup",
    "pip install",
    "install the required dependencies",
    "set up the venv",
)
_PROFILES = {"auto", "ui", "python", "lint", "structural"}


def _strip_workspace_prefix(path: str) -> str:
    return path[len("workspace/"):] if path.startswith("workspace/") else path


def lint_plan(plan: dict) -> tuple[dict, list[str], list[str]]:
    """Apply safe path/dependency fixes and report packet-level issues."""
    fixes: list[str] = []
    issues: list[str] = []
    milestones = plan.get("milestones")
    if not isinstance(milestones, list) or not milestones:
        return plan, fixes, ["plan contains no milestones"]

    ids = [
        str(ms.get("id"))
        for ms in milestones
        if isinstance(ms, dict) and ms.get("id")
    ]
    seen: set[str] = set()
    positions = {ms_id: index for index, ms_id in enumerate(ids)}

    for index, milestone in enumerate(milestones):
        if not isinstance(milestone, dict):
            issues.append(f"milestone #{index + 1} is not a JSON object")
            continue
        label = str(milestone.get("id") or f"#{index + 1}")
        if label in seen:
            issues.append(f"duplicate milestone id '{label}'")
        seen.add(label)
        if not label or label == f"#{index + 1}":
            issues.append(f"milestone #{index + 1} is missing an 'id'")

        haystack = (
            f"{milestone.get('title', '')} {milestone.get('description', '')}"
        ).lower()
        if any(term in haystack for term in _ENV_SETUP_TERMS):
            issues.append(
                f"{label}: environment-setup / pip-install milestones are forbidden"
            )

        deps = milestone.get("depends_on")
        if isinstance(deps, list):
            valid = [dep for dep in deps if str(dep) in ids and str(dep) != label]
            if valid != deps:
                milestone["depends_on"] = valid
                fixes.append(f"{label}: dropped invalid depends_on references")
            forward = [
                str(dep) for dep in valid
                if positions.get(str(dep), -1) >= positions.get(label, index)
            ]
            if forward:
                issues.append(
                    f"{label}: depends_on must reference earlier milestones; "
                    f"move or reorder {forward}"
                )

        targets = milestone.get("target_files")
        if not isinstance(targets, list) or not targets:
            issues.append(f"{label}: 'target_files' must be a non-empty list")
        else:
            stripped = [_strip_workspace_prefix(str(path)) for path in targets]
            if stripped != targets:
                milestone["target_files"] = stripped
                fixes.append(f"{label}: stripped workspace/ prefixes from target_files")

        criteria = milestone.get("acceptance_criteria")
        if not isinstance(criteria, list) or not criteria:
            issues.append(f"{label}: 'acceptance_criteria' must be a non-empty list")
        elif any(not isinstance(item, str) or not item.strip() for item in criteria):
            issues.append(
                f"{label}: acceptance criteria must be non-empty strings"
            )

        profile = str(milestone.get("validation_profile", "auto")).strip().lower()
        if profile not in _PROFILES:
            issues.append(
                f"{label}: unsupported validation_profile '{profile}' "
                "(use auto, ui, python, lint, or structural)"
            )

        if "validation_contract" in milestone:
            issues.append(
                f"{label}: validation_contract is not allowed; provide "
                "acceptance_criteria and validation_profile"
            )

    return plan, fixes, issues
