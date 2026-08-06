"""
Deterministic plan lint — catches structural plan flaws at t=0, before a
single worker cycle is burned.

Two classes of findings:

  FIXES   — applied silently in code (path prefixes, contract retyping,
            dangling depends_on references).
  ISSUES  — cannot be fixed without judgment (missing contract fields,
            policy-denied commands, environment-setup milestones). Reported
            to the Orchestrator for one patch-ops repair pass.

The motivating example: a contract {"type": "shell", "command": "python -m
py_compile main.py"} used to sail through planning, then fail at validation
time with exit 127 — four full worker/validator/replan cycles in a row.
This linter retypes it to a compilable form (or flags it) at plan time.
"""

from __future__ import annotations

import re

from src.sandbox.commands import compile_contract_to_argv
from src.sandbox.policy import validate_shell_script

_PYTEST_CMD_RE = re.compile(r"^\s*(?:python3?\s+-m\s+)?pytest\s+\S+", re.IGNORECASE)
_PY_COMPILE_CMD_RE = re.compile(r"^\s*python3?\s+-m\s+py_compile\s+\S+", re.IGNORECASE)
_FLAKE8_CMD_RE = re.compile(r"^\s*(?:python3?\s+-m\s+)?flake8\s+\S+", re.IGNORECASE)

_ENV_SETUP_RE = re.compile(
    r"environment setup|install (the )?(required )?(dependencies|packages)|"
    r"pip install|set ?up (the )?venv|verify (the )?toolchain",
    re.IGNORECASE,
)

_TEST_SCAFFOLD_REQUIRED_FIELDS = (
    "language",
    "public_api",
    "required_imports",
    "forbidden_definitions",
)


def _strip_workspace_prefix(text: str) -> str:
    return text.replace("workspace/", "") if isinstance(text, str) else text


def lint_plan(plan: dict) -> tuple[dict, list[str], list[str]]:
    """
    Lint a plan deterministically, applying safe fixes in place.

    Args:
        plan: The plan dict (mutated in place for fixes — call before persisting).

    Returns:
        (plan, fixes_applied, issues_remaining)
        fixes_applied:    human-readable list of deterministic repairs made.
        issues_remaining: structural problems needing an Orchestrator repair pass.
    """
    fixes: list[str] = []
    issues: list[str] = []

    milestones = plan.get("milestones")
    if not isinstance(milestones, list) or not milestones:
        return plan, fixes, ["plan contains no milestones"]

    seen_ids: set[str] = set()
    for i, ms in enumerate(milestones):
        if not isinstance(ms, dict):
            issues.append(f"milestone #{i + 1} is not a JSON object")
            continue

        label = str(ms.get("id") or f"#{i + 1}")

        # --- id uniqueness ---------------------------------------------------
        ms_id = str(ms.get("id", "")).strip()
        if not ms_id:
            issues.append(f"milestone #{i + 1} is missing an 'id'")
        elif ms_id in seen_ids:
            issues.append(f"duplicate milestone id '{ms_id}'")
        seen_ids.add(ms_id)

        # --- environment-setup milestones are forbidden ----------------------
        haystack = f"{ms.get('title', '')} {ms.get('description', '')}"
        if _ENV_SETUP_RE.search(haystack):
            issues.append(
                f"{label}: environment-setup / pip-install milestones are forbidden "
                "(pytest, flake8, black are pre-installed; worker uses install_dependency)"
            )

        # --- depends_on references -------------------------------------------
        deps = ms.get("depends_on")
        if isinstance(deps, list):
            valid = [d for d in deps if str(d) in {
                str(m.get("id")) for m in milestones if isinstance(m, dict)
            } and str(d) != ms_id]
            if len(valid) != len(deps):
                dropped = [d for d in deps if d not in valid]
                ms["depends_on"] = valid
                fixes.append(f"{label}: dropped invalid depends_on references {dropped}")

        # --- target_files ------------------------------------------------------
        target_files = ms.get("target_files")
        if not isinstance(target_files, list) or not target_files:
            issues.append(f"{label}: 'target_files' must be a non-empty list")
        else:
            stripped = [_strip_workspace_prefix(str(p)) for p in target_files]
            if stripped != [str(p) for p in target_files]:
                ms["target_files"] = stripped
                fixes.append(f"{label}: stripped workspace/ prefixes from target_files")

        # --- validation contract ---------------------------------------------
        contract = ms.get("validation_contract")
        if not isinstance(contract, dict):
            issues.append(f"{label}: missing 'validation_contract' object")
            continue

        ctype = str(contract.get("type", "")).strip().lower()
        command = str(contract.get("command", ""))

        if not ctype:
            issues.append(f"{label}: validation_contract is missing 'type'")

        cleaned_command = _strip_workspace_prefix(command).strip()
        if cleaned_command != command.strip():
            contract["command"] = cleaned_command
            fixes.append(f"{label}: stripped workspace/ prefix from contract command")

        if ctype == "test_scaffold":
            missing_fields = [
                f for f in _TEST_SCAFFOLD_REQUIRED_FIELDS if f not in contract
            ]
            if missing_fields:
                issues.append(
                    f"{label}: test_scaffold contract is missing fields "
                    f"{missing_fields} (required for stub-based spec validation)"
                )
        elif ctype == "shell" and cleaned_command:
            # Retype shell contracts that match compilable forms so they run
            # via argv instead of the shell path (which produced exit-127 loops).
            if _PYTEST_CMD_RE.match(cleaned_command):
                contract["type"] = "pytest"
                fixes.append(f"{label}: retyped shell contract to 'pytest'")
            elif _FLAKE8_CMD_RE.match(cleaned_command):
                contract["type"] = "lint"
                fixes.append(f"{label}: retyped shell contract to 'lint'")
            elif _PY_COMPILE_CMD_RE.match(cleaned_command):
                contract["type"] = "py_compile"
                fixes.append(f"{label}: retyped shell contract to 'py_compile'")

        # --- contract executability -------------------------------------------
        if cleaned_command:
            argv = compile_contract_to_argv(contract)
            if argv is None:
                verdict = validate_shell_script(cleaned_command, profile="validation")
                if not verdict.allowed:
                    issues.append(
                        f"{label}: contract command is not executable under the "
                        f"validation sandbox policy: {verdict.reason}. "
                        f"Command: {cleaned_command!r}"
                    )
        elif ctype not in ("structural",):
            issues.append(f"{label}: validation_contract has no 'command'")

    return plan, fixes, issues
