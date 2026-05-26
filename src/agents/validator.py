"""
Validator Agent — Phase 4.

Runs the validation contract and renders an adversarial PASS / FAIL / REPLAN
verdict. Two fast-path shortcuts avoid an LLM call when the answer is clear:

  1. Structural audit  — immediately FAILs if the worker illegally edited test
     files outside a designated spec-writing milestone.
  2. Contract exit 0  — immediately PASSes when the shell command exits 0 and
     no structural violation was detected.

The LLM is only invoked when the contract exits non-zero (or was absent) and
the deterministic REPLAN check did not trigger.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from src.llm_client import call_llm, ModelChoice
from src.telemetry import span_llm_call
from src.tools import dispatch
from src.tools.paths import resolve_workspace_path
from src.agents.utils import parse_json_from_text

# ---------------------------------------------------------------------------
# Paths and configuration
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).parent.parent.parent
_CONFIG_DIR = _ROOT / "config"

_VALIDATOR_MD = (_CONFIG_DIR / "validator.md").read_text()

MAX_TOKENS_VALIDATOR = int(os.getenv("MAX_TOKENS_VALIDATOR", "3072"))
MAX_RETRY_CYCLES     = 3


# ---------------------------------------------------------------------------
# Phase 4 — Validation
# ---------------------------------------------------------------------------

def run_validator(
    milestone: dict,
    worker_result: dict,
    retry_count: int,
    is_test_milestone: bool,
    model: ModelChoice,
) -> dict[str, Any]:
    """
    Execute the validation contract and render a verdict.

    Args:
        milestone:         The milestone being validated.
        worker_result:     Result dict returned by run_worker.
        retry_count:       Current retry attempt (0-indexed) — shown in the prompt.
        is_test_milestone: True when this milestone is explicitly a test-writing
                           phase; permits test file edits by the worker.
        model:             LLM backend to use.

    Returns:
        A verdict dict with key "verdict": "PASS" | "FAIL" | "REPLAN".
    """
    ms_id = milestone.get("id", "?")
    contract = milestone.get("validation_contract", {})
    command = contract.get("command", "")
    target_files = milestone.get("target_files", [])

    # --- Structural audit: reject unauthorized test file edits ---------------
    modified_files = worker_result.get("files_modified", [])
    unauthorized_edits = [
        f for f in modified_files
        if "tests/" in f or "test_" in f
    ]
    if unauthorized_edits and not is_test_milestone:
        print(
            f"    [Validator] REJECTED: Worker illegally modified test files: "
            f"{unauthorized_edits}"
        )
        return {
            "verdict": "FAIL",
            "milestone_id": ms_id,
            "errors": [
                f"Specification Gaming Detected: Worker altered test scripts "
                f"{unauthorized_edits} to force a pass."
            ],
            "root_cause": (
                "Worker altered the validation test files instead of "
                "correcting implementation code."
            ),
            "fix_guidance": (
                "Revert all modifications to test files. Correct the implementation "
                "code in your assigned source files so that it correctly passes the "
                "original, unmodified test suite."
            ),
        }

    # --- Run the validation contract command ---------------------------------
    contract_output = ""
    tool_result: dict = {}
    if command:
        print(f"    [Validator] Running: {command}")
        tool_result = dispatch("run_shellscript", {"script": command, "timeout": 120})
        contract_output = (
            f"stdout:\n{tool_result.get('stdout', '')}\n"
            f"stderr:\n{tool_result.get('stderr', '')}\n"
            f"returncode: {tool_result.get('returncode', -1)}"
        )
        print(f"    [Validator] returncode={tool_result.get('returncode')}")

    returncode = tool_result.get("returncode") if command else None

    # --- Deterministic REPLAN check ------------------------------------------
    replan = _detect_contract_replan(milestone, worker_result, contract_output, returncode)
    if replan:
        print(f"    [Validator] Deterministic REPLAN: {replan['replan_guidance']}")
        return replan

    # --- Fast-path PASS (contract exited 0) ----------------------------------
    if returncode == 0:
        return {
            "verdict": "PASS",
            "milestone_id": ms_id,
            "validation_details": "Contract command exited 0.",
        }

    # --- LLM verdict for non-zero exit or commandless milestones -------------
    prompt = (
        f"{_VALIDATOR_MD}\n\n"
        f"---\n\n"
        f"## Milestone Under Review\n"
        f"**ID**: {ms_id}\n"
        f"**Title**: {milestone.get('title', '')}\n"
        f"**Description**: {milestone.get('description', '')}\n"
        f"**Target files (worker may ONLY edit these)**: {target_files}\n\n"
        f"## Validation Contract\n```json\n{json.dumps(contract, indent=2)}\n```\n\n"
        f"## Contract Execution Output\n```\n{contract_output or '(no command run)'}\n```\n\n"
        f"## Worker Handoff\n"
        f"Files modified: {worker_result.get('files_modified', [])}\n"
        f"Summary: {worker_result.get('summary', '')}\n\n"
        f"**Retry attempt**: {retry_count + 1} of {MAX_RETRY_CYCLES}\n"
        f"Emit your PASS, FAIL, or REPLAN JSON verdict now:"
    )

    with span_llm_call("validator", ms_id, model):
        result = call_llm(
            prompt, model=model, max_tokens=MAX_TOKENS_VALIDATOR,
            json_mode=True, enable_thinking=True,
        )

    parsed = parse_json_from_text(result.text)
    if parsed is None:
        if command and "returncode: 0" in contract_output:
            return {
                "verdict": "PASS",
                "milestone_id": ms_id,
                "validation_details": "Contract command exited 0.",
            }
        return {
            "verdict": "FAIL",
            "milestone_id": ms_id,
            "errors": ["Could not parse validator response."],
            "fix_guidance": result.text[:500],
        }

    parsed = _normalize_validator_verdict(parsed, returncode=returncode)

    # Contract success is authoritative — don't let the LLM REPLAN over a green run
    if returncode == 0 and parsed.get("verdict") == "REPLAN":
        print("    [Validator] Overriding REPLAN — contract command returned 0.")
        parsed["verdict"] = "PASS"
        parsed.setdefault("validation_details", "Contract command exited 0.")

    return parsed


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def _detect_contract_replan(
    milestone: dict,
    worker_result: dict,
    contract_output: str,
    returncode: int | None,
) -> dict | None:
    """
    Deterministic REPLAN trigger — fires before the LLM is consulted.

    Currently handles: pytest exit code 5 (no tests collected) when test files
    exist but the -k filter doesn't match any test function names.

    Returns a REPLAN verdict dict, or None if no deterministic trigger applies.
    """
    command = milestone.get("validation_contract", {}).get("command", "")
    modified = worker_result.get("files_modified", [])
    worker_did_not_touch_tests = not any("tests/" in f or "test_" in f for f in modified)

    if returncode == 5 and "pytest" in command:
        test_files = list(resolve_workspace_path(".").glob("tests/test_*.py"))
        if test_files and worker_did_not_touch_tests:
            kw_match = re.search(r"-k\s+(\S+)", command)
            keyword = kw_match.group(1) if kw_match else None
            replan_detail = (
                "Pytest collected zero tests (exit code 5) but test files exist. "
                "This is likely an Orchestrator validation command mismatch."
            )
            if keyword:
                replan_detail += f" The -k filter '{keyword}' may not match any test names."
            return {
                "verdict": "REPLAN",
                "milestone_id": milestone.get("id"),
                "validation_details": replan_detail,
                "errors": ["pytest exit code 5: no tests collected"],
                "root_cause": "Validation contract command does not match existing tests.",
                "fix_guidance": None,
                "replan_guidance": (
                    f"Fix validation_contract.command for {milestone.get('id')}. "
                    f"Current command: {command!r}. "
                    f"Inspect tests/test_*.py function names and use a matching -k expression "
                    f"(e.g. -k tokenize instead of -k tokenizer), or run the specific test "
                    f"functions directly."
                ),
            }

    return None


def _valid_replan_guidance(value: Any) -> bool:
    """True only for a non-empty, meaningful replan instruction string."""
    if not isinstance(value, str):
        return False
    cleaned = value.strip().lower()
    return bool(cleaned) and cleaned not in {"null", "none", "n/a", "na"}


def _normalize_validator_verdict(parsed: dict, *, returncode: int | None) -> dict:
    """
    Coerce and guard the raw LLM verdict dict.

    Rules applied in order:
      1. Normalise the verdict string to PASS / REPLAN / FAIL.
      2. Upgrade to REPLAN if replan_guidance is present but verdict said FAIL.
      3. Downgrade REPLAN to PASS/FAIL when replan_guidance is absent.
    """
    verdict_raw = str(parsed.get("verdict", "FAIL")).strip().upper()

    if verdict_raw.startswith("PASS"):
        parsed["verdict"] = "PASS"
    elif verdict_raw.startswith("REPLAN"):
        parsed["verdict"] = "REPLAN"
    else:
        parsed["verdict"] = "FAIL"

    # Upgrade FAIL → REPLAN when actionable guidance is present
    if parsed["verdict"] != "REPLAN" and _valid_replan_guidance(parsed.get("replan_guidance")):
        parsed["verdict"] = "REPLAN"

    # REPLAN without actionable guidance is meaningless — demote
    if parsed["verdict"] == "REPLAN" and not _valid_replan_guidance(parsed.get("replan_guidance")):
        if returncode == 0:
            print(
                "    [Validator] Ignoring REPLAN with empty guidance — "
                "contract passed (returncode 0)."
            )
            parsed["verdict"] = "PASS"
            parsed.setdefault(
                "validation_details",
                "Contract command exited 0; ignored invalid REPLAN verdict.",
            )
        else:
            print("    [Validator] Ignoring REPLAN with empty guidance — treating as FAIL.")
            parsed["verdict"] = "FAIL"
            parsed.setdefault(
                "errors", ["Validator emitted REPLAN without replan_guidance."]
            )

    return parsed
