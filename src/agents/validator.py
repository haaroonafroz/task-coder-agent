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
from pathlib import Path, PurePosixPath
from typing import Any, Optional

from src.llm_client import call_llm, ModelChoice, resolve_model_config
from src.telemetry import span_llm_call, TelemetryContext
from src.sandbox.commands import execute_contract
from src.sandbox.context import get_sandbox_context
from src.sandbox.dependency_check import (
    check_target_file_dependencies,
    format_missing_dependency_message,
    planned_module_names,
)
from src.sandbox.policy import describe_policy_for_profile, format_policy_reference
from src.tools.paths import resolve_workspace_path
from src.tools.git_ops import git_diff
from src.events import EventEmitter
from src.agents.utils import parse_json_from_text, failure_signature
from src.agents.test_scaffold_validator import (
    build_python_stub_overlay,
    collect_contract,
    python_stub_env_overlay,
    red_phase_contract,
    validate_test_scaffold_structure,
)

# ---------------------------------------------------------------------------
# Paths and configuration
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).parent.parent.parent
_CONFIG_DIR = _ROOT / "config"

_VALIDATOR_MD = (_CONFIG_DIR / "validator.md").read_text()

MAX_TOKENS_VALIDATOR = int(os.getenv("MAX_TOKENS_VALIDATOR", "48000"))
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
    emitter: "EventEmitter | None" = None,
    session: Optional[TelemetryContext] = None,
    plan: Optional[dict] = None,
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
        emitter:           Optional EventEmitter for streaming validation events.
        session:           Optional telemetry context used to bind LLM spans to a
                           Phoenix session (Phase 5).
        plan:              Optional full mission plan — used to recognise
                           planned-but-not-yet-created local modules in the
                           dependency check (TDD red-phase imports).

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
        if emitter:
            emitter.emit(
                "validation.spec_gaming",
                milestone_id=ms_id,
                unauthorized_edits=unauthorized_edits,
            )
            emitter.emit("validation.finished", milestone_id=ms_id, verdict="FAIL")
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
            "failure_signature": failure_signature(
                ms_id, command, f"spec gaming: {unauthorized_edits}", None
            ),
        }

    boundary_fail = _target_file_boundary_fail(milestone, worker_result)
    if boundary_fail:
        print(
            f"    [Validator] REJECTED: Worker modified files outside target_files: "
            f"{boundary_fail['out_of_scope_files']}"
        )
        boundary_fail["failure_signature"] = failure_signature(
            ms_id, command,
            f"out of scope: {boundary_fail['out_of_scope_files']}", None,
        )
        if emitter:
            emitter.emit(
                "validation.spec_gaming",
                milestone_id=ms_id,
                unauthorized_edits=boundary_fail["out_of_scope_files"],
                reason="out_of_scope_target_files",
            )
            emitter.emit("validation.finished", milestone_id=ms_id, verdict="FAIL")
        return boundary_fail

    contract_type = str(contract.get("type", "")).lower()

    # --- Phase-aware test scaffold validation --------------------------------
    scaffold_replan = _test_scaffold_contract_replan(milestone, is_test_milestone)
    if scaffold_replan:
        return scaffold_replan

    if contract_type == "test_scaffold":
        if emitter:
            emitter.emit("validation.started", milestone_id=ms_id, command=command)
        scaffold_verdict = _run_test_scaffold_contract(
            milestone=milestone,
            emitter=emitter,
        )
        if emitter:
            emitter.emit(
                "validation.finished",
                milestone_id=ms_id,
                verdict=scaffold_verdict.get("verdict", "FAIL"),
                path="test_scaffold",
            )
        return scaffold_verdict

    if emitter:
        emitter.emit("validation.started", milestone_id=ms_id, command=command)

    # --- Run the validation contract command ---------------------------------
    contract_output = ""
    tool_result: dict = {}
    if command:
        print(f"    [Validator] Running: {command}")
        tool_result = execute_contract(
            contract,
            timeout=120,
            profile="validation",
        )
        exec_mode = tool_result.get("execution_mode", "unknown")
        exec_python = tool_result.get("python", "")
        print(
            f"    [Validator] mode={exec_mode} python={exec_python} "
            f"returncode={tool_result.get('returncode')}"
        )
        contract_output = (
            f"stdout:\n{tool_result.get('stdout', '')}\n"
            f"stderr:\n{tool_result.get('stderr', '')}\n"
            f"returncode: {tool_result.get('returncode', -1)}\n"
            f"python: {exec_python}\n"
            f"execution_mode: {exec_mode}\n"
            f"policy_denied: {tool_result.get('policy_denied', False)}"
        )
        if emitter:
            emitter.emit(
                "validation.contract_run",
                milestone_id=ms_id,
                returncode=tool_result.get("returncode", -1),
                python=exec_python,
                execution_mode=exec_mode,
                policy_denied=tool_result.get("policy_denied", False),
            )

    returncode = tool_result.get("returncode") if command else None

    # Failure fingerprint for replan dedup + raw output passthrough so the
    # runtime can hand the worker the un-digested contract output on retry.
    sig = failure_signature(ms_id, command, contract_output, returncode)

    def _with_failure_context(verdict: dict) -> dict:
        verdict.setdefault("failure_signature", sig)
        if contract_output:
            verdict.setdefault("contract_output", contract_output[:2000])
        return verdict

    # --- Policy denial (contract never ran) ----------------------------------
    policy_replan = _detect_policy_denial_replan(
        milestone, contract, tool_result, contract_output
    )
    if policy_replan:
        print(f"    [Validator] Policy denial REPLAN: {policy_replan['replan_guidance'][:120]}…")
        if emitter:
            emitter.emit(
                "validation.finished",
                milestone_id=ms_id,
                verdict="REPLAN",
                path="policy_denied",
            )
        return _with_failure_context(policy_replan)

    # --- Deterministic REPLAN check ------------------------------------------
    replan = _detect_contract_replan(milestone, worker_result, contract_output, returncode)
    if replan:
        print(f"    [Validator] Deterministic REPLAN: {replan['replan_guidance']}")
        if emitter:
            emitter.emit(
                "validation.finished",
                milestone_id=ms_id,
                verdict="REPLAN",
                reason="deterministic",
            )
        return _with_failure_context(replan)
    
    # --- TDD red-phase PASS (test-scaffolding milestone) ---------------------
    tdd_pass = _detect_tdd_red_pass(
        milestone, worker_result, contract_output, returncode, is_test_milestone
    )
    if tdd_pass:
        print(f"    [Validator] TDD red-phase PASS: {tdd_pass['validation_details']}")
        if emitter:
            emitter.emit(
                "validation.finished",
                milestone_id=ms_id,
                verdict="PASS",
                path="tdd_red",
            )
        return tdd_pass

    # --- Spec-gaming: all tests skipped on scaffolding milestone -------------
    skip_fail = _detect_all_skipped_spec_gaming(
        milestone, worker_result, contract_output, is_test_milestone
    )
    if skip_fail:
        print(f"    [Validator] Spec-gaming FAIL: all tests skipped")
        if emitter:
            emitter.emit(
                "validation.finished",
                milestone_id=ms_id,
                verdict="FAIL",
                path="all_skipped",
            )
        return _with_failure_context(skip_fail)

    # --- Collect-only PASS (test-scaffolding milestone) ----------------------
    collect_pass = _detect_collect_only_pass(
        milestone, contract, contract_output, returncode, is_test_milestone
    )
    if collect_pass:
        print(f"    [Validator] Collect-only PASS: {collect_pass['validation_details']}")
        if emitter:
            emitter.emit(
                "validation.finished",
                milestone_id=ms_id,
                verdict="PASS",
                path="collect_only",
            )
        return collect_pass

    # --- Fast-path PASS (contract exited 0) ----------------------------------
    if returncode == 0:
        dep_fail = _missing_dependency_fail(
            milestone,
            target_files,
            emitter=emitter,
            planned_modules=planned_module_names(plan),
        )
        if dep_fail:
            return _with_failure_context(dep_fail)

        if emitter:
            emitter.emit(
                "validation.finished",
                milestone_id=ms_id,
                verdict="PASS",
                path="fast_path",
            )
        return {
            "verdict": "PASS",
            "milestone_id": ms_id,
            "validation_details": "Contract command exited 0.",
        }

    # --- LLM verdict for non-zero exit or commandless milestones -------------
    diff_block = _bounded_workspace_diff()
    prompt = (
        f"{_VALIDATOR_MD}\n\n"
        f"---\n\n"
        f"## Milestone Under Review\n"
        f"**ID**: {ms_id}\n"
        f"**Title**: {milestone.get('title', '')}\n"
        f"**Description**: {milestone.get('description', '')}\n"
        f"**Target files (worker may ONLY edit these)**: {target_files}\n\n"
        f"**Test-scaffolding milestone**: {is_test_milestone}\n\n"
        f"## Validation Contract\n```json\n{json.dumps(contract, indent=2)}\n```\n\n"
        f"## Contract Execution Output\n```\n{contract_output or '(no command run)'}\n```\n\n"
        f"{diff_block}"
        f"## Worker Handoff\n"
        f"Files modified: {worker_result.get('files_modified', [])}\n"
        f"Summary: {worker_result.get('summary', '')}\n\n"
        f"**Retry attempt**: {retry_count + 1} of {MAX_RETRY_CYCLES}\n"
    )

    if "policy_denied: True" in (contract_output or ""):
        prompt += (
            f"\n## Sandbox Policy Reference (validation profile)\n"
            f"{format_policy_reference('validation')}\n"
        )

    prompt += (
        f"\nEmit your PASS, FAIL, or REPLAN JSON verdict now:"
    )

    span_model = (
        resolve_model_config(model, "validator").model_name
        if model != "auto" else model
    )
    with span_llm_call("validator", ms_id, span_model, session=session):
        result = call_llm(
            prompt, model=model, max_tokens=MAX_TOKENS_VALIDATOR,
            json_mode=True, role="validator",
        )

    if emitter:
        emitter.emit(
            "llm.call",
            role="validator",
            milestone_id=ms_id,
            model_used=result.model_used,
            tokens_prompt=result.tokens_prompt,
            tokens_generated=result.tokens_generated,
            prefill_ms=result.prefill_ms,
            decode_ms=result.decode_ms,
            total_ms=result.total_ms,
            thinking_level=result.thinking_level,
            fallback_used=result.fallback_used,
        )

    parsed = parse_json_from_text(result.text)
    if parsed is None:
        if command and "returncode: 0" in contract_output:
            if emitter:
                emitter.emit(
                    "validation.finished",
                    milestone_id=ms_id,
                    verdict="PASS",
                    path="unparseable_contract_ok",
                )
            return {
                "verdict": "PASS",
                "milestone_id": ms_id,
                "validation_details": "Contract command exited 0.",
            }
        if emitter:
            emitter.emit(
                "validation.finished",
                milestone_id=ms_id,
                verdict="FAIL",
                path="unparseable",
            )
        return _with_failure_context({
            "verdict": "FAIL",
            "milestone_id": ms_id,
            "errors": ["Could not parse validator response."],
            "fix_guidance": result.text[:500],
        })

    parsed = _normalize_validator_verdict(parsed, returncode=returncode)
    parsed = _block_infrastructure_replan(parsed, contract_output, returncode)

    # Contract success is authoritative — don't let the LLM REPLAN over a green run
    if returncode == 0 and parsed.get("verdict") == "REPLAN":
        print("    [Validator] Overriding REPLAN — contract command returned 0.")
        parsed["verdict"] = "PASS"
        parsed.setdefault("validation_details", "Contract command exited 0.")

    if emitter:
        emitter.emit(
            "validation.finished",
            milestone_id=ms_id,
            verdict=parsed.get("verdict", "FAIL"),
            path="llm",
        )

    return _with_failure_context(parsed)


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

_DIFF_MAX_CHARS = 4000


def _bounded_workspace_diff() -> str:
    """
    Return the uncommitted workspace diff as a prompt block (size-capped).

    The validator must judge spec-gaming and root cause from the ACTUAL code
    changes, not the worker's self-reported summary. Missing diff is not
    fatal — the block is simply omitted.
    """
    try:
        result = git_diff()
    except Exception:
        return ""
    if not result.get("success"):
        return ""
    diff_text = result.get("diff", "") or ""
    if not diff_text.strip() or diff_text.startswith("(no"):
        return ""
    if len(diff_text) > _DIFF_MAX_CHARS:
        head = diff_text[:3000]
        tail = diff_text[-900:]
        diff_text = f"{head}\n... [diff truncated {len(diff_text)} chars total] ...\n{tail}"
    return f"## Workspace Diff (uncommitted changes)\n```diff\n{diff_text}\n```\n\n"

def _run_test_scaffold_contract(
    milestone: dict,
    emitter: Optional[EventEmitter] = None,
) -> dict:
    """
    Validate a spec/test milestone as an external contract.

    This is intentionally stricter than pytest collect-only: it rejects tests
    that embed production objects, then collects and red-runs the tests against
    temporary generated stubs for the declared public API.
    """
    ms_id = milestone.get("id", "?")
    workspace_root = resolve_workspace_path(".")
    structural = validate_test_scaffold_structure(milestone, workspace_root)
    if not structural.ok:
        verdict = "REPLAN" if structural.needs_replan else "FAIL"
        result = {
            "verdict": verdict,
            "milestone_id": ms_id,
            "errors": structural.errors,
            "warnings": structural.warnings,
            "root_cause": (
                "The test-scaffolding contract is underspecified."
                if structural.needs_replan
                else "The test scaffold embeds implementation details or is not a valid external spec."
            ),
            "fix_guidance": (
                "Patch the plan with language, public_api, required_imports, and forbidden_definitions."
                if structural.needs_replan
                else (
                    "Rewrite tests so they import the public API from source modules and do not "
                    "define production classes, enums, functions, or methods inside test files."
                )
            ),
        }
        if structural.needs_replan:
            result["replan_guidance"] = result["fix_guidance"]
        return result

    sandbox = get_sandbox_context()
    if sandbox is None:
        return {
            "verdict": "FAIL",
            "milestone_id": ms_id,
            "errors": ["No active sandbox context for test_scaffold validation."],
            "fix_guidance": "Run scaffold validation inside a session sandbox.",
        }

    stub_root = sandbox.tmp_dir / "validation_stubs" / str(ms_id)
    try:
        build_python_stub_overlay(milestone, sandbox.workspace_root, stub_root)
    except Exception as exc:  # noqa: BLE001
        return {
            "verdict": "REPLAN",
            "milestone_id": ms_id,
            "errors": [f"Could not build test scaffold stubs: {exc}"],
            "root_cause": "The public_api contract is invalid or incomplete.",
            "fix_guidance": "Patch public_api with valid Python module paths and symbol names.",
            "replan_guidance": (
                f"Fix public_api for {ms_id}; it must contain valid Python module paths, "
                "symbol names, kinds, and class methods/enums where needed."
            ),
        }

    env_overlay = python_stub_env_overlay(stub_root, sandbox.workspace_root)

    collect_result = execute_contract(
        collect_contract(milestone),
        timeout=120,
        profile="validation",
        env_overlay=env_overlay,
    )
    if emitter:
        emitter.emit(
            "validation.contract_run",
            milestone_id=ms_id,
            phase="test_scaffold_collect",
            returncode=collect_result.get("returncode", -1),
            python=collect_result.get("python", ""),
            execution_mode=collect_result.get("execution_mode", "unknown"),
            policy_denied=collect_result.get("policy_denied", False),
        )

    if collect_result.get("returncode") != 0:
        return {
            "verdict": "FAIL",
            "milestone_id": ms_id,
            "errors": ["Test scaffold did not collect against generated public API stubs."],
            "validation_details": _format_tool_output(collect_result),
            "fix_guidance": (
                "Fix import names, syntax, fixtures, or test structure so the scaffold "
                "collects against the declared public API stubs."
            ),
        }

    red_result = execute_contract(
        red_phase_contract(milestone),
        timeout=120,
        profile="validation",
        env_overlay=env_overlay,
    )
    if emitter:
        emitter.emit(
            "validation.contract_run",
            milestone_id=ms_id,
            phase="test_scaffold_red",
            returncode=red_result.get("returncode", -1),
            python=red_result.get("python", ""),
            execution_mode=red_result.get("execution_mode", "unknown"),
            policy_denied=red_result.get("policy_denied", False),
        )

    if red_result.get("returncode") == 0:
        return {
            "verdict": "FAIL",
            "milestone_id": ms_id,
            "errors": ["Tests pass against empty generated stubs."],
            "validation_details": _format_tool_output(red_result),
            "root_cause": "The scaffold does not assert behavior strongly enough.",
            "fix_guidance": (
                "Add behavioral assertions that fail while public API symbols are only stubs."
            ),
        }

    if red_result.get("policy_denied"):
        return {
            "verdict": "REPLAN",
            "milestone_id": ms_id,
            "errors": [red_result.get("stderr", "Policy denied")],
            "root_cause": "The scaffold red-phase command is not permitted by sandbox policy.",
            "fix_guidance": "Use a permitted pytest validation command.",
            "replan_guidance": (
                f"Update {ms_id} test_scaffold command to use python -m pytest "
                "<test_file> --collect-only -q or a structured pytest target/args contract."
            ),
        }

    return {
        "verdict": "PASS",
        "milestone_id": ms_id,
        "validation_details": (
            "Test scaffold imports the declared public API, avoids embedded implementation, "
            "collects against generated stubs, and fails red-phase as expected."
        ),
        "warnings": structural.warnings,
    }


def _format_tool_output(result: dict) -> str:
    return (
        f"stdout:\n{result.get('stdout', '')}\n"
        f"stderr:\n{result.get('stderr', '')}\n"
        f"returncode: {result.get('returncode', -1)}\n"
        f"execution_mode: {result.get('execution_mode', 'unknown')}"
    )


def _test_scaffold_contract_replan(
    milestone: dict,
    is_test_milestone: bool,
) -> dict | None:
    """Require phase-aware contracts for all test-writing milestones."""
    if not is_test_milestone:
        return None
    contract = milestone.get("validation_contract", {}) or {}
    if str(contract.get("type", "")).lower() == "test_scaffold":
        return None

    ms_id = milestone.get("id", "?")
    return {
        "verdict": "REPLAN",
        "milestone_id": ms_id,
        "errors": [
            (
                "Test-scaffolding milestones must use validation_contract.type="
                "'test_scaffold' so the harness can validate tests as external specs."
            )
        ],
        "root_cause": "The plan used an implementation-style validation contract for a spec milestone.",
        "fix_guidance": None,
        "replan_guidance": (
            f"Patch {ms_id} to use validation_contract.type='test_scaffold'. "
            "Add language, public_api, required_imports, forbidden_definitions, "
            "and min_assertions. Tests must import the future implementation API "
            "and must not define production objects inside tests."
        ),
    }


def _missing_dependency_fail(
    milestone: dict,
    target_files: list[str],
    *,
    emitter: "EventEmitter | None" = None,
    planned_modules: frozenset[str] = frozenset(),
) -> dict | None:
    """FAIL when target files import third-party packages not installed in the venv."""
    if not target_files:
        return None

    report = check_target_file_dependencies(
        target_files, planned_modules=planned_modules
    )
    if report.ok:
        return None

    ms_id = milestone.get("id", "?")
    message = format_missing_dependency_message(report)
    print(f"    [Validator] REJECTED: {message}")

    if emitter:
        emitter.emit(
            "dependency.missing",
            milestone_id=ms_id,
            packages=report.missing_packages,
            imports=report.missing_imports,
            checked_files=report.checked_files,
            source="validator",
        )
        emitter.emit("validation.finished", milestone_id=ms_id, verdict="FAIL")

    return {
        "verdict": "FAIL",
        "milestone_id": ms_id,
        "errors": [message, *report.errors],
        "root_cause": (
            "Target files import third-party packages that are not installed in "
            "the session venv."
        ),
        "fix_guidance": (
            "Call install_dependency for each missing package "
            f"({', '.join(report.missing_packages) or 'see errors'}), then ensure "
            "target files import successfully before signalling complete."
        ),
    }


def _target_file_boundary_fail(
    milestone: dict,
    worker_result: dict,
) -> dict | None:
    """Fail milestones that modify files outside their declared target_files."""
    target_files = {
        _normalize_workspace_rel(path)
        for path in milestone.get("target_files", [])
        if str(path).strip()
    }
    modified_files = {
        _normalize_workspace_rel(path)
        for path in worker_result.get("files_modified", [])
        if str(path).strip()
    }
    if not target_files or not modified_files:
        return None

    out_of_scope = sorted(modified_files - target_files)
    if not out_of_scope:
        return None

    ms_id = milestone.get("id", "?")
    return {
        "verdict": "FAIL",
        "milestone_id": ms_id,
        "errors": [
            (
                "Worker modified files outside this milestone's target_files: "
                f"{out_of_scope}. Allowed target_files: {sorted(target_files)}"
            )
        ],
        "out_of_scope_files": out_of_scope,
        "root_cause": (
            "The worker exceeded the milestone boundary instead of limiting edits "
            "to the files assigned by the orchestrator."
        ),
        "fix_guidance": (
            "Revert out-of-scope changes and complete this milestone using only "
            f"the declared target_files: {sorted(target_files)}."
        ),
    }


def _normalize_workspace_rel(path: object) -> str:
    raw = str(path).strip().replace("\\", "/")
    while raw.startswith("./"):
        raw = raw[2:]
    if raw.startswith("workspace/"):
        raw = raw[len("workspace/"):]
    normalized = PurePosixPath(raw)
    parts = [part for part in normalized.parts if part not in ("", ".")]
    return str(PurePosixPath(*parts)) if parts else "."

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


def _detect_policy_denial_replan(
    milestone: dict,
    contract: dict,
    tool_result: dict,
    contract_output: str,
) -> dict | None:
    """
    Deterministic REPLAN when the sandbox policy blocked contract execution.

    Surfaces the denial reason and the validation allowlist so the orchestrator
    can rewrite the contract using permitted commands.
    """
    if not tool_result.get("policy_denied"):
        return None

    reason = tool_result.get("stderr", "Policy denied (no reason provided)")
    policy_ref = describe_policy_for_profile("validation")
    command = contract.get("command", "")
    ms_id = milestone.get("id", "?")

    suggested = "python -m pytest <test_file> --collect-only -q"
    if contract.get("type") == "pytest" or "pytest" in command:
        suggested = command
    elif "tests/" in command:
        import re as _re
        match = _re.search(r"tests/\S+\.py", command)
        if match:
            suggested = f"python -m pytest {match.group(0)} --collect-only -q"

    replan_guidance = (
        f"Sandbox policy denied the validation contract — the command never executed.\n"
        f"Denial reason: {reason}\n\n"
        f"Allowed shell commands: {', '.join(policy_ref['shell_commands'])}\n"
        f"Allowed python -m modules: {', '.join(policy_ref['python_modules'])}\n"
        f"Recommended replacements:\n"
        + "\n".join(f"  - {ex}" for ex in policy_ref["recommended_contracts"])
        + f"\n\nUpdate {ms_id} validation_contract.command to a permitted form. "
        f"For test-scaffolding milestones prefer: `{suggested}`. "
        f"Do not use grep, pip install, curl, or bash eval/exec. "
        f"Do not try to bypass policy with string obfuscation."
    )

    return {
        "verdict": "REPLAN",
        "milestone_id": ms_id,
        "validation_details": (
            "Validation contract was blocked by sandbox policy before execution."
        ),
        "errors": [reason],
        "root_cause": (
            "The orchestrator-authored validation_contract.command uses commands "
            "or patterns not permitted in the validation sandbox profile."
        ),
        "fix_guidance": None,
        "replan_guidance": replan_guidance,
        "policy_reference": policy_ref,
    }


def _pytest_ran_in_output(contract_output: str, returncode: int | None) -> bool:
    """True when contract output shows pytest actually executed."""
    out = contract_output.lower()
    if "test session starts" in out:
        return True
    if "tests collected" in out or "test collected" in out:
        return True
    if "error collecting" in out:
        return True
    if "short test summary" in out:
        return True
    if "collected" in out and "error" in out:
        return True
    if returncode in (0, 1, 2, 5) and "pytest" in out:
        return True
    if " passed" in out and "failed" in out:
        return True
    if returncode in (0, 1, 2, 5) and "execution_mode: argv" in out:
        # argv-compiled pytest contracts always invoke pytest
        return True
    return False


def _detect_tdd_red_pass(
    milestone: dict,
    worker_result: dict,
    contract_output: str,
    returncode: int | None,
    is_test_milestone: bool,
) -> dict | None:
    """
    PASS test-scaffolding milestones in TDD red phase.

    Fires when pytest ran, some tests passed (oracle/baseline), and remaining
    failures are only because the implementation module does not exist yet.
    """
    if not is_test_milestone:
        return None
    if returncode != 1:
        return None
    if not _pytest_ran_in_output(contract_output, returncode):
        return None

    out_lower = contract_output.lower()

    # Real infrastructure failures — do not PASS
    if "no module named pytest" in out_lower:
        return None
    if "policy denied" in out_lower:
        return None

    # Expect missing implementation import, not broken test logic
    if "modulenotfounderror" not in out_lower and "no module named" not in out_lower:
        return None

    target_files = milestone.get("target_files", [])
    modified = worker_result.get("files_modified", [])
    touched_tests = any("tests/" in f for f in modified) or any(
        "tests/" in f for f in target_files
    )
    if not touched_tests:
        return None

    return {
        "verdict": "PASS",
        "milestone_id": milestone.get("id"),
        "validation_details": (
            "TDD red phase: pytest ran successfully; oracle/baseline tests passed; "
            "target tests fail only because the implementation module is not present yet."
        ),
    }


def _detect_collect_only_pass(
    milestone: dict,
    contract: dict,
    contract_output: str,
    returncode: int | None,
    is_test_milestone: bool,
) -> dict | None:
    """
    PASS test-scaffolding milestones when --collect-only discovers tests.

    Fires on exit 0 or when output reports N tests collected.
    """
    if not is_test_milestone:
        return None

    command = contract.get("command", "")
    if "--collect-only" not in command and contract.get("mode") != "collect-only":
        return None

    if returncode not in (0, 5) and not _pytest_ran_in_output(contract_output, returncode):
        return None

    collected_match = re.search(r"(\d+)\s+tests?\s+collected", contract_output, re.IGNORECASE)
    if not collected_match and returncode != 0:
        return None

    count = int(collected_match.group(1)) if collected_match else 0
    min_tests = 1
    pass_criteria = contract.get("pass_criteria", "")
    criteria_match = re.search(r"at least\s+(\d+)", pass_criteria, re.IGNORECASE)
    if criteria_match:
        min_tests = int(criteria_match.group(1))

    if count < min_tests and returncode != 0:
        return None

    return {
        "verdict": "PASS",
        "milestone_id": milestone.get("id"),
        "validation_details": (
            f"Test scaffolding: pytest collected {count or 'tests'} successfully "
            f"(collect-only contract)."
        ),
    }


def _detect_all_skipped_spec_gaming(
    milestone: dict,
    worker_result: dict,
    contract_output: str,
    is_test_milestone: bool,
) -> dict | None:
    """FAIL when a test-scaffolding milestone marks every test as skipped."""
    if not is_test_milestone:
        return None

    modified = worker_result.get("files_modified", [])
    target_files = milestone.get("target_files", [])
    touched_tests = any("tests/" in f for f in modified) or any(
        "tests/" in f for f in target_files
    )
    if not touched_tests:
        return None

    for rel_path in set(modified) | set(target_files):
        if "tests/" not in rel_path:
            continue
        try:
            content = resolve_workspace_path(rel_path).read_text(encoding="utf-8")
        except (OSError, ValueError):
            continue
        if "@pytest.mark.skip" not in content:
            return None
        test_defs = re.findall(r"^\s*def\s+test_\w+", content, re.MULTILINE)
        skip_marks = content.count("@pytest.mark.skip")
        if test_defs and skip_marks >= len(test_defs):
            return {
                "verdict": "FAIL",
                "milestone_id": milestone.get("id"),
                "errors": [
                    "Specification gaming: all tests are marked @pytest.mark.skip; "
                    "tests must exercise the target API, not be deferred."
                ],
                "root_cause": (
                    "Worker skipped every test instead of writing real assertions "
                    "for the TDD red phase."
                ),
                "fix_guidance": (
                    "Remove @pytest.mark.skip decorators. Write real tests that import "
                    "the target module and assert expected behavior. Missing "
                    "implementation should fail with import errors, not skipped tests."
                ),
            }

    return None


def _toolchain_evidence_in_output(
    contract_output: str,
    returncode: int | None,
) -> bool:
    """True when contract output shows the session toolchain actually ran."""
    if _pytest_ran_in_output(contract_output, returncode):
        return True
    return "execution_mode: argv" in contract_output.lower()


def _is_shell_interpreter_miss(
    contract_output: str,
    returncode: int | None,
) -> bool:
    """True for shell-mode exit 127 — usually a harness path issue, not a broken venv."""
    if returncode != 127:
        return False
    return "execution_mode: shell" in contract_output.lower()


def _block_infrastructure_replan(
    parsed: dict,
    contract_output: str,
    returncode: int | None,
) -> dict:
    """
    Downgrade LLM REPLANs that blame missing pytest/pip/venv when the toolchain
    already ran, or when shell-mode exit 127 indicates a contract execution path
    issue rather than a broken session environment.
    """
    if parsed.get("verdict") != "REPLAN":
        return parsed

    guidance = str(parsed.get("replan_guidance", "")).lower()
    shell_127 = _is_shell_interpreter_miss(contract_output, returncode)
    infra_markers = (
        "pip install",
        "install pytest",
        "environment setup",
        "testing framework",
        "not installed",
        "correctly installed",
        "virtual environment",
        "virtual environments",
        "path is configured",
        "interpreter",
        "interpreter path",
        "missing interpreter",
        "pytest is available",
        "requirements.txt installation",
        "container/ci environment",
        "infrastructure",
        "sandbox",
        "symlink",
        "command not found",
        "no such file or directory",
        "misconfigured",
        "broken environment",
        "environment's virtual environment",
        "validation environment",
    )
    has_infra_guidance = any(m in guidance for m in infra_markers)

    if not has_infra_guidance and not shell_127:
        return parsed

    if not _toolchain_evidence_in_output(contract_output, returncode) and not shell_127:
        return parsed

    if shell_127:
        print(
            "    [Validator] Blocking infrastructure REPLAN — shell-mode exit 127 "
            "is a contract execution-path issue, not a missing venv."
        )
        parsed["verdict"] = "FAIL"
        parsed["replan_guidance"] = None
        parsed.setdefault("errors", []).append(
            "Rejected infrastructure REPLAN: validation ran in shell mode and "
            "returned exit 127. Prefer `python -m pytest`, `python -m py_compile`, "
            "or `python -m flake8` so the harness compiles the contract to argv."
        )
        parsed["root_cause"] = (
            parsed.get("root_cause")
            or (
                "Validator misclassified a shell-mode command-not-found failure "
                "as a broken session environment."
            )
        )
        return parsed

    print(
        "    [Validator] Blocking infrastructure REPLAN — session toolchain "
        "evidence is present in contract output."
    )
    parsed["verdict"] = "FAIL"
    parsed["replan_guidance"] = None
    parsed.setdefault("errors", []).append(
        "Rejected infrastructure REPLAN: session toolchain is available; failure "
        "is not missing tooling."
    )
    parsed["root_cause"] = (
        parsed.get("root_cause")
        or "Validator misclassified a normal test failure as missing infrastructure."
    )
    return parsed

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
