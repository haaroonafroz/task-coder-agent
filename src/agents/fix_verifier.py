"""Focused FixVerifier for bounded hotfixes.

Uses deterministic checks (scope, py_compile, lint on touched files, targeted
smoke) plus a small LLM verdict. Missing pytest/heavy deps => UNVERIFIED, never
an automatic mission REPLAN.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from src.agents.contracts import normalize_fix_verification
from src.agents.llm_stream_events import stream_context_for
from src.agents.utils import parse_json_from_text
from src.events import EventEmitter
from src.llm_client import ModelChoice, call_llm, resolve_model_config
from src.telemetry import TelemetryContext, span_llm_call
from src.tools import dispatch
from src.tools.git_ops import git_diff

_ROOT = Path(__file__).parent.parent.parent
_FIX_VERIFIER_MD = (_ROOT / "config" / "fix_verifier.md").read_text(encoding="utf-8")
_MAX_TOKENS = 8192


def _bounded_diff(max_chars: int = 6000) -> str:
    try:
        result = git_diff()
    except Exception:
        return ""
    if not result.get("success"):
        return ""
    text = result.get("diff", "") or ""
    if not text.strip() or text.startswith("(no"):
        return ""
    return text[:max_chars]


def _deterministic_checks(target_files: list[str]) -> dict[str, Any]:
    """Run cheap, targeted checks on touched files only."""
    checks: list[dict[str, Any]] = []
    for path in (target_files or [])[:6]:
        if not path.endswith(".py"):
            continue
        compiled = dispatch(
            "run_shellscript",
            {"script": f"python -m py_compile {path}", "profile": "fixverify"},
        )
        checks.append({"check": f"py_compile {path}", **_compact(compiled)})
        linted = dispatch("run_linter", {"target_path": path})
        checks.append({"check": f"lint {path}", **_compact(linted)})
    return {"checks": checks}


def _compact(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "success": result.get("success", False),
        "returncode": result.get("returncode"),
        "stdout": str(result.get("stdout", ""))[:800],
        "stderr": str(result.get("stderr", ""))[:800],
        "policy_denied": result.get("policy_denied", False),
    }


def _fix_criteria(
    review_report: dict[str, Any],
    hotfix_packet: dict[str, Any],
) -> list[str]:
    """Collect the fix criteria the verifier judges against.

    Prefers high-confidence review findings; falls back to the packet's own
    acceptance criteria (pure-hotfix route has no review report).
    """
    actionable = [
        f
        for f in (review_report or {}).get("findings", [])
        if isinstance(f, dict)
        and f.get("severity") in {"blocker", "bug"}
        and f.get("confidence") == "high"
    ]
    criteria = [c for f in actionable for c in (f.get("fix_criteria", []) or [])]
    if not criteria:
        criteria = [
            str(c)
            for c in (hotfix_packet or {}).get("acceptance_criteria", [])
            if str(c).strip()
        ]
    return criteria


def run_fix_verification(
    *,
    review_report: dict[str, Any],
    hotfix_packet: dict[str, Any],
    hotfix_handoff: dict[str, Any],
    model: ModelChoice = "auto",
    session: Optional[TelemetryContext] = None,
    emitter: Optional[EventEmitter] = None,
) -> dict[str, Any]:
    if emitter:
        emitter.emit("fixverify.started")
    target_files = list((hotfix_packet or {}).get("target_files", []))
    deterministic = _deterministic_checks(target_files)
    diff = _bounded_diff()
    criteria = _fix_criteria(review_report, hotfix_packet)
    prompt = (
        f"{_FIX_VERIFIER_MD}\n\n---\n\n"
        f"## Fix criteria\n" + "".join(f"- {c}\n" for c in criteria[:12])
        + f"\n## Target files\n{target_files}\n\n"
        f"## Files modified\n{(hotfix_handoff or {}).get('files_modified', [])}\n\n"
        f"## Deterministic checks\n```json\n{json.dumps(deterministic, indent=2)[:5000]}\n```\n\n"
        f"## Diff\n```diff\n{diff[:6000]}\n```\n\n"
        "Emit the verification JSON now."
    )
    span_model = (
        resolve_model_config(model, "validator").model_name
        if model != "auto"
        else model
    )
    try:
        with span_llm_call("fixverify", "verify", span_model, session=session):
            result = call_llm(
                prompt,
                model=model,
                max_tokens=_MAX_TOKENS,
                json_mode=True,
                role="validator",
                stream_context=stream_context_for(
                    emitter, "fixverify", output_kind="json"
                ),
            )
        parsed = parse_json_from_text(result.text)
        verdict = normalize_fix_verification(parsed).to_dict()
    except Exception as exc:
        verdict = normalize_fix_verification(None).to_dict()
        verdict["summary"] = f"Verifier LLM failed ({exc}); treating as unverified."
        verdict["missing_checks"] = ["verifier LLM unavailable"]
    verdict["deterministic_checks"] = deterministic
    if emitter:
        emitter.emit(
            "fixverify.completed",
            verdict=verdict.get("verdict"),
            summary=str(verdict.get("summary", ""))[:500],
        )
    return verdict
