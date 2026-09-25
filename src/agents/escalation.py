"""Escalation triage: LLM decides stay_hotfix vs escalate_mission.

File count is a guardrail, never the decision. The model judges whether the
remaining work is localized or structural from review findings + diff context.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from src.agents.contracts import normalize_escalation_decision
from src.agents.llm_stream_events import stream_context_for
from src.agents.utils import parse_json_from_text
from src.events import EventEmitter
from src.llm_client import ModelChoice, call_llm, resolve_model_config
from src.telemetry import TelemetryContext, span_llm_call

_ROOT = Path(__file__).parent.parent.parent
_ESCALATION_MD = (_ROOT / "config" / "escalation.md").read_text(encoding="utf-8")
_MAX_TOKENS = 4096


def _summarize_findings(report: dict[str, Any]) -> str:
    findings = report.get("findings", []) if isinstance(report, dict) else []
    lines = []
    for f in findings[:12]:
        if not isinstance(f, dict):
            continue
        lines.append(
            f"- [{f.get('severity')}/{f.get('confidence')}] {f.get('title')}: "
            f"files={f.get('affected_files', [])}"
        )
    return "\n".join(lines) or "(no findings)"


def run_escalation_triage(
    *,
    review_report: Optional[dict[str, Any]] = None,
    hotfix_packet: Optional[dict[str, Any]] = None,
    hotfix_handoff: Optional[dict[str, Any]] = None,
    verification: Optional[dict[str, Any]] = None,
    diff_summary: str = "",
    project_profile: Optional[dict[str, Any]] = None,
    model: ModelChoice = "auto",
    session: Optional[TelemetryContext] = None,
    emitter: Optional[EventEmitter] = None,
) -> dict[str, Any]:
    """Return a normalized escalation decision dict."""
    if emitter:
        emitter.emit("escalation.started")
    report = review_report or {}
    prompt = (
        f"{_ESCALATION_MD}\n\n---\n\n"
        f"## Review summary\n{str(report.get('summary', ''))[:4000]}\n\n"
        f"## Findings\n{_summarize_findings(report)}\n\n"
        f"## Hotfix packet\n```json\n{json.dumps(hotfix_packet or {}, indent=2)[:4000]}\n```\n\n"
        f"## Hotfix handoff\n```json\n{json.dumps(hotfix_handoff or {}, indent=2)[:3000]}\n```\n\n"
        f"## Verification\n```json\n{json.dumps(verification or {}, indent=2)[:3000]}\n```\n\n"
        f"## Diff summary\n{diff_summary[:4000]}\n\n"
        f"## Project\n```json\n{json.dumps(project_profile or {}, indent=2)[:3000]}\n```\n\n"
        "Emit the escalation JSON now."
    )
    span_model = (
        resolve_model_config(model, "triage").model_name if model != "auto" else model
    )
    try:
        with span_llm_call("escalation", "triage", span_model, session=session):
            result = call_llm(
                prompt,
                model=model,
                max_tokens=_MAX_TOKENS,
                json_mode=True,
                role="triage",
                stream_context=stream_context_for(
                    emitter, "escalation", output_kind="json"
                ),
            )
        parsed = parse_json_from_text(result.text)
        decision = normalize_escalation_decision(parsed).to_dict()
    except Exception as exc:
        decision = normalize_escalation_decision(None).to_dict()
        decision["reason"] = f"Escalation LLM failed ({exc}); defaulting to bounded fix."
        decision["source"] = "deterministic"
    if emitter:
        emitter.emit(
            "escalation.completed",
            decision=decision.get("decision"),
            scope_type=decision.get("scope_type"),
            risk=decision.get("risk"),
            reason=decision.get("reason", "")[:500],
        )
    return decision


def deterministic_guard(
    decision: dict[str, Any],
    *,
    target_files: list[str],
) -> dict[str, Any]:
    """Apply safety guardrails without letting file count decide alone.

    - Architectural scope from the LLM is respected.
    - Sensitive areas (migrations, public API, build config) require confirmation:
      downgrade a stay_hotfix to escalate only when risk is high.
    """
    out = dict(decision)
    sensitive = ("migrations/", "api/", "setup.py", "pyproject.toml", "package.json")
    touches_sensitive = any(
        any(s in f for s in sensitive) for f in (target_files or [])
    )
    if (
        out.get("decision") == "stay_hotfix"
        and touches_sensitive
        and out.get("risk") == "high"
    ):
        out["decision"] = "escalate_mission"
        out["source"] = "deterministic"
        out["reason"] = (
            "Sensitive area with high risk; escalating for planned change. "
            + out.get("reason", "")
        )
    return out
