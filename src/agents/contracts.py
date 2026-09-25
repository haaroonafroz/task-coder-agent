"""Typed contracts shared by request routing and focused agent profiles."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal, Optional

ExecutionRoute = Literal["mission", "hotfix", "review"]
RouteOverride = Literal["auto", "mission", "hotfix", "review"]
Confidence = Literal["high", "medium", "low"]


def _strings(value: Any, *, limit: int = 20) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()][:limit]


def _workspace_relative_files(value: Any, workspace_root: Path) -> list[str]:
    files: list[str] = []
    root = workspace_root.resolve()
    for raw in _strings(value, limit=12):
        candidate = Path(raw)
        if candidate.is_absolute():
            try:
                candidate = candidate.resolve().relative_to(root)
            except (OSError, ValueError):
                continue
        text = str(candidate).replace("\\", "/")
        if text.startswith("workspace/"):
            text = text[len("workspace/"):]
        if text and text != "." and ".." not in Path(text).parts and text not in files:
            files.append(text)
    return files


@dataclass
class RouteDecision:
    route: ExecutionRoute
    confidence: Confidence
    rationale: str
    candidate_files: list[str] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    validation_intent: list[str] = field(default_factory=list)
    review_scope: str = "user_request"
    source: str = "triage"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class HotfixResult:
    status: str
    diagnosis: str
    files_modified: list[str] = field(default_factory=list)
    summary: str = ""
    requested_scope: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def normalize_hotfix_result(payload: Any) -> HotfixResult:
    if not isinstance(payload, dict):
        raise ValueError("hotfix result must be an object")
    status = str(payload.get("status", "blocked")).strip().lower()
    if status not in {"complete", "blocked", "request_scope", "cancelled"}:
        status = "blocked"
    reason = str(payload.get("reason", "")).strip()
    summary = str(payload.get("summary", "")).strip()
    return HotfixResult(
        status=status,
        diagnosis=reason or summary or "No diagnosis supplied.",
        files_modified=_strings(payload.get("files_modified", [])),
        summary=summary,
        requested_scope=_strings(
            payload.get("requested_paths", payload.get("requested_files", []))
        ),
    )


def normalize_route_decision(
    report: Optional[dict[str, Any]],
    *,
    workspace_root: Path,
    requested_route: str = "auto",
) -> RouteDecision:
    """Validate a triage response and apply deterministic safe fallbacks."""
    report = report or {}
    raw_route = str(report.get("route", "mission")).strip().lower()
    confidence = str(report.get("confidence", "low")).strip().lower()
    if confidence not in {"high", "medium", "low"}:
        confidence = "low"
    route: ExecutionRoute = (
        raw_route if raw_route in {"mission", "hotfix", "review"} else "mission"
    )  # type: ignore[assignment]
    source = "triage"
    if requested_route in {"mission", "hotfix", "review"}:
        route = requested_route  # type: ignore[assignment]
        confidence = "high"
        source = "override"

    # Triage is advisory. Ambiguous requests take the fully planned path.
    if confidence == "low" and source != "override":
        route = "mission"

    candidate_files = _workspace_relative_files(
        report.get("candidate_files", report.get("affected_files", [])),
        workspace_root,
    )
    fallback_reason = ""
    if route == "hotfix" and (
        not candidate_files or (confidence != "high" and source != "override")
    ):
        route = "mission"
        fallback_reason = (
            "Hotfix routing lacked a high-confidence, evidence-backed file scope; "
            "using the planned mission route."
        )

    return RouteDecision(
        route=route,
        confidence=confidence,  # type: ignore[arg-type]
        rationale=fallback_reason or (
            f"Execution route explicitly requested: {requested_route}."
            if source == "override"
            else (
                str(report.get("rationale") or report.get("summary") or "").strip()
                or "No routing rationale supplied."
            )
        ),
        candidate_files=candidate_files,
        constraints=_strings(
            report.get("constraints", report.get("repair_constraints", []))
        ),
        validation_intent=_strings(
            report.get(
                "validation_intent",
                report.get("regression_requirements", []),
            )
        ),
        review_scope=str(report.get("review_scope", "user_request")).strip()
        or "user_request",
        source=source,
    )


@dataclass
class EscalationDecision:
    decision: str  # "stay_hotfix" | "escalate_mission"
    reason: str
    scope_type: str = "localized"  # localized | cross_cutting | architectural
    risk: str = "low"  # low | medium | high
    suggested_mission_goal: str = ""
    source: str = "escalation_triage"  # escalation_triage | deterministic | human

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class FixVerification:
    verdict: str  # VERIFIED | NEEDS_REWORK | UNVERIFIED | ESCALATE_TO_MISSION
    summary: str
    evidence: list[str] = field(default_factory=list)
    fix_guidance: str = ""
    missing_checks: list[str] = field(default_factory=list)
    escalation_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def normalize_escalation_decision(payload: Any) -> EscalationDecision:
    """Validate an escalation-triage LLM response with safe fallbacks."""
    if not isinstance(payload, dict):
        return EscalationDecision(
            decision="stay_hotfix",
            reason="Escalation response was not an object; defaulting to bounded fix.",
            source="deterministic",
        )
    raw = str(payload.get("decision", "stay_hotfix")).strip().lower()
    decision = raw if raw in {"stay_hotfix", "escalate_mission"} else "stay_hotfix"
    scope = str(payload.get("scope_type", "localized")).strip().lower()
    if scope not in {"localized", "cross_cutting", "architectural"}:
        scope = "localized"
    risk = str(payload.get("risk", "low")).strip().lower()
    if risk not in {"low", "medium", "high"}:
        risk = "low"
    return EscalationDecision(
        decision=decision,
        reason=str(payload.get("reason", "")).strip() or "No reason supplied.",
        scope_type=scope,
        risk=risk,
        suggested_mission_goal=str(payload.get("suggested_mission_goal", "")).strip(),
        source="escalation_triage",
    )


def normalize_fix_verification(payload: Any) -> FixVerification:
    """Validate a FixVerifier LLM response with safe fallbacks."""
    if not isinstance(payload, dict):
        return FixVerification(
            verdict="UNVERIFIED",
            summary="Verifier response was not an object; treating as unverified.",
            missing_checks=["unparseable verifier response"],
        )
    raw = str(payload.get("verdict", "UNVERIFIED")).strip().upper()
    verdict = (
        raw
        if raw in {"VERIFIED", "NEEDS_REWORK", "UNVERIFIED", "ESCALATE_TO_MISSION"}
        else "UNVERIFIED"
    )
    return FixVerification(
        verdict=verdict,
        summary=str(payload.get("summary", "")).strip() or "No summary supplied.",
        evidence=_strings(payload.get("evidence", [])),
        fix_guidance=str(payload.get("fix_guidance", "")).strip(),
        missing_checks=_strings(payload.get("missing_checks", [])),
        escalation_reason=str(payload.get("escalation_reason", "")).strip(),
    )


def build_mission_brief(
    *,
    original_request: str,
    review_report: Optional[dict[str, Any]] = None,
    hotfix_packet: Optional[dict[str, Any]] = None,
    hotfix_handoff: Optional[dict[str, Any]] = None,
    verification: Optional[dict[str, Any]] = None,
    escalation: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Build an enriched mission brief so the Orchestrator never sees a stale request.

    The review phase is DONE when escalation happens. The brief carries the
    confirmed defects, attempted fix, and why a bounded fix is insufficient.
    """
    report = review_report or {}
    findings = report.get("findings", []) if isinstance(report, dict) else []
    actionable = [
        f
        for f in findings
        if isinstance(f, dict)
        and f.get("severity") in {"blocker", "bug"}
        and f.get("confidence") == "high"
    ]
    escalation_dict = escalation or {}
    goal = (
        escalation_dict.get("suggested_mission_goal", "")
        or "Fix the confirmed defects from the completed code review."
    )
    briefing = (
        "The code review phase is COMPLETE. Do not plan another review.\n\n"
        f"Original user request: {original_request.strip()}\n\n"
        f"Mission goal: {goal}\n\n"
        f"Confirmed actionable findings: {len(actionable)}\n"
        f"Review summary: {str(report.get('summary', '')).strip()}\n\n"
        "Fix criteria from review (authoritative acceptance conditions):\n"
        + "".join(
            f"- {c}\n"
            for f in actionable
            for c in (f.get("fix_criteria", []) or [])
        )
        + "\nAttempted bounded fix:\n"
        f"- files: {(hotfix_packet or {}).get('target_files', [])}\n"
        f"- modified: {(hotfix_handoff or {}).get('files_modified', [])}\n"
        f"- verdict: {(hotfix_handoff or {}).get('verdict', 'unknown')}\n\n"
        f"Verification: {str((verification or {}).get('summary', 'none'))}\n"
        f"Escalation reason: {str(escalation_dict.get('reason', ''))}\n"
        f"Scope: {escalation_dict.get('scope_type', 'unknown')} / "
        f"risk {escalation_dict.get('risk', 'unknown')}\n\n"
        "Constraints: start from these known defects; do not re-audit the whole repo."
    )
    return {
        "original_request": original_request,
        "mission_goal": goal,
        "review_report": report,
        "hotfix_packet": hotfix_packet,
        "hotfix_handoff": hotfix_handoff,
        "verification": verification,
        "escalation": escalation,
        "briefing": briefing,
    }


@dataclass
class ReviewFinding:
    severity: str
    confidence: Confidence
    title: str
    issue: str
    evidence: list[str]
    affected_files: list[str]
    fix_criteria: list[str]

    @property
    def actionable(self) -> bool:
        return (
            self.severity in {"blocker", "bug"}
            and self.confidence == "high"
            and bool(self.affected_files)
            and bool(self.fix_criteria)
        )


@dataclass
class ReviewReport:
    verdict: str
    summary: str
    scope: str
    findings: list[ReviewFinding] = field(default_factory=list)
    tool_calls: int = 0

    @property
    def actionable_findings(self) -> list[ReviewFinding]:
        return [finding for finding in self.findings if finding.actionable]

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict,
            "summary": self.summary,
            "scope": self.scope,
            "findings": [asdict(finding) for finding in self.findings],
            "tool_calls": self.tool_calls,
        }


def normalize_review_report(
    payload: Any,
    *,
    workspace_root: Path,
    tool_calls: int = 0,
) -> ReviewReport:
    if not isinstance(payload, dict):
        raise ValueError("review report must be a JSON object")
    verdict = str(payload.get("verdict", "")).strip().lower()
    if verdict not in {"clean", "issues_found"}:
        raise ValueError("review verdict must be clean or issues_found")
    summary = str(payload.get("summary", "")).strip()
    if not summary:
        raise ValueError("review summary must be non-empty")

    findings: list[ReviewFinding] = []
    raw_findings = payload.get("findings", [])
    if not isinstance(raw_findings, list):
        raise ValueError("review findings must be a list")
    for raw in raw_findings[:30]:
        if not isinstance(raw, dict):
            continue
        confidence = str(raw.get("confidence", "low")).lower()
        if confidence not in {"high", "medium", "low"}:
            confidence = "low"
        severity = str(raw.get("severity", "risk")).lower()
        if severity not in {"blocker", "bug", "risk", "style", "nit"}:
            severity = "risk"
        findings.append(
            ReviewFinding(
                severity=severity,
                confidence=confidence,  # type: ignore[arg-type]
                title=str(raw.get("title", "Untitled finding")).strip(),
                issue=str(raw.get("issue", "")).strip(),
                evidence=_strings(raw.get("evidence", [])),
                affected_files=_workspace_relative_files(
                    raw.get("affected_files", []), workspace_root
                ),
                fix_criteria=_strings(raw.get("fix_criteria", [])),
            )
        )
    if verdict == "clean":
        findings = []
    return ReviewReport(
        verdict=verdict,
        summary=summary,
        scope=str(payload.get("scope", "user_request")).strip() or "user_request",
        findings=findings,
        tool_calls=tool_calls,
    )


def hotfix_milestone_from_route(
    decision: RouteDecision,
    user_request: str,
    *,
    milestone_id: str = "HOTFIX",
) -> dict[str, Any]:
    criteria = decision.validation_intent or [
        f"The reported issue is fixed: {user_request.strip()}"
    ]
    suffixes = {Path(path).suffix.lower() for path in decision.candidate_files}
    profile = "ui" if suffixes & {".html", ".css", ".js", ".jsx", ".ts", ".tsx", ".vue"} else "auto"
    return {
        "id": milestone_id,
        "title": "Focused hotfix",
        "description": user_request.strip(),
        "depends_on": [],
        "target_files": decision.candidate_files,
        "acceptance_criteria": criteria,
        "validation_profile": profile,
        "status": "pending",
        "route": "hotfix",
        "constraints": decision.constraints,
    }


def hotfix_milestone_from_review(
    report: ReviewReport,
    *,
    milestone_id: str = "REVIEW-FIX",
) -> dict[str, Any]:
    findings = report.actionable_findings
    files = sorted(
        {path for finding in findings for path in finding.affected_files}
    )
    criteria = [
        criterion
        for finding in findings
        for criterion in finding.fix_criteria
    ]
    return hotfix_milestone_from_route(
        RouteDecision(
            route="hotfix",
            confidence="high",
            rationale="Fix high-confidence defects identified by code review.",
            candidate_files=files,
            constraints=["Do not address non-actionable style or speculative findings."],
            validation_intent=criteria,
            source="review",
        ),
        report.summary,
        milestone_id=milestone_id,
    )
