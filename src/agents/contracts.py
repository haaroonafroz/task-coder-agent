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
