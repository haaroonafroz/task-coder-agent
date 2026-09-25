"""
Pydantic v2 request/response schemas for the Missions Control API.

All API-facing models live here so the routers stay thin and the wire
contract is centralized. Models mirror the dataclasses already in use
(SessionContext, MissionResult, MilestoneHandoff) but with paths
serialized as strings and optional fields relaxed for partial updates.
"""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

from src.llm_client import ModelChoice

ExecutionRoute = Literal["auto", "mission", "hotfix", "review"]


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------

class WorkspaceCreate(BaseModel):
    kind: Literal["managed", "external"] = "managed"
    path: Optional[str] = None
    access_mode: Literal["read_write", "read_only"] = "read_write"
    environment_strategy: Literal["auto", "project", "harness"] = "auto"


class WorkspaceResponse(BaseModel):
    kind: Literal["managed", "external"]
    path: str
    access_mode: Literal["read_write", "read_only"]
    environment_strategy: Literal["auto", "project", "harness"]
    git_root: Optional[str] = None
    sandbox_required: bool = False


class ProjectProfileResponse(BaseModel):
    root: str
    workspace_kind: str
    git: dict[str, Any] = Field(default_factory=dict)
    languages: list[str] = Field(default_factory=list)
    manifests: list[str] = Field(default_factory=list)
    environment: Optional[dict[str, Any]] = None
    detected_commands: dict[str, str] = Field(default_factory=dict)


class WorkspaceInfoResponse(BaseModel):
    workspace: WorkspaceResponse
    project: ProjectProfileResponse


class SessionCreate(BaseModel):
    title: str = Field(..., min_length=1, max_length=200)
    model: ModelChoice = "auto"
    thinking_profile: str = "auto"
    phoenix_project: Optional[str] = None
    workspace: WorkspaceCreate = Field(default_factory=WorkspaceCreate)


class SessionUpdate(BaseModel):
    title: Optional[str] = None
    status: Optional[str] = None
    selected_model: Optional[ModelChoice] = None
    thinking_profile: Optional[str] = None


class SessionResponse(BaseModel):
    session_id: str
    title: str
    status: str
    selected_model: str
    thinking_profile: str
    created_at: str
    phoenix_session_id: Optional[str] = None
    phoenix_project: Optional[str] = None
    workspace_root: str
    plan_path: str
    events_path: str
    workspace: WorkspaceResponse
    project_profile: Optional[dict[str, Any]] = None
    token_usage: dict[str, int] = Field(
        default_factory=lambda: {
            "prompt": 0,
            "generated": 0,
            "calls": 0,
            "estimated_calls": 0,
        }
    )


# ---------------------------------------------------------------------------
# Messages
# ---------------------------------------------------------------------------

class MessageCreate(BaseModel):
    content: str = Field(..., min_length=1)
    trigger_run: bool = True
    model: Optional[ModelChoice] = None
    run_kind: Literal["auto", "new", "resume", "repair"] = "auto"
    execution_route: ExecutionRoute = "auto"
    review_fix_mode: Literal["auto", "ask"] = "ask"


class PendingDecisionResponse(BaseModel):
    type: str
    title: str
    summary: str
    options: list[str] = Field(default_factory=list)
    session_id: str
    created_at: str
    payload: dict[str, Any] = Field(default_factory=dict)


class DecisionCreate(BaseModel):
    action: Literal[
        "apply_fix", "dismiss", "escalate_mission", "run_smoke", "setup_env"
    ]


class DecisionResolveResponse(BaseModel):
    action: str
    session_id: str
    run_id: Optional[str] = None
    detail: str = ""


class MessageResponse(BaseModel):
    id: str
    role: str
    content: str
    ts: str
    run_id: Optional[str] = None
    run_kind: Optional[str] = None
    execution_route: Optional[str] = None


# ---------------------------------------------------------------------------
# Runs
# ---------------------------------------------------------------------------

class RunCreate(BaseModel):
    request: str = Field(..., min_length=1)
    model: Optional[ModelChoice] = None
    run_kind: Literal["auto", "new", "resume", "repair"] = "auto"
    execution_route: ExecutionRoute = "auto"
    review_fix_mode: Literal["auto", "ask"] = "ask"


class RunResponse(BaseModel):
    run_id: str
    session_id: str
    request: str
    status: str  # queued | running | completed | partial | failed | error | cancelled | awaiting_decision
    model: str
    queued_at: str
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    result: Optional[dict[str, Any]] = None
    error: Optional[str] = None
    run_kind: str = "auto"
    execution_route: str = "auto"
    plan_id: Optional[str] = None
    review_fix_mode: str = "ask"


# ---------------------------------------------------------------------------
# Plan / Handoff / Memory (read-through)
# ---------------------------------------------------------------------------

class PlanResponse(BaseModel):
    mission_id: Optional[str] = None
    title: Optional[str] = None
    milestones: list[dict[str, Any]] = Field(default_factory=list)


class HandoffResponse(BaseModel):
    milestone_id: str
    title: str
    verdict: str
    worker_summary: Optional[str] = None
    files_modified: list[str] = Field(default_factory=list)
    tool_calls: Optional[int] = None
    retry_count: Optional[int] = None
    commit_hash: Optional[str] = None
    timestamp: Optional[str] = None
    session_id: Optional[str] = None


# ---------------------------------------------------------------------------
# Workspace
# ---------------------------------------------------------------------------

class WorkspaceNodeResponse(BaseModel):
    name: str
    path: str
    type: str
    size: Optional[int] = None
    children: list["WorkspaceNodeResponse"] = Field(default_factory=list)


class WorkspaceTreeResponse(BaseModel):
    path: str
    tree: str
    entries: list[str] = Field(default_factory=list)
    root: str = "workspace"
    nodes: list[WorkspaceNodeResponse] = Field(default_factory=list)


class WorkspaceFileResponse(BaseModel):
    path: str
    content: str
    size: int
    encoding: str = "utf-8"


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------

class EventResponse(BaseModel):
    ts: str
    type: str
    session_id: str
    data: dict[str, Any] = Field(default_factory=dict)
    index: int


# ---------------------------------------------------------------------------
# Catalogs
# ---------------------------------------------------------------------------

class ModelInfo(BaseModel):
    key: str
    model: str
    base_url: str
    available: bool
    error: Optional[str] = None
    models_by_role: dict[str, str] = Field(default_factory=dict)
    thinking_by_role: dict[str, str] = Field(default_factory=dict)
    context_length: Optional[int] = None
    label: Optional[str] = None
    adapter: Optional[str] = None
    compat: Optional[str] = None
    enabled: bool = True
    api_key_set: bool = False
    discovered_models: list[str] = Field(default_factory=list)


class ToolParamSchema(BaseModel):
    name: str
    type: str
    required: bool
    default: Optional[Any] = None


class ToolInfo(BaseModel):
    name: str
    params: list[ToolParamSchema] = Field(default_factory=list)


class SkillInfo(BaseModel):
    name: str
    keywords: list[str] = Field(default_factory=list)


class SkillDetail(SkillInfo):
    raw_markdown: str


# ---------------------------------------------------------------------------
# Uploads
# ---------------------------------------------------------------------------

class UploadResponse(BaseModel):
    filename: str
    size: int
    path: str


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    status: str  # "ok" | "degraded"
    version: str = "1"
    home: str = ""
    qdrant: dict[str, Any] = Field(default_factory=dict)
    embeddings: str = "none"
    sandbox: dict[str, Any] = Field(default_factory=dict)
    providers_configured: int = 0


class ReadyResponse(BaseModel):
    ready: bool
    checks: dict[str, dict[str, Any]] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Evals (Phase 6)
# ---------------------------------------------------------------------------

class EvalScoreResponse(BaseModel):
    name: str
    score: float
    passed: bool
    summary: str
    details: dict[str, Any] = Field(default_factory=dict)
    evidence: list[str] = Field(default_factory=list)
    kind: str = "deterministic"


class SessionEvalReportResponse(BaseModel):
    session_id: str
    evaluated_at: str
    mission_status: Optional[str] = None
    overall_score: float
    overall_passed: bool
    scores: list[EvalScoreResponse] = Field(default_factory=list)
    event_count: int
    deterministic_only: bool = True
    weights: dict[str, float] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

class SettingsResponse(BaseModel):
    settings: dict[str, Any]
    home: str
    presets: list[dict[str, Any]] = Field(default_factory=list)
    migrated_from_env: bool = False
    container: bool = False


class SettingsUpdate(BaseModel):
    settings: dict[str, Any] = Field(default_factory=dict)


class ProviderProbeRequest(BaseModel):
    base_url: str
    api_key: Optional[str] = None
    provider_id: Optional[str] = None


class ProviderProbeResponse(BaseModel):
    ok: bool
    models: list[str] = Field(default_factory=list)
    error: Optional[str] = None
