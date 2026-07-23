"""Workspace browsing endpoints — read-only, no global workspace mutation."""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException

from src.api.deps import get_session_manager, require_session
from src.api.schemas import (
    HandoffResponse,
    PlanResponse,
    WorkspaceFileResponse,
    WorkspaceNodeResponse,
    WorkspaceTreeResponse,
)
from src.session import SessionContext, SessionManager
import json
from pathlib import Path

router = APIRouter(prefix="/sessions/{sid}", tags=["workspace", "plan", "handoffs", "memory"])


# ---------------------------------------------------------------------------
# Plan
# ---------------------------------------------------------------------------

@router.get("/plan", response_model=PlanResponse)
async def get_plan(
    sid: str,
    manager: SessionManager = Depends(get_session_manager),
) -> PlanResponse:
    ctx = require_session(sid, manager)
    if not ctx.plan_path.exists():
        return PlanResponse()
    try:
        plan = json.loads(ctx.plan_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return PlanResponse()
    return PlanResponse(
        mission_id=plan.get("mission_id"),
        title=plan.get("title"),
        milestones=plan.get("milestones", []),
    )


# ---------------------------------------------------------------------------
# Handoffs
# ---------------------------------------------------------------------------

@router.get("/handoffs", response_model=list[HandoffResponse])
async def list_handoffs(
    sid: str,
    manager: SessionManager = Depends(get_session_manager),
) -> list[HandoffResponse]:
    ctx = require_session(sid, manager)
    if not ctx.handoffs_dir.exists():
        return []
    out = []
    for entry in sorted(ctx.handoffs_dir.glob("*.json")):
        try:
            data = json.loads(entry.read_text(encoding="utf-8"))
            out.append(HandoffResponse(
                milestone_id=data.get("milestone_id", ""),
                title=data.get("title", ""),
                verdict=data.get("verdict", ""),
                worker_summary=data.get("worker_summary"),
                files_modified=data.get("files_modified", []),
                tool_calls=data.get("tool_calls"),
                retry_count=data.get("retry_count"),
                commit_hash=data.get("commit_hash"),
                timestamp=data.get("timestamp"),
                session_id=data.get("session_id"),
            ))
        except (json.JSONDecodeError, OSError):
            continue
    return out


@router.get("/handoffs/{ms_id}", response_model=HandoffResponse)
async def get_handoff(
    sid: str,
    ms_id: str,
    manager: SessionManager = Depends(get_session_manager),
) -> HandoffResponse:
    ctx = require_session(sid, manager)
    if not ctx.handoffs_dir.exists():
        raise HTTPException(status_code=404, detail=f"No handoffs for session '{sid}'")
    # Return the latest handoff whose filename starts with the milestone id.
    candidates = sorted(
        (p for p in ctx.handoffs_dir.glob(f"{ms_id}_*.json")),
        reverse=True,
    )
    if not candidates:
        raise HTTPException(
            status_code=404,
            detail=f"No handoff found for milestone '{ms_id}' in session '{sid}'",
        )
    data = json.loads(candidates[0].read_text(encoding="utf-8"))
    return HandoffResponse(
        milestone_id=data.get("milestone_id", ""),
        title=data.get("title", ""),
        verdict=data.get("verdict", ""),
        worker_summary=data.get("worker_summary"),
        files_modified=data.get("files_modified", []),
        tool_calls=data.get("tool_calls"),
        retry_count=data.get("retry_count"),
        commit_hash=data.get("commit_hash"),
        timestamp=data.get("timestamp"),
        session_id=data.get("session_id"),
    )


# ---------------------------------------------------------------------------
# Memory (read-only in Phase 3)
# ---------------------------------------------------------------------------

@router.get("/memory")
async def get_memory(
    sid: str,
    manager: SessionManager = Depends(get_session_manager),
) -> dict:
    ctx = require_session(sid, manager)
    if not ctx.memory_store_path.exists():
        return {}
    try:
        return json.loads(ctx.memory_store_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


# ---------------------------------------------------------------------------
# Workspace tree + file read
# ---------------------------------------------------------------------------

def _safe_resolve(root: Path, rel: str) -> Path:
    """Resolve a path under an allowed session root with escape guard."""
    rel = rel.strip().lstrip("/")
    if rel in ("", "."):
        return root.resolve()
    candidate = (root / rel).resolve()
    root_resolved = root.resolve()
    if root_resolved not in candidate.parents and candidate != root_resolved:
        raise HTTPException(
            status_code=400,
            detail=f"Path escapes allowed root: {rel}",
        )
    return candidate


def _browse_root(ctx: SessionContext, scope: str) -> Path:
    if scope == "workspace":
        return ctx.workspace_root
    if scope == "session":
        return ctx.root
    raise HTTPException(status_code=400, detail=f"Unknown workspace scope: {scope}")


def _build_tree(
    path: Path,
    base_root: Path,
    depth: int,
    max_depth: int,
    prefix: str = "",
) -> tuple[str, list[str]]:
    """Return (tree_string, flat_entry_list) for a directory."""
    if depth >= max_depth:
        return "", []
    try:
        entries = sorted(path.iterdir(), key=lambda p: (not p.is_dir(), p.name))
    except (OSError, PermissionError):
        return "", []
    lines: list[str] = []
    flat: list[str] = []
    for i, entry in enumerate(entries):
        is_last = i == len(entries) - 1
        connector = "└── " if is_last else "├── "
        lines.append(f"{prefix}{connector}{entry.name}")
        rel = str(entry.relative_to(base_root))
        flat.append(rel)
        if entry.is_dir():
            ext_prefix = prefix + ("    " if is_last else "│   ")
            child_lines, child_flat = _build_tree(
                entry,
                base_root,
                depth + 1,
                max_depth,
                ext_prefix,
            )
            lines.extend(child_lines)
            flat.extend(child_flat)
    return "\n".join(lines), flat


def _build_nodes(
    path: Path,
    base_root: Path,
    depth: int,
    max_depth: int,
) -> list[WorkspaceNodeResponse]:
    if depth >= max_depth:
        return []
    try:
        entries = sorted(path.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
    except (OSError, PermissionError):
        return []

    nodes: list[WorkspaceNodeResponse] = []
    for entry in entries:
        rel = str(entry.relative_to(base_root))
        is_dir = entry.is_dir()
        size = None
        if not is_dir:
            try:
                size = entry.stat().st_size
            except OSError:
                size = None
        nodes.append(
            WorkspaceNodeResponse(
                name=entry.name,
                path=rel,
                type="directory" if is_dir else "file",
                size=size,
                children=_build_nodes(entry, base_root, depth + 1, max_depth)
                if is_dir
                else [],
            )
        )
    return nodes


@router.get("/workspace", response_model=WorkspaceTreeResponse)
async def list_workspace(
    sid: str,
    path: str = "",
    depth: int = 3,
    scope: str = "workspace",
    manager: SessionManager = Depends(get_session_manager),
) -> WorkspaceTreeResponse:
    ctx = require_session(sid, manager)
    root = _browse_root(ctx, scope).resolve()
    target = _safe_resolve(root, path)
    if not target.exists():
        raise HTTPException(status_code=404, detail=f"Path not found: {path}")
    if not target.is_dir():
        raise HTTPException(status_code=400, detail=f"Not a directory: {path}")
    max_depth = max(depth, 1)
    tree, entries = _build_tree(target, root, 0, max_depth)
    nodes = _build_nodes(target, root, 0, max_depth)
    rel = str(target.relative_to(root)) if target != root else ""
    return WorkspaceTreeResponse(
        path=rel or ".",
        tree=tree,
        entries=entries,
        root=scope,
        nodes=nodes,
    )


@router.get("/workspace/file", response_model=WorkspaceFileResponse)
async def read_workspace_file(
    sid: str,
    path: str,
    scope: str = "workspace",
    manager: SessionManager = Depends(get_session_manager),
) -> WorkspaceFileResponse:
    ctx = require_session(sid, manager)
    root = _browse_root(ctx, scope).resolve()
    target = _safe_resolve(root, path)
    if not target.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {path}")
    if not target.is_file():
        raise HTTPException(status_code=400, detail=f"Not a file: {path}")
    try:
        content = target.read_text(encoding="utf-8")
        encoding = "utf-8"
    except UnicodeDecodeError:
        content = target.read_bytes().decode("latin-1", errors="replace")
        encoding = "latin-1"
    size = target.stat().st_size
    rel = str(target.relative_to(root))
    return WorkspaceFileResponse(path=rel, content=content, size=size, encoding=encoding)
