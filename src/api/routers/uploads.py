"""Uploads endpoints — stub for Phase 8 (LlamaParse)."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File

from src.api.deps import get_session_manager, require_session
from src.api.schemas import UploadResponse
from src.session import SessionManager

router = APIRouter(prefix="/sessions/{sid}/uploads", tags=["uploads"])


@router.post("", response_model=UploadResponse, status_code=201)
async def upload_file(
    sid: str,
    file: UploadFile = File(...),
    manager: SessionManager = Depends(get_session_manager),
) -> UploadResponse:
    ctx = require_session(sid, manager)
    ctx.uploads_dir.mkdir(parents=True, exist_ok=True)
    dest = ctx.uploads_dir / Path(file.filename or "upload.bin").name
    content = await file.read()
    dest.write_bytes(content)
    return UploadResponse(
        filename=dest.name,
        size=len(content),
        path=str(dest.relative_to(ctx.root)),
    )


@router.get("", response_model=list[UploadResponse])
async def list_uploads(
    sid: str,
    manager: SessionManager = Depends(get_session_manager),
) -> list[UploadResponse]:
    ctx = require_session(sid, manager)
    if not ctx.uploads_dir.exists():
        return []
    out = []
    for entry in sorted(ctx.uploads_dir.iterdir()):
        if entry.is_file():
            out.append(UploadResponse(
                filename=entry.name,
                size=entry.stat().st_size,
                path=str(entry.relative_to(ctx.root)),
            ))
    return out
