"""Skills catalog endpoints — backed by the DynamicToolRouter."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from src.api.deps import get_router
from src.api.schemas import SkillDetail, SkillInfo
from src.tool_registry import DynamicToolRouter

router = APIRouter(prefix="/skills", tags=["skills"])


@router.get("", response_model=list[SkillInfo])
async def list_skills(
    tool_router: DynamicToolRouter = Depends(get_router),
) -> list[SkillInfo]:
    skills = tool_router._skills
    return [
        SkillInfo(name=s["name"], keywords=s.get("keywords", []))
        for s in skills
    ]


@router.get("/{name}", response_model=SkillDetail)
async def get_skill(
    name: str,
    tool_router: DynamicToolRouter = Depends(get_router),
) -> SkillDetail:
    for s in tool_router._skills:
        if s["name"] == name:
            return SkillDetail(
                name=s["name"],
                keywords=s.get("keywords", []),
                raw_markdown=s.get("raw_markdown", ""),
            )
    raise HTTPException(status_code=404, detail=f"Skill '{name}' not found")
