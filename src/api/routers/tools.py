"""Tools catalog endpoints — schemas only, no execution."""

from __future__ import annotations

import inspect
from typing import Any

from fastapi import APIRouter, HTTPException

from src.api.schemas import ToolInfo, ToolParamSchema
from src.tools import AVAILABLE_TOOLS, _TOOLS
from src.tools.file_ops import read_file, write_file, patch_file, list_directory
from src.tools.git_ops import git_commit, git_diff, view_git_log
from src.tools.system_ops import (
    run_pytest, run_linter, install_dependency,
    search_grep, run_shellscript, uninstall_dependency,
)

router = APIRouter(prefix="/tools", tags=["tools"])

# Map tool name → actual implementation function for schema introspection.
_IMPL = {
    "read_file": read_file,
    "write_file": write_file,
    "patch_file": patch_file,
    "list_directory": list_directory,
    "git_commit": git_commit,
    "git_diff": git_diff,
    "view_git_log": view_git_log,
    "run_pytest": run_pytest,
    "run_linter": run_linter,
    "install_dependency": install_dependency,
    "uninstall_dependency": uninstall_dependency,
    "search_grep": search_grep,
    "run_shellscript": run_shellscript,
}


def _type_name(t) -> str:
    if t is type(None):
        return "None"
    if hasattr(t, "__name__"):
        return t.__name__
    if hasattr(t, "_name"):
        return t._name
    return str(t)


def _tool_params(name: str) -> list[ToolParamSchema]:
    """Introspect a tool implementation function's signature."""
    fn = _IMPL.get(name)
    if fn is None:
        return []
    try:
        sig = inspect.signature(fn)
    except (ValueError, TypeError):
        return []
    params: list[ToolParamSchema] = []
    for pname, param in sig.parameters.items():
        if pname in ("self", "args"):
            continue
        ann = param.annotation if param.annotation is not inspect.Parameter.empty else Any
        required = param.default is inspect.Parameter.empty
        default = None if param.default is inspect.Parameter.empty else param.default
        params.append(ToolParamSchema(
            name=pname,
            type=_type_name(ann),
            required=required,
            default=default,
        ))
    return params


@router.get("", response_model=list[ToolInfo])
async def list_tools() -> list[ToolInfo]:
    return [ToolInfo(name=n, params=_tool_params(n)) for n in AVAILABLE_TOOLS]


@router.get("/{name}", response_model=ToolInfo)
async def get_tool(name: str) -> ToolInfo:
    if name not in AVAILABLE_TOOLS:
        raise HTTPException(status_code=404, detail=f"Unknown tool '{name}'")
    return ToolInfo(name=name, params=_tool_params(name))
