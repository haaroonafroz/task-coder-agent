"""Deterministic schemas and diagnostics for Worker tool calls.

JSON mode guarantees syntactic JSON, not that a model selected a real tool or
provided the right argument names. This module keeps that semantic validation
outside the model so malformed calls become one cheap, actionable retry.
"""

from __future__ import annotations

from typing import Any


TOOL_SCHEMAS: dict[str, dict[str, Any]] = {
    "read_file": {
        "required": {"file_path": str},
        "optional": {},
    },
    "write_file": {
        "required": {"file_path": str, "content": str},
        "optional": {"rewrite": bool},
    },
    "patch_file": {
        "required": {"file_path": str, "search_string": str, "replace_string": str},
        "optional": {},
    },
    "list_directory": {
        "required": {},
        "optional": {"target_dir": str, "max_depth": int},
    },
    "search_grep": {
        "required": {"query": str},
        "optional": {"target_dir": str},
    },
    "run_pytest": {
        "required": {"test_path": str},
        "optional": {"extra_args": str},
    },
    "run_linter": {
        "required": {"target_path": str},
        "optional": {"tool": str, "extra_args": str},
    },
    "install_dependency": {
        "required": {"package_name": str},
        "optional": {},
    },
    "uninstall_dependency": {
        "required": {"package_name": str},
        "optional": {},
    },
    "run_shellscript": {
        "required": {"script": str},
        "optional": {"timeout": int, "profile": str},
    },
    "git_commit": {
        "required": {"message": str},
        "optional": {"stage_paths": list},
    },
    "git_diff": {
        "required": {},
        "optional": {},
    },
    "view_git_log": {
        "required": {},
        "optional": {"limit": int},
    },
    "search_tools": {
        "required": {"query": str},
        "optional": {"limit": int},
    },
}


def validate_tool_call(
    tool_name: str,
    args: Any,
    *,
    active_tools: set[str] | None = None,
) -> dict[str, Any] | None:
    """Return a structured error dict, or ``None`` when the call is valid."""
    if not isinstance(tool_name, str) or not tool_name.strip():
        return {
            "success": False,
            "error_category": "invalid_tool_name",
            "error": "The 'tool' field must be a non-empty string.",
        }
    if not isinstance(args, dict):
        return {
            "success": False,
            "error_category": "invalid_arguments",
            "error": f"Arguments for '{tool_name}' must be a JSON object.",
        }

    schema = TOOL_SCHEMAS.get(tool_name)
    if schema is None:
        known = sorted(TOOL_SCHEMAS)
        return {
            "success": False,
            "error_category": "unknown_tool",
            "error": (
                f"Unknown tool '{tool_name}'. Known tools: {known}. "
                "Use search_tools when the required capability is not listed."
            ),
        }

    if active_tools is not None and tool_name not in active_tools:
        available = sorted(active_tools)
        return {
            "success": False,
            "error_category": "tool_not_available",
            "error": (
                f"Tool '{tool_name}' exists but is not currently available. "
                f"Currently available: {available}. Use search_tools to discover "
                "another capability."
            ),
        }

    required: dict[str, type] = schema["required"]
    optional: dict[str, type] = schema["optional"]
    missing = sorted(name for name in required if name not in args)
    if missing:
        return {
            "success": False,
            "error_category": "invalid_arguments",
            "error": f"Missing required argument(s) for '{tool_name}': {missing}.",
        }

    unexpected = sorted(set(args) - set(required) - set(optional))
    if unexpected:
        return {
            "success": False,
            "error_category": "invalid_arguments",
            "error": (
                f"Unexpected argument(s) for '{tool_name}': {unexpected}. "
                f"Expected keys: {sorted(set(required) | set(optional))}."
            ),
        }

    type_errors: list[str] = []
    for name, expected in {**required, **optional}.items():
        if name not in args:
            continue
        value = args[name]
        # bool is a subclass of int; treat it as a distinct JSON type here.
        valid = isinstance(value, expected) and not (
            expected is int and isinstance(value, bool)
        )
        if not valid:
            type_errors.append(
                f"{name} must be {expected.__name__}, got {type(value).__name__}"
            )
    if type_errors:
        return {
            "success": False,
            "error_category": "invalid_arguments",
            "error": f"Invalid arguments for '{tool_name}': {'; '.join(type_errors)}.",
        }

    if tool_name == "search_tools":
        limit = args.get("limit", 3)
        if not 1 <= limit <= 5:
            return {
                "success": False,
                "error_category": "invalid_arguments",
                "error": "'limit' for search_tools must be between 1 and 5.",
            }

    return None
