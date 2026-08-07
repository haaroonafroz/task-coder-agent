"""Tests for structured progressive tool discovery."""

from __future__ import annotations

from src.tool_registry import DynamicToolRouter


def test_search_tools_returns_names_and_documentation() -> None:
    router = DynamicToolRouter.__new__(DynamicToolRouter)
    router._skills = [
        {
            "id": 1,
            "name": "read_file",
            "raw_markdown": "READ_FILE_DOC",
        },
        {
            "id": 2,
            "name": "run_pytest",
            "raw_markdown": "RUN_PYTEST_DOC",
        },
    ]
    router.fetch_curated_skills = lambda query, top_k=3, rrf_k=60: (
        "READ_FILE_DOC\n\n---\n\nRUN_PYTEST_DOC"
    )

    result = router.search_tools("inspect and test", top_k=3)

    assert result["success"] is True
    assert result["tools"] == ["read_file", "run_pytest"]
    assert result["count"] == 2
    assert "READ_FILE_DOC" in result["documentation"]
