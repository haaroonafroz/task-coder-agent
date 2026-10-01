"""Embedded Qdrant 1.18.2 local persistence."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.settings import get_settings, reset_settings
from src.tool_registry import DynamicToolRouter, _QDRANT_AVAILABLE


pytestmark = pytest.mark.skipif(not _QDRANT_AVAILABLE, reason="qdrant-client not installed")


def test_embedded_qdrant_indexes_and_retrieves(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("TASK_CODER_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("MISSIONS_QDRANT_MODE", "embedded")
    monkeypatch.setenv("MISSIONS_EMBEDDINGS", "none")
    reset_settings()
    settings = get_settings()
    assert settings.qdrant.mode == "embedded"

    skills = tmp_path / "skills.md"
    skills.write_text(
        "\n".join([
            "<!-- SKILL_START: read_file -->",
            "## Skill Name: read_file",
            "**Keywords:** read, file, inspect",
            "Reads a file from the workspace.",
            "<!-- SKILL_END -->",
            "<!-- SKILL_START: run_pytest -->",
            "## Skill Name: run_pytest",
            "**Keywords:** test, pytest, verify",
            "Runs pytest against the project.",
            "<!-- SKILL_END -->",
        ]),
        encoding="utf-8",
    )

    router = DynamicToolRouter(skills)
    status = router.status()
    assert status["mode"] == "embedded"
    assert status["reachable"] is True
    assert status["skill_count"] == 2
    assert status["encoder"] == "none"

    docs = router.fetch_curated_skills("run unit tests with pytest", top_k=1)
    assert "run_pytest" in docs

    router.close()
    router2 = DynamicToolRouter(skills)
    assert router2.status()["reachable"] is True
    assert router2.status()["skill_count"] == 2
    router2.close()


def test_qdrant_off_uses_keyword_fallback(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("TASK_CODER_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("MISSIONS_QDRANT_MODE", "off")
    reset_settings()
    skills = tmp_path / "skills.md"
    skills.write_text(
        "<!-- SKILL_START: read_file -->\n## Skill Name: read_file\n**Keywords:** read\nReads a workspace file.\n<!-- SKILL_END -->\n",
        encoding="utf-8",
    )
    router = DynamicToolRouter(skills)
    assert router.status()["mode"] == "off"
    assert router._client is None
    assert "read_file" in router.fetch_curated_skills("read a file", top_k=1)
