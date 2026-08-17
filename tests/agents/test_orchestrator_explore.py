"""Tests for orchestrator workspace exploration."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.agents.orchestrator_explore import (
    _grep_patterns_from_request,
    build_workspace_orientation,
    should_explore,
)
from src.agents.utils import validate_plan_payload
from src.sandbox import activate_sandbox, deactivate_sandbox
from src.session import SessionContext


@pytest.fixture
def workspace(tmp_path: Path):
    root = tmp_path / "workspace"
    root.mkdir()
    (root / "index.html").write_text(
        '<script>const url = "http://localhost:8080/v1/chat/completions";</script>\n',
        encoding="utf-8",
    )
    (root / "app.js").write_text("console.log('hi');\n", encoding="utf-8")
    session = SessionContext(
        session_id="test-session",
        title="test",
        root=tmp_path,
        workspace_root=root,
        plan_path=tmp_path / "plan.json",
        handoffs_dir=tmp_path / "handoffs",
        memory_store_path=tmp_path / "memory.json",
        events_path=tmp_path / "events.jsonl",
        uploads_dir=tmp_path / "uploads",
        parsed_requirements_dir=tmp_path / "parsed_requirements",
        meta_path=tmp_path / "session.json",
        selected_model="auto",
        created_at="2026-01-01T00:00:00",
        status="created",
    )
    session.ensure_dirs()
    activate_sandbox(session)
    from src.tools.paths import set_workspace_root

    set_workspace_root(root)
    yield root
    deactivate_sandbox()


def test_should_explore_repair_and_greenfield(tmp_path: Path):
    ws = tmp_path / "workspace"
    ws.mkdir()
    (ws / "main.py").write_text("print('hi')\n", encoding="utf-8")
    (ws / "util.py").write_text("def helper(): pass\n", encoding="utf-8")
    assert should_explore("repair", ws)[0] is True
    assert should_explore("new", ws)[0] is True
    assert should_explore("new", None)[0] is False


def test_grep_patterns_extract_files_and_ports():
    patterns = _grep_patterns_from_request(
        'Fix index.html — change localhost:8080 to localhost:8001',
        {"milestones": [{"target_files": ["tests/chat.test.js"]}]},
    )
    assert "index.html" in patterns
    assert "localhost:8080" in patterns or any("8080" in p for p in patterns)


def test_build_workspace_orientation_includes_grep_hits(workspace: Path):
    text = build_workspace_orientation(
        workspace,
        "Update index.html endpoint localhost:8080",
        None,
    )
    assert "Session orientation" in text
    assert "8080" in text or "index.html" in text


def test_validate_plan_payload_rejects_missing_targets():
    err = validate_plan_payload({"milestones": [{"id": "M1", "acceptance_criteria": ["x"]}]})
    assert err is not None
    assert "target_files" in err


def test_validate_plan_payload_accepts_minimal_plan():
    plan = {
        "milestones": [{
            "id": "M1",
            "target_files": ["index.html"],
            "acceptance_criteria": ["Endpoint uses port 8001"],
            "validation_profile": "ui",
        }],
    }
    assert validate_plan_payload(plan) is None
