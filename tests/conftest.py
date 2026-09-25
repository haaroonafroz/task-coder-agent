"""Isolate harness state so tests never touch the developer's ~/.missions or repo .env."""

from __future__ import annotations

import os
import tempfile

import pytest

from src.settings.store import reset_settings


def pytest_configure(config) -> None:
    os.environ.setdefault("TASK_CODER_HOME", tempfile.mkdtemp(prefix="missions-test-"))
    os.environ.setdefault("MISSIONS_EMBEDDINGS", "none")
    os.environ.setdefault("MISSIONS_API_TELEMETRY", "false")
    os.environ.setdefault("MISSIONS_QDRANT_MODE", "embedded")
    for key in (
        "OPENAI_API_KEY",
        "GEMINI_API_KEY",
        "QDRANT_URL",
        "QDRANT_API_KEY",
        "LLM_SPECULATIVE_URL",
        "TARGET_MODEL",
        "MODEL_ALIAS",
        "HF_TOKEN",
    ):
        os.environ.pop(key, None)


@pytest.fixture(autouse=True)
def _isolate_missions_home(tmp_path_factory, monkeypatch):
    home = tmp_path_factory.mktemp("missions-home")
    monkeypatch.setenv("TASK_CODER_HOME", str(home))
    monkeypatch.setenv("MISSIONS_EMBEDDINGS", "none")
    monkeypatch.setenv("MISSIONS_API_TELEMETRY", "false")
    monkeypatch.setenv("MISSIONS_QDRANT_MODE", "embedded")
    monkeypatch.delenv("QDRANT_URL", raising=False)
    monkeypatch.delenv("QDRANT_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("LLM_SPECULATIVE_URL", raising=False)
    monkeypatch.delenv("TARGET_MODEL", raising=False)
    monkeypatch.delenv("MODEL_ALIAS", raising=False)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    reset_settings()
    yield home
    reset_settings()
