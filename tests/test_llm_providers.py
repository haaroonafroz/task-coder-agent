"""Generic OpenAI-compatible provider resolution and adapters."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.llm_client import call_llm, get_model_catalog, resolve_model_config
from src.settings import reset_settings, update_settings


def _add_providers(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("TASK_CODER_HOME", str(tmp_path / "home"))
    reset_settings()
    update_settings({
        "llm": {
            "providers": [
                {
                    "id": "local",
                    "base_url": "http://127.0.0.1:8001/v1",
                    "adapter": "llamacpp_qwen",
                    "model": "qwen-local",
                    "enabled": True,
                },
                {
                    "id": "gemini",
                    "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
                    "adapter": "gemini_openai",
                    "model": "gemini-flash",
                    "enabled": True,
                    "api_key": "g-key",
                },
            ],
            "fallback_order": ["local", "gemini"],
        }
    })


def test_catalog_includes_auto_and_providers(tmp_path, monkeypatch) -> None:
    _add_providers(monkeypatch, tmp_path)
    catalog = get_model_catalog()
    keys = [e["key"] for e in catalog]
    assert keys[0] == "auto"
    assert "local" in keys
    assert "gemini" in keys


def test_no_providers_raises_clear_error(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("TASK_CODER_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    reset_settings()
    with pytest.raises(RuntimeError, match="No LLM providers configured"):
        call_llm("hello", model="auto")


def test_llamacpp_adapter_sends_chat_template_kwargs(tmp_path, monkeypatch) -> None:
    _add_providers(monkeypatch, tmp_path)
    cfg = resolve_model_config("local", "orchestrator")
    assert cfg.adapter == "llamacpp_qwen"
    assert cfg.model_name == "qwen-local"

    captured = {}

    class _Stream:
        def __iter__(self):
            return iter(())

    def fake_create(**kwargs):
        captured.update(kwargs)
        return _Stream()

    client = MagicMock()
    client.chat.completions.create.side_effect = fake_create
    with patch("src.llm_client._build_client", return_value=client):
        call_llm("hi", model="local", role="orchestrator", max_tokens=16)
    assert "chat_template_kwargs" in captured.get("extra_body", {})
    assert captured["extra_body"]["chat_template_kwargs"]["enable_thinking"] is True


def test_gemini_adapter_strips_seed(tmp_path, monkeypatch) -> None:
    _add_providers(monkeypatch, tmp_path)
    captured = {}

    class _Stream:
        def __iter__(self):
            return iter(())

    def fake_create(**kwargs):
        captured.update(kwargs)
        return _Stream()

    client = MagicMock()
    client.chat.completions.create.side_effect = fake_create
    with patch("src.llm_client._build_client", return_value=client):
        call_llm("hi", model="gemini", role="worker", max_tokens=16)
    assert "seed" not in captured
    assert "stream_options" not in captured
