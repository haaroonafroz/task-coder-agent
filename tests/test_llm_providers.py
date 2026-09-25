"""Generic OpenAI-compatible provider resolution and adapters."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.llm_client import (
    apply_unsupported_param,
    call_llm,
    get_model_catalog,
    RequestProfile,
    reset_learned_profiles,
    resolve_model_config,
    resolve_request_profile,
    _candidate_ids,
)
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


def _add_openai_models(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("TASK_CODER_HOME", str(tmp_path / "home"))
    reset_learned_profiles()
    reset_settings()
    update_settings({
        "llm": {
            "providers": [
                {
                    "id": "gpt4o",
                    "base_url": "https://api.openai.com/v1",
                    "adapter": "openai",
                    "model": "gpt-4o",
                    "enabled": True,
                    "api_key": "sk-test",
                    "compat": "auto",
                },
                {
                    "id": "gpt-6-luna",
                    "base_url": "https://api.openai.com/v1",
                    "adapter": "openai",
                    "model": "gpt-6-luna",
                    "enabled": True,
                    "api_key": "sk-test",
                    "compat": "auto",
                },
            ],
            "fallback_order": ["gpt4o", "gpt-6-luna"],
        }
    })


class _Stream:
    def __iter__(self):
        return iter(())


def _status_error(param: str):
    import httpx
    from openai import APIStatusError

    request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    body = {
        "error": {
            "message": (
                f"Unsupported parameter: '{param}' is not supported with this model. "
                f"Use 'max_completion_tokens' instead."
            ),
            "type": "invalid_request_error",
            "param": param,
            "code": "unsupported_parameter",
        }
    }
    response = httpx.Response(400, request=request, json=body)
    return APIStatusError(body["error"]["message"], response=response, body=body)


def test_gpt4o_uses_classic_max_tokens(tmp_path, monkeypatch) -> None:
    _add_openai_models(monkeypatch, tmp_path)
    captured = {}

    def fake_create(**kwargs):
        captured.update(kwargs)
        return _Stream()

    client = MagicMock()
    client.chat.completions.create.side_effect = fake_create
    with patch("src.llm_client._build_client", return_value=client):
        call_llm("hi", model="gpt4o", role="worker", max_tokens=32)
    assert captured.get("max_tokens") == 32
    assert "max_completion_tokens" not in captured
    assert "temperature" in captured
    assert "reasoning_effort" not in captured


def test_gpt6_luna_uses_completion_tokens_without_sampling(tmp_path, monkeypatch) -> None:
    _add_openai_models(monkeypatch, tmp_path)
    captured = {}

    def fake_create(**kwargs):
        captured.update(kwargs)
        return _Stream()

    client = MagicMock()
    client.chat.completions.create.side_effect = fake_create
    with patch("src.llm_client._build_client", return_value=client):
        call_llm("hi", model="gpt-6-luna", role="orchestrator", max_tokens=32)
    assert captured.get("max_completion_tokens") == 32
    assert "max_tokens" not in captured
    assert "temperature" not in captured
    assert "top_p" not in captured
    assert "seed" not in captured
    assert captured.get("reasoning_effort") in {"high", "medium", "low"}


def test_unsupported_max_tokens_retries_same_provider(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("TASK_CODER_HOME", str(tmp_path / "home"))
    reset_learned_profiles()
    reset_settings()
    update_settings({
        "llm": {
            "providers": [{
                "id": "custom",
                "base_url": "https://api.openai.com/v1",
                "adapter": "openai",
                "model": "mystery-model",
                "enabled": True,
                "api_key": "sk-test",
                "compat": "classic",
            }],
            "fallback_order": ["custom"],
        }
    })
    calls: list[dict] = []

    def fake_create(**kwargs):
        calls.append(dict(kwargs))
        if len(calls) == 1:
            raise _status_error("max_tokens")
        return _Stream()

    client = MagicMock()
    client.chat.completions.create.side_effect = fake_create
    with patch("src.llm_client._build_client", return_value=client):
        call_llm("hi", model="custom", role="worker", max_tokens=8)
    assert len(calls) == 2
    assert "max_tokens" in calls[0]
    assert calls[1]["max_completion_tokens"] == 8
    assert "max_tokens" not in calls[1]


def test_request_profile_heuristic_and_override() -> None:
    reset_learned_profiles()
    auto_luna = resolve_request_profile(
        adapter="openai", model_name="gpt-6-luna", compat="auto"
    )
    assert auto_luna.token_field == "max_completion_tokens"
    assert auto_luna.sampling is False
    classic = resolve_request_profile(
        adapter="openai", model_name="gpt-6-luna", compat="classic"
    )
    assert classic.token_field == "max_tokens"
    local = resolve_request_profile(
        adapter="llamacpp_qwen", model_name="gpt-6-luna", compat="auto"
    )
    assert local.token_field == "max_tokens"


def test_apply_unsupported_param_renames_token_field() -> None:
    profile = RequestProfile()
    assert apply_unsupported_param(profile, "max_tokens") is True
    assert profile.token_field == "max_completion_tokens"
    assert apply_unsupported_param(profile, "max_tokens") is False


def test_disabled_provider_falls_back_to_enabled_chain(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("TASK_CODER_HOME", str(tmp_path / "home"))
    reset_learned_profiles()
    reset_settings()
    update_settings({
        "llm": {
            "providers": [
                {
                    "id": "local",
                    "base_url": "http://127.0.0.1:8001/v1",
                    "adapter": "llamacpp_qwen",
                    "model": "qwen",
                    "enabled": False,
                },
                {
                    "id": "gpt4o",
                    "base_url": "https://api.openai.com/v1",
                    "adapter": "openai",
                    "model": "gpt-4o",
                    "enabled": True,
                    "api_key": "sk-test",
                },
            ],
            "fallback_order": ["local", "gpt4o"],
        }
    })
    assert _candidate_ids("local") == ["gpt4o"]
    assert _candidate_ids("auto") == ["gpt4o"]
