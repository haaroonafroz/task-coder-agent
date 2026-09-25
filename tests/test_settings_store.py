"""Settings store, .env migration, and env overlay."""

from __future__ import annotations

from pathlib import Path

from src.settings import get_settings, migrate_from_env_file, reset_settings, update_settings
from src.settings.store import save_settings, settings_path, secrets_path


def test_defaults_boot_without_env(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("TASK_CODER_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    reset_settings()
    settings = get_settings()
    assert settings.qdrant.mode == "embedded"
    assert settings.llm.providers == []
    assert settings.roles.worker.temperature == 0.5
    assert settings.roles.worker.max_tokens == 12288


def test_migrate_from_env_file(tmp_path: Path) -> None:
    env = tmp_path / ".env"
    env.write_text(
        "\n".join([
            "LLM_SPECULATIVE_URL=http://127.0.0.1:8001/v1",
            "TARGET_MODEL=qwen-test",
            "OPENAI_API_KEY=sk-test",
            "OPENAI_LLM_MODEL=gpt-4o",
            "LLM_TEMPERATURE_WORKER=0.2",
            "QDRANT_URL=https://qdrant.example:6333",
            "QDRANT_API_KEY=qkey",
        ]),
        encoding="utf-8",
    )
    settings, secrets = migrate_from_env_file(env)
    assert settings.provider("local") is not None
    assert settings.provider("local").model == "qwen-test"
    assert settings.provider("gpt4o") is not None
    assert secrets["provider.gpt4o"] == "sk-test"
    assert settings.qdrant.mode == "http"
    assert settings.qdrant.url.startswith("https://qdrant")
    assert secrets["qdrant"] == "qkey"
    assert settings.roles.worker.temperature == 0.2


def test_update_settings_persists_and_redacts(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("TASK_CODER_HOME", str(tmp_path / "home"))
    reset_settings()
    update_settings({
        "llm": {
            "providers": [{
                "id": "local",
                "label": "llama.cpp",
                "base_url": "http://127.0.0.1:8001/v1",
                "adapter": "llamacpp_qwen",
                "model": "qwen",
                "enabled": True,
                "api_key": "secret-key",
            }],
            "fallback_order": ["local"],
        }
    })
    settings = get_settings()
    assert settings.provider("local").api_key == "secret-key"
    dumped = settings.persistable_dict()
    assert "api_key" not in dumped["llm"]["providers"][0]
    redacted = settings.redacted_dict()
    assert redacted["llm"]["providers"][0]["api_key"] == ""
    assert redacted["llm"]["providers"][0]["api_key_set"] is True
    assert "secret-key" not in settings_path().read_text(encoding="utf-8")
    assert "secret-key" in secrets_path().read_text(encoding="utf-8")


def test_empty_api_key_preserves_existing(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("TASK_CODER_HOME", str(tmp_path / "home"))
    reset_settings()
    update_settings({
        "llm": {
            "providers": [{
                "id": "openai",
                "base_url": "https://api.openai.com/v1",
                "adapter": "openai",
                "model": "gpt-4o",
                "api_key": "keep-me",
            }]
        }
    })
    from src.api.routers.settings import _preserve_secrets
    current = get_settings()
    patch = {
        "llm": {
            "providers": [{
                "id": "openai",
                "base_url": "https://api.openai.com/v1",
                "adapter": "openai",
                "model": "gpt-4o-mini",
                "api_key": "",
            }]
        }
    }
    preserved = _preserve_secrets(current, patch)
    assert preserved["llm"]["providers"][0]["api_key"] == "keep-me"
