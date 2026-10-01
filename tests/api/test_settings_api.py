"""Settings HTTP API."""

from __future__ import annotations

from fastapi.testclient import TestClient

from src.api.app import create_app


def test_settings_get_and_patch_provider() -> None:
    with TestClient(create_app()) as client:
        res = client.get("/api/v1/settings")
        assert res.status_code == 200
        body = res.json()
        assert "settings" in body
        assert body["settings"]["qdrant"]["mode"] == "embedded"
        assert body["settings"]["llm"]["providers"] == []

        patch = client.patch("/api/v1/settings", json={
            "settings": {
                "llm": {
                    "providers": [{
                        "id": "local",
                        "label": "llama.cpp",
                        "base_url": "http://127.0.0.1:8001/v1",
                        "adapter": "llamacpp_qwen",
                        "model": "qwen",
                        "enabled": True,
                    }],
                    "fallback_order": ["local"],
                    "default_provider": "auto",
                }
            }
        })
        assert patch.status_code == 200
        providers = patch.json()["settings"]["llm"]["providers"]
        assert providers[0]["id"] == "local"
        assert providers[0]["api_key"] == ""

        health = client.get("/api/v1/health")
        assert health.status_code == 200
        assert health.json()["providers_configured"] == 1
        assert health.json()["qdrant"]["mode"] == "embedded"
