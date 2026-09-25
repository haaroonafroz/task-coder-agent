"""Pydantic settings schema for the Missions harness.

This is the source of truth for runtime configuration. Values are persisted
under ``$TASK_CODER_HOME/settings.json`` (non-secrets) and
``$TASK_CODER_HOME/secrets.json`` (API keys). Environment variables remain
an overlay for Docker / CI.
"""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


AdapterName = Literal["generic", "openai", "gemini_openai", "llamacpp_qwen"]
CompatName = Literal["auto", "classic", "openai_reasoning"]
QdrantMode = Literal["embedded", "http", "off"]
EmbeddingBackend = Literal["auto", "hf", "openai", "none"]
SandboxExecutorName = Literal["auto", "bwrap", "native"]
SandboxModeName = Literal["strict", "balanced", "permissive"]
MemoryBackendName = Literal["json", "cognee"]
AgentRoleName = Literal[
    "triage", "orchestrator", "worker", "hotfix", "reviewer", "validator"
]


ADAPTER_UNSUPPORTED_PARAMS: dict[str, set[str]] = {
    "generic": set(),
    "openai": set(),
    "gemini_openai": {"seed", "stream_options"},
    "llamacpp_qwen": set(),
}


class ProviderConfig(BaseModel):
    """One OpenAI-compatible inference connection."""

    id: str
    label: str = ""
    base_url: str
    adapter: AdapterName = "generic"
    model: str = ""
    models_by_role: dict[str, str] = Field(default_factory=dict)
    compat: CompatName = "auto"
    enabled: bool = True
    context_length: Optional[int] = None
    api_key: str = ""
    api_key_set: bool = False

    def display_label(self) -> str:
        return self.label or self.id

    def model_for_role(self, role: str) -> str:
        if role in self.models_by_role and self.models_by_role[role]:
            return self.models_by_role[role]
        return self.model or self.id


class LLMSettings(BaseModel):
    providers: list[ProviderConfig] = Field(default_factory=list)
    default_provider: str = "auto"
    fallback_order: list[str] = Field(default_factory=list)
    seed: int = 42
    context_length: int = 32768


class RoleSettings(BaseModel):
    temperature: float = 0.7
    top_p: float = 0.95
    max_tokens: int = 8192
    thinking: str = "medium"
    thinking_enabled: bool = True


class RolesSettings(BaseModel):
    triage: RoleSettings = Field(
        default_factory=lambda: RoleSettings(
            temperature=0.6, top_p=0.95, max_tokens=8192,
            thinking="low", thinking_enabled=True,
        )
    )
    orchestrator: RoleSettings = Field(
        default_factory=lambda: RoleSettings(
            temperature=0.8, top_p=0.95, max_tokens=24576,
            thinking="xhigh", thinking_enabled=True,
        )
    )
    worker: RoleSettings = Field(
        default_factory=lambda: RoleSettings(
            temperature=0.5, top_p=0.8, max_tokens=12288,
            thinking="low", thinking_enabled=False,
        )
    )
    hotfix: RoleSettings = Field(
        default_factory=lambda: RoleSettings(
            temperature=0.3, top_p=0.8, max_tokens=12288,
            thinking="low", thinking_enabled=True,
        )
    )
    reviewer: RoleSettings = Field(
        default_factory=lambda: RoleSettings(
            temperature=0.4, top_p=0.95, max_tokens=16384,
            thinking="medium", thinking_enabled=True,
        )
    )
    validator: RoleSettings = Field(
        default_factory=lambda: RoleSettings(
            temperature=0.6, top_p=0.95, max_tokens=16384,
            thinking="medium", thinking_enabled=True,
        )
    )

    def for_role(self, role: str) -> RoleSettings:
        try:
            return getattr(self, role)
        except AttributeError as exc:
            raise KeyError(f"Unknown agent role: {role}") from exc


class QdrantSettings(BaseModel):
    mode: QdrantMode = "embedded"
    path: str = ""
    url: str = ""
    api_key: str = ""
    api_key_set: bool = False
    collection: str = "agent_skills"
    dense_name: str = "dense"
    sparse_name: str = "sparse"
    dense_dims: int = 768


class EmbeddingSettings(BaseModel):
    backend: EmbeddingBackend = "auto"
    hf_model: str = "BAAI/bge-base-en-v1.5"
    openai_model: str = "text-embedding-3-small"
    openai_dims: int = 768


class SandboxSettings(BaseModel):
    executor: SandboxExecutorName = "auto"
    mode: SandboxModeName = "balanced"
    require_bwrap: bool = False


class RuntimeSettings(BaseModel):
    max_worker_batch_calls: int = 3
    max_worker_history_turns: int = 40
    max_worker_tool_calls: int = 20
    worker_contract_autorun_max: int = 8
    worker_autorun_stdout_chars: int = 600
    worker_autorun_stderr_chars: int = 400
    max_same_tool_failures: int = 2
    max_consecutive_tool_failures: int = 5
    max_replans_per_milestone: int = 2
    max_hotfix_tool_calls: int = 10
    max_review_tool_calls: int = 12
    worker_ui_nudge_after: int = 4
    worker_ui_strong_nudge_after: int = 8
    max_orchestrator_explore_calls: int = 10
    max_orchestrator_explore_light: int = 5
    orchestrator_explore_enabled: bool = True
    log_level: str = "ERROR"
    sandbox: SandboxSettings = Field(default_factory=SandboxSettings)


class ObservabilitySettings(BaseModel):
    api_telemetry: bool = True
    phoenix_host: str = "127.0.0.1"
    phoenix_port: int = 6006
    phoenix_external: bool = False
    auto_eval: bool = False
    eval_llm_judge: bool = False
    eval_phoenix_export: bool = True


class MemorySettings(BaseModel):
    backend: MemoryBackendName = "json"


class MissionsSettings(BaseModel):
    llm: LLMSettings = Field(default_factory=LLMSettings)
    roles: RolesSettings = Field(default_factory=RolesSettings)
    qdrant: QdrantSettings = Field(default_factory=QdrantSettings)
    embeddings: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    runtime: RuntimeSettings = Field(default_factory=RuntimeSettings)
    observability: ObservabilitySettings = Field(default_factory=ObservabilitySettings)
    memory: MemorySettings = Field(default_factory=MemorySettings)

    def provider(self, provider_id: str) -> Optional[ProviderConfig]:
        for item in self.llm.providers:
            if item.id == provider_id:
                return item
        return None

    def enabled_providers(self) -> list[ProviderConfig]:
        return [item for item in self.llm.providers if item.enabled]

    def fallback_ids(self) -> list[str]:
        enabled = {item.id for item in self.enabled_providers()}
        ordered = [pid for pid in self.llm.fallback_order if pid in enabled]
        if ordered:
            return ordered
        return [item.id for item in self.enabled_providers()]

    def persistable_dict(self) -> dict[str, Any]:
        """JSON-safe dump with secrets stripped."""
        data = self.model_dump()
        for provider in data.get("llm", {}).get("providers", []):
            provider.pop("api_key", None)
            provider.pop("api_key_set", None)
        if "qdrant" in data:
            data["qdrant"].pop("api_key", None)
            data["qdrant"].pop("api_key_set", None)
        return data

    def redacted_dict(self) -> dict[str, Any]:
        data = self.model_dump()
        for provider in data.get("llm", {}).get("providers", []):
            provider["api_key_set"] = bool(provider.get("api_key"))
            provider["api_key"] = ""
        if "qdrant" in data:
            data["qdrant"]["api_key_set"] = bool(data["qdrant"].get("api_key"))
            data["qdrant"]["api_key"] = ""
        return data


PROVIDER_PRESETS: list[dict[str, Any]] = [
    {
        "id": "local",
        "label": "llama.cpp (local)",
        "base_url": "http://127.0.0.1:8001/v1",
        "adapter": "llamacpp_qwen",
        "model": "",
        "context_length": 32768,
    },
    {
        "id": "ollama",
        "label": "Ollama",
        "base_url": "http://127.0.0.1:11434/v1",
        "adapter": "generic",
        "model": "",
    },
    {
        "id": "lmstudio",
        "label": "LM Studio",
        "base_url": "http://127.0.0.1:1234/v1",
        "adapter": "generic",
        "model": "",
    },
    {
        "id": "vllm",
        "label": "vLLM",
        "base_url": "http://127.0.0.1:8000/v1",
        "adapter": "generic",
        "model": "",
    },
    {
        "id": "openai",
        "label": "OpenAI",
        "base_url": "https://api.openai.com/v1",
        "adapter": "openai",
        "model": "gpt-4o",
        "models_by_role": {"worker": "gpt-4o-mini", "hotfix": "gpt-4o-mini"},
    },
    {
        "id": "gpt4o",
        "label": "OpenAI (gpt4o)",
        "base_url": "https://api.openai.com/v1",
        "adapter": "openai",
        "model": "gpt-4o",
        "models_by_role": {"worker": "gpt-4o-mini", "hotfix": "gpt-4o-mini"},
    },
    {
        "id": "gemini",
        "label": "Gemini",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "adapter": "gemini_openai",
        "model": "gemini-3.1-flash-lite",
    },
    {
        "id": "openrouter",
        "label": "OpenRouter",
        "base_url": "https://openrouter.ai/api/v1",
        "adapter": "openai",
        "model": "",
    },
]
