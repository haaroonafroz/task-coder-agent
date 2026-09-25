"""Load, persist, and overlay Missions settings."""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any, Optional

from src.settings.schema import (
    MissionsSettings,
    ProviderConfig,
    PROVIDER_PRESETS,
)

REPO_ROOT = Path(__file__).resolve().parents[2]

_LOCK = threading.RLock()
_SETTINGS: Optional[MissionsSettings] = None
_MIGRATED_FROM_ENV = False


def resolve_home() -> Path:
    """Return the harness state directory.

    Precedence:
      1. ``TASK_CODER_HOME``
      2. Repository root when ``sessions/`` or ``.env`` already exist
         (keeps an existing developer checkout working)
      3. ``~/.missions`` for new users
    """
    env = os.getenv("TASK_CODER_HOME", "").strip()
    if env:
        return Path(env).expanduser().resolve()
    if (REPO_ROOT / "sessions").exists() or (REPO_ROOT / ".env").exists():
        return REPO_ROOT
    return (Path.home() / ".missions").resolve()


def settings_path(home: Optional[Path] = None) -> Path:
    return (home or resolve_home()) / "settings.json"


def secrets_path(home: Optional[Path] = None) -> Path:
    return (home or resolve_home()) / "secrets.json"


def qdrant_path(home: Optional[Path] = None) -> Path:
    return (home or resolve_home()) / "qdrant"


def models_cache_path(home: Optional[Path] = None) -> Path:
    return (home or resolve_home()) / "models"


def ensure_home(home: Optional[Path] = None) -> Path:
    root = home or resolve_home()
    root.mkdir(parents=True, exist_ok=True)
    (root / "sessions").mkdir(parents=True, exist_ok=True)
    (root / "managed-workspaces").mkdir(parents=True, exist_ok=True)
    qdrant_path(root).mkdir(parents=True, exist_ok=True)
    models_cache_path(root).mkdir(parents=True, exist_ok=True)
    return root


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _write_json(path: Path, payload: dict[str, Any], *, mode: int = 0o644) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)
    try:
        os.chmod(path, mode)
    except OSError:
        pass


def _parse_dotenv(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.is_file():
        return values
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if "#" in value and not value.startswith("http"):
            value = value.split("#", 1)[0].strip().strip("'").strip('"')
        if key:
            values[key] = value
    return values


def _as_bool(raw: str, default: bool) -> bool:
    text = raw.strip().lower()
    if not text:
        return default
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return default


def _as_int(raw: str, default: int) -> int:
    try:
        return int(raw.strip())
    except (TypeError, ValueError):
        return default


def _as_float(raw: str, default: float) -> float:
    try:
        return float(raw.strip())
    except (TypeError, ValueError):
        return default


def _preset(preset_id: str) -> dict[str, Any]:
    for item in PROVIDER_PRESETS:
        if item["id"] == preset_id:
            return dict(item)
    return {"id": preset_id, "label": preset_id, "base_url": "", "adapter": "generic"}


def migrate_from_env_file(env_file: Path) -> tuple[MissionsSettings, dict[str, str]]:
    """Build settings + secrets from a legacy ``.env`` file."""
    env = _parse_dotenv(env_file)
    settings = MissionsSettings()
    secrets: dict[str, str] = {}
    providers: list[ProviderConfig] = []
    fallback: list[str] = []

    local_url = env.get("LLM_SPECULATIVE_URL", "").strip()
    local_model = env.get("TARGET_MODEL", "").strip() or env.get("MODEL_ALIAS", "").strip()
    if local_url or local_model or env.get("TARGET_MODEL_GGUF", "").strip():
        spec = _preset("local")
        if local_url:
            spec["base_url"] = local_url
        if local_model:
            spec["model"] = local_model
        ctx = env.get("CONTEXT_LEN", "").strip()
        if ctx:
            spec["context_length"] = _as_int(ctx, spec.get("context_length") or 32768)
        providers.append(ProviderConfig(**spec))
        fallback.append("local")

    gemini_key = env.get("GEMINI_API_KEY", "").strip()
    gemini_model = env.get("GEMINI_MODEL", "").strip()
    if gemini_key or gemini_model:
        spec = _preset("gemini")
        if gemini_model:
            spec["model"] = gemini_model
        providers.append(ProviderConfig(**spec))
        if gemini_key:
            secrets["provider.gemini"] = gemini_key
        fallback.append("gemini")

    openai_key = env.get("OPENAI_API_KEY", "").strip()
    openai_model = env.get("OPENAI_LLM_MODEL", "").strip()
    openai_worker = env.get("OPENAI_LLM_MODEL_ENRICHMENT", "").strip()
    if openai_key or openai_model:
        spec = _preset("gpt4o")
        if openai_model:
            spec["model"] = openai_model
        if openai_worker:
            spec["models_by_role"] = {
                "worker": openai_worker,
                "hotfix": openai_worker,
            }
        providers.append(ProviderConfig(**spec))
        if openai_key:
            secrets["provider.gpt4o"] = openai_key
            secrets["provider.openai"] = openai_key
        fallback.append("gpt4o")

    settings.llm.providers = providers
    settings.llm.fallback_order = fallback
    settings.llm.default_provider = "auto" if len(fallback) > 1 else (fallback[0] if fallback else "auto")
    settings.llm.seed = _as_int(env.get("LLM_SEED", ""), settings.llm.seed)
    settings.llm.context_length = _as_int(env.get("CONTEXT_LEN", ""), settings.llm.context_length)

    global_temp = _as_float(env.get("LLM_TEMPERATURE", ""), 0.7)
    global_top_p = _as_float(env.get("LLM_TOP_P", ""), 0.95)
    for role in ("triage", "orchestrator", "worker", "hotfix", "reviewer", "validator"):
        role_cfg = settings.roles.for_role(role)
        role_cfg.temperature = _as_float(env.get(f"LLM_TEMPERATURE_{role.upper()}", ""), global_temp)
        role_cfg.top_p = _as_float(env.get(f"LLM_TOP_P_{role.upper()}", ""), global_top_p)
        role_cfg.max_tokens = _as_int(
            env.get(f"MAX_TOKENS_{role.upper()}", ""),
            role_cfg.max_tokens,
        )
        effort = env.get(f"LOCAL_REASONING_EFFORT_{role.upper()}", "").strip()
        enabled_raw = env.get(f"LOCAL_ENABLE_THINKING_{role.upper()}", "").strip()
        compact = env.get(f"LOCAL_THINKING_{role.upper()}", "").strip() or env.get(
            f"LOCAL_THINKING_LEVEL_{role.upper()}", ""
        ).strip()
        if effort:
            role_cfg.thinking = effort.lower()
        elif compact:
            role_cfg.thinking = compact.lower()
        if enabled_raw:
            role_cfg.thinking_enabled = _as_bool(enabled_raw, role_cfg.thinking_enabled)
        elif compact:
            role_cfg.thinking_enabled = compact.lower() != "off"

    qdrant_url = env.get("QDRANT_URL", "").strip()
    qdrant_key = env.get("QDRANT_API_KEY", "").strip()
    if qdrant_url:
        settings.qdrant.mode = "http"
        settings.qdrant.url = qdrant_url
        if qdrant_key:
            secrets["qdrant"] = qdrant_key
    else:
        settings.qdrant.mode = "embedded"
    settings.qdrant.collection = env.get("QDRANT_COLLECTION", "") or settings.qdrant.collection
    settings.qdrant.dense_name = env.get("QDRANT_DENSE_VECTOR_NAME", "") or settings.qdrant.dense_name
    settings.qdrant.sparse_name = env.get("QDRANT_SPARSE_VECTOR_NAME", "") or settings.qdrant.sparse_name
    settings.qdrant.dense_dims = _as_int(env.get("QDRANT_DENSE_VECTOR_DIMS", ""), settings.qdrant.dense_dims)

    if env.get("OPENAI_EMBEDDING_MODEL", "").strip():
        settings.embeddings.openai_model = env["OPENAI_EMBEDDING_MODEL"].strip()
    settings.embeddings.openai_dims = _as_int(
        env.get("OPENAI_EMBEDDING_DIMS", ""), settings.embeddings.openai_dims
    )
    hf_token = env.get("HF_TOKEN", "").strip()
    if hf_token:
        secrets["hf"] = hf_token

    settings.observability.phoenix_host = env.get("PHOENIX_HOST", "") or settings.observability.phoenix_host
    settings.observability.phoenix_port = _as_int(
        env.get("PHOENIX_PORT", ""), settings.observability.phoenix_port
    )
    settings.observability.phoenix_external = _as_bool(
        env.get("PHOENIX_EXTERNAL", ""), settings.observability.phoenix_external
    )
    settings.observability.api_telemetry = _as_bool(
        env.get("MISSIONS_API_TELEMETRY", ""), settings.observability.api_telemetry
    )
    settings.observability.auto_eval = _as_bool(
        env.get("MISSIONS_AUTO_EVAL", ""), settings.observability.auto_eval
    )
    settings.observability.eval_llm_judge = _as_bool(
        env.get("MISSIONS_EVAL_LLM_JUDGE", ""), settings.observability.eval_llm_judge
    )
    settings.observability.eval_phoenix_export = _as_bool(
        env.get("MISSIONS_EVAL_PHOENIX_EXPORT", ""),
        settings.observability.eval_phoenix_export,
    )

    settings.memory.backend = (  # type: ignore[assignment]
        env.get("MISSIONS_MEMORY_BACKEND", "").strip().lower() or settings.memory.backend
    )

    rt = settings.runtime
    rt.max_worker_batch_calls = _as_int(env.get("MAX_WORKER_BATCH_CALLS", ""), rt.max_worker_batch_calls)
    rt.max_worker_history_turns = _as_int(env.get("MAX_WORKER_HISTORY_TURNS", ""), rt.max_worker_history_turns)
    rt.max_worker_tool_calls = _as_int(env.get("MAX_WORKER_TOOL_CALLS", ""), rt.max_worker_tool_calls)
    rt.worker_contract_autorun_max = _as_int(
        env.get("WORKER_CONTRACT_AUTORUN_MAX", ""), rt.worker_contract_autorun_max
    )
    rt.max_same_tool_failures = _as_int(env.get("MAX_SAME_TOOL_FAILURES", ""), rt.max_same_tool_failures)
    rt.max_consecutive_tool_failures = _as_int(
        env.get("MAX_CONSECUTIVE_TOOL_FAILURES", ""), rt.max_consecutive_tool_failures
    )
    rt.max_replans_per_milestone = _as_int(
        env.get("MAX_REPLANS_PER_MILESTONE", ""), rt.max_replans_per_milestone
    )
    rt.max_hotfix_tool_calls = _as_int(env.get("MAX_HOTFIX_TOOL_CALLS", ""), rt.max_hotfix_tool_calls)
    rt.max_review_tool_calls = _as_int(env.get("MAX_REVIEW_TOOL_CALLS", ""), rt.max_review_tool_calls)
    rt.log_level = env.get("LOG_LEVEL", "") or rt.log_level
    rt.sandbox.executor = (  # type: ignore[assignment]
        env.get("SANDBOX_EXECUTOR", "").strip().lower() or rt.sandbox.executor
    )
    rt.sandbox.mode = (  # type: ignore[assignment]
        env.get("SANDBOX_MODE", "").strip().lower() or rt.sandbox.mode
    )
    rt.sandbox.require_bwrap = _as_bool(
        env.get("SANDBOX_REQUIRE_BWRAP", ""), rt.sandbox.require_bwrap
    )
    return settings, secrets


def _apply_env_overlay(settings: MissionsSettings, secrets: dict[str, str]) -> None:
    """Environment variables win over JSON (Docker / CI)."""

    def env(name: str) -> str:
        return os.getenv(name, "").strip()

    openai_key = env("OPENAI_API_KEY")
    if openai_key:
        secrets["provider.gpt4o"] = openai_key
        secrets["provider.openai"] = openai_key
        if settings.provider("gpt4o") is None and settings.provider("openai") is None:
            spec = _preset("gpt4o")
            settings.llm.providers.append(ProviderConfig(**spec))
            if "gpt4o" not in settings.llm.fallback_order:
                settings.llm.fallback_order.append("gpt4o")

    gemini_key = env("GEMINI_API_KEY")
    if gemini_key:
        secrets["provider.gemini"] = gemini_key
        if settings.provider("gemini") is None:
            settings.llm.providers.append(ProviderConfig(**_preset("gemini")))
            if "gemini" not in settings.llm.fallback_order:
                settings.llm.fallback_order.append("gemini")

    local_url = env("LLM_SPECULATIVE_URL")
    local_model = env("TARGET_MODEL")
    if local_url or local_model:
        existing = settings.provider("local")
        if existing is None:
            spec = _preset("local")
            if local_url:
                spec["base_url"] = local_url
            if local_model:
                spec["model"] = local_model
            settings.llm.providers.insert(0, ProviderConfig(**spec))
            if "local" not in settings.llm.fallback_order:
                settings.llm.fallback_order.insert(0, "local")
        else:
            if local_url:
                existing.base_url = local_url
            if local_model:
                existing.model = local_model

    qdrant_url = env("QDRANT_URL")
    if qdrant_url:
        settings.qdrant.mode = "http"
        settings.qdrant.url = qdrant_url
    qdrant_key = env("QDRANT_API_KEY")
    if qdrant_key:
        secrets["qdrant"] = qdrant_key
    qdrant_mode = env("MISSIONS_QDRANT_MODE")
    if qdrant_mode in {"embedded", "http", "off"}:
        settings.qdrant.mode = qdrant_mode  # type: ignore[assignment]

    embeddings = env("MISSIONS_EMBEDDINGS")
    if embeddings in {"auto", "hf", "openai", "none"}:
        settings.embeddings.backend = embeddings  # type: ignore[assignment]

    hf_token = env("HF_TOKEN")
    if hf_token:
        secrets["hf"] = hf_token

    if env("SANDBOX_EXECUTOR") in {"auto", "bwrap", "native"}:
        settings.runtime.sandbox.executor = env("SANDBOX_EXECUTOR")  # type: ignore[assignment]
    if env("SANDBOX_MODE") in {"strict", "balanced", "permissive"}:
        settings.runtime.sandbox.mode = env("SANDBOX_MODE")  # type: ignore[assignment]
    if env("SANDBOX_REQUIRE_BWRAP"):
        settings.runtime.sandbox.require_bwrap = _as_bool(
            env("SANDBOX_REQUIRE_BWRAP"), settings.runtime.sandbox.require_bwrap
        )

    if env("MISSIONS_API_TELEMETRY"):
        settings.observability.api_telemetry = _as_bool(
            env("MISSIONS_API_TELEMETRY"), settings.observability.api_telemetry
        )
    if env("MISSIONS_MEMORY_BACKEND") in {"json", "cognee"}:
        settings.memory.backend = env("MISSIONS_MEMORY_BACKEND")  # type: ignore[assignment]
    if env("MISSIONS_AUTO_EVAL"):
        settings.observability.auto_eval = _as_bool(
            env("MISSIONS_AUTO_EVAL"), settings.observability.auto_eval
        )
    if env("MISSIONS_EVAL_LLM_JUDGE"):
        settings.observability.eval_llm_judge = _as_bool(
            env("MISSIONS_EVAL_LLM_JUDGE"), settings.observability.eval_llm_judge
        )
    if env("PHOENIX_HOST"):
        settings.observability.phoenix_host = env("PHOENIX_HOST")
    if env("PHOENIX_PORT"):
        settings.observability.phoenix_port = _as_int(
            env("PHOENIX_PORT"), settings.observability.phoenix_port
        )
    if env("PHOENIX_EXTERNAL"):
        settings.observability.phoenix_external = _as_bool(
            env("PHOENIX_EXTERNAL"), settings.observability.phoenix_external
        )


def _inject_secrets(settings: MissionsSettings, secrets: dict[str, str]) -> None:
    for provider in settings.llm.providers:
        key = secrets.get(f"provider.{provider.id}", "")
        if key:
            provider.api_key = key
        provider.api_key_set = bool(provider.api_key)
    qdrant_key = secrets.get("qdrant", "")
    if qdrant_key:
        settings.qdrant.api_key = qdrant_key
    settings.qdrant.api_key_set = bool(settings.qdrant.api_key)


def _extract_secrets(settings: MissionsSettings) -> dict[str, str]:
    secrets: dict[str, str] = {}
    for provider in settings.llm.providers:
        if provider.api_key:
            secrets[f"provider.{provider.id}"] = provider.api_key
    if settings.qdrant.api_key:
        secrets["qdrant"] = settings.qdrant.api_key
    return secrets


def load_settings(*, persist_migration: bool = True) -> MissionsSettings:
    """Load settings from disk, migrating ``.env`` on first run."""
    global _MIGRATED_FROM_ENV
    home = ensure_home()
    path = settings_path(home)
    secrets = dict(_read_json(secrets_path(home)))
    migrated = False

    if path.is_file():
        payload = _read_json(path)
        settings = MissionsSettings.model_validate(payload or {})
    else:
        env_file = home / ".env"
        if not env_file.is_file() and home.resolve() == REPO_ROOT.resolve():
            env_file = REPO_ROOT / ".env"
        if env_file.is_file():
            settings, migrated_secrets = migrate_from_env_file(env_file)
            secrets.update({k: v for k, v in migrated_secrets.items() if v})
            migrated = True
            _MIGRATED_FROM_ENV = True
        else:
            settings = MissionsSettings()

    if not settings.qdrant.path:
        settings.qdrant.path = str(qdrant_path(home))

    _apply_env_overlay(settings, secrets)
    _inject_secrets(settings, secrets)

    if persist_migration and (migrated or not path.is_file()):
        save_settings(settings, secrets_update=secrets, home=home)

    os.environ.setdefault("HF_HOME", str(models_cache_path(home) / "hf"))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(models_cache_path(home) / "hf"))
    return settings


def save_settings(
    settings: MissionsSettings,
    *,
    secrets_update: Optional[dict[str, str]] = None,
    home: Optional[Path] = None,
) -> MissionsSettings:
    root = ensure_home(home)
    existing_secrets = dict(_read_json(secrets_path(root)))
    extracted = _extract_secrets(settings)
    if secrets_update:
        existing_secrets.update({k: v for k, v in secrets_update.items() if v})
    existing_secrets.update(extracted)
    # Drop empty keys.
    existing_secrets = {k: v for k, v in existing_secrets.items() if v}

    if not settings.qdrant.path:
        settings.qdrant.path = str(qdrant_path(root))

    _write_json(settings_path(root), settings.persistable_dict())
    _write_json(secrets_path(root), existing_secrets, mode=0o600)
    _inject_secrets(settings, existing_secrets)
    return settings


def get_settings() -> MissionsSettings:
    global _SETTINGS
    with _LOCK:
        if _SETTINGS is None:
            _SETTINGS = load_settings()
        return _SETTINGS


def reset_settings() -> MissionsSettings:
    """Drop the cached singleton (used by tests and after PATCH)."""
    global _SETTINGS
    with _LOCK:
        _SETTINGS = load_settings(persist_migration=False)
        return _SETTINGS


def replace_settings(settings: MissionsSettings, *, persist: bool = True) -> MissionsSettings:
    global _SETTINGS
    with _LOCK:
        if persist:
            settings = save_settings(settings)
        _SETTINGS = settings
        return settings


def update_settings(patch: dict[str, Any]) -> MissionsSettings:
    """Deep-merge a JSON patch, persist, and refresh the singleton."""
    current = get_settings().model_dump()
    merged = _deep_merge(current, patch)
    settings = MissionsSettings.model_validate(merged)
    return replace_settings(settings, persist=True)


def migrated_from_env() -> bool:
    return _MIGRATED_FROM_ENV


def _deep_merge(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in patch.items():
        if (
            key in out
            and isinstance(out[key], dict)
            and isinstance(value, dict)
            and key != "models_by_role"
        ):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out
