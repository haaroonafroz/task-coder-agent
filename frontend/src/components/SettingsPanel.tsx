import { useEffect, useState } from "react";
import { api } from "../api/client";
import type { MissionsSettingsPayload, ProviderPreset, SettingsResponse } from "../api/types";

interface Props {
  open: boolean;
  onClose: () => void;
  onSaved?: (data: SettingsResponse) => void;
}

type Tab = "models" | "agent" | "integrations";

export function SettingsPanel({ open, onClose, onSaved }: Props) {
  const [tab, setTab] = useState<Tab>("models");
  const [data, setData] = useState<SettingsResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);
  const [probeMsg, setProbeMsg] = useState<string | null>(null);

  useEffect(() => {
    if (!open) return;
    api.getSettings()
      .then(setData)
      .catch((e) => setError(String(e)));
  }, [open]);

  if (!open) return null;

  const settings = data?.settings;

  const save = async (next: MissionsSettingsPayload) => {
    setSaving(true);
    setError(null);
    try {
      const saved = await api.patchSettings(next as unknown as Record<string, unknown>);
      setData(saved);
      onSaved?.(saved);
    } catch (e) {
      setError(String(e));
    } finally {
      setSaving(false);
    }
  };

  const addPreset = (preset: ProviderPreset) => {
    if (!settings) return;
    if (settings.llm.providers.some((p) => p.id === preset.id)) {
      setError(`Provider '${preset.id}' already exists`);
      return;
    }
    const providers = [
      ...settings.llm.providers,
      {
        id: preset.id,
        label: preset.label,
        base_url: preset.base_url,
        adapter: preset.adapter,
        model: preset.model || "",
        models_by_role: preset.models_by_role || {},
        enabled: true,
        context_length: preset.context_length ?? null,
        api_key: "",
        api_key_set: false,
      },
    ];
    const fallback = settings.llm.fallback_order.includes(preset.id)
      ? settings.llm.fallback_order
      : [...settings.llm.fallback_order, preset.id];
    void save({
      ...settings,
      llm: { ...settings.llm, providers, fallback_order: fallback },
    });
  };

  const updateProvider = (id: string, patch: Record<string, unknown>) => {
    if (!settings) return;
    const providers = settings.llm.providers.map((p) =>
      p.id === id ? { ...p, ...patch } : p
    );
    void save({ ...settings, llm: { ...settings.llm, providers } });
  };

  const removeProvider = (id: string) => {
    if (!settings) return;
    const providers = settings.llm.providers.filter((p) => p.id !== id);
    void save({
      ...settings,
      llm: {
        ...settings.llm,
        providers,
        fallback_order: settings.llm.fallback_order.filter((x) => x !== id),
      },
    });
  };

  const test = async (id: string) => {
    if (!settings) return;
    const provider = settings.llm.providers.find((p) => p.id === id);
    if (!provider) return;
    setProbeMsg("Testing…");
    try {
      const result = await api.testProvider({
        base_url: provider.base_url,
        provider_id: provider.id,
        api_key: provider.api_key || undefined,
      });
      if (result.ok) {
        setProbeMsg(`${id}: ${result.models.length} models (${result.models.slice(0, 4).join(", ")})`);
        if (result.models.length && !provider.model) {
          updateProvider(id, { model: result.models[0] });
        }
      } else {
        setProbeMsg(`${id}: ${result.error || "unreachable"}`);
      }
    } catch (e) {
      setProbeMsg(String(e));
    }
  };

  return (
    <div className="settings-overlay" onClick={onClose}>
      <div className="settings-panel" onClick={(e) => e.stopPropagation()}>
        <div className="panel-header">
          <span>Settings</span>
          <button onClick={onClose}>Close</button>
        </div>
        <div className="settings-tabs">
          {(["models", "agent", "integrations"] as Tab[]).map((item) => (
            <button
              key={item}
              className={tab === item ? "primary" : ""}
              onClick={() => setTab(item)}
            >
              {item}
            </button>
          ))}
        </div>
        <div className="settings-body">
          {error && <div className="settings-error">{error}</div>}
          {data?.container && (
            <div className="settings-banner">
              Running in Docker. Attach host folders only if they are bind-mounted,
              and use the container path in the session form.
            </div>
          )}
          {!settings && <div className="empty-state">Loading…</div>}
          {settings && tab === "models" && (
            <div>
              <p className="settings-help">
                The harness is an OpenAI-compatible client. Point it at llama.cpp,
                Ollama, LM Studio, vLLM, OpenAI, Gemini, or OpenRouter. Inference
                is not started by this app.
              </p>
              <div className="settings-row">
                <label>Default</label>
                <select
                  value={settings.llm.default_provider}
                  onChange={(e) =>
                    void save({
                      ...settings,
                      llm: { ...settings.llm, default_provider: e.target.value },
                    })
                  }
                >
                  <option value="auto">auto</option>
                  {settings.llm.providers.map((p) => (
                    <option key={p.id} value={p.id}>{p.label || p.id}</option>
                  ))}
                </select>
              </div>
              {settings.llm.providers.map((p) => (
                <div key={p.id} className="settings-card">
                  <div className="settings-card-title">
                    <strong>{p.label || p.id}</strong>
                    <label>
                      <input
                        type="checkbox"
                        checked={p.enabled}
                        onChange={(e) => updateProvider(p.id, { enabled: e.target.checked })}
                      />{" "}
                      enabled
                    </label>
                  </div>
                  <input
                    defaultValue={p.base_url}
                    key={`${p.id}-url-${p.base_url}`}
                    placeholder="http://127.0.0.1:8001/v1"
                    onBlur={(e) => {
                      if (e.target.value !== p.base_url) updateProvider(p.id, { base_url: e.target.value });
                    }}
                  />
                  <input
                    defaultValue={p.model}
                    key={`${p.id}-model-${p.model}`}
                    placeholder="model name"
                    onBlur={(e) => {
                      if (e.target.value !== p.model) updateProvider(p.id, { model: e.target.value });
                    }}
                  />
                  <input
                    type="password"
                    placeholder={p.api_key_set ? "API key set — paste to replace" : "API key (optional for local)"}
                    onBlur={(e) => {
                      if (e.target.value) updateProvider(p.id, { api_key: e.target.value });
                    }}
                  />
                  <div className="settings-row">
                    <select
                      value={p.adapter}
                      onChange={(e) => updateProvider(p.id, { adapter: e.target.value })}
                    >
                      <option value="generic">generic</option>
                      <option value="openai">openai</option>
                      <option value="gemini_openai">gemini</option>
                      <option value="llamacpp_qwen">llama.cpp / Qwen</option>
                    </select>
                    <button onClick={() => void test(p.id)}>Test</button>
                    <button className="danger" onClick={() => removeProvider(p.id)}>Remove</button>
                  </div>
                </div>
              ))}
              {probeMsg && <div className="settings-help">{probeMsg}</div>}
              <div className="settings-help">Add a preset:</div>
              <div className="settings-presets">
                {(data?.presets || []).map((preset) => (
                  <button key={preset.id} onClick={() => addPreset(preset)}>
                    {preset.label}
                  </button>
                ))}
              </div>
            </div>
          )}
          {settings && tab === "agent" && (
            <div>
              {Object.entries(settings.roles).map(([role, cfg]) => (
                <div key={role} className="settings-card">
                  <strong>{role}</strong>
                  <div className="settings-row">
                    <label>temp</label>
                    <input
                      type="number"
                      step="0.05"
                      value={cfg.temperature}
                      onChange={(e) =>
                        void save({
                          ...settings,
                          roles: {
                            ...settings.roles,
                            [role]: { ...cfg, temperature: Number(e.target.value) },
                          },
                        })
                      }
                    />
                    <label>max tokens</label>
                    <input
                      type="number"
                      value={cfg.max_tokens}
                      onChange={(e) =>
                        void save({
                          ...settings,
                          roles: {
                            ...settings.roles,
                            [role]: { ...cfg, max_tokens: Number(e.target.value) },
                          },
                        })
                      }
                    />
                  </div>
                  <div className="settings-row">
                    <label>thinking</label>
                    <input
                      value={cfg.thinking}
                      onChange={(e) =>
                        void save({
                          ...settings,
                          roles: {
                            ...settings.roles,
                            [role]: { ...cfg, thinking: e.target.value },
                          },
                        })
                      }
                    />
                    <label>
                      <input
                        type="checkbox"
                        checked={cfg.thinking_enabled}
                        onChange={(e) =>
                          void save({
                            ...settings,
                            roles: {
                              ...settings.roles,
                              [role]: { ...cfg, thinking_enabled: e.target.checked },
                            },
                          })
                        }
                      />{" "}
                      enabled
                    </label>
                  </div>
                </div>
              ))}
            </div>
          )}
          {settings && tab === "integrations" && (
            <div>
              <div className="settings-card">
                <strong>Qdrant</strong>
                <p className="settings-help">
                  Default is a local embedded store (qdrant-client 1.18.0, Qdrant 1.18.x)
                  under the harness home. Cloud HTTP remains optional.
                </p>
                <select
                  value={settings.qdrant.mode}
                  onChange={(e) =>
                    void save({
                      ...settings,
                      qdrant: {
                        ...settings.qdrant,
                        mode: e.target.value as MissionsSettingsPayload["qdrant"]["mode"],
                      },
                    })
                  }
                >
                  <option value="embedded">Local (bundled)</option>
                  <option value="http">Existing HTTP / Cloud</option>
                  <option value="off">Off (keyword search)</option>
                </select>
                {settings.qdrant.mode === "http" && (
                  <>
                    <input
                      value={settings.qdrant.url}
                      placeholder="https://….qdrant.io:6333"
                      onChange={(e) =>
                        void save({
                          ...settings,
                          qdrant: { ...settings.qdrant, url: e.target.value },
                        })
                      }
                    />
                    <input
                      type="password"
                      placeholder={settings.qdrant.api_key_set ? "API key set — paste to replace" : "Qdrant API key"}
                      onChange={(e) =>
                        void save({
                          ...settings,
                          qdrant: { ...settings.qdrant, api_key: e.target.value },
                        })
                      }
                    />
                  </>
                )}
              </div>
              <div className="settings-card">
                <strong>Embeddings</strong>
                <select
                  value={settings.embeddings.backend}
                  onChange={(e) =>
                    void save({
                      ...settings,
                      embeddings: {
                        ...settings.embeddings,
                        backend: e.target.value as MissionsSettingsPayload["embeddings"]["backend"],
                      },
                    })
                  }
                >
                  <option value="auto">auto (local BGE, then OpenAI)</option>
                  <option value="hf">HuggingFace BGE (local)</option>
                  <option value="openai">OpenAI embeddings</option>
                  <option value="none">none (keyword only)</option>
                </select>
              </div>
              <div className="settings-help">Config home: {data?.home}</div>
            </div>
          )}
          {saving && <div className="settings-help">Saving…</div>}
        </div>
      </div>
    </div>
  );
}
