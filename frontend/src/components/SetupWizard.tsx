import { useEffect, useState } from "react";
import { api } from "../api/client";
import type { ProviderPreset, SettingsResponse } from "../api/types";

interface Props {
  onComplete: (data: SettingsResponse) => void;
}

export function SetupWizard({ onComplete }: Props) {
  const [data, setData] = useState<SettingsResponse | null>(null);
  const [kind, setKind] = useState<"local" | "cloud">("local");
  const [presetId, setPresetId] = useState("local");
  const [baseUrl, setBaseUrl] = useState("http://127.0.0.1:8001/v1");
  const [apiKey, setApiKey] = useState("");
  const [model, setModel] = useState("");
  const [models, setModels] = useState<string[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    api.getSettings().then(setData).catch((e) => setError(String(e)));
  }, []);

  const presets = data?.presets || [];
  const applyPreset = (id: string, nextKind: "local" | "cloud") => {
    setKind(nextKind);
    setPresetId(id);
    const preset = presets.find((p) => p.id === id);
    if (preset) {
      setBaseUrl(preset.base_url);
      if (preset.model) setModel(preset.model);
    }
  };

  const test = async () => {
    setBusy(true);
    setError(null);
    try {
      const result = await api.testProvider({
        base_url: baseUrl,
        api_key: apiKey || undefined,
      });
      if (!result.ok) {
        setError(result.error || "Could not reach /v1/models");
        setModels([]);
      } else {
        setModels(result.models);
        if (!model && result.models[0]) setModel(result.models[0]);
      }
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(false);
    }
  };

  const finish = async () => {
    if (!data) return;
    setBusy(true);
    setError(null);
    const preset: ProviderPreset | undefined = presets.find((p) => p.id === presetId);
    const id = presetId || (kind === "local" ? "local" : "openai");
    const provider = {
      id,
      label: preset?.label || id,
      base_url: baseUrl,
      adapter: preset?.adapter || (kind === "local" ? "llamacpp_qwen" : "openai"),
      model,
      models_by_role: preset?.models_by_role || {},
      enabled: true,
      context_length: preset?.context_length ?? null,
      api_key: apiKey,
      api_key_set: Boolean(apiKey),
    };
    try {
      const saved = await api.patchSettings({
        llm: {
          ...data.settings.llm,
          providers: [provider],
          default_provider: "auto",
          fallback_order: [id],
        },
      });
      onComplete(saved);
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="wizard-overlay">
      <div className="wizard-card">
        <h1>Set up Missions</h1>
        <p>
          This app does not download or start a coding model. Point it at a
          local OpenAI-compatible server, or paste a cloud API key.
        </p>
        <div className="settings-row">
          <button className={kind === "local" ? "primary" : ""} onClick={() => applyPreset("local", "local")}>
            Local server
          </button>
          <button className={kind === "cloud" ? "primary" : ""} onClick={() => applyPreset("openai", "cloud")}>
            Cloud API
          </button>
        </div>
        <label>Preset</label>
        <select
          value={presetId}
          onChange={(e) => applyPreset(e.target.value, kind)}
        >
          {presets
            .filter((p) => (kind === "local"
              ? ["local", "ollama", "lmstudio", "vllm"].includes(p.id)
              : ["openai", "gpt4o", "gemini", "openrouter"].includes(p.id)))
            .map((p) => (
              <option key={p.id} value={p.id}>{p.label}</option>
            ))}
        </select>
        <label>Base URL</label>
        <input value={baseUrl} onChange={(e) => setBaseUrl(e.target.value)} />
        <label>API key {kind === "local" ? "(usually empty)" : ""}</label>
        <input type="password" value={apiKey} onChange={(e) => setApiKey(e.target.value)} />
        <div className="settings-row">
          <button onClick={() => void test()} disabled={busy}>Test connection</button>
        </div>
        {models.length > 0 && (
          <>
            <label>Model</label>
            <select value={model} onChange={(e) => setModel(e.target.value)}>
              {models.map((m) => (
                <option key={m} value={m}>{m}</option>
              ))}
            </select>
          </>
        )}
        {models.length === 0 && (
          <>
            <label>Model name</label>
            <input value={model} onChange={(e) => setModel(e.target.value)} placeholder="optional if the server has a default" />
          </>
        )}
        {error && <div className="settings-error">{error}</div>}
        <button className="primary" disabled={busy || !baseUrl} onClick={() => void finish()}>
          {busy ? "Working…" : "Save and continue"}
        </button>
      </div>
    </div>
  );
}
