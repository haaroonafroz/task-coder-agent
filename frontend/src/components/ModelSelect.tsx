import type { ModelChoice, ModelInfo } from "../api/types";

/** Human-readable label for a backend option in the model selector. */
export function formatModelOption(m: ModelInfo): string {
  if (m.key === "auto") return "Auto";

  const orch = m.models_by_role?.orchestrator ?? m.model;
  const worker = m.models_by_role?.worker;
  const keyLabel =
    m.key === "gpt4o" ? "GPT" : m.key.charAt(0).toUpperCase() + m.key.slice(1);

  if (worker && worker !== orch) {
    return `${keyLabel} (${orch} / worker ${worker})`;
  }
  return `${keyLabel} (${orch})`;
}

interface Props {
  value: ModelChoice;
  models: ModelInfo[];
  onChange: (value: ModelChoice) => void;
  className?: string;
}

export function ModelSelect({ value, models, onChange, className }: Props) {
  const options =
    models.length > 0
      ? models
      : ([
          { key: "auto", model: "Auto", base_url: "", available: true, error: null },
          { key: "local", model: "Local", base_url: "", available: true, error: null },
          { key: "gemini", model: "Gemini", base_url: "", available: true, error: null },
          { key: "gpt4o", model: "GPT", base_url: "", available: true, error: null },
        ] as ModelInfo[]);

  return (
    <select
      className={className}
      value={value}
      onChange={(e) => onChange(e.target.value as ModelChoice)}
    >
      {options.map((m) => (
        <option key={m.key} value={m.key} disabled={m.key !== "auto" && !m.available}>
          {formatModelOption(m)}
          {m.key !== "auto" && !m.available ? " (unavailable)" : ""}
        </option>
      ))}
    </select>
  );
}
