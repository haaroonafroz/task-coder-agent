import type { ModelChoice, ModelInfo } from "../api/types";

/** Human-readable label for a backend option in the model selector. */
export function formatModelOption(m: ModelInfo): string {
  if (m.key === "auto") return m.model?.startsWith("(auto") ? "Auto" : "Auto";

  const label = m.label || (m.key === "gpt4o" ? "OpenAI" : m.key);
  const orch = m.models_by_role?.orchestrator ?? m.model;
  const worker = m.models_by_role?.worker;
  if (worker && worker !== orch && orch) {
    return `${label} (${orch} / worker ${worker})`;
  }
  if (orch) return `${label} (${orch})`;
  return String(label);
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
