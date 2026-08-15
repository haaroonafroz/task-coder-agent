import type { LLMMetrics } from "../api/types";

interface Props {
  metrics?: LLMMetrics;
  contextLength?: number | null;
}

function speed(tokens: number, ms: number): string {
  if (!ms || ms <= 0 || !tokens) return "—";
  return `${(tokens / (ms / 1000)).toFixed(1)} tok/s`;
}

export function LLMStatsBar({ metrics, contextLength }: Props) {
  if (!metrics) return null;

  const prefillSpeed = speed(metrics.tokens_prompt, metrics.prefill_ms);
  const decodeSpeed = speed(metrics.tokens_generated, metrics.decode_ms);
  const contextUsed = metrics.tokens_prompt;
  const contextMax = contextLength && contextLength > 0 ? contextLength : null;
  const usagePct =
    contextMax && contextUsed > 0
      ? Math.min(100, Math.round((contextUsed / contextMax) * 100))
      : null;

  return (
    <div className="llm-stats-bar">
      <div className="llm-stats-row">
        <span>{metrics.model_used}</span>
        {metrics.thinking_level && <span>thinking: {metrics.thinking_level}</span>}
        <span>{metrics.total_ms.toFixed(0)} ms total</span>
      </div>
      <div className="llm-stats-row">
        <span>prefill {prefillSpeed}</span>
        <span>decode {decodeSpeed}</span>
        <span>{metrics.tokens_prompt} prompt · {metrics.tokens_generated} generated</span>
      </div>
      {contextMax && (
        <div className="llm-context-row">
          <span>
            context {contextUsed.toLocaleString()} / {contextMax.toLocaleString()}
            {usagePct !== null ? ` (${usagePct}%)` : ""}
          </span>
          <div className="llm-context-track">
            <div
              className="llm-context-fill"
              style={{ width: `${usagePct ?? 0}%` }}
            />
          </div>
        </div>
      )}
    </div>
  );
}
