import type { AgentTurn, ToolCallEntry } from "../api/types";
import { LLMStatsBar } from "./LLMStatsBar";
import { formatTs, personaLabel } from "../hooks/useChatTurns";

interface Props {
  turn: AgentTurn;
  contextLength?: number | null;
}

function ToolRow({ entry }: { entry: ToolCallEntry }) {
  return (
    <div className="tool-call-row">
      <span className="tool-call-name">{entry.tool}</span>
      {entry.reasoning && <span className="tool-call-reason">{entry.reasoning}</span>}
    </div>
  );
}

function formatOutput(text: string, outputKind?: string): string {
  if (!text) return "";
  if (outputKind !== "json") return text;
  try {
    return JSON.stringify(JSON.parse(text), null, 2);
  } catch {
    return text;
  }
}

export function PersonaTurn({ turn, contextLength }: Props) {
  const hasThinking = turn.thinking.trim().length > 0;
  const output = formatOutput(turn.output, turn.metrics?.output_kind);

  return (
    <div className={`message agent-turn ${turn.streaming ? "streaming" : ""}`}>
      <div className="role persona-role">
        <span>{personaLabel(turn)}</span>
        {turn.streaming && <span className="streaming-badge">generating</span>}
      </div>

      {hasThinking && (
        <details className="thinking-block" open={turn.streaming}>
          <summary>Thinking</summary>
          <pre>{turn.thinking}</pre>
        </details>
      )}

      {output && (
        <div className="output-block">
          <pre>{output}</pre>
        </div>
      )}

      {turn.tools.length > 0 && (
        <div className="tool-call-list">
          {turn.tools.map((entry, index) => (
            <ToolRow key={`${entry.tool}-${entry.ts}-${index}`} entry={entry} />
          ))}
        </div>
      )}

      {!turn.streaming && <LLMStatsBar metrics={turn.metrics} contextLength={contextLength} />}

      <div className="ts">{formatTs(turn.ts)}</div>
    </div>
  );
}
