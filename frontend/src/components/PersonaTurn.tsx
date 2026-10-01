import type { AgentTurn, ToolCallEntry } from "../api/types";
import { LLMStatsBar } from "./LLMStatsBar";
import { Markdown } from "./Markdown";
import { formatTs, personaLabel } from "../hooks/useChatTurns";

interface Props {
  turn: AgentTurn;
  contextLength?: number | null;
}

function toolCallJson(entry: ToolCallEntry): string {
  return JSON.stringify(
    {
      tool: entry.tool,
      args: entry.args || {},
      ...(entry.reasoning ? { reasoning: entry.reasoning } : {}),
    },
    null,
    2,
  );
}

function ToolRow({ entry }: { entry: ToolCallEntry }) {
  const detail = toolCallJson(entry);
  return (
    <details className="tool-call-row">
      <summary>
        <span className="tool-call-name">{entry.tool}</span>
        {entry.reasoning && <span className="tool-call-reason">{entry.reasoning}</span>}
      </summary>
      <pre className="tool-call-json">{detail}</pre>
    </details>
  );
}

function visibleOutput(text: string): string {
  const stripped = text.trim();
  if (!stripped) return "";
  if (!stripped.startsWith("{")) return stripped;
  try {
    const parsed = JSON.parse(stripped) as Record<string, unknown>;
    if (typeof parsed.answer === "string" && parsed.answer.trim()) {
      return parsed.answer;
    }
    if (
      parsed.tool
      || parsed.calls
      || parsed.action
      || parsed.status === "complete"
      || parsed.status === "blocked"
      || parsed.status === "request_scope"
    ) {
      return "";
    }
  } catch {
    if (/"tool"\s*:|"calls"\s*:/.test(stripped)) return "";
  }
  return stripped;
}

export function PersonaTurn({ turn, contextLength }: Props) {
  const hasThinking = turn.thinking.trim().length > 0;
  const output = visibleOutput(turn.output);

  return (
    <div className={`message agent-turn ${turn.streaming ? "streaming" : ""}`}>
      <div className="role persona-role">
        <span>{personaLabel(turn)}</span>
        {turn.streaming && <span className="streaming-badge">generating</span>}
      </div>

      {hasThinking && (
        <details className="thinking-block" open={turn.streaming && !output}>
          <summary>Thinking</summary>
          <pre>{turn.thinking}</pre>
        </details>
      )}

      {turn.tools.length > 0 && (
        <div className="tool-call-list">
          {turn.tools.map((entry, index) => (
            <ToolRow key={`${entry.tool}-${entry.ts}-${index}`} entry={entry} />
          ))}
        </div>
      )}

      {output && (
        <div className="output-block">
          <Markdown text={output} />
        </div>
      )}

      {!turn.streaming && <LLMStatsBar metrics={turn.metrics} contextLength={contextLength} />}

      <div className="ts">{formatTs(turn.ts)}</div>
    </div>
  );
}
