import type {
  AgentRole,
  AgentTurn,
  ChatItem,
  LLMMetrics,
  Message,
  SSEEvent,
  ToolCallEntry,
} from "../api/types";

const SYSTEM_EVENT_TYPES = new Set(["session.started", "mission.cancelled"]);

function asString(value: unknown): string {
  return typeof value === "string" ? value : "";
}

function asNumber(value: unknown): number {
  return typeof value === "number" && Number.isFinite(value) ? value : 0;
}

function asRole(value: unknown): AgentRole {
  const role = asString(value);
  if (role === "orchestrator" || role === "worker" || role === "validator" || role === "triage") {
    return role;
  }
  return "worker";
}

function metricsFromData(data: Record<string, unknown>, callId: string): LLMMetrics {
  return {
    call_id: callId,
    role: asRole(data.role),
    milestone_id: asString(data.milestone_id) || undefined,
    phase: asString(data.phase) || undefined,
    model_used: asString(data.model_used),
    tokens_prompt: asNumber(data.tokens_prompt),
    tokens_generated: asNumber(data.tokens_generated),
    prefill_ms: asNumber(data.prefill_ms),
    decode_ms: asNumber(data.decode_ms),
    total_ms: asNumber(data.total_ms),
    thinking_level: asString(data.thinking_level) || undefined,
    output_kind: asString(data.output_kind) || undefined,
    thinking_preview: asString(data.thinking_preview) || undefined,
    output_preview: asString(data.output_preview) || undefined,
    thinking_chars: asNumber(data.thinking_chars) || undefined,
    output_chars: asNumber(data.output_chars) || undefined,
    fallback_used: Boolean(data.fallback_used),
  };
}

function ensureTurn(
  turns: Map<string, AgentTurn>,
  callId: string,
  data: Record<string, unknown>,
  ts: string,
): AgentTurn {
  const existing = turns.get(callId);
  if (existing) return existing;

  const turn: AgentTurn = {
    kind: "agent",
    id: `turn-${callId}`,
    call_id: callId,
    role: asRole(data.role),
    milestone_id: asString(data.milestone_id) || undefined,
    phase: asString(data.phase) || undefined,
    thinking: "",
    output: "",
    tools: [],
    streaming: true,
    ts,
  };
  turns.set(callId, turn);
  return turn;
}

function systemLine(ev: SSEEvent): string {
  if (ev.type === "session.started") {
    const request = asString(ev.data.request);
    return request ? `Session started — ${request}` : "Session started";
  }
  if (ev.type === "mission.cancelled") {
    return "Mission cancelled";
  }
  return ev.type;
}

function isRunSummaryMessage(message: Message, missionSummary?: string): boolean {
  if (message.role !== "assistant") return false;
  if (missionSummary && message.content.trim() === missionSummary.trim()) return true;
  return message.content.startsWith("Run finished");
}

function latestMissionSummary(events: SSEEvent[]): string {
  let summary = "";
  for (const ev of events) {
    if (ev.type !== "mission.complete") continue;
    const text = asString(ev.data.summary_text);
    if (text) summary = text;
  }
  return summary;
}

function missionSummaryFallback(ev: SSEEvent): string {
  const data = ev.data || {};
  const status = asString(data.status) || "finished";
  const passed = asNumber(data.milestones_passed);
  const total = asNumber(data.milestones_total);
  const elapsed = asNumber(data.total_elapsed_ms) / 1000;
  return `Mission ${status} — ${passed}/${total} milestones passed (${elapsed.toFixed(1)}s)`;
}

function findWorkerTurn(turns: Map<string, AgentTurn>, milestoneId?: string): AgentTurn | undefined {
  const all = [...turns.values()].sort((a, b) => a.ts.localeCompare(b.ts));
  if (milestoneId) {
    const match = [...all].reverse().find((turn) => turn.role === "worker" && turn.milestone_id === milestoneId);
    if (match) return match;
  }
  return [...all].reverse().find((turn) => turn.role === "worker");
}

const VALIDATION_PATH_LABELS: Record<string, string> = {
  fast_path: "contract exit 0",
  ui_smoke: "UI smoke",
  llm: "LLM verdict",
  spec_gaming: "spec gaming",
  out_of_scope: "out of scope",
  policy_denied: "policy denied",
  deterministic_replan: "deterministic replan",
  tdd_red: "TDD red phase",
  all_skipped: "all tests skipped",
  collect_only: "collect only",
  unparseable_contract_ok: "contract ok (unparseable LLM)",
  unparseable: "unparseable LLM response",
  missing_dependency: "missing dependency",
};

function formatValidationSummary(data: Record<string, unknown>): string {
  const verdict = asString(data.verdict) || "UNKNOWN";
  const path = asString(data.path);
  const pathLabel = VALIDATION_PATH_LABELS[path] || path;
  const lines: string[] = [`Verdict: ${verdict}`];
  if (pathLabel) lines.push(`Path: ${pathLabel}`);

  const details = asString(data.validation_details);
  if (details) {
    lines.push("", details);
  } else if (path === "fast_path" && verdict === "PASS") {
    lines.push("", "Contract command exited 0.");
  }

  const rootCause = asString(data.root_cause);
  if (rootCause) lines.push("", `Root cause: ${rootCause}`);

  const errors = Array.isArray(data.errors) ? data.errors.map((entry) => asString(entry)).filter(Boolean) : [];
  if (errors.length > 0) {
    lines.push("", "Errors:");
    for (const error of errors) lines.push(`- ${error}`);
  }

  const fixGuidance = asString(data.fix_guidance);
  if (fixGuidance) lines.push("", "Fix guidance:", fixGuidance);

  const replanGuidance = asString(data.replan_guidance);
  if (replanGuidance) lines.push("", "Replan guidance:", replanGuidance);

  return lines.join("\n");
}

function hasValidatorLlmTurn(turns: Map<string, AgentTurn>, milestoneId: string): boolean {
  return [...turns.values()].some(
    (turn) =>
      turn.role === "validator" &&
      turn.milestone_id === milestoneId &&
      (turn.metrics !== undefined || turn.output.trim().length > 0),
  );
}

export function buildChatItems(messages: Message[], events: SSEEvent[]): ChatItem[] {
  const turns = new Map<string, AgentTurn>();
  const finalized = new Set<string>();
  const items: ChatItem[] = [];
  const missionSummaryText = latestMissionSummary(events);

  for (const message of messages) {
    if (message.role === "user") {
      items.push({
        kind: "user",
        id: `message-${message.id}`,
        ts: message.ts,
        content: message.content,
      });
      continue;
    }
    if (isRunSummaryMessage(message, missionSummaryText)) {
      continue;
    }
  }

  for (const [index, ev] of events.entries()) {
    const data = ev.data || {};
    const callId = asString(data.call_id) || `legacy-${index}`;

    if (SYSTEM_EVENT_TYPES.has(ev.type)) {
      items.push({
        kind: "system",
        id: `system-${ev.index ?? index}-${ev.type}`,
        ts: ev.ts,
        content: systemLine(ev),
      });
      continue;
    }

    if (ev.type === "mission.complete") {
      const summary = asString(data.summary_text) || missionSummaryFallback(ev);
      items.push({
        kind: "mission_summary",
        id: `mission-${ev.index ?? index}`,
        ts: ev.ts,
        status: asString(data.status) || "finished",
        content: summary,
        failure_reason: asString(data.failure_reason) || undefined,
      });
      continue;
    }

    if (ev.type === "llm.stream.start") {
      ensureTurn(turns, callId, data, ev.ts);
      continue;
    }

    if (ev.type === "llm.stream.delta") {
      const turn = ensureTurn(turns, callId, data, ev.ts);
      const channel = asString(data.channel);
      const text = asString(data.text);
      if (channel === "thinking") turn.thinking += text;
      else turn.output += text;
      continue;
    }

    if (ev.type === "llm.stream.end" || ev.type === "llm.call") {
      if (finalized.has(callId) && ev.type === "llm.call") continue;
      const turn = ensureTurn(turns, callId, data, ev.ts);
      turn.streaming = false;
      turn.metrics = metricsFromData(data, callId);
      if (!turn.thinking && turn.metrics.thinking_preview) {
        turn.thinking = turn.metrics.thinking_preview;
      }
      if (!turn.output && turn.metrics.output_preview) {
        turn.output = turn.metrics.output_preview;
      }
      finalized.add(callId);
      continue;
    }

    if (ev.type === "tool.called") {
      const entry: ToolCallEntry = {
        tool: asString(data.tool) || "tool",
        reasoning: asString(data.reasoning) || undefined,
        ts: ev.ts,
        milestone_id: asString(data.milestone_id) || undefined,
      };
      const workerTurn = findWorkerTurn(turns, entry.milestone_id);
      if (workerTurn) {
        workerTurn.tools.push(entry);
      } else {
        items.push({
          kind: "system",
          id: `tool-${ev.index ?? index}`,
          ts: ev.ts,
          content: `${entry.tool}${entry.reasoning ? `: ${entry.reasoning}` : ""}`,
        });
      }
      continue;
    }

    if (ev.type === "validation.finished") {
      const milestoneId = asString(data.milestone_id);
      const path = asString(data.path);
      if (path === "llm" && milestoneId && hasValidatorLlmTurn(turns, milestoneId)) {
        continue;
      }
      const validationCallId = `validation-${milestoneId || "unknown"}-${ev.index ?? index}`;
      turns.set(validationCallId, {
        kind: "agent",
        id: `turn-${validationCallId}`,
        call_id: validationCallId,
        role: "validator",
        milestone_id: milestoneId || undefined,
        thinking: "",
        output: formatValidationSummary(data),
        tools: [],
        streaming: false,
        ts: ev.ts,
      });
    }
  }

  for (const turn of turns.values()) {
    items.push(turn);
  }

  return items.sort((a, b) => a.ts.localeCompare(b.ts));
}

export function personaLabel(turn: AgentTurn): string {
  const roleLabels: Record<AgentRole, string> = {
    orchestrator: "Orchestrator",
    worker: "Worker",
    validator: "Validator",
    triage: "Triage",
  };
  const base = roleLabels[turn.role];
  if (turn.milestone_id) return `${base} · ${turn.milestone_id}`;
  if (turn.phase) return `${base} · ${turn.phase}`;
  return base;
}

export function formatTs(ts: string): string {
  return ts.split("T")[1] || ts;
}

export function missionStatusLabel(status: string): string {
  switch (status) {
    case "completed":
      return "Completed";
    case "partial":
      return "Partial";
    case "failed":
      return "Failed";
    case "cancelled":
      return "Cancelled";
    default:
      return status;
  }
}
