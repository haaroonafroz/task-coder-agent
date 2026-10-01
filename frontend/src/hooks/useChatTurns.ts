import type {
  AgentRole,
  AgentTurn,
  ChatItem,
  LLMMetrics,
  Message,
  SSEEvent,
  ToolCallEntry,
} from "../api/types";

const SYSTEM_EVENT_TYPES = new Set([
  "session.started",
  "mission.cancelled",
  "verify_hotfix.started",
  "verify_hotfix.completed",
]);

function asString(value: unknown): string {
  return typeof value === "string" ? value : "";
}

function asNumber(value: unknown): number {
  return typeof value === "number" && Number.isFinite(value) ? value : 0;
}

function asRole(value: unknown): AgentRole {
  const role = asString(value);
  if (
    role === "orchestrator" ||
    role === "worker" ||
    role === "hotfix" ||
    role === "reviewer" ||
    role === "validator" ||
    role === "verify_hotfix" ||
    role === "triage" ||
    role === "ask" ||
    role === "compact"
  ) {
    return role;
  }
  return "worker";
}

function asArgs(value: unknown): Record<string, unknown> | undefined {
  if (!value || typeof value !== "object" || Array.isArray(value)) return undefined;
  return value as Record<string, unknown>;
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
    tokens_estimated: Boolean(data.tokens_estimated),
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

function tryParseJson(text: string): Record<string, unknown> | null {
  const stripped = text.trim();
  if (!stripped.startsWith("{") && !stripped.includes("{")) return null;
  const candidates = [stripped];
  const fence = stripped.match(/```(?:json)?\s*(\{[\s\S]*\})\s*```/);
  if (fence?.[1]) candidates.push(fence[1]);
  const start = stripped.indexOf("{");
  const end = stripped.lastIndexOf("}");
  if (start >= 0 && end > start) candidates.push(stripped.slice(start, end + 1));
  for (const candidate of candidates) {
    try {
      const parsed = JSON.parse(candidate);
      if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) {
        return parsed as Record<string, unknown>;
      }
    } catch {
      continue;
    }
  }
  return null;
}

function protocolCallsFromParsed(parsed: Record<string, unknown>, ts: string): ToolCallEntry[] {
  const rawCalls = Array.isArray(parsed.calls) ? parsed.calls : parsed.tool ? [parsed] : [];
  const calls: ToolCallEntry[] = [];
  for (const item of rawCalls) {
    if (!item || typeof item !== "object" || Array.isArray(item)) continue;
    const row = item as Record<string, unknown>;
    const tool = asString(row.tool || row.name);
    if (!tool) continue;
    calls.push({
      tool,
      reasoning: asString(row.reasoning) || undefined,
      ts,
      args: asArgs(row.args ?? row.arguments),
    });
  }
  return calls;
}

function visibleTextFromParsed(parsed: Record<string, unknown>): string | null {
  const answer = parsed.answer;
  if (typeof answer === "string" && answer.trim()) return answer;

  if (parsed.tool || Array.isArray(parsed.calls)) return "";

  const status = asString(parsed.status);
  if (status === "complete") {
    const summary = asString(parsed.summary);
    return summary || "Milestone complete.";
  }
  if (status === "blocked" || status === "request_scope") {
    return asString(parsed.reason) || asString(parsed.clarification) || status;
  }

  if (asString(parsed.action) === "review") {
    const report = asArgs(parsed.report);
    if (!report) return "";
    const verdict = asString(report.verdict) || "review";
    const summary = asString(report.summary);
    return summary ? `**${verdict}** — ${summary}` : `**${verdict}**`;
  }
  if (asString(parsed.action) === "plan") {
    const plan = asArgs(parsed.plan);
    if (!plan) return "";
    const title = asString(plan.title) || "Plan";
    const milestones = Array.isArray(plan.milestones) ? plan.milestones : [];
    const lines = [`**${title}**`, ""];
    for (const item of milestones) {
      if (!item || typeof item !== "object") continue;
      const ms = item as Record<string, unknown>;
      lines.push(`- **${asString(ms.id) || "M"}** — ${asString(ms.title)}`);
    }
    return lines.join("\n").trim();
  }
  if (typeof parsed.summary === "string" && parsed.summary.trim()) {
    return parsed.summary;
  }
  return null;
}

function applyProtocolBuffer(turn: AgentTurn, buffer: string): void {
  const parsed = tryParseJson(buffer);
  if (!parsed) return;
  const extracted = protocolCallsFromParsed(parsed, turn.ts);
  for (const entry of extracted) mergeProtocolTool(turn, entry);
  const visible = visibleTextFromParsed(parsed);
  if (visible) turn.output = visible;
}

function mergeProtocolTool(turn: AgentTurn, entry: ToolCallEntry): void {
  const duplicate = turn.tools.find(
    (item) =>
      item.tool === entry.tool
      && (item.reasoning || "") === (entry.reasoning || "")
      && JSON.stringify(item.args || {}) === JSON.stringify(entry.args || {}),
  );
  if (duplicate) {
    if (entry.args && !duplicate.args) duplicate.args = entry.args;
    return;
  }
  turn.tools.push(entry);
}

function mergeExecutedTool(turn: AgentTurn, entry: ToolCallEntry): void {
  const pending = turn.tools.find((item) => item.tool === entry.tool && item.success === undefined);
  if (pending) {
    pending.ts = entry.ts;
    pending.success = entry.success;
    pending.milestone_id = pending.milestone_id || entry.milestone_id;
    if (entry.reasoning) pending.reasoning = entry.reasoning;
    if (entry.args) pending.args = entry.args;
    return;
  }
  const same = turn.tools.find(
    (item) => item.tool === entry.tool && (item.reasoning || "") === (entry.reasoning || ""),
  );
  if (same) {
    if (entry.args && !same.args) same.args = entry.args;
    if (entry.success !== undefined) same.success = entry.success;
    return;
  }
  turn.tools.push(entry);
}

function systemLine(ev: SSEEvent): string {
  if (ev.type === "session.started") {
    const request = asString(ev.data.request);
    return request ? `Session started — ${request}` : "Session started";
  }
  if (ev.type === "mission.cancelled") {
    return "Mission cancelled";
  }
  if (ev.type === "verify_hotfix.started") {
    return "Verify-Hotfix started — checking the hotfix against the fix criteria";
  }
  if (ev.type === "verify_hotfix.completed") {
    const verdict = asString(ev.data.verdict) || "UNKNOWN";
    const summary = asString(ev.data.summary);
    return summary
      ? `Verify-Hotfix ${verdict} — ${summary}`
      : `Verify-Hotfix ${verdict}`;
  }
  return ev.type;
}

function isRunSummaryMessage(message: Message, missionSummary?: string): boolean {
  if (message.role !== "assistant") return false;
  if (missionSummary && message.content.trim() === missionSummary.trim()) return true;
  if (message.content.startsWith("Run finished")) return true;
  if (message.content.startsWith("Decision resolved")) return true;
  return false;
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

function findOrchestratorTurn(turns: Map<string, AgentTurn>): AgentTurn | undefined {
  return [...turns.values()]
    .sort((a, b) => a.ts.localeCompare(b.ts))
    .reverse()
    .find((turn) => turn.role === "orchestrator");
}

function findRoleTurn(
  turns: Map<string, AgentTurn>,
  role: AgentRole,
): AgentTurn | undefined {
  return [...turns.values()]
    .sort((a, b) => a.ts.localeCompare(b.ts))
    .reverse()
    .find((turn) => turn.role === role);
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

function resolveTurnId(
  callId: string,
  data: Record<string, unknown>,
  askTurnId: string | null,
  protocolCalls: Set<string>,
): string {
  if (asRole(data.role) === "ask" && askTurnId) return askTurnId;
  if (protocolCalls.has(callId) && asRole(data.role) === "ask" && askTurnId) return askTurnId;
  return callId;
}

export function buildChatItems(messages: Message[], events: SSEEvent[]): ChatItem[] {
  const turns = new Map<string, AgentTurn>();
  const finalized = new Set<string>();
  const protocolCalls = new Set<string>();
  const protocolBuffers = new Map<string, string>();
  const items: ChatItem[] = [];
  const missionSummaryText = latestMissionSummary(events);
  const askAnswers = new Set(
    events
      .filter((ev) => ev.type === "ask.replied")
      .map((ev) => asString(ev.data.answer).trim())
      .filter(Boolean),
  );
  let askTurnId: string | null = null;

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

    if (ev.type === "ask.started") {
      askTurnId = `ask-${ev.index ?? index}`;
      ensureTurn(turns, askTurnId, { role: "ask", phase: "reply" }, ev.ts).streaming = true;
      continue;
    }

    if (ev.type === "mission.complete") {
      if (asString(data.status) === "awaiting_decision") continue;
      const summary = asString(data.summary_text) || missionSummaryFallback(ev);
      if (askAnswers.has(summary.trim())) continue;
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

    if (ev.type === "ask.replied") {
      const answer = asString(data.answer).trim();
      if (!askTurnId) askTurnId = `ask-reply-${ev.index ?? index}`;
      const turn = ensureTurn(turns, askTurnId, { role: "ask", phase: "reply" }, ev.ts);
      if (answer) turn.output = answer;
      turn.streaming = false;
      continue;
    }

    if (ev.type === "llm.stream.start") {
      const role = asRole(data.role);
      if (role === "compact") continue;
      const isProtocol = asString(data.output_kind) === "json" || role === "ask";
      if (isProtocol) protocolCalls.add(callId);
      if (role === "ask" && !askTurnId) {
        askTurnId = `ask-${ev.index ?? index}`;
      }
      const turnId = resolveTurnId(callId, data, askTurnId, protocolCalls);
      const turn = ensureTurn(turns, turnId, data, ev.ts);
      turn.streaming = true;
      if (role === "ask") turn.phase = "reply";
      continue;
    }

    if (ev.type === "llm.stream.delta") {
      if (asRole(data.role) === "compact") continue;
      const turnId = resolveTurnId(callId, data, askTurnId, protocolCalls);
      const turn = ensureTurn(turns, turnId, data, ev.ts);
      const channel = asString(data.channel);
      const text = asString(data.text);
      if (channel === "thinking") {
        turn.thinking += text;
      } else if (protocolCalls.has(callId) || asRole(data.role) === "ask") {
        protocolBuffers.set(callId, (protocolBuffers.get(callId) || "") + text);
        applyProtocolBuffer(turn, protocolBuffers.get(callId) || "");
      } else {
        turn.output += text;
      }
      continue;
    }

    if (ev.type === "llm.stream.end" || ev.type === "llm.call") {
      if (asRole(data.role) === "compact") continue;
      if (finalized.has(callId) && ev.type === "llm.call") continue;
      const turnId = resolveTurnId(callId, data, askTurnId, protocolCalls);
      const turn = ensureTurn(turns, turnId, data, ev.ts);
      const isAsk = asRole(data.role) === "ask" || turn.role === "ask";
      if (protocolCalls.has(callId) || isAsk) {
        const buffer = protocolBuffers.get(callId) || asString(data.output_preview);
        turn.metrics = metricsFromData(data, callId);
        if (!turn.thinking && turn.metrics.thinking_preview) {
          turn.thinking = turn.metrics.thinking_preview;
        }
        if (buffer) applyProtocolBuffer(turn, buffer);
        if (isAsk && turn.output.trim()) turn.streaming = false;
        else if (!isAsk) turn.streaming = false;
      } else {
        turn.streaming = false;
        turn.metrics = metricsFromData(data, callId);
        if (!turn.thinking && turn.metrics.thinking_preview) {
          turn.thinking = turn.metrics.thinking_preview;
        }
        if (!turn.output && turn.metrics.output_preview) {
          turn.output = turn.metrics.output_preview;
        }
      }
      if (ev.type === "llm.stream.end") finalized.add(callId);
      continue;
    }

    if (ev.type === "tool.called") {
      const entry: ToolCallEntry = {
        tool: asString(data.tool) || "tool",
        reasoning: asString(data.reasoning) || undefined,
        ts: ev.ts,
        milestone_id: asString(data.milestone_id) || undefined,
        args: asArgs(data.args),
      };
      const toolRole = asString(data.role);
      if (toolRole === "ask") {
        if (!askTurnId) askTurnId = `ask-tool-${ev.index ?? index}`;
        const turn = ensureTurn(turns, askTurnId, { role: "ask", phase: "reply" }, ev.ts);
        mergeExecutedTool(turn, entry);
        continue;
      }
      if (toolRole === "orchestrator" || toolRole === "reviewer" || toolRole === "hotfix") {
        const role = asRole(toolRole);
        const roleTurn = role === "orchestrator"
          ? findOrchestratorTurn(turns)
          : findRoleTurn(turns, role);
        if (roleTurn) {
          mergeExecutedTool(roleTurn, entry);
        } else {
          items.push({
            kind: "system",
            id: `tool-${ev.index ?? index}`,
            ts: ev.ts,
            content: `${personaLabel({
              role,
              milestone_id: entry.milestone_id,
            } as AgentTurn)} · ${entry.tool}${entry.reasoning ? `: ${entry.reasoning}` : ""}`,
          });
        }
      } else {
        const workerTurn = findWorkerTurn(turns, entry.milestone_id);
        if (workerTurn) {
          mergeExecutedTool(workerTurn, entry);
        } else {
          items.push({
            kind: "system",
            id: `tool-${ev.index ?? index}`,
            ts: ev.ts,
            content: `${entry.tool}${entry.reasoning ? `: ${entry.reasoning}` : ""}`,
          });
        }
      }
      continue;
    }

    if (ev.type === "tool.result") {
      const toolRole = asString(data.role);
      const toolName = asString(data.tool);
      const success = data.success === true;
      const target =
        toolRole === "ask" && askTurnId
          ? turns.get(askTurnId)
          : toolRole === "orchestrator"
            ? findOrchestratorTurn(turns)
            : toolRole === "reviewer" || toolRole === "hotfix"
              ? findRoleTurn(turns, asRole(toolRole))
              : findWorkerTurn(turns, asString(data.milestone_id) || undefined);
      if (target && toolName) {
        const pending = [...target.tools].reverse().find(
          (item) => item.tool === toolName && item.success === undefined,
        );
        if (pending) pending.success = success;
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
    if (turn.role === "compact") continue;
    items.push(turn);
  }

  const shownAsk = new Set(
    [...turns.values()]
      .filter((turn) => turn.role === "ask")
      .map((turn) => turn.output.trim())
      .filter(Boolean),
  );
  const askTurns = [...turns.values()]
    .filter((turn) => turn.role === "ask")
    .sort((a, b) => a.ts.localeCompare(b.ts));
  for (const message of messages) {
    if (message.role !== "assistant") continue;
    if (isRunSummaryMessage(message, missionSummaryText)) continue;
    const content = message.content.trim();
    if (!content) continue;
    const askTurn =
      (askTurnId ? turns.get(askTurnId) : undefined)
      || askTurns.find((turn) => !turn.output.trim())
      || askTurns[askTurns.length - 1];
    if (askTurn) {
      if (!askTurn.output.trim() || askTurn.output.trim().startsWith("{")) {
        askTurn.output = content;
      }
      askTurn.streaming = false;
      continue;
    }
    if (shownAsk.has(content)) continue;
    const fallbackId = `ask-message-${message.id}`;
    turns.set(fallbackId, {
      kind: "agent",
      id: `turn-${fallbackId}`,
      call_id: fallbackId,
      role: "ask",
      phase: "reply",
      thinking: "",
      output: content,
      tools: [],
      streaming: false,
      ts: message.ts,
    });
    items.push(turns.get(fallbackId)!);
  }

  return items.sort((a, b) => a.ts.localeCompare(b.ts));
}

export function personaLabel(turn: AgentTurn): string {
  const roleLabels: Record<AgentRole, string> = {
    orchestrator: "Orchestrator",
    worker: "Worker",
    hotfix: "Hotfix",
    reviewer: "Code Review",
    validator: "Validator",
    verify_hotfix: "Verify-Hotfix",
    triage: "Triage",
    ask: "Ask",
    compact: "Compact",
  };
  const base = roleLabels[turn.role];
  if (turn.milestone_id) return `${base} · ${turn.milestone_id}`;
  if (turn.phase && turn.role !== "ask") return `${base} · ${turn.phase}`;
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
    case "awaiting_decision":
      return "Awaiting decision";
    default:
      return status;
  }
}
