import type { SSEEvent } from "../api/types";

export interface SessionTokenTotals {
  prompt: number;
  generated: number;
  total: number;
  calls: number;
  estimated: boolean;
}

export interface AgentToolCount {
  tool: string;
  count: number;
}

export interface AgentUsage {
  role: string;
  prompt: number;
  generated: number;
  total: number;
  calls: number;
  estimated: boolean;
  tools: AgentToolCount[];
  toolCalls: number;
}

export interface SessionUsageSummary extends SessionTokenTotals {
  byRole: AgentUsage[];
}

const ROLE_ORDER = [
  "triage",
  "reviewer",
  "orchestrator",
  "worker",
  "hotfix",
  "validator",
  "verify_hotfix",
];

function asNumber(value: unknown): number {
  return typeof value === "number" && Number.isFinite(value) ? value : 0;
}

function asString(value: unknown): string {
  return typeof value === "string" ? value : "";
}

function roleLabel(role: string): string {
  if (role === "verify_hotfix") return "verify hotfix";
  return role || "unknown";
}

function emptyRole(role: string): AgentUsage {
  return {
    role,
    prompt: 0,
    generated: 0,
    total: 0,
    calls: 0,
    estimated: false,
    tools: [],
    toolCalls: 0,
  };
}

/** Sum unique LLM calls and tool uses, grouped by agent role. */
export function summarizeSessionUsage(events: SSEEvent[]): SessionUsageSummary {
  const byCall = new Map<string, {
    role: string;
    prompt: number;
    generated: number;
    estimated: boolean;
  }>();
  let fallback = 0;
  for (const ev of events) {
    if (ev.type !== "llm.stream.end" && ev.type !== "llm.call") continue;
    const data = ev.data || {};
    const prompt = Math.max(0, Math.floor(asNumber(data.tokens_prompt)));
    const generated = Math.max(0, Math.floor(asNumber(data.tokens_generated)));
    const estimated = Boolean(data.tokens_estimated);
    const role = asString(data.role) || "unknown";
    const callId = typeof data.call_id === "string" && data.call_id ? data.call_id : "";
    if (!callId) {
      fallback += 1;
      byCall.set(`__anon-${ev.index ?? fallback}-${ev.ts}`, {
        role,
        prompt,
        generated,
        estimated,
      });
      continue;
    }
    const prior = byCall.get(callId);
    if (prior && ev.type === "llm.call") continue;
    byCall.set(callId, { role, prompt, generated, estimated });
  }

  const roles = new Map<string, AgentUsage>();
  const bump = (role: string): AgentUsage => {
    const existing = roles.get(role);
    if (existing) return existing;
    const created = emptyRole(role);
    roles.set(role, created);
    return created;
  };

  for (const item of byCall.values()) {
    const row = bump(item.role);
    row.prompt += item.prompt;
    row.generated += item.generated;
    row.calls += 1;
    if (item.estimated) row.estimated = true;
  }

  const toolCounts = new Map<string, Map<string, number>>();
  for (const ev of events) {
    if (ev.type !== "tool.called") continue;
    const data = ev.data || {};
    const role = asString(data.role) || "unknown";
    const tool = asString(data.tool) || "tool";
    let counts = toolCounts.get(role);
    if (!counts) {
      counts = new Map();
      toolCounts.set(role, counts);
    }
    counts.set(tool, (counts.get(tool) || 0) + 1);
    bump(role);
  }

  for (const [role, counts] of toolCounts) {
    const row = bump(role);
    row.tools = [...counts.entries()]
      .map(([tool, count]) => ({ tool, count }))
      .sort((a, b) => b.count - a.count || a.tool.localeCompare(b.tool));
    row.toolCalls = row.tools.reduce((sum, item) => sum + item.count, 0);
  }

  let prompt = 0;
  let generated = 0;
  let estimatedCalls = false;
  for (const row of roles.values()) {
    row.total = row.prompt + row.generated;
    prompt += row.prompt;
    generated += row.generated;
    if (row.estimated) estimatedCalls = true;
  }

  const byRole = [...roles.values()].sort((a, b) => {
    const ai = ROLE_ORDER.indexOf(a.role);
    const bi = ROLE_ORDER.indexOf(b.role);
    const ao = ai === -1 ? ROLE_ORDER.length : ai;
    const bo = bi === -1 ? ROLE_ORDER.length : bi;
    if (ao !== bo) return ao - bo;
    return a.role.localeCompare(b.role);
  });

  return {
    prompt,
    generated,
    total: prompt + generated,
    calls: byCall.size,
    estimated: estimatedCalls,
    byRole,
  };
}

/** @deprecated use summarizeSessionUsage */
export function summarizeSessionTokens(events: SSEEvent[]): SessionTokenTotals {
  const usage = summarizeSessionUsage(events);
  return {
    prompt: usage.prompt,
    generated: usage.generated,
    total: usage.total,
    calls: usage.calls,
    estimated: usage.estimated,
  };
}

export function formatTokenCount(value: number): string {
  return Math.max(0, Math.floor(value)).toLocaleString();
}

export function formatRoleLabel(role: string): string {
  return roleLabel(role);
}
