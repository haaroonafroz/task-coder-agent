import { useEffect, useMemo, useRef, useState } from "react";
import type { Session, Message, ModelChoice, SSEEvent } from "../api/types";
import { ModelSelect } from "./ModelSelect";
import { useModels } from "../hooks";

interface Props {
  session: Session | null;
  messages: Message[];
  sending: boolean;
  connected: boolean;
  events: SSEEvent[];
  onSend: (content: string, triggerRun: boolean, model?: string) => void;
}

type FeedItem =
  | { kind: "message"; id: string; ts: string; message: Message }
  | { kind: "event"; id: string; ts: string; event: SSEEvent };

function eventRole(type: string): string {
  if (type.startsWith("triage.")) return "triage";
  if (type.startsWith("plan.") || type.includes("replan")) return "orchestrator";
  if (type.startsWith("worker.") || type.startsWith("tool.")) return "worker";
  if (type.startsWith("validation.")) return "validator";
  if (type.startsWith("milestone.")) return "milestone";
  return "system";
}

function eventTitle(ev: SSEEvent): string {
  const data = ev.data || {};
  if (ev.type === "triage.started") return "Analyzing the current workspace for repair";
  if (ev.type === "triage.completed") {
    return `Triage completed (${String(data.confidence ?? "unknown")} confidence)`;
  }
  if (ev.type === "triage.failed") return "Triage unavailable; continuing with repair planning";
  if (ev.type === "mission.audit") {
    return data.passed === true ? "Completion audit passed" : "Completion audit found incomplete milestones";
  }
  if (ev.type === "plan.created") {
    return `Plan created: ${String(data.title ?? "untitled mission")}`;
  }
  if (ev.type === "plan.updated") return "Plan updated after replanning";
  if (ev.type === "milestone.started") {
    return `Started ${String(data.milestone_id ?? "milestone")}: ${String(data.title ?? "")}`;
  }
  if (ev.type === "milestone.passed") {
    return `${String(data.milestone_id ?? "Milestone")} passed`;
  }
  if (ev.type === "milestone.replan") {
    return `Replan requested for ${String(data.milestone_id ?? "milestone")}`;
  }
  if (ev.type === "tool.called") {
    const tool = String(data.tool ?? "tool");
    const reasoning = typeof data.reasoning === "string" ? data.reasoning : "";
    return reasoning ? `${tool}: ${reasoning}` : `${tool} called`;
  }
  if (ev.type === "tool.result") {
    return `${String(data.tool ?? "tool")} completed`;
  }
  if (ev.type === "validation.contract_run") {
    return `Validation contract exited ${String(data.returncode ?? "?")}`;
  }
  if (ev.type === "validation.finished") {
    return `Validation ${String(data.verdict ?? "finished")}`;
  }
  if (ev.type === "worker.invalid_json") {
    return `Worker emitted invalid JSON (${String(data.attempt ?? "?")}/${String(
      data.max_attempts ?? "?"
    )})`;
  }
  if (ev.type === "mission.complete") {
    return `Mission ${String(data.status ?? "complete")}`;
  }
  if (typeof data.reason === "string") return data.reason;
  if (typeof data.status === "string") return data.status;
  return ev.type;
}

function shouldShowDetails(ev: SSEEvent): boolean {
  return Object.keys(ev.data || {}).length > 0;
}

function formatTs(ts: string): string {
  return ts.split("T")[1] || ts;
}

export function ChatPanel({ session, messages, sending, connected, events, onSend }: Props) {
  const [input, setInput] = useState("");
  const [model, setModel] = useState<ModelChoice>("auto");
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const { models } = useModels();

  const feed = useMemo<FeedItem[]>(() => {
    const messageItems: FeedItem[] = messages.map((message) => ({
      kind: "message",
      id: `message-${message.id}`,
      ts: message.ts,
      message,
    }));
    const eventItems: FeedItem[] = events.map((event, index) => ({
      kind: "event",
      id: `event-${event.index ?? index}-${event.ts}-${event.type}`,
      ts: event.ts,
      event,
    }));
    return [...messageItems, ...eventItems].sort((a, b) => a.ts.localeCompare(b.ts));
  }, [messages, events]);

  useEffect(() => {
    if (session) {
      setModel((session.selected_model as ModelChoice) || "auto");
    }
  }, [session]);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [feed.length]);

  const handleSend = () => {
    if (!input.trim() || sending) return;
    onSend(input.trim(), true, model);
    setInput("");
  };

  if (!session) {
    return (
      <div className="panel">
        <div className="empty-state">
          Select or create a session to start chatting.
        </div>
      </div>
    );
  }

  return (
    <div className="panel chat-container">
      <div className="panel-header">
        <span>{session.title}</span>
        <span style={{ display: "flex", alignItems: "center", gap: 6 }}>
          <span className={`conn-dot ${connected ? "connected" : "disconnected"}`} />
          <span style={{ fontSize: 11, color: "var(--text-muted)" }}>
            {connected ? "Live" : "Offline"}
          </span>
        </span>
      </div>

      <div className="chat-messages">
        {feed.length === 0 && (
          <div className="empty-state" style={{ fontSize: 12 }}>
            No messages yet. Send a request below to start a mission.
          </div>
        )}
        {feed.map((item) => {
          if (item.kind === "message") {
            const msg = item.message;
            return (
              <div key={item.id} className={`message ${msg.role}`}>
                <div className="role">
                  {msg.role}
                  {msg.run_id && <span className="run-chip">run:{msg.run_id.slice(0, 8)}</span>}
                </div>
                <div className="content">{msg.content}</div>
                <div className="ts">{formatTs(msg.ts)}</div>
              </div>
            );
          }
          const ev = item.event;
          const milestoneId =
            typeof ev.data?.milestone_id === "string" ? ev.data.milestone_id : null;
          return (
            <div key={item.id} className="message event-message">
              <div className="role">
                {eventRole(ev.type)}
                {milestoneId && <span className="run-chip">{milestoneId}</span>}
              </div>
              <div className="content">
                <div className="event-title">{eventTitle(ev)}</div>
                <div className="event-type-label">{ev.type}</div>
                {shouldShowDetails(ev) && (
                  <details className="event-details">
                    <summary>details</summary>
                    <pre>{JSON.stringify(ev.data, null, 2)}</pre>
                  </details>
                )}
              </div>
              <div className="ts">{formatTs(ev.ts)}</div>
            </div>
          );
        })}
        <div ref={messagesEndRef} />
      </div>

      <div className="chat-composer">
        <div className="composer-row">
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                handleSend();
              }
            }}
            placeholder="Describe what to build or fix..."
            rows={2}
          />
        </div>
        <div className="composer-row" style={{ justifyContent: "space-between" }}>
          <ModelSelect
            className="model-select"
            value={model}
            models={models}
            onChange={setModel}
          />
          <button
            className="primary"
            disabled={sending || !input.trim()}
            onClick={handleSend}
          >
            {sending ? "Sending..." : "Send & Run"}
          </button>
        </div>
      </div>
    </div>
  );
}
