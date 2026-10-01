import { useEffect, useMemo, useRef, useState } from "react";
import type {
  ChatMode,
  DecisionAction,
  PendingDecision,
  ReviewFixMode,
  Session,
  Message,
  ModelChoice,
  SSEEvent,
} from "../api/types";
import { api } from "../api/client";
import { ModelSelect } from "./ModelSelect";
import { PersonaTurn } from "./PersonaTurn";
import { DecisionBanner } from "./DecisionBanner";
import { Markdown } from "./Markdown";
import { useModels } from "../hooks";
import { buildChatItems, formatTs, missionStatusLabel } from "../hooks/useChatTurns";

interface Props {
  session: Session | null;
  messages: Message[];
  sending: boolean;
  connected: boolean;
  events: SSEEvent[];
  decision: PendingDecision | null;
  resolvingDecision: boolean;
  onSend: (
    content: string,
    triggerRun: boolean,
    model?: string,
    reviewFixMode?: ReviewFixMode,
    chatMode?: ChatMode,
  ) => void;
  onResolveDecision: (action: DecisionAction) => void;
  onCancelRun?: () => void;
}

const MODE_COPY: Record<ChatMode, { placeholder: string; button: string; empty: string }> = {
  ask: {
    placeholder: "Ask about the project, preferences, or a completed build…",
    button: "Send",
    empty: "Ask mode — talk through what to build. Switch to Plan or Build when you are ready.",
  },
  plan: {
    placeholder: "Describe the mission, or type changes to the draft plan…",
    button: "Update plan",
    empty: "Plan mode — the orchestrator drafts milestones. Approve to build, or send changes.",
  },
  build: {
    placeholder: "Describe what to build or fix…",
    button: "Send & Run",
    empty: "No messages yet. Send a request below to start a mission.",
  },
};

function isExitCommand(value: string): boolean {
  const trimmed = value.trim().toLowerCase();
  return trimmed === "/exit" || trimmed.startsWith("/exit ");
}

export function ChatPanel({
  session,
  messages,
  sending,
  connected,
  events,
  decision,
  resolvingDecision,
  onSend,
  onResolveDecision,
  onCancelRun,
}: Props) {
  const [input, setInput] = useState("");
  const [model, setModel] = useState<ModelChoice>("auto");
  const [reviewFixMode, setReviewFixMode] = useState<ReviewFixMode>("ask");
  const [chatMode, setChatMode] = useState<ChatMode>("ask");
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const { models } = useModels();

  const contextLength = useMemo(() => {
    const selected = models.find((entry) => entry.key === model);
    const local = models.find((entry) => entry.adapter === "llamacpp_qwen" || entry.key === "local");
    const auto = models.find((entry) => entry.key === "auto");
    return selected?.context_length ?? local?.context_length ?? auto?.context_length ?? null;
  }, [models, model]);

  const feed = useMemo(
    () => buildChatItems(messages, events),
    [messages, events],
  );

  useEffect(() => {
    if (!session) return;
    const next = (session.selected_model as ModelChoice) || "auto";
    const info = models.find((entry) => entry.key === next);
    if (next !== "auto" && (!info || info.enabled === false || !info.available)) {
      setModel("auto");
      return;
    }
    setModel(next);
  }, [session, models]);

  useEffect(() => {
    if (!session) return;
    const next = session.chat_mode;
    if (next === "ask" || next === "plan" || next === "build") {
      setChatMode(next);
    }
  }, [session]);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [feed]);

  const persistMode = (next: ChatMode) => {
    setChatMode(next);
    if (session) {
      void api.patchSession(session.session_id, { chat_mode: next } as Partial<Session>);
    }
  };

  const handleSend = () => {
    const trimmed = input.trim();
    if (!trimmed) return;
    if (isExitCommand(trimmed)) {
      setInput("");
      onCancelRun?.();
      return;
    }
    if (sending) return;
    onSend(trimmed, true, model, reviewFixMode, chatMode);
    setInput("");
  };

  if (!session) {
    return (
      <div className="panel">
        <div className="empty-state">Select or create a session to start chatting.</div>
      </div>
    );
  }

  const copy = MODE_COPY[chatMode];

  return (
    <div className="panel chat-container">
      <div className="panel-header">
        <span title={session.workspace?.path || session.workspace_root}>
          {session.title}
          <span style={{ marginLeft: 8, fontSize: 10, color: "var(--text-muted)" }}>
            {session.workspace?.kind === "external" ? "external" : "managed"}
            {session.workspace?.access_mode === "read_only" ? " · read only" : ""}
          </span>
        </span>
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
            {copy.empty}
          </div>
        )}
        {feed.map((item) => {
          if (item.kind === "user") {
            return (
              <div key={item.id} className="message user">
                <div className="role">You</div>
                <div className="content">{item.content}</div>
                <div className="ts">{formatTs(item.ts)}</div>
              </div>
            );
          }
          if (item.kind === "system") {
            return (
              <div key={item.id} className="message system-message">
                <div className="content"><Markdown text={item.content} /></div>
                <div className="ts">{formatTs(item.ts)}</div>
              </div>
            );
          }
          if (item.kind === "mission_summary") {
            return (
              <div key={item.id} className={`message mission-summary mission-${item.status}`}>
                <div className="role persona-role">
                  <span>Mission recap</span>
                  <span className={`mission-status-badge status-${item.status}`}>
                    {missionStatusLabel(item.status)}
                  </span>
                </div>
                <div className="mission-summary-body">
                  <Markdown text={item.content} />
                </div>
                <div className="ts">{formatTs(item.ts)}</div>
              </div>
            );
          }
          return (
            <PersonaTurn
              key={item.id}
              turn={item}
              contextLength={contextLength}
            />
          );
        })}
        <div ref={messagesEndRef} />
      </div>

      {decision && (
        <DecisionBanner
          decision={decision}
          resolving={resolvingDecision}
          onResolve={onResolveDecision}
        />
      )}

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
            placeholder={copy.placeholder}
            rows={2}
          />
        </div>
        <div className="composer-row" style={{ justifyContent: "space-between" }}>
          <span style={{ display: "flex", alignItems: "center", gap: 6, flexWrap: "wrap" }}>
            <select
              className="chat-mode-select"
              value={chatMode}
              onChange={(e) => persistMode(e.target.value as ChatMode)}
              aria-label="Chat mode"
              title="Ask: converse only. Plan: orchestrator draft. Build: full mission."
            >
              <option value="ask">Ask</option>
              <option value="plan">Plan</option>
              <option value="build">Build</option>
            </select>
            <ModelSelect
              className="model-select"
              value={model}
              models={models}
              onChange={setModel}
            />
            {chatMode === "build" && (
              <select
                value={reviewFixMode}
                onChange={(e) => setReviewFixMode(e.target.value as ReviewFixMode)}
                aria-label="Review fix mode"
                title="Ask: pause when review finds defects. Auto: apply fixes without asking."
                style={{ fontSize: 12 }}
              >
                <option value="ask">Review fixes: ask me</option>
                <option value="auto">Review fixes: auto-apply</option>
              </select>
            )}
          </span>
          <button
            className="primary"
            disabled={
              !input.trim()
              || (sending && !isExitCommand(input))
            }
            onClick={handleSend}
          >
            {sending && isExitCommand(input)
              ? "Stop"
              : sending
                ? "Sending..."
                : copy.button}
          </button>
        </div>
      </div>
    </div>
  );
}
