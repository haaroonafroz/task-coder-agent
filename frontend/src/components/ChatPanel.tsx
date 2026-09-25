import { useEffect, useMemo, useRef, useState } from "react";
import type {
  DecisionAction,
  PendingDecision,
  ReviewFixMode,
  Session,
  Message,
  ModelChoice,
  SSEEvent,
} from "../api/types";
import { ModelSelect } from "./ModelSelect";
import { PersonaTurn } from "./PersonaTurn";
import { DecisionBanner } from "./DecisionBanner";
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
  ) => void;
  onResolveDecision: (action: DecisionAction) => void;
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
}: Props) {
  const [input, setInput] = useState("");
  const [model, setModel] = useState<ModelChoice>("auto");
  const [reviewFixMode, setReviewFixMode] = useState<ReviewFixMode>("ask");
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
    if (session) {
      setModel((session.selected_model as ModelChoice) || "auto");
    }
  }, [session]);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [feed]);

  const handleSend = () => {
    if (!input.trim() || sending) return;
    onSend(input.trim(), true, model, reviewFixMode);
    setInput("");
  };

  if (!session) {
    return (
      <div className="panel">
        <div className="empty-state">Select or create a session to start chatting.</div>
      </div>
    );
  }

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
            No messages yet. Send a request below to start a mission.
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
                <div className="content">{item.content}</div>
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
                <pre className="mission-summary-body">{item.content}</pre>
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
            placeholder="Describe what to build or fix..."
            rows={2}
          />
        </div>
        <div className="composer-row" style={{ justifyContent: "space-between" }}>
          <span style={{ display: "flex", alignItems: "center", gap: 6 }}>
            <ModelSelect
              className="model-select"
              value={model}
              models={models}
              onChange={setModel}
            />
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
          </span>
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
